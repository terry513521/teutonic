#!/usr/bin/env python3
"""Submit an already-uploaded Hippius challenger to the chain.

Use this when the model is already on Hippius Hub and you have its repo id
and manifest digest. It posts the same v4 reveal commitment as
submit_challenger.py, but without requiring a verdict.json from offline eval.

    v4|{repo}|sha256:{manifest_digest}|{author_hotkey}

Run this on the host where the wallet lives.

Coldkey gate: the validator rejects Hippius repos whose full id does not
contain the first 8 ss58 chars of your coldkey (case-insensitive). This
script enforces that locally before broadcasting.

Usage:
    python3 scripts/mining/submit_uploaded_challenger.py \\
      --repo "ranupthestairs/teutonic-5gb7e4b4-v8" \\
      --digest "61548ee8925ee1778751d8e7155058e67c51fa68db6b4f5ceeb3f3582cacf5ba" \\
      --wallet-name peter \\
      --hotkey 3_7 \\
      --netuid 3 \\
      --network finney
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import bittensor as bt

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_store import ModelRef, build_reveal_v4  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [submit-uploaded] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("submit_uploaded_challenger")

COLDKEY_PREFIX_LEN = 8
BARE_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def normalize_digest(digest: str) -> str:
    """Accept sha256:<hex>, hf:<hex>, or bare 64-char manifest hex."""
    value = (digest or "").strip()
    if value.startswith(("sha256:", "hf:")):
        return value
    if BARE_SHA256_RE.match(value):
        return f"sha256:{value.lower()}"
    return value


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Submit a pre-uploaded Hippius challenger to Teutonic SN3",
    )
    ap.add_argument("--repo", required=True,
                    help="Hippius repo id (e.g. namespace/teutonic-<coldkey8>-v8)")
    ap.add_argument("--digest", required=True,
                    help="OCI manifest digest (sha256:<hex> or bare 64-char hex)")
    ap.add_argument("--wallet-name", default="teutonic",
                    help="Bittensor wallet name")
    ap.add_argument("--hotkey", default="h0",
                    help="Bittensor hotkey name")
    ap.add_argument("--netuid", type=int, default=3)
    ap.add_argument("--network", default="finney")
    ap.add_argument("--blocks-until-reveal", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true",
                    help="Build and print payload without submitting on chain")
    args = ap.parse_args()

    digest = normalize_digest(args.digest)
    try:
        model_ref = ModelRef(args.repo, digest)
    except ValueError as exc:
        log.error("invalid Hippius model ref: %s", exc)
        sys.exit(2)

    wallet = bt.Wallet(name=args.wallet_name, hotkey=args.hotkey)
    log.info("wallet hotkey: %s", wallet.hotkey.ss58_address)

    coldkey_ss58 = wallet.coldkeypub.ss58_address
    expected_prefix = coldkey_ss58[:COLDKEY_PREFIX_LEN]
    if expected_prefix.lower() not in args.repo.lower():
        log.error(
            "Hippius repo '%s' does NOT contain your coldkey prefix '%s' "
            "(first %d chars of %s).\n"
            "    The validator will reject this submission with "
            "`coldkey_required` and your tx will be wasted.\n"
            "    Rename your Hippius repo or Hippius namespace so its full id "
            "contains '%s' (case-insensitive substring) anywhere — e.g.\n"
            "        %s/<chain.name>-%s-v1\n"
            "    then re-upload and rerun this script.",
            args.repo, expected_prefix, COLDKEY_PREFIX_LEN, coldkey_ss58,
            expected_prefix,
            args.repo.split("/", 1)[0] if "/" in args.repo else "<your-hippius-namespace>",
            expected_prefix,
        )
        sys.exit(6)
    log.info("coldkey gate ok: repo '%s' contains coldkey prefix '%s'",
             args.repo, expected_prefix)

    payload = build_reveal_v4(model_ref, wallet.hotkey.ss58_address)
    log.info("payload: %s", payload)

    if args.dry_run:
        log.info("[dry-run] not submitting")
        return

    sub = bt.Subtensor(network=args.network)
    try:
        meta = sub.metagraph(args.netuid)
        if wallet.hotkey.ss58_address not in meta.hotkeys:
            log.error("hotkey not registered on netuid %d", args.netuid)
            sys.exit(4)
        uid = meta.hotkeys.index(wallet.hotkey.ss58_address)
        log.info("registered as uid=%d", uid)

        revealed = sub.get_revealed_commitment_by_hotkey(args.netuid, wallet.hotkey.ss58_address)
        if revealed:
            log.warning(
                "hotkey already has %d reveal(s) on chain; validator may de-dupe this submission",
                len(revealed),
            )

        resp = sub.set_reveal_commitment(
            wallet=wallet,
            netuid=args.netuid,
            data=payload,
            blocks_until_reveal=args.blocks_until_reveal,
            wait_for_revealed_execution=False,
        )
        if resp.success:
            log.info("reveal committed: %s -- validator should pick up after reveal", resp.message)
        else:
            log.error("commitment failed: %s", resp.message)
            sys.exit(5)
    finally:
        try:
            sub.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
