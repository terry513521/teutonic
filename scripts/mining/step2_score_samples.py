#!/usr/bin/env python3
"""Step 2: pull training shards and score sample sequences with the king."""
from __future__ import annotations

import argparse
from pathlib import Path

from challenger_step_lib import read_json, score_samples, write_json
from train_challenger import download_shard, fetch_manifest, load_shard, log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/root/teutonic-mining/work",
                    help="Pipeline work directory")
    ap.add_argument("--king-dir", default="",
                    help="King model dir; defaults to king_dir in <work>/king.json or <work>/king")
    ap.add_argument("--n-shards", type=int, default=2,
                    help="Number of training dataset shards to download")
    ap.add_argument("--shard-start", type=int, default=0,
                    help="Index of first training shard")
    ap.add_argument("--eval-shard", type=int, default=10,
                    help="Reserved held-out shard index; must not overlap training shards")
    ap.add_argument("--n-score", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--scored-out", default="",
                    help="Output scored JSONL; defaults to <work>/scored_samples.jsonl")
    ap.add_argument("--summary-out", default="",
                    help="Output summary JSON; defaults to <work>/score_summary.json")
    args = ap.parse_args()

    work = Path(args.work)
    cache = work / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    scored_out = Path(args.scored_out) if args.scored_out else work / "scored_samples.jsonl"
    summary_out = Path(args.summary_out) if args.summary_out else work / "score_summary.json"

    if args.king_dir:
        king_dir = Path(args.king_dir)
    else:
        meta_path = work / "king.json"
        king_dir = Path(read_json(meta_path)["king_dir"]) if meta_path.exists() else work / "king"

    train_shard_idxs = list(range(args.shard_start, args.shard_start + args.n_shards))
    if args.eval_shard in train_shard_idxs:
        raise ValueError("eval_shard cannot overlap training shards")

    manifest = fetch_manifest(cache)
    shards = []
    shard_records = []
    for idx in train_shard_idxs:
        key = manifest["shards"][idx]["key"]
        path = cache / Path(key).name
        download_shard(key, path)
        arr, _ = load_shard(path)
        log.info("loaded training shard %d: %d sequences", idx, len(arr))
        shards.append(arr)
        shard_records.append({"idx": idx, "key": key, "path": str(path), "n_sequences": len(arr)})

    summary = score_samples(str(king_dir), shards, args.n_score, args.seed, args.device, scored_out)
    summary.update({
        "king_dir": str(king_dir),
        "train_shards": shard_records,
        "eval_shard": args.eval_shard,
        "seed": args.seed,
    })
    write_json(summary_out, summary)
    log.info("step2 complete: scored=%s summary=%s", scored_out, summary_out)


if __name__ == "__main__":
    main()
