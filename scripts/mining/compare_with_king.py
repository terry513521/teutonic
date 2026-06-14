#!/usr/bin/env python3
"""Compare a fine-tuned challenger against the current downloaded king."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from challenger_step_lib import (
    download_king_from_hippius,
    paired_eval_datasets,
    read_json,
    write_json,
)
from step6_eval_verdict import (
    load_dataset_specs,
    load_eval_sets,
    load_train_shard_exclusions,
)
from train_challenger import allocate_weighted_counts, fetch_king, log, sha256_dir


def has_model_files(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file() and any(
        path.glob(pattern) for pattern in ("*.safetensors", "*.bin")
    )


def resolve_king_dir(work: Path, king_dir_arg: str) -> tuple[Path, dict]:
    if king_dir_arg:
        return Path(king_dir_arg), {}

    meta_path = work / "king.json"
    meta = read_json(meta_path) if meta_path.exists() else {}
    return Path(meta.get("king_dir", work / "king")), meta


def _king_identity(king: dict) -> tuple[str, str]:
    repo = king.get("model_repo") or king.get("hf_repo") or ""
    revision = king.get("king_digest") or king.get("king_revision") or king.get("revision") or ""
    return repo, revision


def _metadata_matches_king(meta: dict, repo: str, revision: str) -> bool:
    dashboard_king = meta.get("dashboard_king") or {}
    meta_repo = meta.get("king_repo") or dashboard_king.get("model_repo") or dashboard_king.get("hf_repo")
    meta_revision = (
        meta.get("king_revision")
        or dashboard_king.get("king_digest")
        or dashboard_king.get("king_revision")
        or dashboard_king.get("revision")
    )
    if repo and meta_repo != repo:
        return False
    if revision and meta_revision != revision:
        return False
    return bool(meta_repo)


def _local_current_king_candidates(king_dir: Path, metadata_out: Path) -> list[tuple[Path, Path]]:
    work = metadata_out.parent
    candidates = [
        (metadata_out, king_dir),
        (work / "king.json", work / "king"),
    ]
    seen = set()
    unique = []
    for meta_path, default_dir in candidates:
        key = (meta_path, default_dir)
        if key not in seen:
            unique.append((meta_path, default_dir))
            seen.add(key)
    return unique


def refresh_current_king(king_dir: Path, metadata_out: Path, download_workers: int) -> tuple[Path, dict]:
    """Fetch the live dashboard king and materialize it for comparison."""
    king = fetch_king()
    repo, revision = _king_identity(king)

    for meta_path, default_dir in _local_current_king_candidates(king_dir, metadata_out):
        if not meta_path.exists():
            continue
        meta = read_json(meta_path)
        candidate_dir = Path(meta.get("king_dir", default_dir))
        if _metadata_matches_king(meta, repo, revision) and has_model_files(candidate_dir):
            log.info(
                "reusing downloaded current king: repo=%s revision=%s dir=%s",
                repo,
                (revision or "HEAD")[:19],
                candidate_dir,
            )
            meta.update({
                "king_repo": repo,
                "king_revision": revision,
                "king_hash": meta.get("king_hash") or sha256_dir(candidate_dir),
                "king_dir": str(candidate_dir),
                "dashboard_king": king,
            })
            write_json(metadata_out, meta)
            return candidate_dir, meta

    repo, revision = download_king_from_hippius(king, king_dir, download_workers)
    meta = {
        "king_repo": repo,
        "king_revision": revision,
        "king_hash": sha256_dir(king_dir),
        "king_dir": str(king_dir),
        "dashboard_king": king,
    }
    write_json(metadata_out, meta)
    return king_dir, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/workspace/teutonic-mining/work",
                    help="Pipeline work directory")
    ap.add_argument("--king-dir", default="",
                    help="King model dir; defaults to <work>/current_king when refreshing")
    ap.add_argument("--refresh-current-king", dest="refresh_current_king",
                    action="store_true", default=True,
                    help="Fetch the live dashboard and re-download the current king before comparing (default)")
    ap.add_argument("--no-refresh-current-king", dest="refresh_current_king",
                    action="store_false",
                    help="Use an existing --king-dir instead of downloading the live current king")
    ap.add_argument("--king-metadata-out", default="",
                    help="Metadata JSON for the refreshed king; defaults to <work>/current_king.json")
    ap.add_argument("--download-workers", type=int, default=8,
                    help="Parallel workers for current king download")
    ap.add_argument("--challenger-dir", required=True,
                    help="Fine-tuned standalone model dir to compare against the king")
    ap.add_argument("--datasets-config", default="",
                    help="Optional JSON file/list overriding DEFAULT_DATASETS")
    ap.add_argument("--n-shards-per-dataset", type=int, default=1,
                    help="Number of eval shards to consider per dataset manifest")
    ap.add_argument("--shard-start", type=int, default=10,
                    help="First eval shard when not using --random-shards")
    ap.add_argument("--random-shards", action="store_true",
                    help="Randomly sample eval shards per dataset with --seed")
    ap.add_argument("--train-summary", default="",
                    help="Score summary JSON used to exclude train shards; defaults to <work>/score_summary.json")
    ap.add_argument("--include-train-shards", action="store_true",
                    help="Do not exclude shards used during step2 scoring")
    ap.add_argument("--n-eval", type=int, default=2000,
                    help="Total evaluation sequences across datasets")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--comparison-out", default="",
                    help="Comparison JSON; defaults to <work>/comparison.json")
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    if args.refresh_current_king:
        king_dir = Path(args.king_dir) if args.king_dir else work / "current_king"
        king_metadata_out = (
            Path(args.king_metadata_out) if args.king_metadata_out else work / "current_king.json"
        )
        king_dir, king_meta = refresh_current_king(king_dir, king_metadata_out, args.download_workers)
    else:
        king_dir, king_meta = resolve_king_dir(work, args.king_dir)
    challenger_dir = Path(args.challenger_dir)
    comparison_out = Path(args.comparison_out) if args.comparison_out else work / "comparison.json"

    if not has_model_files(king_dir):
        raise FileNotFoundError(
            f"king model dir is missing/incomplete: {king_dir}. "
            "Run /workspace/run/step1_download_king.sh first or pass --king-dir."
        )
    if not has_model_files(challenger_dir):
        raise FileNotFoundError(
            f"challenger model dir is missing/incomplete: {challenger_dir}. "
            "Pass --challenger-dir pointing at a merged standalone fine-tuned model."
        )

    datasets = load_dataset_specs(args.datasets_config)
    weights = [float(spec["weight"]) for spec in datasets]
    counts = allocate_weighted_counts(args.n_eval, weights)
    sample_counts = {
        spec["name"]: count
        for spec, count in zip(datasets, counts)
    }
    log.info("comparison eval allocation across datasets: %s", sample_counts)

    excluded_by_dataset = {}
    if not args.include_train_shards:
        excluded_by_dataset = load_train_shard_exclusions(work, args.train_summary)
        if excluded_by_dataset:
            log.info("excluding train shards from comparison eval: %s", {
                dataset: sorted(shards)
                for dataset, shards in sorted(excluded_by_dataset.items())
            })

    eval_sets = load_eval_sets(
        work,
        datasets,
        sample_counts,
        args.n_shards_per_dataset,
        args.seed,
        args.random_shards,
        args.shard_start,
        excluded_by_dataset,
    )

    verdict = paired_eval_datasets(
        str(king_dir),
        str(challenger_dir),
        eval_sets,
        args.device,
        batch_size=args.batch_size,
        n_bootstrap=args.n_bootstrap,
    )
    result = {
        "king_repo": king_meta.get("king_repo"),
        "king_revision": king_meta.get("king_revision"),
        "king_hash": king_meta.get("king_hash") or sha256_dir(king_dir),
        "challenger_hash": sha256_dir(challenger_dir),
        "king_dir": str(king_dir),
        "challenger_dir": str(challenger_dir),
        "datasets_config": datasets,
        "sample_counts": sample_counts,
        "n_eval_requested": args.n_eval,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "train_shards_excluded": {
            dataset: sorted(shards)
            for dataset, shards in sorted(excluded_by_dataset.items())
        },
        "comparison": verdict,
        "summary": {
            "challenger_better_nats_per_token": verdict["mu_hat"],
            "lower_confidence_bound": verdict["lcb"],
            "passes_teutonic_delta": verdict["accepted"],
            "avg_king_loss": verdict["avg_king_loss"],
            "avg_challenger_loss": verdict["avg_chall_loss"],
        },
        "ts": time.time(),
    }
    write_json(comparison_out, result)
    log.info(
        "comparison complete: mu_hat=%.6f lcb=%.6f accepted=%s out=%s",
        verdict["mu_hat"],
        verdict["lcb"],
        verdict["accepted"],
        comparison_out,
    )


if __name__ == "__main__":
    main()
