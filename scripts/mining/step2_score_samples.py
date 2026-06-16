#!/usr/bin/env python3
"""Step 2: load cached weighted multi-dataset shards and score samples with the king."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import shutil
from pathlib import Path
from urllib.parse import urlparse

from challenger_step_lib import (
    default_scoring_devices,
    load_cached_shard,
    read_json,
    score_samples,
    write_json,
)
from train_challenger import (
    fetch_manifest_url,
    log,
    sha256_dir,
)


DEFAULT_KING_URL = ""
DEFAULT_KING_REVISION = "main"
MODEL_ALLOW_PATTERNS = [
    "*.safetensors",
    "*.bin",
    "*.json",
    "*.py",
    "tokenizer*",
    "special_tokens*",
    "*.model",
    "*.txt",
]

DEFAULT_MIN_FREE_GB = 5.0
DEFAULT_DATASETS_CONFIG = Path(__file__).parents[2] / "run" / "datasets.json"


def repo_from_hf_link(model: str) -> str:
    model = model.strip()
    if not model:
        raise ValueError("model URL/repo cannot be empty")

    if "://" not in model:
        return model.removeprefix("models/").strip("/")

    parsed = urlparse(model)
    if parsed.netloc not in {"huggingface.co", "www.huggingface.co"}:
        raise ValueError(f"expected a huggingface.co model URL, got {model!r}")

    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if parts and parts[0] == "models":
        parts = parts[1:]
    if len(parts) < 2:
        raise ValueError(f"could not parse Hugging Face model repo from {model!r}")
    return "/".join(parts[:2])


def has_transformers_model_files(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    return any(path.glob(pattern) for pattern in ("*.safetensors", "*.bin"))


def require_free_space(path: Path, min_free_gb: float) -> None:
    if min_free_gb <= 0:
        return
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    free_gb = usage.free / 1e9
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"not enough free disk under {path}: {free_gb:.2f} GB available, "
            f"need at least {min_free_gb:.2f} GB. Step2 writes token-heavy JSONL "
            "and summary files; free space by deleting old merged models or dataset caches, "
            "or rerun with --min-free-gb 0 to bypass this check."
        )


def snapshot_download_model(**kwargs) -> None:
    try:
        snapshot_download = importlib.import_module("huggingface_hub").snapshot_download
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "huggingface-hub is required only when step2 needs to download a missing king "
            "model. Run /workspace/run/setup_finetune_env.sh or install huggingface-hub, "
            "or run step1 first so --king-dir already contains a complete model."
        ) from exc
    snapshot_download(**kwargs)


def king_repo_from_meta(meta: dict) -> str:
    dashboard_king = meta.get("dashboard_king") or {}
    return (
        meta.get("king_repo")
        or meta.get("model_repo")
        or meta.get("hf_repo")
        or dashboard_king.get("model_repo")
        or dashboard_king.get("hf_repo")
        or ""
    )


def king_revision_from_meta(meta: dict) -> str:
    dashboard_king = meta.get("dashboard_king") or {}
    return (
        meta.get("king_revision")
        or meta.get("revision")
        or dashboard_king.get("king_revision")
        or dashboard_king.get("king_digest")
        or dashboard_king.get("revision")
        or ""
    )


def candidate_king_meta_paths(work: Path) -> list[Path]:
    paths = [
        work / "king.json",
        work.parent / "work" / "king.json",
        Path("/workspace/teutonic-mining/work/king.json"),
    ]
    unique = []
    seen = set()
    for path in paths:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def resolve_king_dir(
    work: Path,
    king_dir_arg: str,
    repo_arg: str,
    model_url: str,
    revision_arg: str,
    hf_token: str,
    download_workers: int,
) -> tuple[Path, dict]:
    meta_path = work / "king.json"
    meta = {}
    for candidate in candidate_king_meta_paths(work):
        if candidate.exists():
            meta_path = candidate
            meta = read_json(candidate)
            break

    king_dir = Path(king_dir_arg) if king_dir_arg else Path(meta.get("king_dir", work / "king"))

    if has_transformers_model_files(king_dir):
        if not meta:
            meta.update({
                "king_hash": sha256_dir(king_dir),
                "king_dir": str(king_dir),
            })
        return king_dir, meta

    repo = repo_arg.strip() or king_repo_from_meta(meta)
    revision = revision_arg.strip() or king_revision_from_meta(meta)
    if not repo:
        if model_url.strip():
            repo = repo_from_hf_link(model_url)
            revision = revision or DEFAULT_KING_REVISION
        else:
            raise FileNotFoundError(
                f"king model dir is missing/incomplete: {king_dir}. "
                "Run /workspace/run/step1_download_king.sh first, or pass --repo/--model-url "
                "and --revision explicitly."
            )

    log.info(
        "king model dir missing/incomplete, downloading repo=%s revision=%s -> %s",
        repo,
        revision or "HEAD",
        king_dir,
    )
    king_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download_model(
        repo_id=repo,
        revision=revision or None,
        local_dir=str(king_dir),
        allow_patterns=MODEL_ALLOW_PATTERNS,
        ignore_patterns="optimizer*",
        token=hf_token or None,
        max_workers=download_workers,
    )

    if not has_transformers_model_files(king_dir):
        raise FileNotFoundError(f"downloaded king model is still incomplete: {king_dir}")

    meta.update({
        "king_repo": repo,
        "king_revision": revision,
        "king_hash": sha256_dir(king_dir),
        "king_dir": str(king_dir),
        "dashboard_king": meta.get("dashboard_king") or {
            "model_repo": repo,
            "hf_repo": repo,
            "king_revision": revision,
            "king_digest": revision,
            "source": model_url,
        },
    })
    write_json(meta_path, meta)
    return king_dir, meta


def load_dataset_specs(datasets_config: str) -> list[dict]:
    if not datasets_config:
        raise ValueError("--datasets-config is required; use run/datasets.json")
    path = Path(datasets_config)
    data = json.loads(path.read_text()) if path.exists() else json.loads(datasets_config)
    if isinstance(data, dict):
        data = data.get("datasets", data.get("items", []))
    if not isinstance(data, list) or not data:
        raise ValueError("datasets config must be a non-empty list")
    specs = [dict(spec) for spec in data]

    missing_weights = [spec for spec in specs if "weight" not in spec]
    if missing_weights:
        totals = [target_samples_for_spec(spec, fallback_total=1) for spec in specs]
        total = sum(totals)
        if total <= 0:
            raise ValueError("dataset sample totals must be positive to infer weights")
        for spec, count in zip(specs, totals):
            spec.setdefault("weight", count / total)
    weight_total = sum(float(spec.get("weight", 0.0)) for spec in specs)
    if weight_total <= 0:
        raise ValueError("dataset weights must sum to a positive value")
    for spec in specs:
        spec["weight"] = float(spec["weight"]) / weight_total
    return specs


def target_samples_for_spec(spec: dict, fallback_total: int | None = None) -> int:
    if "samples" in spec:
        return int(spec["samples"])
    if "target_samples" in spec:
        return int(spec["target_samples"])
    if "samples_per_shard" in spec:
        return int(spec.get("n_shards", 1)) * int(spec["samples_per_shard"])
    if fallback_total is not None:
        return fallback_total
    raise ValueError(
        f"dataset {spec.get('name', '<unknown>')} needs samples, target_samples, "
        "or samples_per_shard"
    )


def cached_manifest_records(dataset_cache: Path, manifest: dict) -> list[tuple[int, dict, Path]]:
    shards_dir = dataset_cache / "shards"
    if not shards_dir.is_dir():
        return []
    cached_paths = sorted(
        path
        for path in shards_dir.glob("*.npy")
        if path.is_file() and path.stat().st_size > 1024
    )
    manifest_by_name = {
        Path(shard_info["key"]).name: (idx, shard_info)
        for idx, shard_info in enumerate(manifest["shards"])
    }
    records = []
    for path in cached_paths:
        manifest_record = manifest_by_name.get(path.name)
        if manifest_record is not None:
            idx, shard_info = manifest_record
            records.append((idx, shard_info, path))
    return records


def select_cached_records(
    records: list[tuple[int, dict, Path]],
    n_shards: int,
    seed: int,
) -> list[tuple[int, dict, Path]]:
    if n_shards > len(records):
        raise ValueError(
            f"requested {n_shards} cached shards, but only {len(records)} cached shards are available"
        )
    return sorted(
        random.Random(seed).sample(records, n_shards),
        key=lambda item: item[2].name,
    )


def load_weighted_dataset_shards(
    work: Path,
    datasets: list[dict],
    sample_counts: dict[str, int],
    n_shards_per_dataset: int,
    seed: int,
    dataset_cache_root: Path | None = None,
) -> tuple[list, list[dict]]:
    shards = []
    shard_records = []

    for spec_idx, spec in enumerate(datasets):
        name = spec["name"]
        manifest_url = spec["manifest_url"]
        weight = float(spec["weight"])
        target_samples = int(sample_counts[name])
        samples_per_shard = int(spec.get("samples_per_shard", 0) or 0)
        requested_n_shards = int(spec.get("n_shards", n_shards_per_dataset))
        dataset_cache = (dataset_cache_root or (work / "cache" / "datasets")) / name
        manifest = fetch_manifest_url(dataset_cache, manifest_url)
        cached_records = cached_manifest_records(dataset_cache, manifest)
        if not cached_records:
            raise ValueError(
                f"dataset {name} has no cached shards under {dataset_cache}. "
                "Run run/download_dataset_shards.sh before scoring."
            )
        n_shards = min(requested_n_shards, len(cached_records))
        if n_shards < requested_n_shards:
            log.info(
                "dataset %s: requested %d cached shards but only %d available; using %d",
                name,
                requested_n_shards,
                len(cached_records),
                n_shards,
            )
        selected_records = select_cached_records(
            cached_records,
            n_shards,
            seed + spec_idx,
        )
        log.info(
            "dataset %s: weight=%.2f target_samples=%d samples_per_shard=%s cached_shards=%s tokenizer=%s",
            name,
            weight,
            target_samples,
            samples_per_shard or "auto",
            [path.name for _, _, path in selected_records],
            manifest.get("tokenizer"),
        )

        for manifest_shard_idx, shard_info, path in selected_records:
            key = shard_info["key"]
            log.info("using cached shard: %s", path)
            arr, _ = load_cached_shard(path)
            log.info(
                "loaded dataset %s shard %d: %d sequences",
                name,
                manifest_shard_idx,
                len(arr),
            )
            record = {
                "dataset": name,
                "dataset_weight": weight,
                "target_samples": target_samples,
                "manifest_url": manifest_url,
                "manifest_tokenizer": manifest.get("tokenizer"),
                "shard_idx": manifest_shard_idx,
                "shard_key": key,
                "path": str(path),
                "source_file": shard_info.get("source_file"),
            }
            if samples_per_shard:
                record["target_samples_per_shard"] = samples_per_shard
            shards.append(arr)
            shard_records.append(record)

    return shards, shard_records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/workspace/teutonic-mining/work",
                    help="Pipeline work directory")
    ap.add_argument("--king-dir", default="/workspace/teutonic-mining/work/king",
                    help="King model dir; defaults to king_dir in <work>/king.json or <work>/king")
    ap.add_argument("--datasets-config", default=str(DEFAULT_DATASETS_CONFIG),
                    help="Dataset JSON config; defaults to run/datasets.json")
    ap.add_argument("--n-shards-per-dataset", type=int, default=1,
                    help="Fallback cached shard count only for dataset specs without n_shards")
    ap.add_argument("--n-score", type=int, default=20000,
                    help="Total sequences to score when dataset specs do not set samples_per_shard")
    ap.add_argument("--seed", type=int, default=None,
                    help="Random seed; omitted means choose a new seed greater than 100")
    ap.add_argument(
        "--device",
        default=default_scoring_devices(),
        help="CUDA device(s) for scoring: cuda:0, 0,1,2,3, or auto/all (default: all visible GPUs)",
    )
    ap.add_argument("--per-device-batch-size", type=int, default=8,
                    help="Sequences scored per GPU per forward pass")
    ap.add_argument("--lm-head-chunk", type=int, default=int(os.environ.get("STEP2_LM_HEAD_CHUNK", "512")),
                    help="Token positions per LM-head loss chunk; higher is faster but uses more VRAM")
    ap.add_argument("--empty-cache-every", type=int, default=int(os.environ.get("STEP2_EMPTY_CACHE_EVERY", "0")),
                    help="Call torch.cuda.empty_cache every N scoring batches; 0 disables")
    ap.add_argument("--attn-implementation", default=os.environ.get("STEP2_ATTN_IMPLEMENTATION", "auto"),
                    help="Transformers attention backend for scoring: auto, flash_attention_2, sdpa, or eager")
    ap.add_argument("--model-url", default=DEFAULT_KING_URL,
                    help="Hugging Face model URL or repo to download if king is incomplete")
    ap.add_argument("--repo", default="",
                    help="Hugging Face repo id; overrides --model-url and metadata")
    ap.add_argument("--revision", default="",
                    help="Hugging Face revision; defaults to metadata or main")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""),
                    help="Hugging Face token; defaults to HF_TOKEN")
    ap.add_argument("--download-workers", type=int, default=16,
                    help="Parallel workers if step2 must download a missing/incomplete king")
    ap.add_argument("--dataset-cache", default="",
                    help="Dataset cache root; defaults to <work>/cache/datasets")
    ap.add_argument("--scored-out", default="",
                    help="Output scored JSONL; defaults to <work>/scored_samples.jsonl")
    ap.add_argument("--summary-out", default="",
                    help="Output summary JSON; defaults to <work>/score_summary.json")
    ap.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB,
                    help="Require this much free disk before scoring (0 disables check)")
    args = ap.parse_args()
    if args.seed is None:
        args.seed = random.SystemRandom().randint(101, 2**32 - 1)
        log.info("no --seed provided; generated step2 seed=%d", args.seed)

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    dataset_cache_root = Path(args.dataset_cache) if args.dataset_cache else None
    scored_out = Path(args.scored_out) if args.scored_out else work / "scored_samples.jsonl"
    summary_out = Path(args.summary_out) if args.summary_out else work / "score_summary.json"
    require_free_space(scored_out.parent, args.min_free_gb)

    king_dir, king_meta = resolve_king_dir(
        work,
        args.king_dir,
        args.repo,
        args.model_url,
        args.revision,
        args.hf_token,
        args.download_workers,
    )

    datasets = load_dataset_specs(args.datasets_config)
    explicit_counts = any(
        key in spec
        for spec in datasets
        for key in ("samples", "target_samples", "samples_per_shard")
    )
    if explicit_counts:
        counts = [target_samples_for_spec(spec) for spec in datasets]
        n_score = sum(counts)
    else:
        from train_challenger import allocate_weighted_counts

        weights = [float(spec["weight"]) for spec in datasets]
        counts = allocate_weighted_counts(args.n_score, weights)
        n_score = args.n_score
    sample_counts = {
        spec["name"]: count
        for spec, count in zip(datasets, counts)
    }
    log.info("sample allocation across datasets: %s", sample_counts)

    shards, shard_records = load_weighted_dataset_shards(
        work,
        datasets,
        sample_counts,
        args.n_shards_per_dataset,
        args.seed,
        dataset_cache_root,
    )

    summary = score_samples(
        str(king_dir),
        shards,
        n_score,
        args.seed,
        args.device,
        scored_out,
        shard_records=shard_records,
        per_device_batch_size=args.per_device_batch_size,
        attn_implementation=args.attn_implementation,
        lm_head_chunk=args.lm_head_chunk,
        empty_cache_every=args.empty_cache_every,
    )
    summary.update({
        "king_dir": str(king_dir),
        "king_repo": king_meta.get("king_repo"),
        "king_revision": king_meta.get("king_revision"),
        "king_hash": king_meta.get("king_hash"),
        "datasets_config": datasets,
        "dataset_cache": str(dataset_cache_root or (work / "cache" / "datasets")),
        "sample_counts": sample_counts,
        "n_score_requested": n_score,
        "per_device_batch_size": args.per_device_batch_size,
        "attn_implementation": args.attn_implementation,
        "lm_head_chunk": args.lm_head_chunk,
        "empty_cache_every": args.empty_cache_every,
        "train_shards": shard_records,
        "seed": args.seed,
    })
    write_json(summary_out, summary)
    log.info("step2 complete: scored=%s summary=%s", scored_out, summary_out)


if __name__ == "__main__":
    main()
