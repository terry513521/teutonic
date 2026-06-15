#!/usr/bin/env python3
"""Select sample sets from cached Teutonic dataset shards.

The input shards are .npy files containing flat uint32 token IDs. This script
reshapes each shard into (n_samples, seq_len), samples rows, and writes one .npy
array per dataset source.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


DEFAULT_CONFIG = Path(__file__).with_name("million_sample_datasets.json")
DEFAULT_CACHE = Path("/workspace/datasets")
DEFAULT_OUT_DIR = Path("/workspace/teutonic-mining/selected_samples")
DEFAULT_SUMMARY = Path("/workspace/teutonic-mining/selected_samples/selected_1000000_summary.json")

def generated_seed() -> int:
    return random.SystemRandom().randint(101, 2**32 - 1)


def load_specs(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("datasets", data.get("items", []))
    if not isinstance(data, list) or not data:
        raise ValueError("config must be a non-empty list")
    return data


def allocate_counts(total: int, ratios: list[float]) -> list[int]:
    if total <= 0:
        raise ValueError("--total must be positive")
    ratio_sum = sum(ratios)
    if ratio_sum <= 0:
        raise ValueError("ratios must sum to a positive number")

    normalized = [ratio / ratio_sum for ratio in ratios]
    raw = [total * ratio for ratio in normalized]
    counts = [int(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(raw)), key=lambda idx: raw[idx] - counts[idx], reverse=True)
    for idx in order[:remainder]:
        counts[idx] += 1
    return counts


def load_shard(path: Path, seq_len: int) -> np.ndarray:
    flat = np.load(path, mmap_mode="r")
    if flat.ndim != 1:
        flat = flat.reshape(-1)
    usable = (flat.shape[0] // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"shard has no complete samples for seq_len={seq_len}: {path}")
    return flat[:usable].reshape(-1, seq_len)


def cached_shards(dataset_cache: Path) -> list[Path]:
    shards_dir = dataset_cache / "shards"
    if not shards_dir.is_dir():
        shards_dir = dataset_cache
    if not shards_dir.is_dir():
        return []
    return sorted(
        path
        for path in shards_dir.glob("*.npy")
        if path.is_file() and path.stat().st_size > 1024
    )


def selected_indices_by_shard(
    rng: np.random.Generator,
    shard_lengths: list[int],
    target: int,
    replace: bool = False,
) -> list[np.ndarray]:
    available = sum(shard_lengths)
    if target > available and not replace:
        raise ValueError(f"requested {target} samples, but only {available} cached samples are available")

    global_indices = np.sort(rng.choice(available, size=target, replace=replace))
    offsets = np.cumsum([0, *shard_lengths])
    selected: list[np.ndarray] = []
    for shard_idx, length in enumerate(shard_lengths):
        start = offsets[shard_idx]
        end = offsets[shard_idx + 1]
        left = np.searchsorted(global_indices, start, side="left")
        right = np.searchsorted(global_indices, end, side="left")
        selected.append((global_indices[left:right] - start).astype(np.int64, copy=False))
    return selected


def write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(summary, indent=2) + "\n")
    tmp.replace(path)


def dataset_output_path(out_dir: Path, total: int, name: str) -> Path:
    safe_name = name.replace("/", "_")
    return out_dir / f"selected_{total}_{safe_name}.npy"


def main() -> None:
    parser = argparse.ArgumentParser(description="Select 1,000,000 weighted samples from cached shards")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Dataset ratio JSON config")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE), help="Dataset cache root")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Directory for per-dataset .npy outputs")
    parser.add_argument("--out", default="", help="Optional combined selected .npy output")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY), help="Selection summary JSON output")
    parser.add_argument("--total", type=int, default=1_000_000, help="Total samples to select")
    parser.add_argument("--samples-per-dataset", type=int, default=0,
                        help="Select this many samples from each datasource; ignores ratios for counts")
    parser.add_argument("--seq-len", type=int, default=2048, help="Token sequence length per sample")
    parser.add_argument("--seed", type=int, default=None, help="Random seed; omitted chooses >100")
    parser.add_argument("--allow-replacement", action="store_true",
                        help="Allow duplicate rows when a datasource has fewer cached samples than requested")
    parser.add_argument("--dry-run", action="store_true", help="Validate counts and cached capacity only")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files")
    args = parser.parse_args()

    config_path = Path(args.config)
    cache_root = Path(args.cache)
    out_dir = Path(args.out_dir)
    combined_out_path = Path(args.out) if args.out else None
    summary_path = Path(args.summary)
    seed = args.seed if args.seed is not None else generated_seed()
    rng = np.random.default_rng(seed)

    if args.seq_len <= 0:
        raise ValueError("--seq-len must be positive")
    if args.samples_per_dataset < 0:
        raise ValueError("--samples-per-dataset cannot be negative")
    if combined_out_path and combined_out_path.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"output exists; pass --overwrite to replace: {combined_out_path}")

    specs = load_specs(config_path)
    ratios = [float(spec["ratio"]) for spec in specs]
    if args.samples_per_dataset:
        counts = [args.samples_per_dataset for _spec in specs]
    else:
        counts = allocate_counts(args.total, ratios)
    requested_total = sum(counts)

    summary = {
        "config": str(config_path),
        "cache": str(cache_root),
        "out_dir": str(out_dir),
        "combined_out": str(combined_out_path) if combined_out_path else None,
        "total": requested_total,
        "ratio_total": args.total if not args.samples_per_dataset else None,
        "samples_per_dataset": args.samples_per_dataset or None,
        "seq_len": args.seq_len,
        "seed": seed,
        "allow_replacement": args.allow_replacement,
        "dry_run": args.dry_run,
        "datasets": [],
    }
    write_summary(summary_path, summary)

    combined_selected = None
    created_paths: list[Path] = []
    if combined_out_path is not None and not args.dry_run:
        combined_out_path.parent.mkdir(parents=True, exist_ok=True)
        combined_selected = np.lib.format.open_memmap(
            combined_out_path,
            mode="w+",
            dtype=np.uint32,
            shape=(requested_total, args.seq_len),
        )
        created_paths.append(combined_out_path)

    cursor = 0
    try:
        for spec, target in zip(specs, counts):
            name = spec["name"]
            dataset_out_path = dataset_output_path(out_dir, target, name)
            if dataset_out_path.exists() and not args.overwrite and not args.dry_run:
                raise FileExistsError(f"output exists; pass --overwrite to replace: {dataset_out_path}")
            paths = cached_shards(cache_root / name)
            if not paths:
                raise ValueError(f"no cached .npy shards found for {name} under {cache_root / name / 'shards'}")

            lengths = [len(load_shard(path, args.seq_len)) for path in paths]
            available_samples = sum(lengths)
            use_replacement = target > available_samples
            if use_replacement and not args.allow_replacement:
                raise ValueError(
                    f"{name}: requested {target} samples, but only {available_samples} cached samples "
                    "are available. Download more shards or pass --allow-replacement to allow duplicates."
                )
            picks = selected_indices_by_shard(rng, lengths, target, replace=use_replacement)
            start = cursor
            dataset_cursor = 0
            shard_records = []
            dataset_selected = None
            if not args.dry_run:
                out_dir.mkdir(parents=True, exist_ok=True)
                dataset_selected = np.lib.format.open_memmap(
                    dataset_out_path,
                    mode="w+",
                    dtype=np.uint32,
                    shape=(target, args.seq_len),
                )
                created_paths.append(dataset_out_path)

            print(
                f"[select] {name}: target={target} cached_shards={len(paths)} "
                f"replacement={use_replacement} out={dataset_out_path}",
                flush=True,
            )
            for path, length, local_indices in zip(paths, lengths, picks):
                if len(local_indices) == 0:
                    continue
                end = cursor + len(local_indices)
                dataset_end = dataset_cursor + len(local_indices)
                if dataset_selected is not None:
                    shard = load_shard(path, args.seq_len)
                    rows = shard[local_indices]
                    dataset_selected[dataset_cursor:dataset_end] = rows
                    if combined_selected is not None:
                        combined_selected[cursor:end] = rows
                dataset_cursor = dataset_end
                cursor = end
                shard_records.append({
                    "path": str(path),
                    "available_samples": int(length),
                    "selected_samples": int(len(local_indices)),
                    "first_selected_indices": local_indices[:10].astype(int).tolist(),
                })

            dataset_summary = {
                "name": name,
                "ratio": float(spec["ratio"]),
                "target_samples": int(target),
                "output_path": str(dataset_out_path),
                "output_start": int(start),
                "output_end": int(cursor),
                "cached_shards": len(paths),
                "available_samples": int(available_samples),
                "sampling_with_replacement": use_replacement,
                "shards": shard_records,
            }
            summary["datasets"].append(dataset_summary)
            write_summary(summary_path, summary)
            if dataset_selected is not None:
                dataset_selected.flush()

        if cursor != requested_total:
            raise RuntimeError(f"internal error: wrote {cursor} samples, expected {requested_total}")
        if combined_selected is not None:
            combined_selected.flush()
    except Exception:
        for path in created_paths:
            if path.exists():
                path.unlink()
        raise

    if args.dry_run:
        print("[select] dry run complete; no .npy output written", flush=True)
    else:
        print(f"[select] complete: per-dataset outputs in {out_dir}", flush=True)
        if combined_out_path is not None:
            print(f"[select] combined: {combined_out_path}", flush=True)
    print(f"[select] summary:  {summary_path}", flush=True)


if __name__ == "__main__":
    main()
