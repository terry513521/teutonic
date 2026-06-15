#!/usr/bin/env python3
"""Download random dataset shards listed in run/datasets.json.

Only downloads shard files. It does not score, load, or train on them.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_CONFIG = Path(__file__).with_name("datasets.json")
DEFAULT_CACHE = Path("/workspace/teutonic-mining/cache/datasets")


def random_seed_gt_100() -> int:
    return random.SystemRandom().randint(101, 2**32 - 1)


def hippius_root_from_manifest_url(manifest_url: str) -> str:
    parsed = urlparse(manifest_url)
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if not parts:
        raise ValueError(f"invalid manifest url: {manifest_url}")
    return f"{parsed.scheme}://{parsed.netloc}/{parts[0]}"


def fetch_manifest(dataset_cache: Path, manifest_url: str) -> dict:
    dataset_cache.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_cache / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            if not isinstance(manifest.get("shards"), list) or not manifest["shards"]:
                raise ValueError("cached manifest has no shards")
        except Exception:
            print(f"[download] cached manifest invalid, refetching {manifest_url}", flush=True)
            manifest_path.unlink(missing_ok=True)
    if not manifest_path.exists():
        print(f"[download] manifest {manifest_url}", flush=True)
        tmp = manifest_path.with_suffix(".json.tmp")
        try:
            subprocess.check_call(["curl", "-fsSL", "-o", str(tmp), manifest_url])
            tmp.replace(manifest_path)
        finally:
            if tmp.exists():
                tmp.unlink()
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest.get("shards"), list) or not manifest["shards"]:
        raise ValueError(f"manifest has no shards: {manifest_url}")
    manifest["_root"] = hippius_root_from_manifest_url(manifest_url)
    return manifest


def select_shards(n_shards: int, manifest_count: int, seed: int) -> list[int]:
    if n_shards > manifest_count:
        raise ValueError(f"requested {n_shards} shards, manifest only has {manifest_count}")
    return sorted(random.Random(seed).sample(range(manifest_count), n_shards))


def download_shard(root: str, shard_key: str, out: Path) -> str:
    tmp = out.with_suffix(out.suffix + ".tmp")
    if out.exists() and out.stat().st_size > 1024:
        print(f"[download] cached {out} ({out.stat().st_size / 1e9:.2f} GB)", flush=True)
        if tmp.exists():
            tmp.unlink()
        return "cached"
    if out.exists():
        print(f"[download] removing incomplete shard {out}", flush=True)
        out.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)
    if tmp.exists():
        tmp.unlink()
    url = f"{root}/{shard_key}"
    print(f"[download] {url} -> {out}", flush=True)
    try:
        subprocess.check_call(["curl", "-fL", "--retry", "5", "--retry-delay", "2", "-o", str(tmp), url])
        tmp.replace(out)
    finally:
        if tmp.exists():
            tmp.unlink()
    return "downloaded"


def write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(summary, indent=2) + "\n")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download random Teutonic dataset shards")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Dataset JSON config")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE), help="Dataset cache root")
    parser.add_argument("--seed", type=int, default=None, help="Random seed; omitted chooses >100")
    parser.add_argument("--summary", default="", help="Summary JSON path")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random_seed_gt_100()
    config_path = Path(args.config)
    cache_root = Path(args.cache)
    summary_path = Path(args.summary) if args.summary else cache_root / "random_shard_download_summary.json"

    datasets = json.loads(config_path.read_text())
    planned = []
    print("[download] validating manifests before shard downloads", flush=True)
    for spec_idx, spec in enumerate(datasets):
        name = spec["name"]
        n_shards = int(spec["n_shards"])
        dataset_cache = cache_root / name
        manifest = fetch_manifest(dataset_cache, spec["manifest_url"])
        shard_indices = select_shards(n_shards, len(manifest["shards"]), seed + spec_idx)
        planned.append((spec_idx, spec, dataset_cache, manifest, shard_indices))

    summary = {
        "config": str(config_path),
        "cache": str(cache_root),
        "seed": seed,
        "datasets": [],
    }
    write_summary(summary_path, summary)
    print(f"[download] seed={seed}", flush=True)

    for _spec_idx, spec, dataset_cache, manifest, shard_indices in planned:
        name = spec["name"]
        n_shards = int(spec["n_shards"])
        print(f"[download] {name}: selected random shards {shard_indices}", flush=True)

        dataset_summary = {
            "name": name,
            "requested_shards": n_shards,
            "selected_indices": shard_indices,
            "shards": [],
        }
        summary["datasets"].append(dataset_summary)
        write_summary(summary_path, summary)

        for idx in shard_indices:
            shard = manifest["shards"][idx]
            key = shard["key"]
            out = dataset_cache / "shards" / Path(key).name
            status = download_shard(manifest["_root"], key, out)
            dataset_summary["shards"].append({
                "idx": idx,
                "key": key,
                "path": str(out),
                "status": status,
                "size_bytes": out.stat().st_size if out.exists() else 0,
            })
            write_summary(summary_path, summary)

    print(f"[download] complete; summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
