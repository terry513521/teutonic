#!/usr/bin/env python3
"""Download random dataset shards listed in run/datasets.json.

Only downloads shard files. It does not score, load, or train on them.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random
import shutil
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


def resolve_downloader(preference: str) -> str:
    if preference == "auto":
        return "aria2c" if shutil.which("aria2c") else "curl"
    if preference == "aria2c" and not shutil.which("aria2c"):
        raise RuntimeError("aria2c requested but not found. Install it or use --downloader curl.")
    if preference == "curl" and not shutil.which("curl"):
        raise RuntimeError("curl requested but not found")
    return preference


def download_shard(
    root: str,
    shard_key: str,
    out: Path,
    downloader: str,
    connections: int,
    retries: int,
) -> str:
    tmp = out.with_suffix(out.suffix + ".tmp")
    if out.exists() and out.stat().st_size > 1024:
        print(f"[download] cached {out} ({out.stat().st_size / 1e9:.2f} GB)", flush=True)
        return "cached"
    if out.exists():
        print(f"[download] removing incomplete shard {out}", flush=True)
        out.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)
    url = f"{root}/{shard_key}"
    resume_note = " resume" if tmp.exists() and tmp.stat().st_size > 0 else ""
    print(f"[download] {url} -> {out} [{downloader}{resume_note}]", flush=True)
    try:
        if downloader == "aria2c":
            subprocess.check_call([
                "aria2c",
                "-c",
                "-x", str(connections),
                "-s", str(connections),
                "-k", "1M",
                "--file-allocation=none",
                "--disk-cache=256M",
                "--max-tries", str(retries),
                "--retry-wait", "2",
                "--summary-interval", "30",
                "--allow-overwrite=true",
                "--auto-file-renaming=false",
                "-d", str(out.parent),
                "-o", tmp.name,
                url,
            ])
        else:
            subprocess.check_call([
                "curl",
                "-fL",
                "-C", "-",
                "--retry", str(retries),
                "--retry-delay", "2",
                "--connect-timeout", "30",
                "--speed-time", "60",
                "--speed-limit", "1024",
                "-o", str(tmp),
                url,
            ])
        tmp.replace(out)
    finally:
        # Keep partial .tmp files so the next run can resume instead of starting over.
        pass
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
    parser.add_argument("--jobs", type=int, default=4,
                        help="Number of shard downloads to run concurrently")
    parser.add_argument("--downloader", choices=("auto", "curl", "aria2c"), default="auto",
                        help="Downloader backend. auto uses aria2c when installed, otherwise curl")
    parser.add_argument("--connections", type=int, default=8,
                        help="Connections per file when using aria2c")
    parser.add_argument("--retries", type=int, default=10,
                        help="Retry count for shard downloads")
    args = parser.parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")
    if args.connections <= 0:
        raise ValueError("--connections must be positive")
    if args.retries <= 0:
        raise ValueError("--retries must be positive")

    downloader = resolve_downloader(args.downloader)

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
        "jobs": args.jobs,
        "downloader": downloader,
        "connections": args.connections if downloader == "aria2c" else None,
        "retries": args.retries,
        "datasets": [],
    }
    write_summary(summary_path, summary)
    print(
        f"[download] seed={seed} jobs={args.jobs} downloader={downloader}",
        flush=True,
    )

    tasks = []
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
            tasks.append({
                "dataset_summary": dataset_summary,
                "root": manifest["_root"],
                "idx": idx,
                "key": key,
                "out": out,
            })

    def run_task(task: dict) -> dict:
        out = task["out"]
        status = download_shard(
            task["root"],
            task["key"],
            out,
            downloader,
            args.connections,
            args.retries,
        )
        return {
            "idx": task["idx"],
            "key": task["key"],
            "path": str(out),
            "status": status,
            "size_bytes": out.stat().st_size if out.exists() else 0,
            "dataset_summary": task["dataset_summary"],
        }

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_task, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            dataset_summary = result.pop("dataset_summary")
            dataset_summary["shards"].append(result)
            dataset_summary["shards"].sort(key=lambda row: row["idx"])
            write_summary(summary_path, summary)

    print(f"[download] complete; summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
