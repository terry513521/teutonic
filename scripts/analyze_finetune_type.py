#!/usr/bin/env python3
"""Heuristically classify a fine-tuned checkpoint as merged LoRA or full fine-tune.

Usage:
    python scripts/analyze_finetune_type.py --base /path/to/original --tuned /path/to/fine-tuned

The script compares matching safetensors weights in two Transformers-style model
directories. It cannot prove provenance, but it can show whether weight changes
are sparse and concentrated in typical LoRA target modules or spread broadly.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open


DEFAULT_LORA_HINTS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "dense",
    "fc",
)


def load_weight_map(model_dir: Path) -> dict[str, Path]:
    index = model_dir / "model.safetensors.index.json"
    single = model_dir / "model.safetensors"

    if index.exists():
        data = json.loads(index.read_text())
        weight_map = data.get("weight_map", {})
        if not weight_map:
            raise ValueError(f"{index} does not contain a weight_map")
        return {name: model_dir / shard for name, shard in weight_map.items()}

    if single.exists():
        with safe_open(single, framework="pt", device="cpu") as f:
            return {name: single for name in f.keys()}

    raise FileNotFoundError(
        f"No safetensors checkpoint found in {model_dir}. Expected "
        "model.safetensors or model.safetensors.index.json."
    )


def tensor_stats(
    base_file: Path,
    tuned_file: Path,
    name: str,
    *,
    atol: float,
    chunk_size: int,
) -> dict:
    with safe_open(base_file, framework="pt", device="cpu") as base_f, safe_open(
        tuned_file, framework="pt", device="cpu"
    ) as tuned_f:
        base = base_f.get_tensor(name)
        tuned = tuned_f.get_tensor(name)

    if base.shape != tuned.shape:
        return {
            "name": name,
            "shape": tuple(base.shape),
            "changed": True,
            "shape_mismatch": True,
            "max_abs": math.inf,
            "mean_abs": math.inf,
            "rel_l2": math.inf,
            "changed_frac": 1.0,
            "numel": base.numel(),
        }

    base_flat = base.reshape(-1)
    tuned_flat = tuned.reshape(-1)
    numel = base_flat.numel()
    max_abs = 0.0
    sum_abs = 0.0
    changed_count = 0
    diff_sq = 0.0
    base_sq = 0.0

    for start in range(0, numel, chunk_size):
        end = min(start + chunk_size, numel)
        b = base_flat[start:end].float()
        t = tuned_flat[start:end].float()
        diff = t - b
        abs_diff = diff.abs()
        max_abs = max(max_abs, float(abs_diff.max().item()) if abs_diff.numel() else 0.0)
        sum_abs += float(abs_diff.sum().item())
        changed_count += int((abs_diff > atol).sum().item())
        diff_sq += float((diff * diff).sum().item())
        base_sq += float((b * b).sum().item())

    mean_abs = sum_abs / max(numel, 1)
    rel_l2 = math.sqrt(diff_sq) / max(math.sqrt(base_sq), 1e-12)

    return {
        "name": name,
        "shape": tuple(base.shape),
        "changed": changed_count > 0,
        "shape_mismatch": False,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rel_l2": rel_l2,
        "changed_frac": changed_count / max(numel, 1),
        "numel": numel,
    }


def module_bucket(name: str) -> str:
    parts = name.split(".")
    if len(parts) >= 2 and parts[-1] in {"weight", "bias"}:
        return parts[-2]
    return parts[-1]


def is_lora_target_like(name: str, hints: Iterable[str]) -> bool:
    return any(hint in name for hint in hints)


def classify(
    *,
    total_tensors: int,
    changed_tensors: int,
    changed_lora_like: int,
    changed_norm_or_embed: int,
) -> tuple[str, str]:
    if total_tensors == 0:
        return "unknown", "no comparable tensors were found"

    changed_ratio = changed_tensors / total_tensors
    lora_like_ratio = changed_lora_like / max(changed_tensors, 1)

    if changed_ratio >= 0.85:
        return (
            "likely full fine-tune",
            "most tensors changed, which is the usual pattern for full-model training",
        )

    if changed_ratio <= 0.55 and lora_like_ratio >= 0.75 and changed_norm_or_embed == 0:
        return (
            "likely merged LoRA",
            "changes are sparse and concentrated in typical LoRA target modules",
        )

    if changed_ratio <= 0.70:
        return (
            "likely adapter/partial fine-tune",
            "only a subset of tensors changed, but the pattern is not clean enough for LoRA",
        )

    return (
        "ambiguous, leaning full fine-tune",
        "many tensors changed, but not enough for a confident full-fine-tune call",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path, help="Original/base model directory")
    parser.add_argument("--tuned", required=True, type=Path, help="Fine-tuned model directory")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance for changed values")
    parser.add_argument(
        "--rel-l2-threshold",
        type=float,
        default=1e-7,
        help="Tensor is counted as meaningfully changed only above this relative L2 delta",
    )
    parser.add_argument("--top", type=int, default=30, help="Number of changed tensors to print")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2_000_000,
        help="Elements processed at once per tensor",
    )
    parser.add_argument(
        "--lora-hints",
        default=",".join(DEFAULT_LORA_HINTS),
        help="Comma-separated substrings treated as typical LoRA target modules",
    )
    args = parser.parse_args()

    base_map = load_weight_map(args.base)
    tuned_map = load_weight_map(args.tuned)
    common = sorted(set(base_map) & set(tuned_map))
    missing_in_tuned = sorted(set(base_map) - set(tuned_map))
    extra_in_tuned = sorted(set(tuned_map) - set(base_map))
    hints = tuple(h.strip() for h in args.lora_hints.split(",") if h.strip())

    changed = []
    unchanged_count = 0
    bucket_counts: Counter[str] = Counter()
    changed_lora_like = 0
    changed_norm_or_embed = 0
    total_numel = 0
    changed_numel = 0

    for i, name in enumerate(common, start=1):
        stats = tensor_stats(
            base_map[name],
            tuned_map[name],
            name,
            atol=args.atol,
            chunk_size=args.chunk_size,
        )
        total_numel += stats["numel"]
        meaningful = stats["changed"] and stats["rel_l2"] > args.rel_l2_threshold

        if meaningful:
            changed.append(stats)
            changed_numel += int(stats["changed_frac"] * stats["numel"])
            bucket_counts[module_bucket(name)] += 1
            if is_lora_target_like(name, hints):
                changed_lora_like += 1
            lowered = name.lower()
            if any(token in lowered for token in ("embed", "norm", "lm_head")):
                changed_norm_or_embed += 1
        else:
            unchanged_count += 1

        if i % 100 == 0:
            print(f"processed {i}/{len(common)} tensors...", flush=True)

    changed.sort(key=lambda row: row["rel_l2"], reverse=True)
    label, reason = classify(
        total_tensors=len(common),
        changed_tensors=len(changed),
        changed_lora_like=changed_lora_like,
        changed_norm_or_embed=changed_norm_or_embed,
    )

    print("\n=== Finetune Type Heuristic ===")
    print(f"result: {label}")
    print(f"reason: {reason}")
    print()
    print(f"comparable tensors: {len(common)}")
    print(f"meaningfully changed tensors: {len(changed)} ({len(changed) / max(len(common), 1):.1%})")
    print(f"unchanged tensors: {unchanged_count} ({unchanged_count / max(len(common), 1):.1%})")
    print(f"changed values above atol: {changed_numel:,} / {total_numel:,}")
    print(f"changed tensors matching LoRA hints: {changed_lora_like} / {max(len(changed), 1)}")
    print(f"changed norm/embed/lm_head tensors: {changed_norm_or_embed}")
    print(f"missing tensors in tuned: {len(missing_in_tuned)}")
    print(f"extra tensors in tuned: {len(extra_in_tuned)}")

    print("\nTop changed module buckets:")
    for bucket, count in bucket_counts.most_common(20):
        print(f"  {bucket}: {count}")

    print(f"\nTop {min(args.top, len(changed))} changed tensors by relative L2:")
    for row in changed[: args.top]:
        print(
            f"  {row['name']} | shape={row['shape']} | "
            f"rel_l2={row['rel_l2']:.3e} | max_abs={row['max_abs']:.3e} | "
            f"changed_frac={row['changed_frac']:.3%}"
        )


if __name__ == "__main__":
    main()
