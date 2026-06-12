#!/usr/bin/env python3
"""Step 5: merge the trained LoRA adapter into the base king weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from challenger_step_lib import merge_lora_local, read_json, write_json
from train_challenger import log, sha256_dir


def has_adapter_weights(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "adapter_config.json").is_file()
        and (
            (path / "adapter_model.safetensors").is_file()
            or (path / "adapter_model.bin").is_file()
        )
    )


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def find_best_adapter(work: Path, lora_out_dir_arg: str) -> tuple[Path, dict]:
    lora_out = Path(lora_out_dir_arg) if lora_out_dir_arg else work / "lora_out"
    candidates = []
    if (lora_out / "trainer_state.json").exists():
        candidates.append(lora_out / "trainer_state.json")
    candidates.extend(sorted(lora_out.glob("checkpoint-*/trainer_state.json")))

    best_info: dict = {"lora_output_dir": str(lora_out)}
    for state_path in sorted(candidates, key=lambda p: checkpoint_step(p.parent), reverse=True):
        state = json.loads(state_path.read_text())
        best_checkpoint = state.get("best_model_checkpoint")
        best_metric = state.get("best_metric")
        if not best_checkpoint:
            continue
        adapter_dir = Path(best_checkpoint)
        if has_adapter_weights(adapter_dir):
            best_info.update({
                "adapter_dir": str(adapter_dir),
                "best_metric": best_metric,
                "best_global_step": state.get("best_global_step"),
                "trainer_state": str(state_path),
                "selection": "trainer_best_model_checkpoint",
            })
            return adapter_dir, best_info
        best_info.update({
            "missing_best_adapter_dir": str(adapter_dir),
            "missing_best_metric": best_metric,
            "missing_trainer_state": str(state_path),
        })

    if has_adapter_weights(lora_out / "best_adapter"):
        adapter_dir = lora_out / "best_adapter"
        best_info.update({
            "adapter_dir": str(adapter_dir),
            "selection": "best_adapter_dir",
        })
        return adapter_dir, best_info

    checkpoints = sorted(lora_out.glob("checkpoint-*"), key=checkpoint_step, reverse=True)
    for adapter_dir in checkpoints:
        if has_adapter_weights(adapter_dir):
            best_info.update({
                "adapter_dir": str(adapter_dir),
                "selection": "latest_checkpoint_with_adapter_weights",
            })
            return adapter_dir, best_info

    raise FileNotFoundError(
        f"Could not find LoRA adapter weights in {lora_out}. Expected "
        "adapter_model.safetensors or adapter_model.bin in the trainer best checkpoint, "
        "best_adapter, or checkpoint-* directories."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/workspace/teutonic-mining/work",
                    help="Pipeline work directory")
    ap.add_argument("--king-dir", default="/workspace/teutonic-mining/work/king",
                    help="King model dir; defaults to king_dir in <work>/king.json or <work>/king")
    ap.add_argument("--adapter-dir", default="",
                    help="Adapter dir override; defaults to trainer best checkpoint in <work>/lora_out")
    ap.add_argument("--lora-out-dir", default="",
                    help="LoRA training output dir; defaults to <work>/lora_out")
    ap.add_argument("--merged-dir", default="",
                    help="Merged model output dir; defaults to <work>/merged")
    ap.add_argument("--max-shard-size", default="4.3GB",
                    help="Split model.safetensors into shards of at most this size "
                         "(Transformers format, e.g. 4.3GB). Use empty string for a single file.")
    ap.add_argument("--metadata-out", default="",
                    help="Merge metadata JSON; defaults to <work>/merged.json")
    args = ap.parse_args()

    work = Path(args.work)
    if args.king_dir:
        king_dir = Path(args.king_dir)
    else:
        king_meta = work / "king.json"
        king_dir = Path(read_json(king_meta)["king_dir"]) if king_meta.exists() else work / "king"

    if args.adapter_dir:
        adapter_dir = Path(args.adapter_dir)
        adapter_info = {"adapter_dir": str(adapter_dir), "selection": "explicit_adapter_dir"}
    else:
        adapter_dir, adapter_info = find_best_adapter(work, args.lora_out_dir)

    if not has_adapter_weights(adapter_dir):
        raise FileNotFoundError(
            f"adapter dir is missing LoRA weights: {adapter_dir}. "
            "Expected adapter_config.json plus adapter_model.safetensors or adapter_model.bin."
        )

    merged_dir = Path(args.merged_dir) if args.merged_dir else work / "merged"
    metadata_out = Path(args.metadata_out) if args.metadata_out else work / "merged.json"

    merge_lora_local(
        str(king_dir),
        adapter_dir,
        merged_dir,
        max_shard_size=args.max_shard_size,
    )
    write_json(metadata_out, {
        "king_dir": str(king_dir),
        "adapter_dir": str(adapter_dir),
        "adapter_selection": adapter_info,
        "merged_dir": str(merged_dir),
        "challenger_hash": sha256_dir(merged_dir),
    })
    log.info("step5 complete: merged=%s metadata=%s", merged_dir, metadata_out)


if __name__ == "__main__":
    main()
