#!/usr/bin/env python3
"""Step 4: train a LoRA adapter from curriculum JSONL files."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from challenger_step_lib import read_json, write_json
from train_challenger import log, run_lora_training


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/workspace/teutonic-mining/work",
                    help="Pipeline work directory")
    ap.add_argument("--king-dir", default="/workspace/teutonic-mining/work/king",
                    help="King model dir; defaults to king_dir in <work>/king.json or <work>/king")
    ap.add_argument("--bundle", default="/workspace/teutonic-mining/bundle",
                    help="Path to training_bundle directory")
    ap.add_argument("--train-data", default="",
                    help="Train JSONL; defaults to <work>/curriculum/train.jsonl")
    ap.add_argument("--val-data", default="",
                    help="Validation JSONL; defaults to <work>/curriculum/val.jsonl")
    ap.add_argument("--output-dir", default="/workspace/teutonic-mining/work/lora_out1",
                    help="LoRA output dir; defaults to <work>/lora_out")
    ap.add_argument("--n-gpus", type=int, default=1)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.01,
                    help="AdamW weight decay for LoRA training")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05,
                    help="LoRA dropout probability")
    ap.add_argument("--lora-init", default=True,
                    help="LoRA init_lora_weights value, e.g. true, false, pissa, pissa_niter_16")
    ap.add_argument("--lora-target-modules", default="",
                    help="Comma-separated LoRA target module suffixes; empty uses training bundle defaults")
    ap.add_argument("--adapter-type", choices=["lora", "adalora", "vera", "loha", "lokr", "kokr"],
                    default="lora", help="PEFT adapter type; kokr aliases lokr")
    ap.add_argument("--use-adalora", dest="adapter_type", action="store_const", const="adalora",
                    help="Use AdaLoRA adapter instead of regular LoRA")
    ap.add_argument("--use-vera", dest="adapter_type", action="store_const", const="vera",
                    help="Use VeRA adapter instead of regular LoRA")
    ap.add_argument("--use-loha", dest="adapter_type", action="store_const", const="loha",
                    help="Use LoHa adapter instead of regular LoRA")
    ap.add_argument("--use-lokr", "--use-kokr", dest="adapter_type", action="store_const", const="lokr",
                    help="Use LoKr adapter instead of regular LoRA")
    ap.add_argument("--use-qlora", action="store_true",
                    help="Load the base model in 4-bit before adapter training")
    ap.add_argument("--qlora-quant-type", default="nf4", choices=["nf4", "fp4"])
    ap.add_argument("--use-unsloth", action="store_true",
                    help="Use Unsloth FastLanguageModel backend for LoRA/QLoRA training")
    ap.add_argument("--unsloth-gradient-checkpointing", default="unsloth",
                    help='Unsloth gradient checkpointing mode, usually "unsloth" or "true"')
    ap.add_argument("--unsloth-random-state", type=int, default=3407)
    ap.add_argument("--use-delta-lora", action="store_true",
                    help="Reserved opt-in for Delta-LoRA; raises if unsupported by installed PEFT")
    ap.add_argument("--use-pissa", action="store_true", help="Use PiSSA LoRA initialization")
    ap.add_argument("--use-corda", action="store_true", help="Use CorDA LoRA initialization")
    ap.add_argument("--use-eva", action="store_true", help="Use EVA LoRA initialization")
    ap.add_argument("--eva-rho", type=float, default=2.0)
    ap.add_argument("--eva-tau", type=float, default=0.99)
    ap.add_argument("--eva-whiten", action="store_true")
    ap.add_argument("--use-loftq", action="store_true", help="Use LoftQ LoRA initialization")
    ap.add_argument("--loftq-bits", type=int, default=4)
    ap.add_argument("--loftq-iter", type=int, default=1)
    ap.add_argument("--use-lora-ga", action="store_true",
                    help="Use LoRA-GA initialization if supported by installed PEFT")
    ap.add_argument("--lora-ga-direction", default="ArB2r",
                    choices=["ArBr", "A2rBr", "ArB2r", "random"])
    ap.add_argument("--lora-ga-scale", default="stable",
                    choices=["stable", "weight_svd", "gd_scale", "unit"])
    ap.add_argument("--lora-ga-stable-gamma", type=int, default=16)
    ap.add_argument("--adalora-target-r", type=int, default=8)
    ap.add_argument("--adalora-init-r", type=int, default=12)
    ap.add_argument("--adalora-tinit", type=int, default=0)
    ap.add_argument("--adalora-tfinal", type=int, default=0)
    ap.add_argument("--adalora-delta-t", type=int, default=1)
    ap.add_argument("--adalora-beta1", type=float, default=0.85)
    ap.add_argument("--adalora-beta2", type=float, default=0.85)
    ap.add_argument("--adalora-orth-reg-weight", type=float, default=0.5)
    ap.add_argument("--vera-projection-prng-key", type=int, default=0)
    ap.add_argument("--vera-save-projection", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--vera-d-initial", type=float, default=0.1)
    ap.add_argument("--loha-module-dropout", type=float, default=0.0)
    ap.add_argument("--lokr-module-dropout", type=float, default=0.0)
    ap.add_argument("--lokr-decompose-both", action="store_true")
    ap.add_argument("--lokr-decompose-factor", type=int, default=-1)
    ap.add_argument("--use-dora", action="store_true",
                    help="Enable DoRA magnitude decomposition in PEFT LoRA")
    ap.add_argument("--use-rslora", action="store_true",
                    help="Enable rank-stabilized LoRA scaling in PEFT")
    ap.add_argument("--use-loraplus", action="store_true",
                    help="Use the LoRA+ optimizer for adapter training")
    ap.add_argument("--loraplus-lr-ratio", type=float, default=16.0,
                    help="LoRA+ learning-rate ratio between LoRA B and A")
    ap.add_argument("--eval-steps", type=int, default=150,
                    help="Run validation every N optimizer steps")
    ap.add_argument("--save-steps", type=int, default=150,
                    help="Save checkpoints every N optimizer steps")
    ap.add_argument("--save-total-limit", type=int, default=0,
                    help="Max checkpoints to retain (0 = keep all)")
    ap.add_argument("--seed", type=int, default=None,
                    help="Seed for training shuffling and adapter initialization; omitted chooses >100")
    ap.add_argument("--metadata-out", default="",
                    help="Adapter metadata JSON; defaults to <work>/adapter.json")
    args = ap.parse_args()
    if args.seed is None:
        args.seed = random.SystemRandom().randint(101, 2**32 - 1)
        log.info("no --seed provided; generated step4 seed=%d", args.seed)

    work = Path(args.work)
    if args.king_dir:
        king_dir = Path(args.king_dir)
    else:
        meta_path = work / "king.json"
        king_dir = Path(read_json(meta_path)["king_dir"]) if meta_path.exists() else work / "king"
    train_data = Path(args.train_data) if args.train_data else work / "curriculum" / "train.jsonl"
    val_data = Path(args.val_data) if args.val_data else work / "curriculum" / "val.jsonl"
    output_dir = Path(args.output_dir) if args.output_dir else work / "lora_out"
    metadata_out = Path(args.metadata_out) if args.metadata_out else work / "adapter.json"

    adapter = run_lora_training(
        str(king_dir), train_data, val_data, output_dir, args.n_gpus, args, Path(args.bundle),
    )
    write_json(metadata_out, {
        "king_dir": str(king_dir),
        "train_data": str(train_data),
        "val_data": str(val_data),
        "lora_output_dir": str(output_dir),
        "adapter_dir": str(adapter),
        "micro_batch": args.micro_batch,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "epochs": args.epochs,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_init": args.lora_init,
        "lora_target_modules": args.lora_target_modules,
        "adapter_type": args.adapter_type,
        "use_qlora": args.use_qlora,
        "qlora_quant_type": args.qlora_quant_type,
        "use_unsloth": args.use_unsloth,
        "unsloth_gradient_checkpointing": args.unsloth_gradient_checkpointing,
        "unsloth_random_state": args.unsloth_random_state,
        "use_delta_lora": args.use_delta_lora,
        "use_pissa": args.use_pissa,
        "use_corda": args.use_corda,
        "use_eva": args.use_eva,
        "eva_rho": args.eva_rho,
        "eva_tau": args.eva_tau,
        "eva_whiten": args.eva_whiten,
        "use_loftq": args.use_loftq,
        "loftq_bits": args.loftq_bits,
        "loftq_iter": args.loftq_iter,
        "use_lora_ga": args.use_lora_ga,
        "lora_ga_direction": args.lora_ga_direction,
        "lora_ga_scale": args.lora_ga_scale,
        "lora_ga_stable_gamma": args.lora_ga_stable_gamma,
        "adalora_target_r": args.adalora_target_r,
        "adalora_init_r": args.adalora_init_r,
        "adalora_tinit": args.adalora_tinit,
        "adalora_tfinal": args.adalora_tfinal,
        "adalora_delta_t": args.adalora_delta_t,
        "adalora_beta1": args.adalora_beta1,
        "adalora_beta2": args.adalora_beta2,
        "adalora_orth_reg_weight": args.adalora_orth_reg_weight,
        "vera_projection_prng_key": args.vera_projection_prng_key,
        "vera_save_projection": args.vera_save_projection,
        "vera_d_initial": args.vera_d_initial,
        "loha_module_dropout": args.loha_module_dropout,
        "lokr_module_dropout": args.lokr_module_dropout,
        "lokr_decompose_both": args.lokr_decompose_both,
        "lokr_decompose_factor": args.lokr_decompose_factor,
        "use_dora": args.use_dora,
        "use_rslora": args.use_rslora,
        "use_loraplus": args.use_loraplus,
        "loraplus_lr_ratio": args.loraplus_lr_ratio,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "seed": args.seed,
    })
    log.info("step4 complete: adapter=%s metadata=%s", adapter, metadata_out)


if __name__ == "__main__":
    main()
