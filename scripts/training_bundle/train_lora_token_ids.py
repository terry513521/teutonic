#!/usr/bin/env python3
import argparse
import ctypes
import inspect
import json
import os
import random
import site
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from peft.optimizers import create_loraplus_optimizer


def set_training_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _optional_peft_class(name: str):
    import peft

    return getattr(peft, name, None)


def preload_cuda_runtime() -> None:
    for site_dir in site.getsitepackages():
        base = Path(site_dir) / "nvidia"
        for cuda_runtime in base.glob("cuda_runtime/lib/libcudart.so*"):
            try:
                ctypes.CDLL(str(cuda_runtime), mode=ctypes.RTLD_GLOBAL)
                print(f"preloaded CUDA runtime: {cuda_runtime}", flush=True)
                return
            except OSError:
                continue


def add_model_dir_to_pythonpath(model_dir: str) -> None:
    resolved = str(Path(model_dir).resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
        print(f"added model dir to python path: {resolved}", flush=True)


def patch_transformers_masking_utils() -> None:
    try:
        import transformers.masking_utils as masking_utils
    except ImportError:
        return

    create_causal_mask = masking_utils.create_causal_mask
    if "cache_position" in inspect.signature(create_causal_mask).parameters:
        return

    def create_causal_mask_compat(*args, cache_position=None, **kwargs):
        past_key_values = kwargs.get("past_key_values")
        original_get_mask_sizes = getattr(past_key_values, "get_mask_sizes", None)
        if cache_position is not None and original_get_mask_sizes is not None:
            def get_mask_sizes_compat(q_length_or_cache_position, layer_idx):
                if hasattr(q_length_or_cache_position, "shape"):
                    return original_get_mask_sizes(q_length_or_cache_position, layer_idx)
                return original_get_mask_sizes(cache_position, layer_idx)

            past_key_values.get_mask_sizes = get_mask_sizes_compat
        try:
            return create_causal_mask(*args, **kwargs)
        finally:
            if original_get_mask_sizes is not None:
                past_key_values.get_mask_sizes = original_get_mask_sizes

    masking_utils.create_causal_mask = create_causal_mask_compat
    print("patched transformers.masking_utils.create_causal_mask compatibility", flush=True)


def load_tokenizer_optional(base_model: str):
    for use_fast in (True, False):
        try:
            return AutoTokenizer.from_pretrained(
                base_model,
                use_fast=use_fast,
                trust_remote_code=True,
            )
        except Exception as exc:
            print(f"tokenizer load failed use_fast={use_fast}: {exc}", flush=True)
    print("continuing without tokenizer; token-id training does not require one", flush=True)
    return None


class TokenIdsDataset(Dataset):
    def __init__(self, path, seq_len):
        self.rows = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    ids = obj["input_ids"][:seq_len]
                    if len(ids) < seq_len:
                        continue
                    self.rows.append(ids)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        ids = self.rows[idx]
        x = torch.tensor(ids, dtype=torch.long)
        return {
            "input_ids": x,
            "attention_mask": torch.ones_like(x),
            "labels": x.clone(),
        }


@dataclass
class Collator:
    def __call__(self, features):
        return {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
        }


def _parse_init_value(value):
    if not isinstance(value, str):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes"}:
        return True
    if lowered in {"0", "false", "no"}:
        return False
    return value


def _single_enabled_init(args) -> str | None:
    enabled = [
        name for enabled, name in (
            (args.use_pissa, "pissa"),
            (args.use_corda, "corda"),
            (args.use_eva, "eva"),
            (args.use_loftq, "loftq"),
            (args.use_lora_ga, "lora-ga"),
        )
        if enabled
    ]
    if len(enabled) > 1:
        raise ValueError(f"only one LoRA initialization method can be enabled at a time: {enabled}")
    return enabled[0] if enabled else None


def _build_peft_config(args, target_modules):
    adapter_type = args.adapter_type.lower().replace("_", "-")
    init_method = _single_enabled_init(args)
    lora_init = _parse_init_value(args.lora_init)
    if init_method == "lora-ga" and lora_init is True:
        # LoRA-GA is configured through lora_ga_config, not init_lora_weights.
        lora_init = True
    elif init_method and lora_init is True:
        lora_init = init_method
    elif init_method:
        raise ValueError(
            f"--use-{init_method} conflicts with explicit --lora-init={args.lora_init!r}; "
            "use one initialization selector"
        )

    if args.use_delta_lora:
        raise RuntimeError(
            "Delta-LoRA was requested, but PEFT 0.19.1 in this environment does not expose "
            "a Delta-LoRA adapter/config. Leave --use-delta-lora disabled or add a compatible implementation."
        )

    if adapter_type == "adalora":
        AdaLoraConfig = _optional_peft_class("AdaLoraConfig")
        if AdaLoraConfig is None:
            raise RuntimeError("AdaLoRA requested, but this PEFT version has no AdaLoraConfig")
        return AdaLoraConfig(
            r=args.lora_r,
            target_r=args.adalora_target_r,
            init_r=args.adalora_init_r,
            tinit=args.adalora_tinit,
            tfinal=args.adalora_tfinal,
            deltaT=args.adalora_delta_t,
            beta1=args.adalora_beta1,
            beta2=args.adalora_beta2,
            orth_reg_weight=args.adalora_orth_reg_weight,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            init_lora_weights=lora_init,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
            use_dora=args.use_dora,
            use_rslora=args.use_rslora,
        )

    if adapter_type == "vera":
        VeraConfig = _optional_peft_class("VeraConfig")
        if VeraConfig is None:
            raise RuntimeError("VeRA requested, but this PEFT version has no VeraConfig")
        return VeraConfig(
            r=args.lora_r,
            target_modules=target_modules,
            vera_dropout=args.lora_dropout,
            projection_prng_key=args.vera_projection_prng_key,
            save_projection=args.vera_save_projection,
            d_initial=args.vera_d_initial,
            bias="none",
            task_type="CAUSAL_LM",
        )

    if adapter_type == "loha":
        LoHaConfig = _optional_peft_class("LoHaConfig")
        if LoHaConfig is None:
            raise RuntimeError("LoHa requested, but this PEFT version has no LoHaConfig")
        return LoHaConfig(
            r=args.lora_r,
            alpha=args.lora_alpha,
            rank_dropout=args.lora_dropout,
            module_dropout=args.loha_module_dropout,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )

    if adapter_type in {"lokr", "kokr"}:
        LoKrConfig = _optional_peft_class("LoKrConfig")
        if LoKrConfig is None:
            raise RuntimeError("LoKr requested, but this PEFT version has no LoKrConfig")
        return LoKrConfig(
            r=args.lora_r,
            alpha=args.lora_alpha,
            rank_dropout=args.lora_dropout,
            module_dropout=args.lokr_module_dropout,
            decompose_both=args.lokr_decompose_both,
            decompose_factor=args.lokr_decompose_factor,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )

    if adapter_type != "lora":
        raise ValueError(f"unknown adapter type: {args.adapter_type!r}")

    config_kwargs = {}
    if args.use_loftq:
        LoftQConfig = _optional_peft_class("LoftQConfig")
        if LoftQConfig is None:
            raise RuntimeError("LoftQ requested, but this PEFT version has no LoftQConfig")
        config_kwargs["loftq_config"] = LoftQConfig(
            loftq_bits=args.loftq_bits,
            loftq_iter=args.loftq_iter,
        )
    if args.use_eva:
        EvaConfig = _optional_peft_class("EvaConfig")
        if EvaConfig is None:
            raise RuntimeError("EVA requested, but this PEFT version has no EvaConfig")
        config_kwargs["eva_config"] = EvaConfig(
            rho=args.eva_rho,
            tau=args.eva_tau,
            whiten=args.eva_whiten,
        )
    if args.use_lora_ga:
        LoraGAConfig = _optional_peft_class("LoraGAConfig")
        if LoraGAConfig is None:
            raise RuntimeError("LoRA-GA requested, but this PEFT version has no LoraGAConfig")
        config_kwargs["lora_ga_config"] = LoraGAConfig(
            direction=args.lora_ga_direction,
            scale=args.lora_ga_scale,
            stable_gamma=args.lora_ga_stable_gamma,
        )

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights=lora_init,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        use_dora=args.use_dora,
        use_rslora=args.use_rslora,
        **config_kwargs,
    )


def _build_unsloth_model(args, target_modules):
    if args.adapter_type.lower().replace("_", "-") != "lora":
        raise RuntimeError("Unsloth backend currently supports regular LoRA only; disable adapter variants")
    unsupported = [
        name for enabled, name in (
            (args.use_dora, "DoRA"),
            (args.use_delta_lora, "Delta-LoRA"),
            (args.use_pissa, "PiSSA"),
            (args.use_corda, "CorDA"),
            (args.use_eva, "EVA"),
            (args.use_lora_ga, "LoRA-GA"),
        )
        if enabled
    ]
    if unsupported:
        raise RuntimeError(
            "Unsloth backend does not support these options in this trainer: "
            f"{', '.join(unsupported)}"
        )

    try:
        from unsloth import FastLanguageModel
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "USE_UNSLOTH=1 was requested, but `unsloth` is not installed in this environment. "
            "Install a CUDA-compatible Unsloth build first or leave USE_UNSLOTH=0."
        ) from exc

    dtype = torch.bfloat16 if torch.cuda.is_available() else None
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.seq_len,
        dtype=dtype,
        load_in_4bit=args.use_qlora,
    )
    loftq_config = None
    if args.use_loftq:
        LoftQConfig = _optional_peft_class("LoftQConfig")
        if LoftQConfig is None:
            raise RuntimeError("LoftQ requested, but this PEFT version has no LoftQConfig")
        loftq_config = LoftQConfig(loftq_bits=args.loftq_bits, loftq_iter=args.loftq_iter)
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing=args.unsloth_gradient_checkpointing,
        random_state=args.unsloth_random_state,
        use_rslora=args.use_rslora,
        loftq_config=loftq_config,
    )
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--micro-batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-init", default="true",
                    help="LoRA init_lora_weights value: true, false, pissa, pissa_niter_16, etc.")
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-target-modules", type=str, default=None,
                    help="comma-separated module name suffixes; defaults to a Quasar-aware set")
    ap.add_argument("--adapter-type", choices=["lora", "adalora", "vera", "loha", "lokr", "kokr"],
                    default="lora", help="PEFT adapter type; kokr is accepted as an alias for lokr")
    ap.add_argument("--use-adalora", dest="adapter_type", action="store_const", const="adalora",
                    help="Use AdaLoRA adapter instead of regular LoRA")
    ap.add_argument("--use-vera", dest="adapter_type", action="store_const", const="vera",
                    help="Use VeRA adapter instead of regular LoRA")
    ap.add_argument("--use-loha", dest="adapter_type", action="store_const", const="loha",
                    help="Use LoHa adapter instead of regular LoRA")
    ap.add_argument("--use-lokr", "--use-kokr", dest="adapter_type", action="store_const", const="lokr",
                    help="Use LoKr adapter instead of regular LoRA")
    ap.add_argument("--use-qlora", action="store_true",
                    help="Load the base model in 4-bit with bitsandbytes before adapter training")
    ap.add_argument("--qlora-quant-type", default="nf4", choices=["nf4", "fp4"])
    ap.add_argument("--use-unsloth", action="store_true",
                    help="Use Unsloth FastLanguageModel backend for LoRA/QLoRA training")
    ap.add_argument("--unsloth-gradient-checkpointing", default="unsloth",
                    help='Unsloth gradient checkpointing mode, usually "unsloth" or "true"')
    ap.add_argument("--unsloth-random-state", type=int, default=3407)
    ap.add_argument("--use-delta-lora", action="store_true",
                    help="Reserved opt-in for Delta-LoRA; raises if unsupported by installed PEFT")
    ap.add_argument("--use-pissa", action="store_true",
                    help="Use PiSSA LoRA initialization")
    ap.add_argument("--use-corda", action="store_true",
                    help="Use CorDA LoRA initialization if supported by installed PEFT")
    ap.add_argument("--use-eva", action="store_true",
                    help="Use EVA LoRA initialization")
    ap.add_argument("--eva-rho", type=float, default=2.0)
    ap.add_argument("--eva-tau", type=float, default=0.99)
    ap.add_argument("--eva-whiten", action="store_true")
    ap.add_argument("--use-loftq", action="store_true",
                    help="Use LoftQ LoRA initialization")
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
                    help="Enable DoRA magnitude decomposition")
    ap.add_argument("--use-rslora", action="store_true",
                    help="Enable rank-stabilized LoRA scaling")
    ap.add_argument("--use-loraplus", action="store_true",
                    help="Use the LoRA+ optimizer for adapter training")
    ap.add_argument("--loraplus-lr-ratio", type=float, default=16.0,
                    help="LoRA+ learning-rate ratio between LoRA B and A")
    ap.add_argument("--eval-steps", type=int, default=50,
                    help="Run validation every N optimizer steps")
    ap.add_argument("--save-steps", type=int, default=50,
                    help="Save checkpoints every N optimizer steps")
    ap.add_argument("--save-total-limit", type=int, default=0,
                    help="Max checkpoints to retain (0 = keep all)")
    ap.add_argument("--seed", type=int, default=None,
                    help="Seed for trainer shuffling and adapter initialization; omitted chooses >100")
    args = ap.parse_args()
    if args.seed is None:
        args.seed = random.SystemRandom().randint(101, 2**32 - 1)
        print(f"no --seed provided; generated training seed={args.seed}", flush=True)

    set_training_seed(args.seed)
    preload_cuda_runtime()
    add_model_dir_to_pythonpath(args.base_model)
    patch_transformers_masking_utils()

    # Quasar module names: attn uses q_proj/k_proj/v_proj/o_proj
    # like Qwen3, but FFN paths split into the dense SwiGLU (ffn.gate / ffn.up
    # / ffn.down) and BigMac MoE (shared_experts.{i}.{gate,up,down},
    # w_down_proj, w_up_proj). LoRA on `experts_w12` / `experts_w3` is not
    # supported — they are nn.Parameter blocks, not Linear layers. Override
    # via --lora-target-modules if you want a custom set.
    target_modules = args.lora_target_modules.split(",") if args.lora_target_modules else [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate", "up", "down",
        "w_down_proj", "w_up_proj",
    ]
    if args.use_unsloth:
        model, tokenizer = _build_unsloth_model(args, target_modules)
    else:
        tokenizer = load_tokenizer_optional(args.base_model)
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "use_safetensors": True,
            "trust_remote_code": True,
        }
        if args.use_qlora:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=args.qlora_quant_type,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                bnb_4bit_use_double_quant=True,
            )
        model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
        peft_cfg = _build_peft_config(args, target_modules)
        model = get_peft_model(model, peft_cfg)
    model.config.use_cache = False
    model.print_trainable_parameters()

    train_ds = TokenIdsDataset(args.train_data, args.seq_len)
    val_ds = TokenIdsDataset(args.val_data, args.seq_len)

    optimizer = None
    if args.use_loraplus:
        optimizer = create_loraplus_optimizer(
            model,
            optimizer_cls=torch.optim.AdamW,
            lr=args.learning_rate,
            loraplus_lr_ratio=args.loraplus_lr_ratio,
            weight_decay=args.weight_decay,
        )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit or None,
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=True,
        report_to="none",
        ddp_find_unused_parameters=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=Collator(),
        optimizers=(optimizer, None),
    )

    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "best_adapter"))
    if tokenizer is not None:
        tokenizer.save_pretrained(os.path.join(args.output_dir, "best_adapter"))


if __name__ == "__main__":
    main()
