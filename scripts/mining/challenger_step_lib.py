"""Shared helpers for stepwise Teutonic challenger training scripts."""
from __future__ import annotations

import collections
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
import shutil
import ctypes
import inspect
import site
import sys
import time
from pathlib import Path

import numpy as np
import torch
from hippius_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_challenger import (
    allocate_weighted_counts,
    compute_per_seq_loss,
    EVAL_ALPHA,
    EVAL_DELTA,
    filter_indices_by_vocab,
    log,
    SEQ_LEN,
    sha256_dir,
)

HIPPIUS_MODEL_ALLOW_PATTERNS = [
    "*.safetensors",
    "*.json",
    "*.py",
    "tokenizer*",
    "special_tokens*",
    "*.model",
    "*.txt",
]

DEFAULT_ATTN_IMPLEMENTATION = os.environ.get("STEP2_ATTN_IMPLEMENTATION", "auto")


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, data: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(out)


def load_cached_shard(path: Path, seq_len: int = SEQ_LEN) -> tuple[np.ndarray, int]:
    """Load a cached .npy shard lazily so scoring does not pre-read whole files."""
    arr = np.load(path, mmap_mode="r")
    if arr.ndim not in (1, 2):
        raise ValueError(f"unexpected shard shape {arr.shape}")
    flat = arr.reshape(-1)
    n_seq = flat.size // seq_len
    if n_seq <= 0:
        raise ValueError(f"shard {path} has too few tokens for seq_len={seq_len}")
    return flat[: n_seq * seq_len].reshape(n_seq, seq_len), seq_len


def configure_cuda_inference() -> None:
    """Enable safe CUDA inference fast paths before model loading."""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)


def load_causal_lm_for_scoring(
    model_dir: str,
    device: str,
    attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION,
):
    base_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": device},
        "use_safetensors": True,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation and attn_implementation != "auto":
        attempts = [attn_implementation]
    else:
        attempts = []
        if importlib.util.find_spec("flash_attn") is not None:
            attempts.append("flash_attention_2")
        attempts.extend(["sdpa", ""])

    errors = []
    for impl in attempts:
        kwargs = dict(base_kwargs)
        if impl:
            kwargs["attn_implementation"] = impl
        try:
            model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
            log.info("loaded scorer on %s with attn_implementation=%s", device, impl or "default")
            break
        except Exception as exc:
            errors.append(f"{impl or 'default'}: {exc}")
            if attn_implementation and attn_implementation != "auto":
                raise
            log.info("attn_implementation=%s unavailable on %s; trying fallback", impl or "default", device)
    else:
        raise RuntimeError("failed to load scorer model: " + " | ".join(errors))

    if hasattr(model, "config"):
        model.config.use_cache = False
    model.eval()
    return model


def default_scoring_devices() -> str:
    """All visible CUDA devices, or cpu when CUDA is unavailable."""
    if not torch.cuda.is_available():
        return "cpu"
    device_count = torch.cuda.device_count()
    if device_count <= 0:
        return "cpu"
    return ",".join(f"cuda:{idx}" for idx in range(device_count))


def parse_scoring_devices(device: str) -> list[str]:
    """Parse `--device` for sample scoring.

    Accepts existing single-device forms (`2`, `cuda:2`, `cpu`) plus comma
    separated CUDA devices such as `2,3` or `cuda:2,cuda:3`. Use `auto` to
    score on every visible CUDA device.
    """
    if str(device or "").strip().lower() in {"auto", "all"}:
        device = default_scoring_devices()
    raw_parts = [part.strip() for part in str(device or "").split(",") if part.strip()]
    if not raw_parts:
        raise ValueError("device cannot be empty")
    devices = []
    for part in raw_parts:
        if part.isdigit():
            devices.append(f"cuda:{part}")
        else:
            devices.append(part)
    if len(devices) > 1 and any(not dev.startswith("cuda") for dev in devices):
        raise ValueError("multi-device scoring currently requires CUDA devices")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        valid_devices = []
        for dev in devices:
            if not dev.startswith("cuda"):
                valid_devices.append(dev)
                continue
            if dev == "cuda":
                idx = torch.cuda.current_device()
            else:
                try:
                    idx = int(dev.split(":", 1)[1])
                except (IndexError, ValueError) as exc:
                    raise ValueError(f"invalid CUDA device specifier: {dev}") from exc
            if 0 <= idx < device_count:
                valid_devices.append(f"cuda:{idx}")
            else:
                log.warning(
                    "skipping unavailable CUDA device %s; only %d visible CUDA device(s)",
                    dev,
                    device_count,
                )
        devices = valid_devices
    elif any(dev.startswith("cuda") for dev in devices):
        log.warning("CUDA requested (%s) but torch.cuda is unavailable; using CPU", ",".join(devices))
        devices = ["cpu"]
    if not devices:
        raise ValueError("no usable scoring devices after validating --device")
    return devices


def jsonl_rows(path: str | Path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def preload_cuda_runtime() -> None:
    """Make Python-packaged CUDA runtimes visible to custom model extensions."""
    candidates = []
    for site_dir in site.getsitepackages():
        base = Path(site_dir) / "nvidia"
        candidates.extend(base.glob("cuda_runtime/lib/libcudart.so*"))
    for cuda_runtime in candidates:
        try:
            ctypes.CDLL(str(cuda_runtime), mode=ctypes.RTLD_GLOBAL)
            log.info("preloaded CUDA runtime: %s", cuda_runtime)
            return
        except OSError:
            continue


def add_model_dir_to_pythonpath(model_dir: str | Path) -> None:
    """Support custom model files that import sibling files as top-level modules."""
    resolved = str(Path(model_dir).resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
        log.info("added model dir to python path: %s", resolved)


def patch_transformers_masking_utils() -> None:
    """Bridge Quasar custom code to the installed Transformers mask helper."""
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
    log.info("patched transformers.masking_utils.create_causal_mask compatibility")


def download_king_from_hippius(king: dict, out_dir: Path, max_workers: int) -> tuple[str, str]:
    repo = king.get("model_repo") or king.get("hf_repo")
    if not repo:
        raise KeyError(f"dashboard king missing model_repo/hf_repo; keys={sorted(king.keys())}")
    revision = king.get("king_digest") or king.get("king_revision") or king.get("revision") or ""

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("downloading king from Hippius: repo=%s revision=%s -> %s",
             repo, (revision or "HEAD")[:19], out_dir)
    snapshot_download(
        repo_id=repo,
        revision=revision or None,
        local_dir=str(out_dir),
        allow_patterns=HIPPIUS_MODEL_ALLOW_PATTERNS,
        ignore_patterns="optimizer*",
        max_workers=max_workers,
    )
    return repo, revision


def write_king_metadata(path: Path, king: dict, king_dir: Path, repo: str, revision: str) -> None:
    write_json(path, {
        "king_repo": repo,
        "king_revision": revision,
        "king_hash": sha256_dir(king_dir),
        "king_dir": str(king_dir),
        "dashboard_king": king,
    })


def score_samples(
    king_dir: str,
    shards: list[np.ndarray],
    n_score: int,
    seed: int,
    device: str,
    out_path: Path,
    shard_records: list[dict] | None = None,
    per_device_batch_size: int = 8,
    attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION,
    lm_head_chunk: int = 512,
    empty_cache_every: int = 0,
) -> dict:
    if shard_records is not None and len(shard_records) != len(shards):
        raise ValueError("shard_records length must match shards length")
    if per_device_batch_size <= 0:
        raise ValueError("per_device_batch_size must be positive")
    if lm_head_chunk <= 0:
        raise ValueError("lm_head_chunk must be positive")
    if empty_cache_every < 0:
        raise ValueError("empty_cache_every cannot be negative")

    rng = np.random.default_rng(seed)
    cands = []
    dataset_targets: dict[str, int] = {}
    dataset_to_shards: dict[str, list[int]] = collections.defaultdict(list)
    per_shard_targets: dict[int, int] = {}
    if shard_records and any("target_samples_per_shard" in record for record in shard_records):
        for s_idx, record in enumerate(shard_records):
            target = int(record.get("target_samples_per_shard", 0))
            if target <= 0:
                continue
            per_shard_targets[s_idx] = target
            shard = shards[s_idx]
            if len(shard) == 0:
                continue
            idxs = rng.choice(len(shard), size=min(target, len(shard)), replace=False)
            for j in idxs:
                cands.append((s_idx, int(j)))
    elif shard_records and any("target_samples" in record for record in shard_records):
        for s_idx, record in enumerate(shard_records):
            dataset = record.get("dataset", str(s_idx))
            dataset_to_shards[dataset].append(s_idx)
            dataset_targets[dataset] = int(record.get("target_samples", n_score))
        for dataset, shard_idxs in dataset_to_shards.items():
            target = dataset_targets[dataset]
            n_take = max((target * 2) // max(len(shard_idxs), 1), 32)
            for s_idx in shard_idxs:
                shard = shards[s_idx]
                if len(shard) == 0:
                    continue
                idxs = rng.choice(len(shard), size=min(n_take, len(shard)), replace=False)
                for j in idxs:
                    cands.append((s_idx, int(j)))
    else:
        for s_idx, shard in enumerate(shards):
            if len(shard) == 0:
                continue
            n_take = max((n_score * 2) // max(len(shards), 1), 32)
            idxs = rng.choice(len(shard), size=min(n_take, len(shard)), replace=False)
            for j in idxs:
                cands.append((s_idx, int(j)))
    rng.shuffle(cands)

    devices = parse_scoring_devices(device)
    log.info("scoring %d samples with king on %s", len(cands), ",".join(devices))
    configure_cuda_inference()
    preload_cuda_runtime()
    add_model_dir_to_pythonpath(king_dir)
    patch_transformers_masking_utils()
    scorers = []
    for scorer_device in devices:
        log.info("loading king scorer replica on %s", scorer_device)
        model = load_causal_lm_for_scoring(
            king_dir,
            scorer_device,
            attn_implementation=attn_implementation,
        )
        scorers.append((model, scorer_device))
    vocab_size = min(
        int(getattr(model.config, "vocab_size", None) or model.lm_head.out_features)
        for model, _ in scorers
    )

    valid_cands = []
    valid_by_dataset: dict[str, int] = collections.defaultdict(int)
    valid_by_shard: dict[int, int] = collections.defaultdict(int)
    invalid_count = 0
    for s_idx, j in cands:
        tokens = shards[s_idx][j]
        if int(tokens.min()) < 0 or int(tokens.max()) >= vocab_size:
            invalid_count += 1
            continue
        if per_shard_targets:
            if valid_by_shard[s_idx] >= per_shard_targets[s_idx]:
                continue
            valid_by_shard[s_idx] += 1
        elif dataset_targets:
            dataset = shard_records[s_idx].get("dataset", str(s_idx)) if shard_records else str(s_idx)
            if valid_by_dataset[dataset] >= dataset_targets[dataset]:
                continue
            valid_by_dataset[dataset] += 1
        valid_cands.append((s_idx, j))
        if per_shard_targets:
            if all(valid_by_shard[s_idx] >= target for s_idx, target in per_shard_targets.items()):
                break
        elif dataset_targets:
            if all(valid_by_dataset[d] >= target for d, target in dataset_targets.items()):
                break
        elif len(valid_cands) >= n_score:
            break
    cands = valid_cands
    if invalid_count:
        log.info("dropped %d sampled sequences with token ids outside vocab_size=%d",
                 invalid_count, vocab_size)
    expected_count = sum(per_shard_targets.values()) if per_shard_targets else n_score
    if len(cands) < expected_count:
        log.warning("only %d/%d sampled sequences fit vocab_size=%d",
                    len(cands), expected_count, vocab_size)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    losses_by_order: list[float | None] = [None] * len(cands)

    for scorer_idx, (model, scorer_device) in enumerate(scorers):
        warmup_indices = range(scorer_idx, len(cands), len(scorers))[:per_device_batch_size]
        if len(warmup_indices) == 0:
            continue
        warmup_toks = np.stack([
            shards[cands[order_idx][0]][cands[order_idx][1]]
            for order_idx in warmup_indices
        ], axis=0)
        log.info("warming scorer replica on %s with %d samples", scorer_device, warmup_toks.shape[0])
        compute_per_seq_loss(model, warmup_toks, scorer_device, chunk=lm_head_chunk)

    def score_partition(scorer_idx: int, scorer: tuple) -> list[tuple[int, float]]:
        model, scorer_device = scorer
        assigned = range(scorer_idx, len(cands), len(scorers))
        results: list[tuple[int, float]] = []
        t0 = time.time()
        for start in range(0, len(assigned), per_device_batch_size):
            order_indices = assigned[start:start + per_device_batch_size]
            toks = np.stack([
                shards[cands[order_idx][0]][cands[order_idx][1]]
                for order_idx in order_indices
            ], axis=0)
            batch_losses = compute_per_seq_loss(model, toks, scorer_device, chunk=lm_head_chunk)
            results.extend(
                (order_idx, float(loss))
                for order_idx, loss in zip(order_indices, batch_losses)
            )
            del toks, batch_losses
            batch_idx = start // per_device_batch_size + 1
            if empty_cache_every and batch_idx % empty_cache_every == 0 and scorer_device.startswith("cuda"):
                with torch.cuda.device(scorer_device):
                    torch.cuda.empty_cache()
            done = min(start + per_device_batch_size, len(assigned))
            if start == 0 or done == len(assigned) or (start // per_device_batch_size) % 20 == 0:
                log.info(
                    "gpu=%s scored %d/%d assigned samples | total=%d | %.1fs",
                    scorer_device,
                    done,
                    len(assigned),
                    len(cands),
                    time.time() - t0,
                )
        return results

    if len(scorers) == 1:
        for order_idx, loss in score_partition(0, scorers[0]):
            losses_by_order[order_idx] = loss
    else:
        with ThreadPoolExecutor(max_workers=len(scorers)) as pool:
            futures = [
                pool.submit(score_partition, scorer_idx, scorer)
                for scorer_idx, scorer in enumerate(scorers)
            ]
            for future in futures:
                for order_idx, loss in future.result():
                    losses_by_order[order_idx] = loss

    missing = sum(loss is None for loss in losses_by_order)
    if missing:
        raise RuntimeError(f"internal scoring error: {missing} samples were not scored")

    losses = []
    dataset_losses: dict[str, list[float]] = collections.defaultdict(list)
    with out_path.open("w") as f:
        for order_idx, ((s_idx, j), loss) in enumerate(zip(cands, losses_by_order)):
            tok = shards[s_idx][j].tolist()
            arr = np.asarray(tok)
            unique_r = float(len(set(tok)) / len(tok))
            rep_r = float(np.mean(arr[1:] == arr[:-1])) if len(arr) > 1 else 0.0
            ngrams = [tuple(tok[k:k + 4]) for k in range(len(tok) - 3)]
            rep_ng = 1.0 - len(set(ngrams)) / len(ngrams) if ngrams else 0.0
            shard_meta = shard_records[s_idx] if shard_records else {}
            losses.append(float(loss))
            row = {
                "shard": s_idx,
                "idx": j,
                "loss": float(loss),
                "unique_r": unique_r,
                "rep_r": rep_r,
                "rep_ng4": rep_ng,
                "tokens": tok,
            }
            for key in (
                "dataset",
                "dataset_weight",
                "target_samples",
                "target_samples_per_shard",
                "manifest_url",
                "manifest_tokenizer",
                "shard_idx",
                "shard_key",
                "path",
                "source_file",
            ):
                if key in shard_meta:
                    row[key] = shard_meta[key]
            if shard_records:
                dataset_losses[row.get("dataset", "unknown")].append(float(loss))
            f.write(json.dumps(row) + "\n")
            if order_idx == 0 or (order_idx + 1) % (per_device_batch_size * len(scorers) * 20) == 0:
                log.info("wrote %d/%d scored samples", order_idx + 1, len(cands))

    for model, _scorer_device in scorers:
        del model
    scorers.clear()
    for scorer_device in devices:
        if scorer_device.startswith("cuda"):
            with torch.cuda.device(scorer_device):
                torch.cuda.empty_cache()

    by_dataset: dict[str, dict] = {}
    if dataset_losses:
        for dataset, values in sorted(dataset_losses.items()):
            arr = np.asarray(values, dtype=np.float64)
            by_dataset[dataset] = {
                "n_scored": int(arr.size),
                "loss_min": float(arr.min()) if arr.size else None,
                "loss_max": float(arr.max()) if arr.size else None,
                "loss_mean": float(arr.mean()) if arr.size else None,
            }

    summary = {
        "n_scored": len(losses),
        "loss_min": float(np.min(losses)) if losses else None,
        "loss_max": float(np.max(losses)) if losses else None,
        "loss_mean": float(np.mean(losses)) if losses else None,
        "scored_path": str(out_path),
    }
    if by_dataset:
        summary["datasets"] = by_dataset
    log.info("wrote scored samples -> %s", out_path)
    return summary


def bucket_for_row(row: dict, p50: float, p85: float) -> str:
    if row["rep_r"] > 0.2 or row["rep_ng4"] > 0.5 or row["unique_r"] < 0.05:
        return "suspicious"
    if row["loss"] >= p85:
        return "hard"
    if row["loss"] >= p50 * 0.8:
        return "general"
    return "easy"


def _sample_rows(rng: np.random.Generator, src: list[dict], n: int) -> list[dict]:
    if not src or n <= 0:
        return []
    if n >= len(src):
        return list(src)
    selected = rng.choice(len(src), size=n, replace=False)
    return [src[int(k)] for k in selected]


def _select_bucket_mix(pool: list[dict], n_total: int, rng: np.random.Generator) -> list[dict]:
    general = [r for r in pool if r["bucket"] == "general"]
    hard = [r for r in pool if r["bucket"] == "hard"]
    easy = [r for r in pool if r["bucket"] == "easy"]
    n_general = int(n_total * 0.4)
    n_hard = int(n_total * 0.5)
    n_easy = n_total - n_general - n_hard
    rows: list[dict] = []
    rows.extend(_sample_rows(rng, general, n_general))
    rows.extend(_sample_rows(rng, hard, n_hard))
    rows.extend(_sample_rows(rng, easy, n_easy))
    return rows


def build_curriculum(
    scored_path: Path,
    out_dir: Path,
    train_per_iter: int,
    val_size: int,
    seed: int,
) -> dict:
    rows = list(jsonl_rows(scored_path))
    if not rows:
        raise ValueError(f"no scored rows found in {scored_path}")

    losses = np.asarray([r["loss"] for r in rows])
    p50 = float(np.percentile(losses, 50))
    p85 = float(np.percentile(losses, 85))
    for row in rows:
        row["bucket"] = bucket_for_row(row, p50, p85)

    counts = {b: sum(1 for r in rows if r["bucket"] == b)
              for b in ("general", "hard", "easy", "suspicious")}
    clean = [r for r in rows if r["bucket"] != "suspicious"]

    rng = np.random.default_rng(seed + 1)
    dataset_weights: dict[str, float] = {}
    for row in clean:
        dataset = row.get("dataset")
        if dataset and dataset not in dataset_weights and "dataset_weight" in row:
            dataset_weights[dataset] = float(row["dataset_weight"])

    def row_key(row: dict) -> tuple:
        return (
            row.get("dataset", ""),
            row.get("shard_key", row.get("shard", "")),
            row["idx"],
        )

    if dataset_weights:
        ordered = sorted(dataset_weights)
        weights = [dataset_weights[dataset] for dataset in ordered]
        clean_by_dataset: dict[str, list[dict]] = collections.defaultdict(list)
        for row in clean:
            clean_by_dataset[row.get("dataset", "unknown")].append(row)
        for ds_rows in clean_by_dataset.values():
            rng.shuffle(ds_rows)

        val_alloc = allocate_weighted_counts(val_size, weights)
        val_rows = []
        for dataset, n_val in zip(ordered, val_alloc):
            val_rows.extend(clean_by_dataset[dataset][:n_val])
        val_keys = {row_key(r) for r in val_rows}
        pool = [r for r in clean if row_key(r) not in val_keys]

        train_alloc = allocate_weighted_counts(train_per_iter, weights)
        train_rows = []
        for dataset, n_train in zip(ordered, train_alloc):
            ds_pool = [r for r in pool if r.get("dataset") == dataset]
            train_rows.extend(_select_bucket_mix(ds_pool, n_train, rng))
    else:
        rng.shuffle(clean)
        val_rows = clean[:val_size]
        val_keys = {row_key(r) for r in val_rows}
        pool = [r for r in clean if row_key(r) not in val_keys]
        train_rows = _select_bucket_mix(pool, train_per_iter, rng)
    rng.shuffle(train_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    bucketed_path = out_dir / "scored_bucketed.jsonl"

    with train_path.open("w") as f:
        for row in train_rows:
            f.write(json.dumps({"input_ids": row["tokens"]}) + "\n")
    with val_path.open("w") as f:
        for row in val_rows:
            f.write(json.dumps({"input_ids": row["tokens"]}) + "\n")
    with bucketed_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    summary = {
        "counts": counts,
        "datasets": {
            dataset: {
                "total": sum(1 for r in rows if r.get("dataset", "unknown") == dataset),
                "clean": sum(1 for r in clean if r.get("dataset", "unknown") == dataset),
                "train": sum(1 for r in train_rows if r.get("dataset", "unknown") == dataset),
                "val": sum(1 for r in val_rows if r.get("dataset", "unknown") == dataset),
            }
            for dataset in sorted({r.get("dataset", "unknown") for r in rows})
        },
        "p50": p50,
        "p85": p85,
        "train": len(train_rows),
        "val": len(val_rows),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "bucketed_path": str(bucketed_path),
    }
    write_json(out_dir / "scoring.json", summary)
    log.info("wrote curriculum train=%d val=%d -> %s",
             len(train_rows), len(val_rows), out_dir)
    return summary


def paired_eval_datasets(
    king_dir: str,
    chall_dir: str,
    eval_sets: list[dict],
    device: str,
    batch_size: int = 8,
    n_bootstrap: int = 10000,
    alpha: float = EVAL_ALPHA,
) -> dict:
    """Run one paired bootstrap over samples drawn from multiple datasets."""
    delta = EVAL_DELTA
    preload_cuda_runtime()
    add_model_dir_to_pythonpath(king_dir)
    add_model_dir_to_pythonpath(chall_dir)
    patch_transformers_masking_utils()
    log.info("paired_eval_datasets: loading king %s on %s", king_dir, device)
    king = AutoModelForCausalLM.from_pretrained(
        king_dir, torch_dtype=torch.bfloat16, device_map={"": device},
        use_safetensors=True, trust_remote_code=True,
    )
    king.eval()
    log.info("paired_eval_datasets: loading challenger %s on %s", chall_dir, device)
    chall = AutoModelForCausalLM.from_pretrained(
        chall_dir, torch_dtype=torch.bfloat16, device_map={"": device},
        use_safetensors=True, trust_remote_code=True,
    )
    chall.eval()

    king_vocab_size = getattr(king.config, "vocab_size", None) or king.lm_head.out_features
    chall_vocab_size = getattr(chall.config, "vocab_size", None) or chall.lm_head.out_features
    vocab_size = min(int(king_vocab_size), int(chall_vocab_size))

    diffs = []
    per_dataset: dict[str, dict] = {}
    king_sum = chall_sum = 0.0
    n_done = 0
    requested_total = 0
    dropped_total = 0
    t0 = time.time()

    for eval_set in eval_sets:
        dataset = eval_set["dataset"]
        shard = eval_set["shard"]
        indices = list(eval_set["indices"])
        requested_total += len(indices)
        indices, dropped = filter_indices_by_vocab(shard, indices, vocab_size)
        dropped_total += dropped
        if dropped:
            log.info(
                "dropped %d/%d %s eval sequences outside vocab_size=%d",
                dropped,
                len(eval_set["indices"]),
                dataset,
                vocab_size,
            )
        ds_diffs = []
        ds_king_sum = ds_chall_sum = 0.0
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            toks = [shard[j].tolist() for j in batch_idx]
            kl = compute_per_seq_loss(king, toks, device)
            cl = compute_per_seq_loss(chall, toks, device)
            for k, c in zip(kl, cl):
                diff = k - c
                diffs.append(diff)
                ds_diffs.append(diff)
                king_sum += k
                chall_sum += c
                ds_king_sum += k
                ds_chall_sum += c
                n_done += 1
            if n_done and (n_done // batch_size) % 5 == 0:
                log.info(
                    "eval %d/%d | mu_hat=%.6f | king=%.4f chall=%.4f | %.1fs",
                    n_done,
                    requested_total,
                    float(np.mean(diffs)),
                    king_sum / n_done,
                    chall_sum / n_done,
                    time.time() - t0,
                )

        ds_n = len(ds_diffs)
        per_dataset[dataset] = {
            "n_eval": ds_n,
            "n_eval_requested": len(eval_set["indices"]),
            "n_eval_dropped_vocab": dropped,
            "avg_king_loss": ds_king_sum / ds_n if ds_n else None,
            "avg_chall_loss": ds_chall_sum / ds_n if ds_n else None,
            "mu_hat": float(np.mean(ds_diffs)) if ds_diffs else None,
            "manifest_url": eval_set.get("manifest_url"),
            "shard_key": eval_set.get("shard_key"),
            "target_samples": eval_set.get("target_samples"),
            "weight": eval_set.get("weight"),
        }

    if not diffs:
        raise ValueError(f"no eval sequences fit vocab_size={vocab_size}")

    diffs_arr = np.asarray(diffs, dtype=np.float64)
    mu_hat = float(diffs_arr.mean())
    boot = np.empty(n_bootstrap)
    rng = np.random.default_rng(0xB007)
    for b in range(n_bootstrap):
        boot[b] = diffs_arr[rng.integers(0, len(diffs_arr), size=len(diffs_arr))].mean()
    lcb = float(np.quantile(boot, alpha))
    accepted = lcb > delta
    res = {
        "n_eval": n_done,
        "mu_hat": mu_hat,
        "lcb": lcb,
        "delta": delta,
        "alpha": alpha,
        "accepted": accepted,
        "avg_king_loss": king_sum / n_done,
        "avg_chall_loss": chall_sum / n_done,
        "n_eval_requested": requested_total,
        "n_eval_dropped_vocab": dropped_total,
        "datasets": per_dataset,
        "elapsed_s": time.time() - t0,
    }
    log.info("paired_eval_datasets: mu_hat=%.6f lcb=%.6f accepted=%s",
             mu_hat, lcb, accepted)
    del king, chall
    torch.cuda.empty_cache()
    return res


def merge_lora_local(
    base_model: str,
    adapter: Path,
    out: Path,
    max_shard_size: str = "4.3GB",
) -> Path:
    log.info("merging LoRA %s into %s -> %s", adapter, base_model, out)
    from peft import PeftModel

    preload_cuda_runtime()
    add_model_dir_to_pythonpath(base_model)
    patch_transformers_masking_utils()
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"safe_serialization": True}
    if max_shard_size:
        save_kwargs["max_shard_size"] = max_shard_size
        log.info("saving merged model with max_shard_size=%s", max_shard_size)
    merged.save_pretrained(str(out), **save_kwargs)
    try:
        tok = AutoTokenizer.from_pretrained(
            base_model,
            use_fast=True,
            trust_remote_code=True,
        )
        tok.save_pretrained(str(out))
    except Exception as exc:
        log.warning("tokenizer save skipped: %s", exc)

    for pattern in ("config.json", "generation_config.json", "*.py", "tokenizer*", "special_tokens*", "*.model"):
        for src in Path(base_model).glob(pattern):
            if src.is_file():
                shutil.copy(src, out / src.name)

    del base, merged
    torch.cuda.empty_cache()
    log.info("merged model saved to %s", out)
    return out
