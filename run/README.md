# Teutonic Fine-Tuning Runner

This directory contains local helpers for the stepwise mining/fine-tuning flow in
`scripts/mining/`.

## 1. Create the environment

```bash
run/setup_finetune_env.sh
source .venv/bin/activate
```

The setup script installs the project, `scripts/mining/requirements.txt`, PEFT,
Accelerate, Hippius Hub, and a CUDA 12.8 PyTorch wheel. CUDA 12.8 is intentional
for Blackwell GPUs, where older CUDA wheels can install but fail at runtime.

## 2. Run the step pipeline

```bash
run/run_mining_steps.sh all
```

Or run each step independently:

```bash
run/step1_download_king.sh
run/step2_score_samples.sh
run/step3_build_curriculum.sh
run/step4_train_lora.sh
run/step5_merge_lora.sh
run/step6_eval_verdict.sh
```

To compare any standalone fine-tuned model directory against the live current king:

```bash
CHALLENGER_DIR=/path/to/merged-model run/compare_with_king.sh
```

If you used the default stepwise flow, `CHALLENGER_DIR` defaults to
`/workspace/teutonic-mining/work-stepwise/merged`. The comparison script
re-downloads the current dashboard king into
`/workspace/teutonic-mining/work-stepwise/current_king` so it does not reuse the
older model you fine-tuned from.

The default work directory is `/workspace/teutonic-mining/work-stepwise`. Outputs
include:

- `king/` and `king.json`
- `scored_samples.jsonl` and `score_summary.json`
- `curriculum/train.jsonl` and `curriculum/val.jsonl`
- `lora_out/` and `adapter.json`
- `merged/` and `merged.json`
- `verdict.json`

## Useful overrides

`step2_score_samples.py` defaults to 30,000 samples per dataset source.
For non-Quasar sources it selects up to 10 random shards and samples 3,000
rows per shard:

- `automathtext-v2`: 30,000 samples across up to 10 shards
- `quasar-sn3`: 30,000 samples with dataset-level random sampling
- `ultradata-math`: 30,000 samples across up to 10 shards
- `finewebedu`: 30,000 samples across up to 10 shards

That produces up to 120,000 scored rows. If a non-Quasar source has fewer than
10 available shards, Step 2 uses the available shard count and still samples
up to 3,000 rows per selected shard.
Each row preserves the original
`train_challenger.py` scoring format: `shard`, `idx`, `loss`, `unique_r`,
`rep_r`, `rep_ng4`, and `tokens`, with dataset metadata added for weighted
curriculum selection. Step 2 selects shards randomly by default and downloads
missing dataset shards into `<work>/cache/datasets`. It generates a new seed for
each run unless you set `SEED`, then selects samples randomly within each
selected shard. Set `CACHE_ONLY=1` only when you want to restrict selection to
already cached local shards.

For reproducible shard/sample selection:

```bash
SEED=123 run/step2_score_samples.sh
```

Step 3 defaults to `TRAIN_PER_ITER=15000` and `VAL_SIZE=600`, using a harder
curriculum mix: 40% general, 50% hard, 10% easy.

For debugging, contiguous shard selection is still available:

```bash
SEQUENTIAL_SHARDS=1 SHARD_START=0 run/step2_score_samples.sh
```

```bash
N_SCORE=60000 TRAIN_PER_ITER=16000 VAL_SIZE=1000 run/run_mining_steps.sh step2 step3
N_GPUS=1 MICRO_BATCH=2 GRAD_ACCUM=2 LR=1e-5 EPOCHS=2 WARMUP_RATIO=0.03 EVAL_STEPS=150 run/run_mining_steps.sh step4
N_EVAL=5000 BATCH_SIZE=4 run/run_mining_steps.sh step6
```

`WARMUP_RATIO` defaults to `0.03`. `EVAL_STEPS` controls how often step 4 runs
validation during LoRA training. `SAVE_STEPS` defaults to the same value so
best-checkpoint loading stays aligned.

Step 6 reads `<work>/score_summary.json` by default and excludes any training
shards recorded there before selecting evaluation shards. It then allocates
`N_EVAL` across the same dataset weights and samples rows randomly from the held
out shard for each dataset.

`compare_with_king.sh` first refreshes the current dashboard king, then uses the
same evaluation process and writes `<work>/comparison.json`. Positive
`challenger_better_nats_per_token` means the fine-tuned model has lower loss than
the live current king on the sampled eval set.

Use default `KING_SOURCE=dashboard` for the live current king. Use
`KING_SOURCE=hf` or `KING_SOURCE=hippius` only when you want to bypass the live
dashboard and download a specific model; those modes require `KING_MODEL_URL` or
`KING_REPO`:

```bash
KING_SOURCE=hf \
KING_MODEL_URL=https://huggingface.co/org/model \
KING_REVISION=main \
run/run_mining_steps.sh step1
```

The runner defaults to the newer weighted multi-dataset scorer
`step2_score_samples.py`. To use the older single-manifest scorer:

```bash
STEP2_IMPL=legacy run/run_mining_steps.sh step2
```
