# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QEdpediaCN-Qwen is a QLoRA fine-tuning project that trains Chinese educational QA models using the Qwen2.5 family (1.5B/3B/7B/14B) on the Fineweb-Edu-Chinese-V2.2 dataset. The pipeline preprocesses data, benchmarks the base model, trains with validation early stopping via Transformers' native `EarlyStoppingCallback`, merges LoRA adapters into the base model, and evaluates the merged model.

## Commands

```bash
# Full pipeline (preprocess -> train -> test eval)
python main.py --config configs/config.yaml

# Data preprocessing only
python data_preprocess.py --config configs/config.yaml

# Training only (requires --timestamp)
python sft_train.py --config configs/config.yaml --timestamp "20260512153000"

# Evaluation only (requires --mode and --timestamp)
python test_eval.py --config configs/config.yaml --mode validation --timestamp "20260512153000"
python test_eval.py --config configs/config.yaml --mode test --timestamp "20260512153000"

# Install dependencies
pip install -r requirements.txt

# Wandb setup (required before training)
wandb login
```

## Architecture

The pipeline is orchestrated by `main.py` which generates a digit-only timestamp (e.g., `20260513153000`) and runs:

1. **Preprocessing** (`data_preprocess.py`): Loads raw JSONL from `data/raw/`, formats into Instruction/Output SFT format, shuffles (seed 42), splits into train/val/test sets based on config sizes, saves to `data/processed/train.jsonl`, `val.jsonl`, `test.jsonl`. Supports `use_processed_data` config flag to skip reprocessing and load `sft_data.jsonl` directly.

2. **Benchmark Evaluation (Pre-Training)** (`test_eval.py`): Evaluates the base model on both validation and test sets before training, establishing baseline perplexity.

3. **Training** (`sft_train.py`): Loads base model with 4-bit quantization, applies LoRA to all linear layers (q/k/v/o_proj, gate/up/down_proj), trains via `SFTTrainer` with validation early stopping. Evaluates on val set every `eval_steps` steps; stops when `eval_loss` hasn't improved by `early_stopping_threshold` for `early_stopping_patience` evaluations. Saves best LoRA adapters to `outputs/lora_adapter/<timestamp>/`. Logs full training loss history to `outputs/train_stats.json`.

4. **Merge LoRA** (`src/model_utils.py`): Merges trained LoRA adapters into the base model at full precision (bfloat16/float16) and saves to `models/current_model`. Frees GPU memory after saving.

5. **Post-Training Evaluation** (`test_eval.py`): Evaluates the merged model on both validation and test sets, enabling direct comparison with pre-training benchmarks.

6. **Config**: All hyperparameters in `configs/config.yaml` (paths, LoRA params, batch size, context length, early stopping params, data sizes).

## Logging

Both `sft_train.py` and `test_eval.py` append to their respective JSON log files. Each entry includes:
- `timestamp`: passed from the orchestrator (shared across a pipeline run)
- `config`: snapshot of all training hyperparameters at time of run
- Training: `train_loss_history` (step + loss pairs), `total_training_steps`
- Evaluation: `mode` ("validation" or "test"), `test_loss`, `perplexity`

Log files:
- `outputs/train_stats.json` — training loss history per run
- `outputs/eval_stats.json` — evaluation metrics per run

## Key Implementation Details

- HuggingFace mirror is set via `HF_ENDPOINT=https://hf-mirror.com` in `main.py` for Chinese network access.
- `PYTHONUTF8=1` is set in `main.py` to fix `UnicodeDecodeError` on Windows with Chinese locale (trl reads Jinja templates without specifying encoding).
- Early stopping uses `EarlyStoppingCallback` with `steps` strategy, `eval_loss` metric, `load_best_model_at_end=True`.
- `SFTConfig` (from `trl`) replaces `TrainingArguments`; `max_length` replaces deprecated `max_seq_length`.
- `formatting_func` is called per-example (`batched=False`) in newer trl — must return a single string, not a list.
- All numeric config values are explicitly cast (`int()`, `float()`) to avoid type errors from YAML string parsing.
- LoRA adapters are saved with a timestamp subdirectory: `outputs/lora_adapter/<timestamp>/`.
- `merge_lora_to_base` loads the base model at full precision (not quantized) for accurate merging, then frees GPU memory after saving.
- `use_processed_data` config flag: when `true`, loads `data/processed/sft_data.jsonl` directly; when `false`, processes raw data and saves it as `sft_data.jsonl`.
- Virtual environment: `.venv-QEDpediaCN-Qwen` (non-standard name, gitignored).
- Gitignored directories: `.venv-QEDpediaCN-Qwen/`, `hf_cache/`, `data/`, `models/`, `__pycache__/`.

## Code Conventions

- Module-level docstrings on every Python file (Description, Usage, Dependencies).
- Function docstrings required (Description, Args, Returns).
- Type hints mandatory on all function signatures.
- `snake_case` for variables/functions, `PascalCase` for classes.
- Inline comments explain "why", not "what".
- YAML for config, JSON for machine-generated outputs.
