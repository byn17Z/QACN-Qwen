# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QEdpediaCN-Qwen is a QLoRA fine-tuning project that trains Chinese educational QA models using the Qwen2.5 family (1.5B/3B/7B/14B) on the Fineweb-Edu-Chinese-V2.2 dataset. The pipeline preprocesses data into train/val/test splits, trains with validation early stopping via Transformers' native `EarlyStoppingCallback`, then evaluates on the test set.

## Commands

```bash
# Full pipeline (preprocess -> train -> test eval)
python main.py --config configs/config.yaml

# Data preprocessing only
python data_preprocess.py --config configs/config.yaml

# Training only (requires --timestamp)
python sft_train.py --config configs/config.yaml --timestamp "2026-05-12T15:30:00"

# Evaluation only (requires --mode and --timestamp)
python test_eval.py --config configs/config.yaml --mode validation --timestamp "2026-05-12T15:30:00"
python test_eval.py --config configs/config.yaml --mode test --timestamp "2026-05-12T15:30:00"

# Install dependencies
pip install -r requirements.txt

# Wandb setup (required before training)
wandb login
```

## Architecture

The pipeline is orchestrated by `main.py` which generates a timestamp and runs:

1. **Preprocessing** (`data_preprocess.py`): Loads raw JSONL from `data/raw/`, formats into Instruction/Output SFT format, shuffles (seed 42), splits into train/val/test sets based on config sizes, saves to `data/processed/train.jsonl`, `val.jsonl`, `test.jsonl`.

2. **Training** (`sft_train.py`): Loads base model with 4-bit quantization, applies LoRA to all linear layers (q/k/v/o_proj, gate/up/down_proj), trains via `SFTTrainer` with validation early stopping. Evaluates on val set every `eval_steps` steps; stops when `eval_loss` hasn't improved by `early_stopping_threshold` for `early_stopping_patience` evaluations. Saves best LoRA adapters to `outputs/lora_adapter/`. Logs full training loss history to `outputs/train_stats.json`.

3. **Test Evaluation** (`test_eval.py`): Calculates perplexity on test set, logs to `outputs/eval_stats.json` with mode mark and config snapshot. Supports `--mode validation` or `--mode test` to select dataset.

4. **Config**: All hyperparameters in `configs/config.yaml` (paths, LoRA params, batch size, context length, early stopping params, data sizes).

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
- Early stopping uses `EarlyStoppingCallback` with `steps` strategy, `eval_loss` metric, `load_best_model_at_end=True`.
- LoRA targets all linear layers with r=32, alpha=64, dropout=0.1.
- Virtual environment: `.venv-QEDpediaCN-Qwen` (non-standard name, gitignored).
- Gitignored directories: `.venv-QEDpediaCN-Qwen/`, `hf_cache/`, `data/`, `models/`, `__pycache__/`.

## Code Conventions

- Module-level docstrings on every Python file (Description, Usage, Dependencies).
- Function docstrings required (Description, Args, Returns).
- Type hints mandatory on all function signatures.
- `snake_case` for variables/functions, `PascalCase` for classes.
- Inline comments explain "why", not "what".
- YAML for config, JSON for machine-generated outputs.
