# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QACN-Qwen is a QLoRA fine-tuning project that trains Chinese educational QA models using the Qwen2.5 family (1.5B/3B/7B/14B) on the Fineweb-Edu-Chinese-V2.2 dataset. The pipeline preprocesses data, benchmarks the base model, trains with validation early stopping via Transformers' native `EarlyStoppingCallback`, merges LoRA adapters into the base model, and evaluates the merged model.

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

# RAG testing (requires --mode and --timestamp)
python rag/rag_test.py --config configs/config.yaml --mode validation --timestamp "20260512153000"

# Deployment (FastAPI + Gradio)
python deploy.py
python deploy.py --config configs/config.yaml
python deploy.py --host 127.0.0.1 --port 9000

# Install dependencies (includes fastapi, gradio for deployment)
pip install -r requirements.txt

# Wandb setup (required before training)
wandb login
```

## Architecture

The pipeline is orchestrated by `main.py` which generates a digit-only timestamp (e.g., `20260513153000`) and runs:

1. **Preprocessing** (`data_preprocess.py`): Loads raw JSONL from `data/raw/`, formats into Instruction/Output SFT format, preserves `raw_content` field for RAG knowledge base, shuffles (seed 42), splits into train/val/test sets based on config sizes, saves to `data/processed/train.jsonl`, `val.jsonl`, `test.jsonl`. Supports `use_processed_data` config flag to skip reprocessing and load `sft_data.jsonl` directly.

2. **RAG Knowledge Base** (`rag/rag_db.py`): Builds or loads the ChromaDB vector store from `raw_content` fields in `sft_data.jsonl`. Controlled by `build_rag_db` config flag — `true` rebuilds from scratch, `false` loads existing DB. Only runs when any RAG inference flag is enabled. Runs after preprocessing since it reads `sft_data.jsonl`.

3. **Benchmark Evaluation (Pre-Training)** (`test_eval.py`): Evaluates the base model on both validation and test sets before training, establishing baseline perplexity.

4. **Training** (`sft_train.py`): Loads base model with 4-bit quantization, applies LoRA to all linear layers (q/k/v/o_proj, gate/up/down_proj), trains via `SFTTrainer` with validation early stopping. Evaluates on val set every `eval_steps` steps; stops when `eval_loss` hasn't improved by `early_stopping_threshold` for `early_stopping_patience` evaluations. Saves best LoRA adapters to `outputs/lora_adapter/<timestamp>/`. Logs full training loss history to `outputs/train_stats.json`. Supports context-aware formatting via RAG config flags.

5. **Merge LoRA** (`src/model_utils.py`): Merges trained LoRA adapters into the base model at full precision (bfloat16/float16) and saves to `models/current_model`. Frees GPU memory after saving.

6. **Post-Training Evaluation** (`test_eval.py`): Evaluates the merged model on both validation and test sets, enabling direct comparison with pre-training benchmarks. Supports context-aware formatting via RAG config flags.

7. **RAG Testing** (`rag/rag_test.py`): Standalone module to test RAG retrieval quality (Precision@k, Recall@k, MRR) and compare model perplexity across three context modes: no context, labeled context, and RAG-retrieved context.

8. **Deployment** (`deploy.py`): Serves the merged model via FastAPI REST API + Gradio web UI. Loads model with 4-bit BitsAndBytes NF4 quantization (configurable). Supports optional RAG integration controlled by `rag.rag_inference_deploy` flag — when enabled, loads ChromaDB, embedder, and reranker at startup. RAG can be toggled per-request via API parameter or Gradio checkbox. Components:
   - `src/inference_engine.py` — `InferenceEngine` class holding model, tokenizer, RAG components; single `generate()` method
   - `src/api.py` — FastAPI router with `/health`, `/v1/models`, `/v1/chat/completions` endpoints
   - `src/gradio_app.py` — Gradio chat interface with RAG toggle and generation parameter sliders

9. **Config**: All hyperparameters in `configs/config.yaml` (paths, LoRA params, batch size, context length, early stopping params, data sizes, RAG flags, deploy settings).

## RAG Module (`rag/`)

The RAG module provides retrieval-augmented generation capabilities. Components:
- **Embedder**: `BAAI/bge-small-zh-v1.5` — encodes text into 512-dim vectors
- **Reranker**: `BAAI/bge-reranker-base` — cross-encoder for reranking retrieved candidates
- **Database**: ChromaDB — persistent vector store at `rag/db/`

Files:
- `rag/rag_db.py` — builds and loads the knowledge DB from `raw_content` fields in processed data
- `rag/rag_retrieve.py` — retrieves top-k candidates with embedder, reranks with reranker, returns top-n
- `rag/rag_test.py` — tests retrieval quality (Precision@k, Recall@k, MRR) and compares perplexity across context modes

RAG config flags in `configs/config.yaml` under `rag:` section:
- `build_rag_db` — `true` rebuilds DB from `sft_data.jsonl`; `false` loads existing DB
- `rag_inference_deploy` — integrate RAG when deploying
- `context_mask_test` / `context_mask_train` — hide context entirely during eval/training
- `rag_inference_test` / `rag_inference_train` — `true` uses RAG-retrieved docs; `false` uses labeled `raw_content`; ignored when context mask is `true`

Data format: raw data has `raw_content` field as labeled knowledge context. `data_preprocess.py` preserves this field in processed output.

Detailed implementation plan: see `rag.md`.

## Deployment (`deploy.py`)

The deployment server loads the merged model and serves it via FastAPI + Gradio.

```bash
python deploy.py                              # defaults from config
python deploy.py --config configs/config.yaml # explicit config
python deploy.py --host 127.0.0.1 --port 9000 # override host/port
```

Endpoints:
- `GET /health` — health check (model path, RAG status)
- `GET /v1/models` — model metadata (family, size, quantization)
- `POST /v1/chat/completions` — inference with `ChatRequest` JSON body
- `GET /gradio` — Gradio web UI

### Startup Sequence

1. `load_config()` → resolve model path (`deploy.model_path_override` > `model.current_model_path` > `model.base_model_path`)
2. Load tokenizer + model (4-bit NF4 via `BitsAndBytesConfig` if `deploy.quantization=="4bit"`, else bf16/fp16)
3. If `rag.rag_inference_deploy` is `true`: load ChromaDB collection, embedder, reranker via existing `rag/` module
4. Construct `InferenceEngine` (holds model, tokenizer, RAG components)
5. Create FastAPI app, include API router, mount Gradio at `/gradio`
6. `uvicorn.run()`

### InferenceEngine (`src/inference_engine.py`)

`generate()` method flow: validate input → resolve generation params (request overrides > config defaults) → RAG retrieval if enabled (graceful degradation on failure) → build prompt matching training format → `model.generate()` under `torch.inference_mode()` → decode generated tokens only → return response + metadata.

Prompt format (must match training):
- With RAG: `Context: {docs}\nInstruction: {q}\nInput: {inp}\nOutput:`
- Without RAG: `Instruction: {q}\nInput: {inp}\nOutput:`

### Error Handling

- Startup: model path not found → exit 1; RAG DB missing when required → exit 1
- Runtime: empty instruction → 400; CUDA OOM → 500 + clear cache; RAG failure → log warning, proceed without context

### Config

Deploy config in `configs/config.yaml` under `deploy:` section:
- `quantization` — `"4bit"` (BitsAndBytes NF4) or `"none"` (bf16/fp16)
- `model_path_override` — non-empty overrides `current_model_path`
- `rag_enabled_default` — default RAG toggle for Gradio UI
- Generation defaults: `max_new_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty`, `do_sample`

Detailed implementation plan: see `deploy.md`. Implementation details: see `deployment_details.md`.

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
- Gitignored directories: `.venv-QEDpediaCN-Qwen/`, `hf_cache/`, `data/`, `models/`, `rag/db/`, `rag/models/`, `__pycache__/`.

## Code Conventions

- Module-level docstrings on every Python file (Description, Usage, Dependencies).
- Function docstrings required (Description, Args, Returns).
- Type hints mandatory on all function signatures.
- `snake_case` for variables/functions, `PascalCase` for classes.
- Inline comments explain "why", not "what".
- YAML for config, JSON for machine-generated outputs.
