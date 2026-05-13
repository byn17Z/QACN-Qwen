# QACN-Qwen

QACN-Qwen is an SFT project dedicated to developing educational Question-Answering models in Chinese language based on the **Qwen2.5** model family and the **Fineweb-Edu-Chinese-V2.2** dataset.

## Overview

The project trains a Qwen2.5 model using **4-bit QLoRA** with **validation early stopping** via Transformers' native `EarlyStoppingCallback`. The data is split into train, validation, and test sets, and training stops automatically when validation loss stops improving. After training, LoRA adapters are merged into the base model and evaluated on both validation and test sets, enabling direct comparison with pre-training benchmarks.

The pipeline supports **Retrieval-Augmented Generation (RAG)** with configurable context modes: labeled context from training data, RAG-retrieved context, or no context.

## Core Architecture

The training workflow follows a structured pipeline:
1.  **Preprocessing:** Converts raw dataset files into SFT (Instruction/Output) format and splits them into train, validation, and test sets. Supports a `use_processed_data` flag to skip reprocessing and load previously saved data directly.
2.  **RAG Knowledge Base:** Builds or loads a ChromaDB vector store from `raw_content` fields for retrieval-augmented context (only runs when RAG inference flags are enabled). Runs after preprocessing since it reads the processed data.
3.  **Benchmark Evaluation (Pre-Training):** Evaluates the base model on validation and test sets to establish baseline perplexity.
4.  **Training:**
    - **Load:** Initializes the base model using 4-bit quantization.
    - **Train:** Executes SFT training on the train set with LoRA. Supports context-aware formatting with labeled or RAG-retrieved context.
    - **Early Stopping:** Evaluates on the validation set at regular step intervals. Training stops when validation loss plateaus (configurable patience and threshold).
    - **Save:** Saves the best LoRA adapters to `outputs/lora_adapter/<timestamp>/`.
    - **Log:** Records full training loss history to `outputs/train_stats.json`.
5.  **Merge LoRA:** Merges trained LoRA adapters into the base model at full precision and saves to `models/current_model`.
6.  **Post-Training Evaluation:** Evaluates the merged model on validation and test sets, enabling direct comparison with pre-training benchmarks.
7.  **RAG Testing:** Standalone module to test retrieval quality (Precision@k, Recall@k, MRR) and compare perplexity across context modes.
8.  **Deployment:** Serves the merged model via FastAPI REST API + Gradio web UI. Loads model with 4-bit quantization (configurable). Supports optional RAG integration with per-request toggle.

## Technical Stack

- **Model Family:** Qwen2.5 (1.5B, 3B, 7B, 14B)
- **Dataset:** Fineweb-Edu-Chinese-V2.2 (sft_qa subset)
- **Technique:** 4-bit QLoRA (via `bitsandbytes`, `peft`)
- **Orchestration:** `trl`, `transformers`, `accelerate`
- **RAG:** `sentence-transformers`, `chromadb`, `BAAI/bge-small-zh-v1.5` (embedder), `BAAI/bge-reranker-base` (reranker)
- **Deployment:** `fastapi`, `gradio`, `uvicorn`
- **Logging:** `wandb`, local JSON stats (`train_stats.json`, `eval_stats.json`)

## Project Structure

```text
├── data/
│   ├── raw/                # Original jsonl files
│   └── processed/          # Train/Val/Test split datasets
├── configs/
│   └── config.yaml         # Training hyperparameters and paths
├── models/                 # Base and merged model weights
├── outputs/                # LoRA adapters and JSON logs
├── rag/
│   ├── rag_db.py           # Knowledge base construction and loading
│   ├── rag_retrieve.py     # Retrieval and reranking logic
│   ├── rag_test.py         # RAG testing (retrieval quality + perplexity)
│   ├── db/                 # ChromaDB vector store (gitignored)
│   └── models/             # Embedder and reranker models (gitignored)
├── src/
│   ├── utils.py            # Config loading utilities
│   ├── model_utils.py      # LoRA merge utilities
│   ├── inference_engine.py # Model + RAG inference engine
│   ├── api.py              # FastAPI router and schemas
│   └── gradio_app.py       # Gradio chat interface
├── data_preprocess.py      # Data engineering pipeline
├── sft_train.py            # QLoRA training with early stopping
├── test_eval.py            # Evaluation & benchmarking
├── main.py                 # Main pipeline orchestrator
└── deploy.py               # Deployment server (FastAPI + Gradio)
```

## Setup & Installation

### Prerequisites
- Python 3.12+
- CUDA 12.6+ (compatible with PyTorch 2.9.1)
- 1x GPU (e.g. A10 24GB)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/byn17Z/QACN-Qwen.git
   cd QACN-Qwen
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download [models](https://huggingface.co/Qwen) -> models/ and [raw data](https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.2) -> data/raw/.

## Usage

### 1. Configuration
Modify `configs/config.yaml` to set your paths, model size, and hyperparameters:
```yaml
training:
  batch_size: 128
  num_train_epochs: 20
  learning_rate: 1e-4
  train_set_size: 4096
  val_set_size: 512
  test_set_size: 512
  eval_steps: 100
  early_stopping_patience: 3
  early_stopping_threshold: 0.01

dataset:
  use_processed_data: false  # Set true to skip reprocessing raw data

rag:
  build_rag_db: true         # true = rebuild DB; false = load existing
  rag_inference_deploy: false # Enable RAG for deployment
  context_mask_test: false   # Hide context during testing
  rag_inference_test: false  # true = use RAG docs; false = labeled context
  context_mask_train: false  # Hide context during training
  rag_inference_train: false # true = use RAG docs; false = labeled context

deploy:
  quantization: "4bit"       # "4bit" (BitsAndBytes NF4) or "none" (bf16/fp16)
  max_new_tokens: 512
  temperature: 0.7
```

### 2. Run the Pipeline
The `main.py` script automates the entire process from preprocessing to benchmark evaluation, training, LoRA merging, and post-training evaluation:
```bash
python main.py --config configs/config.yaml
```

### 3. Run Individual Stages
```bash
# Preprocessing only
python data_preprocess.py --config configs/config.yaml

# Training only
python sft_train.py --config configs/config.yaml --timestamp "20260512153000"

# Evaluation (validation or test mode)
python test_eval.py --config configs/config.yaml --mode validation --timestamp "20260512153000"
python test_eval.py --config configs/config.yaml --mode test --timestamp "20260512153000"

# RAG testing
python rag/rag_test.py --config configs/config.yaml --mode validation --timestamp "20260512153000"
```

### 4. Deploy the Model
After training and merging, serve the model via FastAPI + Gradio:
```bash
python deploy.py
python deploy.py --host 127.0.0.1 --port 9000
```

- **API:** `http://localhost:8000/v1/chat/completions` (POST, JSON body)
- **Gradio UI:** `http://localhost:8000/gradio` (browser)
- **Health check:** `http://localhost:8000/health`

Set `rag.rag_inference_deploy: true` in `configs/config.yaml` to enable RAG at deployment time.

## Known Issues

### Windows `UnicodeDecodeError: 'gbk' codec can't decode byte` during training
On Windows with Chinese locale, Python defaults to GBK encoding for file I/O. The `trl` library reads Jinja template files without specifying `encoding="utf-8"`, causing a `UnicodeDecodeError`. The pipeline sets `PYTHONUTF8=1` in `main.py` to force UTF-8 mode, which all subprocesses inherit. If you run `sft_train.py` standalone, set the environment variable manually:
```bash
set PYTHONUTF8=1
python sft_train.py --config configs/config.yaml --timestamp "20260512153000"
```

## Logging

Each pipeline run generates a shared timestamp. Both training and evaluation scripts append structured JSON logs with this timestamp and a config snapshot:

- `outputs/train_stats.json` — full training loss history per run
- `outputs/eval_stats.json` — evaluation metrics per run (with `mode` mark: "validation" or "test")
- `outputs/rag_test_stats.json` — RAG retrieval metrics and perplexity comparison across context modes

## Monitoring
Training progress and evaluation metrics are logged to **Weights & Biases (wandb)**. Ensure you are logged in:
```bash
wandb login
```
