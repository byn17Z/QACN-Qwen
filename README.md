# QEdpediaCN-Qwen

QEdpediaCN-Qwen is an SFT project dedicated to developing educational Question-Answering models in Chinese language based on the **Qwen2.5** model family and the **Fineweb-Edu-Chinese-V2.2** dataset.

## Overview

The project trains a Qwen2.5 model using **4-bit QLoRA** with **validation early stopping** via Transformers' native `EarlyStoppingCallback`. The data is split into train, validation, and test sets, and training stops automatically when validation loss stops improving. After training, the model is evaluated on the test set.

## Core Architecture

The training workflow follows a structured pipeline:
1.  **Preprocessing:** Converts raw dataset files into SFT (Instruction/Output) format and splits them into train, validation, and test sets.
2.  **Training:**
    - **Load:** Initializes the base model using 4-bit quantization.
    - **Train:** Executes SFT training on the train set with LoRA.
    - **Early Stopping:** Evaluates on the validation set at regular step intervals. Training stops when validation loss plateaus (configurable patience and threshold).
    - **Save:** Saves the best LoRA adapters to `outputs/lora_adapter/`.
    - **Log:** Records full training loss history to `outputs/train_stats.json`.
3.  **Evaluation:** Benchmarks the trained model against the test set, logs metrics to `outputs/eval_stats.json`.

## Technical Stack

- **Model Family:** Qwen2.5 (1.5B, 3B, 7B, 14B)
- **Dataset:** Fineweb-Edu-Chinese-V2.2 (sft_qa subset)
- **Technique:** 4-bit QLoRA (via `bitsandbytes`, `peft`)
- **Orchestration:** `trl`, `transformers`, `accelerate`
- **Logging:** `wandb`, local JSON stats (`train_stats.json`, `eval_stats.json`)

## Project Structure

```text
├── data/
│   ├── raw/                # Original jsonl files
│   └── processed/          # Train/Val/Test split datasets
├── configs/
│   └── config.yaml         # Training hyperparameters and paths
├── models/                 # Base model weights
├── outputs/                # LoRA adapters and JSON logs
├── src/                    # Core utilities
├── data_preprocess.py      # Data engineering pipeline
├── sft_train.py            # QLoRA training with early stopping
├── test_eval.py            # Evaluation & benchmarking
└── main.py                 # Main pipeline orchestrator
```

## Setup & Installation

### Prerequisites
- Python 3.12+
- CUDA 12.6+ (compatible with PyTorch 2.9.1)
- 1x GPU (e.g. A10 24GB)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/byn17Z/QEdpediaCN-Qwen.git
   cd QEdpediaCN-Qwen
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
```

### 2. Run the Pipeline
The `main.py` script automates the entire process from preprocessing to training to test evaluation:
```bash
python main.py --config configs/config.yaml
```

### 3. Run Individual Stages
```bash
# Preprocessing only
python data_preprocess.py --config configs/config.yaml

# Training only
python sft_train.py --config configs/config.yaml --timestamp "2026-05-12T15:30:00"

# Evaluation (validation or test mode)
python test_eval.py --config configs/config.yaml --mode validation --timestamp "2026-05-12T15:30:00"
python test_eval.py --config configs/config.yaml --mode test --timestamp "2026-05-12T15:30:00"
```

## Logging

Each pipeline run generates a shared timestamp. Both training and evaluation scripts append structured JSON logs with this timestamp and a config snapshot:

- `outputs/train_stats.json` — full training loss history per run
- `outputs/eval_stats.json` — evaluation metrics per run (with `mode` mark: "validation" or "test")

## Monitoring
Training progress and evaluation metrics are logged to **Weights & Biases (wandb)**. Ensure you are logged in:
```bash
wandb login
```
