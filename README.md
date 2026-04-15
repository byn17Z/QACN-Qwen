# QEdpediaCN-Qwen

QEdpediaCN-Qwen is an SFT project dedicated to developing educational Question-Answering models in Chinese language based on the **Qwen2.5** model family and the **Fineweb-Edu-Chinese-V2.2** dataset.

## Overview

The project employs an **iterative training loop** where the model is incrementally improved through data shards, evaluated, and merged back into a base model using **4-bit QLoRA**. This approach allows for efficient training on consumer-grade hardware while maintaining high performance.

## Core Architecture

The training workflow follows a structured pipeline:
1.  **Preprocessing:** Converts raw dataset files into SFT (Instruction/Output) format and splits them into deterministic shards.
2.  **Iterative Loop:**
    - **Load:** Initializes the current model using 4-bit quantization.
    - **Train:** Executes SFT training on a specific data shard using LoRA.
    - **Archive:** Saves LoRA adapters for versioning and recovery.
    - **Atomic Merge:** Merges LoRA adapters into the base model and updates the "current" model.
    - **Evaluate:** Benchmarks the model against a static test set and logs metrics.
3.  **Finalization:** Aggregates statistics and exports the final model (optionally to GGUF).

## Technical Stack

- **Model Family:** Qwen2.5 (1.5B, 3B, 7B, 14B)
- **Dataset:** Fineweb-Edu-Chinese-V2.2 (sft_qa subset)
- **Technique:** 4-bit QLoRA (via `bitsandbytes`, `peft`)
- **Orchestration:** `trl`, `transformers`, `accelerate`
- **Logging:** `wandb`, local JSON stats

## 📁 Project Structure

```text
├── data/
│   ├── raw/                # Original jsonl files
│   └── processed/          # Sharded SFT datasets
├── configs/
│   └── config.yaml         # Training hyperparameters and paths
├── models/                 # Models weights
│   └── [base_model]/       # Starting model weights
├── outputs/                # LoRA adapters and eval logs
├── src/                    # Core utilities (model management, etc.)
├── data_preprocess.py      # Data engineering pipeline
├── sft_train.py            # QLoRA training logic
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
  num_iterations: 4
  learning_rate: 2e-4
```

### 2. Run the Pipeline
The `main.py` script automates the entire process from preprocessing to final evaluation:
```bash
python main.py --config configs/config.yaml
```

## Monitoring
Training progress and evaluation metrics are logged to **Weights & Biases (wandb)**. Ensure you are logged in:
```bash
wandb login
```
