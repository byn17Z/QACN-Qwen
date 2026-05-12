
# GEMINI.md - QEdpediaCN-Qwen

## Project Overview
**QEdpediaCN-Qwen** is a fine-tuning project aimed at developing high-quality Chinese Educational Question-Answering (QA) models. It leverages the **Qwen2.5** model family and the **Fineweb-Edu-Chinese-V2.2** dataset (specifically the `sft_qa` subset) to create models with strong pedagogical logic and accurate knowledge.

The project trains the model using **4-bit QLoRA** with **validation early stopping** via Transformers' native `EarlyStoppingCallback`. The data is split into train, validation, and test sets, and training stops automatically when validation loss stops improving.

## Core Architecture
The training workflow is orchestrated as follows:
1.  **Preprocessing:** Convert raw dataset files into SFT (Instruction/Output) format and split them into train, validation, and test sets.
2.  **Training:**
    - **Load:** Initialize the base model using 4-bit quantization.
    - **Train:** Perform SFT training on the train set using **4-bit QLoRA**.
    - **Early Stopping:** Evaluate on the validation set at regular step intervals. Stop when validation loss plateaus (configurable patience and threshold).
    - **Save Best Model:** Save the LoRA adapters with the lowest validation loss to `outputs/lora_adapter/`.
3.  **Evaluation:** Benchmark the trained model against the test set and log metrics.

## Technical Workflow
The project follows a five-stage implementation roadmap:

### Stage 1: Environment & Foundation
*   **Base Model Initialization:** Download and place the starting base model (e.g., `Qwen2.5-3B-Instruct`) into `models/`.
*   **Infrastructure Verification:** Ensure the `.venv-QEDpediaCN-Qwen` environment contains `PEFT`, `TRL`, `bitsandbytes`, and `Transformers`.
*   **Config Validation:** Finalize `configs/config.yaml` with local paths and hardware constraints.

### Stage 2: Data Engineering (`data_preprocess.py`)
*   **Ingestion:** Load the `sft_qa` subset of the `Fineweb-Edu-Chinese-V2.2` dataset.
*   **Formatting:** Standardize samples into `Instruction/Output` format.
*   **Splitting:** Shuffle (seed 42) and split into train, validation, and test sets based on `train_set_size`, `val_set_size`, and `test_set_size` in config. Save as `train.jsonl`, `val.jsonl`, and `test.jsonl`.

### Stage 3: SFT Training with Early Stopping (`sft_train.py`)
*   **4-bit Quantization:** Load the base model using `bitsandbytes`.
*   **LoRA Config:** Target all linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
*   **Training:** Execute the `SFTTrainer` on the train set with validation early stopping.
*   **Early Stopping:** Uses Transformers' `EarlyStoppingCallback` with `steps` strategy, `eval_loss` metric, and configurable patience/threshold.
*   **Save Best:** Save the best LoRA adapters to `outputs/lora_adapter/`.

### Stage 4: Evaluation & Monitoring (`test_eval.py`)
*   **Benchmark:** Run against the test set.
*   **Metrics:** Track loss and perplexity.
*   **Telemetry:** Log results to **wandb** and `outputs/eval_stats.json`.

### Stage 5: Orchestration (`main.py`)
*   Automate the full pipeline: `Preprocess -> Train`.

## Technical Stack
- **Language:** Python
- **Base Models:** Qwen2.5 (1.5B, 3B, 7B, 14B)
- **Dataset:** Fineweb-Edu-Chinese-V2.2 (SFT-enhanced version)
- **Libraries:** `PyTorch`, `Transformers`, `PEFT`, `TRL`, `Accelerate`, `Datasets`, `wandb`, `PyYAML`

## Project Structure
- `configs/`: YAML configuration files (hyperparameters, paths).
- `data/`: Raw and processed datasets.
- `models/`: Storage for base model weights.
- `outputs/`: LoRA adapters, training logs, and evaluation metrics.
- `src/`: Modular logic for utilities.
- `main.py`: Orchestrator script.
- `data_preprocess.py`: Data formatting and splitting.
- `sft_train.py`: QLoRA training with early stopping.
- `test_eval.py`: Evaluation and logging.

## Development Conventions
### Code Style
- **Docstrings:** Required for all modules and functions.
- **Type Hints:** Mandatory for all function signatures.
- **Naming:** `snake_case` for variables/functions, `PascalCase` for classes.

## Usage
```bash
# Run full pipeline
python main.py --config configs/config.yaml
```

## Status
**Phase:** Core Logic Implemented (Stages 2-5).
**Running Environment:** Ubuntu 24.04, Python 3.12, PyTorch 2.9.1 (CUDA 12.6), 1x A10 24GB GPU (or 1x RTX3060 6GB).
Documentation and directory structures are established.
All core scripts (`data_preprocess.py`, `sft_train.py`, `test_eval.py`, `main.py`) have been implemented.
**Next Steps:** Verify the full pipeline execution and tune early stopping parameters.
