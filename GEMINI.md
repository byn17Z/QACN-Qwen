
# GEMINI.md - QEdpediaCN-Qwen

## Project Overview
**QEdpediaCN-Qwen** is an iterative fine-tuning project aimed at developing high-quality Chinese Educational Question-Answering (QA) models. It leverages the **Qwen2.5** model family and the **Fineweb-Edu-Chinese-V2.2** dataset (specifically the `sft_qa` subset) to create models with strong pedagogical logic and accurate knowledge.

The project employs an **iterative training loop** where the model is incrementally improved through shards of data, evaluated, and then merged back into a base model using 4-bit QLoRA.

## Core Architecture
The training workflow is orchestrated as follows:
1.  **Preprocessing:** Convert raw dataset files into SFT (Instruction/Output) format and split them into multiple training shards.
2.  **Iterative Loop:**
    - **Load:** Initialize the current model from `models/current_model/` using 4-bit quantization.
    - **Train:** Perform one iteration of SFT training on a specific data shard using **4-bit QLoRA**.
    - **Archive:** Save LoRA adapters to `outputs/lora_iter_N/`.
    - **Atomic Merge:** Merge the new LoRA adapters into the current base model, validate integrity, and update the model in `models/current_model/`.
    - **Evaluate:** Run benchmarks on a held-out test set and log metrics to `wandb` and local JSON files.
3.  **Finalization:** Aggregate statistics across all iterations and export the final model (optionally to GGUF for local deployment).

## Technical Workflow
The project follows a six-stage implementation roadmap:

### Stage 1: Environment & Foundation
*   **Base Model Initialization:** Download and place the starting base model (e.g., `Qwen2.5-1.5B-Instruct`) into `models/current_model/`.
*   **Infrastructure Verification:** Ensure the `.venv-QEDpediaCN-Qwen` environment contains `PEFT`, `TRL`, `bitsandbytes`, and `Transformers`.
*   **Config Validation:** Finalize `configs/config.yaml` with local paths and hardware constraints.

### Stage 2: Data Engineering (`data_preprocess.py`)
*   **Ingestion:** Load the `sft_qa` subset of the `Fineweb-Edu-Chinese-V2.2` dataset.
*   **Formatting:** Standardize samples into `Instruction/Output` format.
*   **Sharding:** Implement a deterministic split into $N$ shards (e.g., 1024 samples each) for the iterative loop.

### Stage 3: The Iterative Training Script (`sft_train.py`)
*   **4-bit Quantization:** Load the model from `models/current_model/` using `bitsandbytes`.
*   **LoRA Config:** Target key attention layers (`q_proj`, `v_proj`).
*   **Shard Training:** Execute the `SFTTrainer` on the current iteration's specific shard.
*   **Adapter Archiving:** Save LoRA weights to `outputs/lora_iter_{i}/`.

### Stage 4: Atomic Model Management (`src/model_utils.py`)
*   **Merge Logic:** Utility to merge the base model with the latest LoRA adapters.
*   **Integrity Check:** Verify the merged model's reload capability.
*   **Atomic Swap:** Replace `models/current_model` with the newly merged weights safely.

### Stage 5: Evaluation & Monitoring (`test_eval.py`)
*   **Benchmark:** Run against a static `test_set` created during preprocessing.
*   **Metrics:** Track loss, perplexity, and QA accuracy.
*   **Telemetry:** Log results to **wandb** and `outputs/eval_stats.json`.

### Stage 6: Orchestration (`main.py`)
*   Automate the full loop: `Preprocess -> [Train -> Merge -> Eval] x N -> Finalize`.

## Technical Stack
- **Language:** Python
- **Base Models:** Qwen2.5 (1.5B, 3B, 7B, 14B)
- **Dataset:** Fineweb-Edu-Chinese-V2.2 (SFT-enhanced version)
- **Libraries:** `PyTorch`, `Transformers`, `PEFT`, `TRL`, `Accelerate`, `Datasets`, `wandb`, `PyYAML`

## Project Structure
- `configs/`: YAML configuration files (hyperparameters, paths).
- `data/`: Raw and processed datasets.
- `models/`: Storage for base and current merged models.
- `outputs/`: LoRA adapters, training logs, and evaluation metrics.
- `src/`: Modular logic for model loading, merging, and utilities.
- `main.py`: Orchestrator script.
- `data_preprocess.py`: Data formatting and sharding.
- `sft_train.py`: QLoRA training logic.
- `test_eval.py`: Evaluation and logging.

## Development Conventions
### Code Style
- **Docstrings:** Required for all modules and functions.
- **Type Hints:** Mandatory for all function signatures.
- **Naming:** `snake_case` for variables/functions, `PascalCase` for classes.

## Usage (Planned)
```bash
# Run full pipeline
python main.py --config configs/config.yaml
```

## Status
**Phase:** Core Logic Implemented (Stages 2-6).
**Running Environment:** Ubuntu 24.04, Python 3.12, PyTorch 2.9.1 (CUDA 12.6), 1x A10 24GB GPU (or 1x RTX3060 6GB).
Documentation and directory structures are established.
All core scripts (`data_preprocess.py`, `sft_train.py`, `src/model_utils.py`, `test_eval.py`, `main.py`) have been implemented.
**Next Steps:** Verify the full pipeline execution and address any remaining data encoding issues.
