# QEdpediaCN-Qwen

## Environment

The Python virtual environment is built in `/.venv-QEDpediaCN-Qwen`. 

## Goal

Fine-tune a question-answering model based on the `Qwen2.5` family (selectable: 1.5B, 3B, 7B, or 14B) with samples in the dataset `Fineweb-Edu-Chinese-V2.2`. 

## Basic Information

* **Language:** `python`
* **Base Model:** `Qwen2.5-1.5B/3B/7B/14B-Instruct`
* **Dataset:** `Fineweb-Edu-Chinese-V2.2`
* **SFT Method:** `4bit-QLoRA`

## Dependency Packages

* `PyTorch`, `Transformers`, `PEFT`, `TRL`, `Accelerate`, `Datasets`, `wandb`, `PyYAML`

## Project Structure

```text
QEdpediaCN-Qwen/
├── configs/            # YAML configurations for hyperparameters and prompts
├── data/               # Raw and processed datasets (JSON)
├── models/             # Local storage for base model weights
├── outputs/            # LoRA adapters and training logs
├── src/                # Modular logic (utilities)
├── data_preprocess.py  # Script: Raw -> SFT format & Train/Val/Test split
├── sft_train.py        # Script: SFT training with early-stopping validation
├── test_eval.py        # Script: Evaluation logic
└── main.py             # Orchestrator
```

## Details

### 1. Configuration (`configs/config.yaml`)
* Use **YAML** for human-readable input settings (hyperparameters, prompt templates, paths).
* Use **JSON** for machine-generated outputs (metrics, processed data).

### 2. Data Preprocessing (`data_preprocess.py`)
* **Rewrite:** Convert raw data into SFT format using defined prompt templates.
* **Split:** Shuffle (seed 42) and split into train, validation, and test sets based on `train_set_size`, `val_set_size`, and `test_set_size` in config.
* **Store:** Save to `data/processed/train.jsonl`, `val.jsonl`, and `test.jsonl`.

### 3. SFT Training with Early Stopping (`sft_train.py`)
* **Load:** Initialize the base model in 4-bit quantization.
* **Train:** Run SFT training on the full train set using **4bit QLoRA**.
* **Validation Early Stopping:** Use Transformers' native `EarlyStoppingCallback` with `steps` strategy, `eval_loss` metric, and `load_best_model_at_end`.
* **Save:** Save the best LoRA adapters to `outputs/lora_adapter/`.

### 4. Testing & Evaluating (`test_eval.py`)
* Run evaluation on the test set after training.
* Log results to `outputs/eval_stats.json` and `wandb`.

### 5. Main Execution (`main.py`)
* Orchestrate the pipeline: Preprocess -> Train.

---
**Initial Test Case:**
* **Base Model:** `Qwen2.5-1.5B`
* **Train Set Size:** 1024
* **Batch Size:** 128
* **Context Length:** 512

## Code Style Guide

To maintain readability and facilitate collaboration, all code must follow these documentation standards:

### 1. Module Documentation
Every Python file must start with a header docstring containing:
* **Description:** A brief summary of the module's purpose.
* **Usage:** Example of how to run the script or import its core logic.
* **Dependencies:** Key libraries used.

### 2. Function Documentation
Every function must include a docstring using the following format:
* **Description:** What the function does.
* **Args:** Name, type, and description of each parameter.
* **Returns:** Type and description of the return value.

**Example:**
```python
def load_config(config_path: str) -> dict:
    '''
    Loads training configuration from a YAML file.
    
    Args:
        config_path (str): The absolute or relative path to the .yaml file.
        
    Returns:
        dict: A dictionary containing nested configuration parameters.
    '''
    ...
```

### 3. General Rules
* **Comments:** Use inline comments sparingly to explain "why" something is done, not "what" is being done.
* **Typing:** Use Python type hints (`List`, `Dict`, `Optional`, etc.) for all function signatures.
* **Naming:** Use `snake_case` for functions/variables and `PascalCase` for classes.



## Workflow

### Stage 1: Environment & Foundation
   * **Base Model Initialization**: Download and place the starting base model (e.g., Qwen2.5-1.5B-Instruct) into `models/`.
   * **Infrastructure Verification**: Ensure the .venv-QEDpediaCN-Qwen environment is correctly populated with the dependencies listed above.
   * **Config Validation**: Finalize `configs/config.yaml` to match the local paths and intended hardware constraints.

### Stage 2: Data Engineering (`data_preprocess.py`)
   * **Ingestion**: Download the `sft_qa` subset of the Fineweb-Edu-Chinese-V2.2 dataset.
   * **Formatting**: Convert raw samples into the Instruction/Output format required by TRL.
   * **Splitting**: Shuffle and split into train, validation, and test sets.
       * Save as `train.jsonl`, `val.jsonl`, and `test.jsonl` in `data/processed/`.

### Stage 3: SFT Training with Early Stopping (`sft_train.py`)
   1. **4-bit Quantization**: Load the base model using bitsandbytes.
   2. **LoRA Configuration**: Initialize LoRA adapters targeting all linear layers.
   3. **Training**: Execute SFTTrainer on the full train set with validation early stopping.
   4. **Early Stopping**: Uses Transformers' `EarlyStoppingCallback` with `steps` strategy, `eval_loss` metric, and configurable patience/threshold.
   5. **Save Best Model**: The best checkpoint (lowest eval_loss) is saved to `outputs/lora_adapter/`.

### Stage 4: Evaluation & Monitoring (`test_eval.py`)
   * **Benchmark Suite**: Run the model against the test set.
   * **Metrics Tracking**: Calculate loss and perplexity.
   * **Telemetry**: Push results to Weights & Biases (wandb) and append to `outputs/eval_stats.json`.

### Stage 5: Orchestration (`main.py`)
   The orchestrator automates the entire lifecycle:

   ```
   1 # Conceptual logic for main.py
   2 preprocess_data()
   3 train_with_early_stopping()
   ```
