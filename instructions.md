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
├── models/             # Local storage for base and current merged models
├── outputs/            # Iteration-specific LoRA weights and training logs
├── src/                # Modular logic (model loading, merging, utilities)
├── data_preprocess.py  # Script: Raw -> SFT format & Sharding
├── sft_train.py        # Script: Single iteration SFT logic
├── test_eval.py        # Script: Evaluation logic
└── main.py             # Orchestrator
```

## Details

### 1. Configuration (`configs/config.yaml`)
* Use **YAML** for human-readable input settings (hyperparameters, prompt templates, paths).
* Use **JSON** for machine-generated outputs (metrics, processed data).

### 2. Data Preprocessing (`data_preprocess.py`)
* **Rewrite:** Convert raw data into SFT format using defined prompt templates.
* **Store:** Save to `data/processed/sft_data.json`.
* **Shard:** Load, randomize, and split into `TRAIN_SET_NUM` shards (size `TRAIN_SET_SIZE`) and a test set (size `TEST_SET_SIZE`).

### 3. Iterative SFT Training (`sft_train.py`)
* **Load:** Initialize the model from `models/current_model/` in 4-bit.
* **Train:** Implement one iteration of SFT training with **4bit QLoRA**.
* **Archive:** Save the LoRA adapters to `outputs/lora_iter_N/` to maintain a historical record of all iterations.
* **Safe Overwrite (Atomic Merge):**
    1. Merge the new LoRA weights with the current model into a temporary directory.
    2. Validate the integrity of the merged model.
    3. Replace the old model in `models/current_model/` with the new version.

### 4. Testing & Evaluating (`test_eval.py`)
* Run evaluation on the test set after each iteration.
* Log results to `outputs/eval_stats.json` and `wandb`.

### 5. Main Execution (`main.py`)
* Orchestrate the full lifecycle: Setup -> Preprocess -> Iterative Loop (Train -> Eval -> Merge).
* Plot and print final statistics across all iterations.

---
**Initial Test Case:**
* **Base Model:** `Qwen2.5-1.5B`
* **Train Set Size:** 1024 (per iteration)
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
   * **Base Model Initialization**: Download and place the starting base model (e.g., Qwen2.5-1.5B-Instruct) into models/current_model/.
   * **Infrastructure Verification**: Ensure the .venv-QEDpediaCN-Qwen environment is correctly populated with the dependencies listed in instructions.md (PEFT, TRL, etc.).
   * **Config Validation**: Finalize configs/config.yaml to match the local paths and intended hardware constraints.

### Stage 2: Data Engineering (data_preprocess.py)
   * **Ingestion**: Download the sft_qa subset of the Fineweb-Edu-Chinese-V2.2 dataset.
   * **Formatting**: Convert raw samples into the Instruction/Output format required by TRL.
   * **Sharding**: Implement a deterministic sharding strategy.
       * Example: Split the dataset into $N$ shards (e.g., 1024 samples each) to facilitate the iterative learning loop.
       * Save shards as separate JSON files or indexed entries in a master metadata file.

### Stage 3: The Iterative Training Loop (sft_train.py)
  This script will be designed to run a single "turn" of the training process:
   1. **4-bit Quantization**: Load the model from models/current_model/ using bitsandbytes.
   2. **LoRA Configuration**: Initialize LoRA adapters (Targeting q_proj, v_proj, etc.).
   3. **Shard Training**: Execute the SFTTrainer on the current iteration's data shard.
   4. **Adapter Archiving**: Save the resulting LoRA weights to outputs/lora_iter_{i}/ for version control.

### Stage 4: Atomic Model Management (src/model_utils.py)
  To ensure the "Base Model" evolves without corruption:
   * **Merge Logic**: Implement a utility to merge models/current_model + outputs/lora_iter_{i} into a temporary directory.
   * **Integrity Check**: Verify the merged model can be reloaded without error.
   * **Atomic Swap**: Replace the contents of models/current_model with the new merged weights.

### Stage 5: Evaluation & Monitoring (test_eval.py)
   * **Benchmark Suite**: Implement logic to run the model against a static test_set (created during preprocessing).
   * **Metrics Tracking**: Calculate loss, perplexity, or specific QA accuracy metrics.
   * **Telemetry**: Push results to Weights & Biases (wandb) and append to a local outputs/eval_stats.json.

### Stage 6: Orchestration (main.py)
  The final orchestrator will automate the entire lifecycle:

   ```
   1 # Conceptual logic for main.py
   2 for i in range(NUM_ITERATIONS):
   3     run_training(shard=i)
   4     perform_merge(iteration=i)
   5     run_evaluation(iteration=i)
   6     generate_iteration_plots()
   ```
 