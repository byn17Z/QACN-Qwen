"""
Description: Main orchestrator script for the QEdpediaCN-Qwen training pipeline.
Usage: python main.py --config configs/config.yaml
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTHONUTF8"] = "1"
import subprocess
import argparse
import sys
from datetime import datetime
from src.utils import load_config
from src.model_utils import merge_lora_to_base

def run_command(command: list):
    """Utility to run shell commands and check for errors."""
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")

def main(config_path: str):
    config = load_config(config_path)
    python_path = sys.executable
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # 1. Preprocess Data
    print("=== Stage 1: Data Preprocessing ===")
    run_command([python_path, "data_preprocess.py", "--config", config_path])

    # 2. Benchmark Evaluation (before training)
    print("\n=== Stage 2: Benchmark Evaluation (Pre-Training) ===")
    run_command([python_path, "test_eval.py", "--config", config_path, "--mode", "validation", "--timestamp", timestamp])
    run_command([python_path, "test_eval.py", "--config", config_path, "--mode", "test", "--timestamp", timestamp])

    # 3. Train
    print("\n=== Stage 3: Training ===")
    run_command([python_path, "sft_train.py", "--config", config_path, "--timestamp", timestamp])

    # 3.5. Merge LoRA to base model
    print("\n=== Stage 3.5: Merging LoRA Adapter ===")
    lora_adapter_path = os.path.join(config["logging"]["output_dir"], "lora_adapter", timestamp)
    current_model_path = config["model"]["current_model_path"]
    merge_lora_to_base(config["model"]["base_model_path"], lora_adapter_path, current_model_path)

    # 4. Post-Training Evaluation
    print("\n=== Stage 4: Post-Training Evaluation ===")
    run_command([python_path, "test_eval.py", "--config", config_path, "--mode", "validation", "--timestamp", timestamp])
    run_command([python_path, "test_eval.py", "--config", config_path, "--mode", "test", "--timestamp", timestamp])

    print("\n=== Pipeline Complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QEdpediaCN-Qwen Orchestrator")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    main(args.config)
