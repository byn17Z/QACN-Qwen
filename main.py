"""
Description: Main orchestrator script for the QEdpediaCN-Qwen training pipeline.
Usage: python main.py --config configs/config.yaml
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import subprocess
import argparse
import sys
from datetime import datetime
from src.utils import load_config

def run_command(command: list):
    """Utility to run shell commands and check for errors."""
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")

def main(config_path: str):
    config = load_config(config_path)
    python_path = sys.executable
    timestamp = datetime.now().isoformat()

    # 1. Preprocess Data
    print("=== Stage 1: Data Preprocessing ===")
    run_command([python_path, "data_preprocess.py", "--config", config_path])

    # 2. Train
    print("\n=== Stage 2: Training ===")
    run_command([python_path, "sft_train.py", "--config", config_path, "--timestamp", timestamp])

    # 3. Evaluate on test set
    print("\n=== Stage 3: Test Evaluation ===")
    run_command([python_path, "test_eval.py", "--config", config_path, "--mode", "test", "--timestamp", timestamp])

    print("\n=== Pipeline Complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QEdpediaCN-Qwen Orchestrator")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    main(args.config)
