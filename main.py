"""
Description: Main orchestrator script for the QEdpediaCN-Qwen iterative training loop.
Usage: python main.py --config configs/config.yaml
"""

import os
# 启用国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import subprocess
import argparse
import sys
from src.utils import load_config
from src.model_utils import merge_lora_to_base, initialize_current_model

def run_command(command: list):
    """Utility to run shell commands and check for errors."""
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        # Optional: raise exception or exit
        # exit(result.returncode)

def main(config_path: str):
    config = load_config(config_path)
    training_config = config["training"]
    model_config = config["model"]
    logging_config = config["logging"]
    
    num_iterations = training_config["num_iterations"]
    python_path = sys.executable
    
    # 1. Preprocess Data
    print("=== Stage 1: Data Preprocessing ===")
    run_command([python_path, "data_preprocess.py", "--config", config_path])
    
    # Ensure current_model directory is initialized or sft_train handles it
    # We'll let sft_train handle the fallback to base_model for the first iteration.
    
    # 2. Iterative Loop
    for i in range(num_iterations):
        print(f"\n=== Iteration {i}/{num_iterations-1} ===")
        
        # A. Train
        print(f"--- Training Shard {i} ---")
        run_command([python_path, "sft_train.py", "--config", config_path, "--shard_idx", str(i)])
        
        # B. Merge
        print(f"--- Merging LoRA {i} ---")
        # Determine the base model for this merge
        # If it's the first iteration and current_model doesn't exist, use base_model
        current_model_path = model_config["current_model_path"]
        base_to_merge_with = current_model_path
        if not os.path.exists(current_model_path) or not os.listdir(current_model_path):
            base_to_merge_with = model_config["base_model_path"]
            
        lora_path = os.path.join(logging_config["output_dir"], f"lora_iter_{i}")
        
        merge_lora_to_base(
            base_model_path=base_to_merge_with,
            lora_adapter_path=lora_path,
            output_path=current_model_path
        )
        
        # C. Evaluate
        print(f"--- Evaluating Iteration {i} ---")
        run_command([python_path, "test_eval.py", "--config", config_path])
        
    print("\n=== Pipeline Complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QEdpediaCN-Qwen Orchestrator")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    main(args.config)
