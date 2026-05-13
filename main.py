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

def run_command(command: list) -> int:
    """Utility to run shell commands. Returns the process return code."""
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
    return result.returncode

def main(config_path: str):
    config = load_config(config_path)
    python_path = sys.executable
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # 0.5. Knowledge Distillation (optional, must run before preprocessing)
    distill_config = config.get("distill", {})
    if distill_config.get("enabled", False):
        print("=== Stage 0.5: Knowledge Distillation ===")
        rc = run_command([python_path, "distill.py", "--config", config_path, "--yes"])
        if rc != 0:
            print("Distillation failed. Aborting pipeline.")
            return

    # 1. Preprocess Data
    print("\n=== Stage 1: Data Preprocessing ===")
    rc = run_command([python_path, "data_preprocess.py", "--config", config_path])
    if rc != 0:
        print("Preprocessing failed. Aborting pipeline.")
        return

    # 1.5. RAG Knowledge Base (after preprocessing, since it reads sft_data.jsonl)
    rag_config = config.get("rag", {})
    any_rag_flag = (
        rag_config.get("rag_inference_test", False)
        or rag_config.get("rag_inference_train", False)
        or rag_config.get("rag_inference_deploy", False)
    )

    if any_rag_flag:
        if rag_config.get("build_rag_db", False):
            print("\n=== Stage 1.5: Building RAG Knowledge Base ===")
            from rag.rag_db import build_knowledge_db
            sft_data_path = os.path.join(config["dataset"]["processed_data_path"], "sft_data.jsonl")
            if not os.path.exists(sft_data_path):
                print(f"Error: {sft_data_path} not found. Run preprocessing with use_processed_data: false first.")
                return
            build_knowledge_db(sft_data_path, rag_config["db_path"], rag_config["embedder_path"])
        else:
            print("\n=== Stage 1.5: Loading Existing RAG Knowledge Base ===")
            from rag.rag_db import load_knowledge_db
            load_knowledge_db(rag_config["db_path"], rag_config["embedder_path"])

    # 2. Benchmark Evaluation (before training)
    print("\n=== Stage 2: Benchmark Evaluation (Pre-Training) ===")
    rc = run_command([python_path, "test_eval.py", "--config", config_path, "--mode", "validation", "--timestamp", timestamp])
    if rc != 0:
        print("Pre-training validation eval failed. Aborting pipeline.")
        return
    rc = run_command([python_path, "test_eval.py", "--config", config_path, "--mode", "test", "--timestamp", timestamp])
    if rc != 0:
        print("Pre-training test eval failed. Aborting pipeline.")
        return

    # 3. Train
    print("\n=== Stage 3: Training ===")
    rc = run_command([python_path, "sft_train.py", "--config", config_path, "--timestamp", timestamp])
    if rc != 0:
        print("Training failed. Aborting pipeline.")
        return

    # 3.5. Merge LoRA to base model
    print("\n=== Stage 3.5: Merging LoRA Adapter ===")
    lora_adapter_path = os.path.join(config["logging"]["output_dir"], "lora_adapter", timestamp)
    current_model_path = config["model"]["current_model_path"]
    merge_lora_to_base(config["model"]["base_model_path"], lora_adapter_path, current_model_path)

    # 4. Post-Training Evaluation (use distinct timestamp suffix)
    post_timestamp = timestamp + "-post"
    print("\n=== Stage 4: Post-Training Evaluation ===")
    run_command([python_path, "test_eval.py", "--config", config_path, "--mode", "validation", "--timestamp", post_timestamp])
    run_command([python_path, "test_eval.py", "--config", config_path, "--mode", "test", "--timestamp", post_timestamp])

    print("\n=== Pipeline Complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QEdpediaCN-Qwen Orchestrator")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    main(args.config)
