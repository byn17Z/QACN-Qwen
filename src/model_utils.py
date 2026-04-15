"""
Description: Utility functions for model merging and management.
Usage: from src.model_utils import merge_lora_to_base
Dependencies: transformers, peft, torch, shutil
"""

import os
import torch
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_lora_to_base(base_model_path: str, lora_adapter_path: str, output_path: str):
    """
    Merges LoRA adapters into a base model and saves the result.
    
    Args:
        base_model_path (str): Path to the base model.
        lora_adapter_path (str): Path to the LoRA adapter directory.
        output_path (str): Path where the merged model will be saved.
    """
    print(f"Merging LoRA from {lora_adapter_path} into base model {base_model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # Load base model in FP16/BF16 for merging
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map="cpu", # Merge on CPU to save VRAM, or "auto" if enough VRAM
        trust_remote_code=True,
    )
    
    # Load LoRA model
    model = PeftModel.from_pretrained(
        base_model,
        lora_adapter_path,
        torch_dtype=torch_dtype,
    )
    
    # Merge and unload
    print("Merging weights...")
    merged_model = model.merge_and_unload()
    
    # Save merged model
    print(f"Saving merged model to {output_path}...")
    if os.path.exists(output_path):
        # Create a backup or temp name to ensure atomic-like swap later
        temp_output_path = output_path + "_temp"
        if os.path.exists(temp_output_path):
            shutil.rmtree(temp_output_path)
        merged_model.save_pretrained(temp_output_path)
        tokenizer.save_pretrained(temp_output_path)
        
        # Safe swap
        backup_path = output_path + "_backup"
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        
        os.rename(output_path, backup_path)
        os.rename(temp_output_path, output_path)
        shutil.rmtree(backup_path)
    else:
        merged_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
    print("Merge complete.")

def initialize_current_model(base_model_path: str, current_model_path: str):
    """
    Initializes the current_model directory with the base model if it doesn't exist.
    Actually, we can just copy the base model or symlink it. 
    Copying is safer for atomic updates.
    """
    if not os.path.exists(current_model_path) or not os.listdir(current_model_path):
        print(f"Initializing current model from {base_model_path}...")
        os.makedirs(os.path.dirname(current_model_path), exist_ok=True)
        # We can use shutil.copytree, but it might be large. 
        # For the first iteration, sft_train.py already handles falling back to base_model_path.
        # But for consistency, let's copy if it doesn't exist.
        shutil.copytree(base_model_path, current_model_path, dirs_exist_ok=True)
        print("Initialization complete.")
