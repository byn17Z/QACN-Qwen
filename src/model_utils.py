"""
Description: Utility functions for model management.
Usage: from src.model_utils import merge_lora_to_base
Dependencies: transformers, peft, torch
"""

import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_lora_to_base(base_model_path: str, lora_adapter_path: str, output_path: str):
    """
    Merges a LoRA adapter into the base model and saves the merged model.

    Args:
        base_model_path (str): Path to the base model.
        lora_adapter_path (str): Path to the LoRA adapter directory.
        output_path (str): Path to save the merged model.
    """
    print(f"Loading base model from: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from: {lora_adapter_path}")
    model = PeftModel.from_pretrained(model, lora_adapter_path)

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("Merge complete.")

    # Free memory
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
