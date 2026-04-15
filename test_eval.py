"""
Description: Script for evaluating the current model on the held-out test set.
Usage: python test_eval.py --config configs/config.yaml
Dependencies: transformers, torch, datasets, numpy
"""

import os
import torch
import json
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from src.utils import load_config

def calculate_perplexity(model, tokenizer, dataset, max_length=512):
    model.eval()
    nlls = []
    
    print("Evaluating perplexity...")
    for example in tqdm(dataset):
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output_text = example.get("output", "")
        
        full_text = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output_text}"
        
        encodings = tokenizer(full_text, return_tensors="pt", max_length=max_length, truncation=True)
        input_ids = encodings.input_ids.to(model.device)
        target_ids = input_ids.clone()
        
        # We only want to calculate loss on the output part (optional but better)
        # For simplicity now, we calculate on the whole sequence
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
            nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item(), torch.stack(nlls).mean().item()

def main(config_path: str):
    config = load_config(config_path)
    model_config = config["model"]
    logging_config = config["logging"]
    
    model_path = model_config["current_model_path"]
    if not os.path.exists(model_path) or not os.listdir(model_path):
        model_path = model_config["base_model_path"]
    
    test_shard_path = os.path.join(
        os.path.dirname(config["dataset"]["processed_data_path"]),
        "shards",
        "test.jsonl"
    )
    
    print(f"Loading model for evaluation: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Load in 8-bit or FP16 for eval to save memory
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Loading test set: {test_shard_path}")
    test_dataset = load_dataset("json", data_files=test_shard_path, split="train")
    
    ppl, loss = calculate_perplexity(model, tokenizer, test_dataset, max_length=config["training"]["context_length"])
    
    metrics = {
        "test_loss": loss,
        "perplexity": ppl
    }
    
    print(f"\nEvaluation Results:")
    print(f"  - Loss: {loss:.4f}")
    print(f"  - Perplexity: {ppl:.4f}")
    
    # Save results
    eval_output_path = os.path.join(logging_config["output_dir"], "eval_stats.json")
    os.makedirs(logging_config["output_dir"], exist_ok=True)
    
    # Append to existing results if any
    results = []
    if os.path.exists(eval_output_path):
        with open(eval_output_path, 'r') as f:
            try:
                results = json.load(f)
            except:
                results = []
    
    results.append(metrics)
    with open(eval_output_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Metrics saved to {eval_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model on the test set.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    main(args.config)
