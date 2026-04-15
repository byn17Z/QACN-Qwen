"""
Description: Script for performing one iteration of SFT training using 4-bit QLoRA.
Usage: python sft_train.py --config configs/config.yaml --shard_idx 0
Dependencies: transformers, peft, trl, bitsandbytes, torch, datasets
"""

import os
import torch
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
from src.utils import load_config

def main(config_path: str, shard_idx: int):
    # Load configuration
    config = load_config(config_path)
    model_config = config["model"]
    training_config = config["training"]
    logging_config = config["logging"]

    # Paths
    model_path = model_config["current_model_path"]
    # If the current_model_path doesn't exist (first iteration), fall back to base_model_path
    if not os.path.exists(model_path) or not os.listdir(model_path):
        print(f"Current model path {model_path} not found or empty. Using base model: {model_config['base_model_path']}")
        model_path = model_config["base_model_path"]

    shard_path = os.path.join(
        os.path.dirname(config["dataset"]["processed_data_path"]),
        "shards",
        f"train_shard_{shard_idx}.jsonl"
    )
    output_dir = os.path.join(logging_config["output_dir"], f"lora_iter_{shard_idx}")

    print(f"Starting training for iteration {shard_idx}...")
    print(f"Loading model from: {model_path}")
    print(f"Loading shard from: {shard_path}")

    # 1. 4-bit Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )

    # 2. Load Model and Tokenizer
    # check if model_path exists
    if not os.path.exists(model_path) or not os.listdir(model_path):
        print(f"Error: Model path {model_path} not found or empty. Please check your configuration.")
        return
    else:
        print(f"Model path {model_path} found. Proceeding to load the model.")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # 3. Prepare for QLoRA
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=training_config["lora_r"],
        lora_alpha=training_config["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Target all linear layers
        lora_dropout=training_config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # 4. Load Dataset
    # check if shard_path exists
    if not os.path.exists(shard_path):
        print(f"Error: Shard path {shard_path} not found. Please check your data preprocessing step.")
        return
    else: 
        print(f"Shard path {shard_path} found. Proceeding to load the dataset.")
    dataset = load_dataset("json", data_files=shard_path, split="train")

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            text = f"Instruction: {example['instruction'][i]}\nInput: {example['input'][i]}\nOutput: {example['output'][i]}"
            output_texts.append(text)
        return output_texts

    # 5. Trainer Setup
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1, # Adjust based on GPU memory
        gradient_accumulation_steps=training_config["batch_size"], # Simulate large batch size
        learning_rate=training_config["learning_rate"],
        num_train_epochs=training_config.get("num_train_epochs", 1),
        logging_steps=10,
        save_strategy="no",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="wandb" if logging_config["use_wandb"] else "none",
        run_name=f"{logging_config['project_name']}-iter-{shard_idx}"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=training_config["context_length"],
        dataset_text_field="text", # Not used because we use formatting_func
        formatting_func=formatting_prompts_func,
    )

    # 6. Train and Save
    print("Training...")
    trainer.train()
    
    print(f"Saving LoRA adapters to {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Optional: Verify that the LoRA adapters were saved correctly
    if os.path.exists(output_dir) and os.listdir(output_dir):
        if os.path.exists(os.path.join(output_dir, "pytorch_model.bin")) and os.path.exists(os.path.join(output_dir, "tokenizer_config.json")):
            print(f"LoRA adapters saved successfully in {output_dir}.")
        else: 
            print(f"Warning: LoRA adapters saved in {output_dir}, but expected files not found. Please check the saving process.")
    else:
        print(f"Error: Failed to save LoRA adapters in {output_dir}. Please check the training output and saving process.")
    
    print("Iteration complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one iteration of QLoRA SFT.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--shard_idx", type=int, required=True, help="Index of the data shard to train on")
    args = parser.parse_args()
    
    main(args.config, args.shard_idx)
