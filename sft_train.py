"""
Description: Script for SFT training using 4-bit QLoRA.
Usage: python sft_train.py --config configs/config.yaml
Dependencies: transformers, peft, trl, bitsandbytes, torch, datasets
"""

import os
import json
import torch
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
from src.utils import load_config

def main(config_path: str, timestamp: str):
    # Load configuration
    config = load_config(config_path)
    model_config = config["model"]
    training_config = config["training"]
    logging_config = config["logging"]

    # Paths
    model_path = model_config["base_model_path"]
    train_data_path = os.path.join(config["dataset"]["processed_data_path"], "train.jsonl")
    val_data_path = os.path.join(config["dataset"]["processed_data_path"], "val.jsonl")
    output_dir = os.path.join(logging_config["output_dir"], "lora_adapter")

    print(f"Loading model from: {model_path}")
    print(f"Loading training data from: {train_data_path}")
    print(f"Loading validation data from: {val_data_path}")

    # 1. 4-bit Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )

    # 2. Load Model and Tokenizer
    if not os.path.exists(model_path) or not os.listdir(model_path):
        print(f"Error: Model path {model_path} not found or empty. Please check your configuration.")
        return

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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=training_config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # 4. Load Datasets
    if not os.path.exists(train_data_path):
        print(f"Error: Training data {train_data_path} not found. Please run data_preprocess.py first.")
        return
    if not os.path.exists(val_data_path):
        print(f"Error: Validation data {val_data_path} not found. Please run data_preprocess.py first.")
        return

    train_dataset = load_dataset("json", data_files=train_data_path, split="train")
    val_dataset = load_dataset("json", data_files=val_data_path, split="train")

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            text = f"Instruction: {example['instruction'][i]}\nInput: {example['input'][i]}\nOutput: {example['output'][i]}"
            output_texts.append(text)
        return output_texts

    # 5. Trainer Setup
    eval_steps = training_config["eval_steps"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=training_config["batch_size"],
        learning_rate=training_config["learning_rate"],
        num_train_epochs=training_config.get("num_train_epochs", 1),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="wandb" if logging_config["use_wandb"] else "none",
        run_name=f"{logging_config['project_name']}-sft"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        max_seq_length=training_config["context_length"],
        formatting_func=formatting_prompts_func,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=training_config["early_stopping_patience"],
                early_stopping_threshold=training_config["early_stopping_threshold"],
            )
        ],
    )

    # 6. Train and Save
    print("Training...")
    trainer.train()

    print(f"Saving LoRA adapters to {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 7. Log training outputs
    train_logs = [entry for entry in trainer.state.log_history if "loss" in entry and "eval_loss" not in entry]

    train_metrics = {
        "timestamp": timestamp,
        "config": {
            "base_model_path": model_config["base_model_path"],
            "context_length": training_config["context_length"],
            "learning_rate": training_config["learning_rate"],
            "batch_size": training_config["batch_size"],
            "lora_r": training_config["lora_r"],
            "lora_alpha": training_config["lora_alpha"],
            "lora_dropout": training_config["lora_dropout"],
            "num_train_epochs": training_config["num_train_epochs"],
            "train_set_size": training_config["train_set_size"],
            "val_set_size": training_config["val_set_size"],
            "test_set_size": training_config["test_set_size"],
        },
        "train_loss_history": [
            {"step": entry["step"], "loss": entry["loss"]}
            for entry in train_logs
        ],
        "total_training_steps": trainer.state.global_step,
    }

    train_stats_path = os.path.join(logging_config["output_dir"], "train_stats.json")
    os.makedirs(logging_config["output_dir"], exist_ok=True)

    results = []
    if os.path.exists(train_stats_path):
        with open(train_stats_path, 'r') as f:
            try:
                results = json.load(f)
            except:
                results = []

    results.append(train_metrics)
    with open(train_stats_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Training stats saved to {train_stats_path}")
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA SFT Training")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp string to record in the log")
    args = parser.parse_args()

    main(args.config, args.timestamp)
