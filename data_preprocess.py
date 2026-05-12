"""
Description: Script for preprocessing the Fineweb-Edu-Chinese-V2.2 dataset.
Usage: python data_preprocess.py --config configs/config.yaml
Dependencies: datasets, pyyaml (via src.utils)
"""

import os
os.environ['HF_HOME'] = './hf_cache'
import json
from datasets import load_dataset
from typing import Dict, Any
from src.utils import load_config

def format_dataset_entry(entry: Dict[str, Any]) -> Dict[str, str]:
    '''
    Formats a single dataset entry into the Instruction/Output format for SFT.

    Args:
        entry (Dict[str, Any]): A single entry from the raw dataset.

    Returns:
        Dict[str, str]: A dictionary with 'instruction', 'input', and 'output' keys.
    '''
    instruction = entry.get("instruction", "")
    input_text = entry.get("input", "")
    output_text = entry.get("output", "")
    return {"instruction": instruction, "input": input_text, "output": output_text}

def main(config_path: str):
    '''
    Main function to orchestrate data preprocessing: loading, formatting, and splitting.

    Args:
        config_path (str): Path to the configuration YAML file.
    '''
    config = load_config(config_path)
    dataset_config = config["dataset"]
    training_config = config["training"]

    raw_data_dir = dataset_config["raw_data_path"]
    output_dir = dataset_config["processed_data_path"]
    use_processed_data = dataset_config.get("use_processed_data", False)
    train_set_size = int(training_config["train_set_size"])
    val_set_size = int(training_config["val_set_size"])
    test_set_size = int(training_config["test_set_size"])

    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(output_dir):
        print(f"Error: Failed to create output directory at {output_dir}")
        return
    else:
        print(f"Output directory is set to: {output_dir}")

    sft_data_path = os.path.join(output_dir, "sft_data.jsonl")

    if use_processed_data:
        print(f"Loading processed dataset from: {sft_data_path}")
        processed_dataset = load_dataset('json', data_files=sft_data_path, split='train')
    else:
        print(f"Loading raw dataset: {dataset_config['name']}")
        dataset = load_dataset('json', data_dir=raw_data_dir, split='train')

        print(f"Formatting {len(dataset)} entries...")
        processed_dataset = dataset.map(format_dataset_entry, remove_columns=dataset.column_names)

        print(f"Saving processed dataset to: {sft_data_path}")
        processed_dataset.to_json(sft_data_path, force_ascii=False)
        print(f"Processed dataset saved ({len(processed_dataset)} entries)")

    # Shuffle and split into train, val, and test sets
    shuffled_dataset = processed_dataset.shuffle(seed=42)

    total_available_samples = len(shuffled_dataset)
    requested_total = train_set_size + val_set_size + test_set_size
    if requested_total > total_available_samples:
        print(f"Warning: Requested total ({requested_total}) exceeds available samples ({total_available_samples}). Adjusting proportionally.")
        ratio = total_available_samples / requested_total
        train_set_size = int(train_set_size * ratio)
        val_set_size = int(val_set_size * ratio)
        test_set_size = total_available_samples - train_set_size - val_set_size

    train_dataset = shuffled_dataset.select(range(train_set_size))
    val_dataset = shuffled_dataset.select(range(train_set_size, train_set_size + val_set_size))
    test_dataset = shuffled_dataset.select(range(train_set_size + val_set_size, train_set_size + val_set_size + test_set_size))

    # Save train, val, and test sets
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")
    test_path = os.path.join(output_dir, "test.jsonl")

    train_dataset.to_json(train_path, force_ascii=False)
    print(f"Saved train set to {train_path} ({len(train_dataset)} samples)")

    val_dataset.to_json(val_path, force_ascii=False)
    print(f"Saved val set to {val_path} ({len(val_dataset)} samples)")

    test_dataset.to_json(test_path, force_ascii=False)
    print(f"Saved test set to {test_path} ({len(test_dataset)} samples)")

    # Save metadata
    processed_data_info = {
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
    }

    with open(os.path.join(output_dir, "dataset_info.json"), 'w', encoding='utf-8') as f:
        json.dump(processed_data_info, f, ensure_ascii=False, indent=4)

    print(f"Dataset info saved to {os.path.join(output_dir, 'dataset_info.json')}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Data preprocessing script.")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the configuration YAML file.')
    args = parser.parse_args()
    main(args.config)
