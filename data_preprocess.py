"""
Description: Script for preprocessing the Fineweb-Edu-Chinese-V2.2 dataset.
Usage: python data_preprocess.py --config configs/config.yaml
Dependencies: datasets, pyyaml (via src.utils)
"""

import os
os.environ['HF_HOME'] = './hf_cache'  # Set Hugging Face cache directory
import re
import json
from datasets import load_dataset
from typing import Dict, Any, List
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

# def remove_surrogates(text):
#     '''
#     Remove surrogate characters from the text that cannot be encoded in UTF-8.
    
#     Args:
#         text (str): The input text that may contain surrogate characters.
        
#     Returns:
#         str: The cleaned text with surrogate characters removed.
#     '''
#     if text is None:
#         return ""
#     # 直接过滤所有无法用 utf-8 存储的字符
#     return text.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")

# def clean_example(ex):
#     '''
#     Clean a dataset example by removing surrogate characters from all fields.
    
#     Args:
#         ex (Dict[str, Any]): A single dataset entry with 'instruction', 'input', and 'output' fields.

#     Returns:
#         Dict[str, str]: The cleaned dataset entry with surrogate characters removed.
#     '''
#     return {
#         "instruction": remove_surrogates(ex["instruction"]),
#         "input": remove_surrogates(ex["input"]),
#         "output": remove_surrogates(ex["output"])
#     }

# def decode_unicode(text):
#     '''
#     Decode unicode escape sequences in the text (e.g., "\\u4e2d\\u6587" -> "中文").
    
#     Args:
#         text (str): The input text with potential unicode escape sequences.

#     Returns:
#         str: The decoded text with unicode characters.
#     '''
#     if text is None:
#         return ""
#     return str(text).encode("utf-8").decode("unicode_escape")

# def safe_decode_unicode(text):
#     '''
#     Safely decode unicode escape sequences in the text, while ignoring invalid ones.
    
#     Args:
#         text (str): The input text with potential unicode escape sequences.
        
#     Returns:
#         str: The decoded text with unicode characters, ignoring invalid escape sequences.
#     '''
#     if text is None or text == "":
#         return ""
    
#     text = str(text)
    
#     # 核心：严格只处理标准的 \u4e00 这种中文转义，忽略非法 \U
#     # 1. 先尝试正常解码
#     try:
#         return text.encode("utf-8").decode("unicode_escape")
#     except:
#         pass
    
#     # 2. 解码失败 → 只替换合法的 \uXXXX，保留其他内容
#     def replace_match(match):
#         try:
#             return match.group(0).encode("utf-8").decode("unicode_escape")
#         except:
#             return match.group(0)  # 坏的转义原样返回
    
#     # 只匹配标准中文unicode：\uXXXX
#     text = re.sub(r'\\u[0-9a-fA-F]{4}', replace_match, text)
#     return text

# def decode_all_fields(example):
#     '''
#     Decode unicode escape sequences in all relevant fields of the dataset entry.
    
#     Args:
#         example (Dict[str, Any]): A single dataset entry with 'instruction', 'input', and 'output' fields.
        
#     Returns:
#         Dict[str, Any]: The dataset entry with decoded text.
#     '''
#     example["instruction"] = safe_decode_unicode(example["instruction"])
#     example["input"] = safe_decode_unicode(example["input"])
#     example["output"] = safe_decode_unicode(example["output"])
#     return example

# def remove_surrogates(text):
#     """Remove surrogate characters from the text that cannot be encoded in UTF-8."""
#     if not text:
#         return ""
#     text = str(text)
#     # 【底层编码级过滤】直接丢弃所有无法UTF-8编码的字符（包括所有surrogates）
#     return text.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")

# def decode_unicode_escape(text):
#     """Safe decode unicode escape sequences in the text, ignoring invalid ones, and preserving the original text structure."""
#     # if not text:
#     #     return ""
#     # 正则匹配标准的 \uXXXX 格式，只解码这部分
#     def replace_match(match):
#         try:
#             return match.group(0).encode("utf-8").decode("unicode_escape")
#         except:
#             return match.group(0)  # 异常字符原样保留
#     return re.sub(r'\\u[0-9a-fA-F]{4}', replace_match, text)

# def process_data(example):
#     # 第一步：清理非法字符
#     instruction = remove_surrogates(example["instruction"])
#     input_txt = remove_surrogates(example["input"])
#     output = remove_surrogates(example["output"])
    
#     # 第二步：解码转义字符
#     example["instruction"] = decode_unicode_escape(instruction)
#     example["input"] = decode_unicode_escape(input_txt)
#     example["output"] = decode_unicode_escape(output)
    
#     return example

def main(config_path: str):
    '''
    Main function to orchestrate data preprocessing: loading, formatting, and sharding.
    
    Args:
        config_path (str): Path to the configuration YAML file.
    '''
    config = load_config(config_path)
    dataset_config = config["dataset"]
    training_config = config["training"]

    raw_data_dir = dataset_config["raw_data_path"]
    output_dir = dataset_config["processed_data_path"]
    train_set_size = training_config["train_subset_size"] * training_config["num_iterations"]  # Total training samples across all iterations
    test_set_size = training_config["test_set_size"]

    # Ensure output directory exists
    # output_dir = os.path.abspath(processed_data_path)
    os.makedirs(output_dir, exist_ok=True)
    # check if output_dir created successfully
    if not os.path.exists(output_dir):
        print(f"Error: Failed to create output directory at {output_dir}")
        return
    else:
        print(f"Output directory is set to: {output_dir}")

    print(f"Loading dataset: {dataset_config["name"]}")
    # Load the sft_qa split of the dataset
    dataset = load_dataset('json', data_dir=raw_data_dir, split='train')  # Assuming the raw data is in JSON format and using 'train' split for processing
    # dataset = dataset.select_columns(["instruction", "input", "output"])  # Keep only relevant columns

    print(f"Formatting {len(dataset)} entries...")
    # Format the dataset entries into the required structure for SFT
    processed_dataset = dataset.map(format_dataset_entry, remove_columns=dataset.column_names)

    # print("Removing surrogate characters from all fields...")
    # # Remove surrogate characters from all fields to ensure clean text data
    # cleaned_dataset = processed_dataset.map(clean_example)

    # print("Decoding unicode escape sequences in all fields...")
    # # Decode unicode escape sequences in the instruction, input, and output fields to ensure proper text representation
    # decoded_dataset = cleaned_dataset.map(decode_all_fields)

    # decoded_dataset = processed_dataset.map(process_data)

    # Shuffle and split into train and test sets
    shuffled_dataset = processed_dataset.shuffle(seed=42)
    
    # Adjust sizes if the dataset is smaller than requested
    total_available_samples = len(shuffled_dataset)
    if (train_set_size + test_set_size) > total_available_samples:
        print(f"Warning: Requested train+test size ({train_set_size + test_set_size}) exceeds total available samples ({total_available_samples}). Adjusting sizes.")
        if total_available_samples > test_set_size:
            train_set_size = total_available_samples - test_set_size
        else: # not enough samples even for test set
            test_set_size = total_available_samples
            train_set_size = 0
    
    # Create a small test set for evaluation
    test_dataset = shuffled_dataset.select(range(test_set_size))
    train_dataset = shuffled_dataset.select(range(test_set_size, test_set_size + train_set_size))

    # Save processed dataset
    # We will create a 'shards' directory and save individual shards for iterative training
    shards_dir = os.path.join(output_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)
    # check if shards_dir created successfully
    if not os.path.exists(shards_dir):
        print(f"Error: Failed to create shards directory at {shards_dir}")
        return
    else: 
        print(f"Shards directory is set to: {shards_dir}")

    print(f"Saving shards to {shards_dir}...")

    # Save training shards
    num_iterations = training_config["num_iterations"]
    samples_per_shard = training_config["train_subset_size"]

    for i in range(num_iterations):
        start_idx = i * samples_per_shard
        end_idx = start_idx + samples_per_shard
        shard = train_dataset.select(range(start_idx, end_idx))
        shard_path = os.path.join(shards_dir, f"train_shard_{i}.jsonl")
        shard.to_json(shard_path, force_ascii=False)
        print(f"  - Saved shard {i} to {shard_path} ({len(shard)} samples)")
        # Optional: Verify the shard was saved correctly
        if os.path.exists(shard_path):
            print(f"    Verified shard {i} saved successfully.")
        else:
            print(f"    Error: Failed to save shard {i} at {shard_path}")

    # Save test set
    test_path = os.path.join(shards_dir, "test.jsonl")
    test_dataset.to_json(test_path, force_ascii=False)
    print(f"Saved test set to {test_path} ({len(test_dataset)} samples)")
    # Optional: Verify the test set was saved correctly
    if os.path.exists(test_path):
        print("Verified test set saved successfully.")
    else:
        print(f"Error: Failed to save test set at {test_path}")

    # Also save the combined metadata/file for reference if needed
    processed_data_info = {
        "num_shards": num_iterations,
        "samples_per_shard": samples_per_shard,
        "test_samples": len(test_dataset),
        "total_train_samples": len(train_dataset),
        "shards_dir": shards_dir
    }

    with open(os.path.join(output_dir, "dataset_info.json"), 'w', encoding='utf-8') as f:
        json.dump(processed_data_info, f, ensure_ascii=False, indent=4)

    print(f"Dataset info saved to {os.path.join(output_dir, 'dataset_info.json')}")
    # Final verification of the processed dataset
    print("\nFinal verification of processed dataset:")
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total test samples: {len(test_dataset)}")
    print(f"Number of shards created: {num_iterations}")
    print(f"Samples per shard: {samples_per_shard}")
    # check file existence and sample content
    for i in range(num_iterations):
        shard_path = os.path.join(shards_dir, f"train_shard_{i}.jsonl")
        if os.path.exists(shard_path):
            shard_ds = load_dataset('json', data_files=shard_path, split='train')
            print(f"Shard {i} samples: {len(shard_ds)}")
            print(f"Sample 0 Instruction: {shard_ds[0]['instruction'][:50]}...")
        else:
            print(f"Error: Shard file {shard_path} does not exist.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Data preprocessing script.")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the configuration YAML file.')
    args = parser.parse_args()
    main(args.config)

    # Verification
    config = load_config(args.config)
    output_dir = config["dataset"]["processed_data_path"]
    info_path = os.path.join(output_dir, "dataset_info.json")
    
    if os.path.exists(info_path):
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
            print("\nVerification:")
            print(f"Total shards: {info['num_shards']}")
            
            # Check the first shard
            first_shard_path = os.path.join(info['shards_dir'], "train_shard_0.jsonl")
            if os.path.exists(first_shard_path):
                shard_ds = load_dataset('json', data_files=first_shard_path, split='train')
                print(f"First shard samples: {len(shard_ds)}")
                print(f"Sample 0 Instruction: {shard_ds[0]['instruction'][:50]}...")
            
            # Check test set
            test_path = os.path.join(info['shards_dir'], "test.jsonl")
            if os.path.exists(test_path):
                test_ds = load_dataset('json', data_files=test_path, split='train')
                print(f"Test set samples: {len(test_ds)}")