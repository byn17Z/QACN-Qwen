"""
Description: Knowledge distillation script. Reads raw data, calls DeepSeek API
             to generate high-quality teacher responses, saves distilled data.
Usage: python distill.py --config configs/config.yaml [--yes]
Dependencies: aiohttp, pyyaml (via src.utils)
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTHONUTF8"] = "1"
import json
import time
import random
import asyncio
import argparse
import glob
from datetime import datetime
from typing import Dict, Any, List, Optional
import aiohttp
from src.utils import load_config

# DeepSeek pricing per 1M tokens (as of 2026-05-14, v4-pro with 75% discount)
PRICING = {
    "deepseek-v4-pro": {
        "input_cache_hit": 0.003625,
        "input_cache_miss": 0.435,
        "output": 0.87,
    },
    "deepseek-v4-flash": {
        "input_cache_hit": 0.0005,
        "input_cache_miss": 0.06,
        "output": 0.12,
    },
}

SYSTEM_PROMPT = (
    "你是一位专业的中国教育问答助手。请根据提供的参考资料，生成高质量、准确、详细的教育问答回答。\n"
    "要求：\n"
    "1. 回答必须基于参考资料中的事实信息\n"
    "2. 使用清晰的中文，适当使用markdown格式\n"
    "3. 包含关键知识点的解释和扩展\n"
    "4. 如果参考资料是外文，请准确翻译并整合信息\n"
    "5. 回答应具有教育价值，适合学生学习使用"
)


def build_teacher_prompt(entry: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Build the chat messages for the DeepSeek teacher model.

    Args:
        entry: A raw data dict with keys: instruction, input, raw_content.

    Returns:
        List of message dicts for the OpenAI-compatible chat API.
    """
    instruction = entry.get("instruction", "")
    input_text = entry.get("input", "")
    raw_content = entry.get("raw_content", "")

    user_parts = []
    if raw_content:
        user_parts.append(f"参考资料：\n{raw_content}")
    user_parts.append(f"问题：{instruction}")
    if input_text:
        user_parts.append(f"附加输入：{input_text}")
    user_parts.append("请生成一个高质量的教育回答：")

    user_message = "\n\n".join(user_parts)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def load_raw_data(raw_data_dir: str, sample_size: int = 0, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load raw data entries from JSONL files.

    Args:
        raw_data_dir: Path to directory containing raw JSONL files.
        sample_size: If >0, randomly sample this many entries.
        seed: Random seed for sampling.

    Returns:
        List of dicts, each with at least: instruction, input, raw_content, output.
    """
    jsonl_files = sorted(glob.glob(os.path.join(raw_data_dir, "*.jsonl")))
    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {raw_data_dir}")

    entries = []
    for filepath in jsonl_files:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

    print(f"Loaded {len(entries)} entries from {len(jsonl_files)} files")

    if sample_size > 0 and sample_size < len(entries):
        random.seed(seed)
        entries = random.sample(entries, sample_size)
        print(f"Sampled {sample_size} entries (seed={seed})")

    return entries


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load checkpoint mapping {line_index: distilled_entry}.

    Returns:
        Dict with 'metadata' and 'entries' keys.
    """
    if not os.path.exists(checkpoint_path):
        return {"metadata": {}, "entries": {}}

    with open(checkpoint_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(checkpoint_path: str, checkpoint: Dict[str, Any]) -> None:
    """Atomically write the checkpoint file."""
    tmp_path = checkpoint_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, checkpoint_path)


def estimate_cost(entries: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    """
    Rough cost estimate using DeepSeek pricing.
    Uses len(text)/1.5 as Chinese char-to-token approximation.
    """
    total_input_chars = sum(
        len(e.get("instruction", "")) + len(e.get("input", "")) + len(e.get("raw_content", ""))
        for e in entries
    )
    total_input_tokens_est = int(total_input_chars / 1.5)
    total_output_tokens_est = int(total_input_tokens_est * 1.5)

    rates = PRICING.get(model, PRICING["deepseek-v4-pro"])
    cost_cache_hit = (
        (total_input_tokens_est / 1_000_000) * rates["input_cache_hit"]
        + (total_output_tokens_est / 1_000_000) * rates["output"]
    )
    cost_cache_miss = (
        (total_input_tokens_est / 1_000_000) * rates["input_cache_miss"]
        + (total_output_tokens_est / 1_000_000) * rates["output"]
    )

    return {
        "total_entries": len(entries),
        "estimated_input_tokens": total_input_tokens_est,
        "estimated_output_tokens": total_output_tokens_est,
        "estimated_cost_usd_range": [round(cost_cache_hit, 4), round(cost_cache_miss, 4)],
        "model": model,
    }


async def call_deepseek_api(
    session: aiohttp.ClientSession,
    messages: List[Dict[str, str]],
    config: Dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """
    Call DeepSeek API with retry and backoff on 429/5xx.

    Returns:
        Dict with keys: content, reasoning_content, usage, error, latency.
    """
    distill_config = config["distill"]
    api_base_url = distill_config["api_base_url"]
    model = distill_config["model"]
    thinking_mode = distill_config.get("thinking_mode", True)
    reasoning_effort = distill_config.get("reasoning_effort", "high")
    max_retries = int(distill_config.get("max_retries", 5))
    base_backoff = float(distill_config.get("base_backoff", 1.0))
    max_backoff = float(distill_config.get("max_backoff", 60.0))
    request_timeout = int(distill_config.get("request_timeout", 120))

    request_body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }

    if thinking_mode:
        request_body["thinking"] = {"type": "enabled"}
        request_body["reasoning_effort"] = reasoning_effort
    else:
        request_body["temperature"] = 0.7
        request_body["top_p"] = 0.9

    url = f"{api_base_url}/chat/completions"

    for attempt in range(max_retries + 1):
        start_time = time.time()
        try:
            async with semaphore:
                timeout = aiohttp.ClientTimeout(total=request_timeout)
                async with session.post(url, json=request_body, timeout=timeout) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        choice = data["choices"][0]["message"]
                        return {
                            "content": choice.get("content", ""),
                            "reasoning_content": choice.get("reasoning_content", ""),
                            "usage": data.get("usage", {}),
                            "error": None,
                            "latency": time.time() - start_time,
                        }
                    elif resp.status == 429:
                        if attempt < max_retries:
                            retry_after = resp.headers.get("Retry-After")
                            if retry_after:
                                wait = float(retry_after)
                            else:
                                wait = min(base_backoff * (2 ** attempt) + random.uniform(0, 1), max_backoff)
                            print(f"  Rate limited (429), waiting {wait:.1f}s (attempt {attempt + 1}/{max_retries + 1})")
                            await asyncio.sleep(wait)
                        else:
                            error_text = await resp.text()
                            return {
                                "content": "",
                                "reasoning_content": "",
                                "usage": None,
                                "error": f"HTTP 429 after {max_retries + 1} attempts",
                                "latency": time.time() - start_time,
                            }
                    elif resp.status >= 500:
                        if attempt < max_retries:
                            wait = min(base_backoff * (2 ** attempt) + random.uniform(0, 1), max_backoff)
                            error_text = await resp.text()
                            print(f"  Server error ({resp.status}), waiting {wait:.1f}s: {error_text[:200]}")
                            await asyncio.sleep(wait)
                        else:
                            error_text = await resp.text()
                            return {
                                "content": "",
                                "reasoning_content": "",
                                "usage": None,
                                "error": f"HTTP {resp.status} after {max_retries + 1} attempts: {error_text[:500]}",
                                "latency": time.time() - start_time,
                            }
                    else:
                        error_text = await resp.text()
                        return {
                            "content": "",
                            "reasoning_content": "",
                            "usage": None,
                            "error": f"HTTP {resp.status}: {error_text[:500]}",
                            "latency": time.time() - start_time,
                        }
        except asyncio.TimeoutError:
            if attempt < max_retries:
                wait = min(base_backoff * (2 ** attempt) + random.uniform(0, 1), max_backoff)
                print(f"  Timeout, waiting {wait:.1f}s (attempt {attempt + 1}/{max_retries + 1})")
                await asyncio.sleep(wait)
            else:
                return {
                    "content": "",
                    "reasoning_content": "",
                    "usage": None,
                    "error": f"Timeout after {max_retries + 1} attempts",
                    "latency": time.time() - start_time,
                }
        except Exception as e:
            if attempt < max_retries:
                wait = min(base_backoff * (2 ** attempt) + random.uniform(0, 1), max_backoff)
                print(f"  Error: {e}, waiting {wait:.1f}s (attempt {attempt + 1}/{max_retries + 1})")
                await asyncio.sleep(wait)
            else:
                return {
                    "content": "",
                    "reasoning_content": "",
                    "usage": None,
                    "error": str(e)[:500],
                    "latency": time.time() - start_time,
                }


async def run_distillation(config: Dict[str, Any], skip_confirm: bool = False) -> str:
    """
    Main distillation loop.

    Returns:
        Path to the saved distilled data file.
    """
    distill_config = config["distill"]
    raw_data_dir = distill_config["raw_data_dir"]
    distilled_data_path = distill_config["distilled_data_path"]
    checkpoint_path = distill_config["checkpoint_path"]
    sample_size = int(distill_config.get("sample_size", 0))
    seed = int(distill_config.get("seed", 42))
    max_concurrent = int(distill_config.get("max_concurrent", 5))
    model = distill_config.get("model", "deepseek-v4-pro")

    # Load data
    entries = load_raw_data(raw_data_dir, sample_size, seed)

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    completed = checkpoint.get("entries", {})

    # Filter out already completed entries
    pending_indices = [
        i for i in range(len(entries))
        if str(i) not in completed
        or (completed[str(i)].get("error") and not completed[str(i)].get("distilled_output"))
    ]

    print(f"Total entries: {len(entries)}")
    print(f"Already completed: {len(completed)}")
    print(f"Pending (including failed): {len(pending_indices)}")

    if not pending_indices:
        print("All entries already processed. Writing final output...")
        save_distilled_data(entries, completed, distilled_data_path)
        return distilled_data_path

    # Cost estimate
    pending_entries = [entries[i] for i in pending_indices]
    cost_info = estimate_cost(pending_entries, model)
    print(f"\nCost estimate for {cost_info['total_entries']} pending entries:")
    print(f"  Model: {cost_info['model']}")
    print(f"  Estimated input tokens: {cost_info['estimated_input_tokens']:,}")
    print(f"  Estimated output tokens: {cost_info['estimated_output_tokens']:,}")
    print(f"  Estimated cost (USD): ${cost_info['estimated_cost_usd_range'][0]:.4f} - ${cost_info['estimated_cost_usd_range'][1]:.4f}")

    if not skip_confirm:
        response = input("\nProceed with distillation? [y/N] ").strip().lower()
        if response != "y":
            print("Aborted.")
            return ""

    # Save cost log
    cost_log_path = distill_config.get("cost_log_path", "")
    if cost_log_path:
        cost_dir = os.path.dirname(cost_log_path)
        if cost_dir:
            os.makedirs(cost_dir, exist_ok=True)
        with open(cost_log_path, "w", encoding="utf-8") as f:
            json.dump(cost_info, f, indent=2)

    # Initialize checkpoint metadata
    checkpoint["metadata"] = {
        "model": model,
        "thinking_mode": distill_config.get("thinking_mode", True),
        "started_at": datetime.now().isoformat(),
        "total_entries": len(entries),
        "completed_entries": len(completed),
        "failed_entries": 0,
    }
    if "entries" not in checkpoint:
        checkpoint["entries"] = {}

    # Run distillation
    semaphore = asyncio.Semaphore(max_concurrent)
    api_key = os.environ.get(distill_config.get("api_key_env", "DEEPSEEK_API_KEY"), "")
    if not api_key:
        raise ValueError(
            f"API key not found. Set the {distill_config.get('api_key_env', 'DEEPSEEK_API_KEY')} "
            f"environment variable."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    batch_size = min(max_concurrent * 10, 100)
    total_batches = (len(pending_indices) + batch_size - 1) // batch_size
    processed_count = 0
    failed_count = 0

    async with aiohttp.ClientSession(headers=headers) as session:
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(pending_indices))
            batch_indices = pending_indices[batch_start:batch_end]

            print(f"\nBatch {batch_idx + 1}/{total_batches} ({len(batch_indices)} entries)...")

            tasks = []
            for idx in batch_indices:
                entry = entries[idx]
                messages = build_teacher_prompt(entry)
                tasks.append(call_deepseek_api(session, messages, config, semaphore))

            results = await asyncio.gather(*tasks)

            # Update checkpoint
            for idx, result in zip(batch_indices, results):
                checkpoint["entries"][str(idx)] = {
                    "distilled_output": result["content"],
                    "distilled_reasoning": result["reasoning_content"] or "",
                    "usage": result["usage"],
                    "error": result["error"],
                    "latency": result.get("latency", 0),
                }
                if result["error"]:
                    failed_count += 1
                else:
                    processed_count += 1

            # Save checkpoint after each batch
            checkpoint["metadata"]["total_processed"] = len(checkpoint["entries"])
            checkpoint["metadata"]["successful_entries"] = processed_count
            checkpoint["metadata"]["failed_entries"] = failed_count
            checkpoint["metadata"]["last_updated"] = datetime.now().isoformat()
            save_checkpoint(checkpoint_path, checkpoint)

            print(f"  Completed: {processed_count}, Failed: {failed_count}")

    # Write final output
    print(f"\nDistillation complete. Writing output to {distilled_data_path}...")
    save_distilled_data(entries, checkpoint["entries"], distilled_data_path)

    # Print summary
    total_tokens_in = sum(
        v.get("usage", {}).get("prompt_tokens", 0)
        for v in checkpoint["entries"].values()
        if v.get("usage")
    )
    total_tokens_out = sum(
        v.get("usage", {}).get("completion_tokens", 0)
        for v in checkpoint["entries"].values()
        if v.get("usage")
    )
    rates = PRICING.get(model, PRICING["deepseek-v4-pro"])
    actual_cost = (total_tokens_in / 1_000_000) * rates["input_cache_miss"] + \
                  (total_tokens_out / 1_000_000) * rates["output"]

    print(f"\nSummary:")
    print(f"  Total processed: {processed_count}")
    print(f"  Total failed: {failed_count}")
    print(f"  Total input tokens: {total_tokens_in:,}")
    print(f"  Total output tokens: {total_tokens_out:,}")
    print(f"  Estimated cost: ${actual_cost:.4f}")
    print(f"  Output: {distilled_data_path}")

    # Save cost summary
    if cost_log_path:
        cost_summary = {
            "model": model,
            "total_entries": len(entries),
            "processed": processed_count,
            "failed": failed_count,
            "total_input_tokens": total_tokens_in,
            "total_output_tokens": total_tokens_out,
            "estimated_cost_usd": round(actual_cost, 4),
            "completed_at": datetime.now().isoformat(),
        }
        with open(cost_log_path, "w", encoding="utf-8") as f:
            json.dump(cost_summary, f, indent=2)

    return distilled_data_path


def save_distilled_data(
    entries: List[Dict[str, Any]],
    completed: Dict[str, Dict[str, Any]],
    output_path: str,
) -> None:
    """
    Merge raw entries with distilled results and save.

    Each output line has original fields plus:
      - distilled_output: str
      - distilled_reasoning: str
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, entry in enumerate(entries):
            result = completed.get(str(i), {})
            output_entry = {
                "instruction": entry.get("instruction", ""),
                "input": entry.get("input", ""),
                "output": entry.get("output", ""),
                "raw_content": entry.get("raw_content", ""),
                "distilled_output": result.get("distilled_output", ""),
                "distilled_reasoning": result.get("distilled_reasoning", ""),
            }
            f.write(json.dumps(output_entry, ensure_ascii=False) + "\n")

    print(f"Saved {len(entries)} entries to {output_path}")


def main(config_path: str, skip_confirm: bool = False):
    """
    Main entry point for the distillation script.

    Args:
        config_path: Path to the configuration YAML file.
        skip_confirm: If True, skip the cost confirmation prompt.
    """
    config = load_config(config_path)
    distill_config = config.get("distill", {})

    if not distill_config.get("enabled", False):
        print("Distillation is disabled in config. Set distill.enabled: true to run.")
        return

    asyncio.run(run_distillation(config, skip_confirm))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Distillation from DeepSeek API")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--yes", action="store_true", help="Skip cost confirmation prompt")
    args = parser.parse_args()

    main(args.config, args.yes)
