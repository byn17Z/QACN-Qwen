# Model Choice Guide: QEdpediaCN-Qwen

This document outlines the recommended base models for the Chinese Educational QA project based on available hardware and specific tasks (SFT Training vs. Inference Deployment).

## 1. High-Performance Training (A10 GPU - 24GB VRAM)
Recommended for the primary development phase to create the "smartest" version of the model.

| Choice | Model | VRAM (4-bit QLoRA) | Logic/Reasoning | Remarks |
| :--- | :--- | :--- | :--- | :--- |
| **Balanced** | **Qwen2.5-7B** | ~6.5 GB | High | **Recommended.** Very stable, allows for large batch sizes and 8k+ context. Easy to deploy later on consumer cards. |
| **Max Logic** | **Qwen2.5-14B**| ~11 GB | Superior | Best use of A10. Capture deep educational logic. Requires `gradient_checkpointing`. |
| **Edge Case** | **InternLM2.5-20B** | ~15 GB | Excellent | Alternative for "teacher-like" tone. Tight on 24GB; keep context < 2048. |

### Remarks for A10 Training:
*   **Optimizer:** Use `paged_adamw_32bit` to ensure stability.
*   **Context:** Aim for `context_length=4096` to capture complete educational explanations. This fits comfortably in 24GB VRAM when using 4-bit quantization and gradient checkpointing.
*   **Training:** `batch_size=128` on A10 for more frequent updates, which generally leads to better convergence and more stable training.
*   **Quantization:** Use `bitsandbytes` 4-bit for training to maximize headroom.

---

## 2. Local Deployment (RTX 3060 - 6GB VRAM)
Recommended for the final application interface or lightweight testing.

| Choice | Model | VRAM (4-bit Quant) | Inference Speed | Remarks |
| :--- | :--- | :--- | :--- | :--- |
| **Performance**| **Qwen2.5-7B** | ~5.3 GB | Moderate | **Recommended.** Maximum intelligence for 6GB. Use `GGUF` or `AWQ`. Context must be short (< 2k). |
| **Smoothness** | **Qwen2.5-3B** | ~2.5 GB | Very Fast | Best user experience. Allows for long multi-turn conversations without lag. |
| **Ultra-Light** | **Qwen2.5-1.5B**| ~1.5 GB | Blazing | Good for mobile/low-end edge testing. |

### Remarks for 6GB Inference:
*   **Format:** Use **GGUF** (via llama.cpp/Ollama) to manage memory fragmentation better than raw Transformers.
*   **Optimization:** Enable `Flash Attention 2`.
*   **VRAM Management:** Close browsers and other GPU-heavy apps to avoid OOM crashes.

---

## 3. Local Training / SFT (RTX 3060 - 6GB VRAM)
Recommended only if the A10 is unavailable and you need to iterate locally.

| Choice | Model | VRAM (4-bit QLoRA) | Feasibility | Remarks |
| :--- | :--- | :--- | :--- | :--- |
| **Standard** | **Qwen2.5-1.5B**| ~4.5 GB | **Feasible** | Best for local experiments. Can handle 1024-2048 context. |
| **Tight Fit** | **Qwen2.5-3B** | ~5.8 GB | **Risky** | Requires `max_seq_length=512` and batch size 1. No overhead left. |
| **Impossible** | **Qwen2.5-7B** | > 8 GB | **No** | Weights + Gradients will exceed 6GB immediately. |

### Remarks for 6GB SFT:
*   **Mandatory:** `gradient_checkpointing=True`.
*   **Accumulation:** Use `gradient_accumulation_steps=8` to simulate a real batch size while keeping `per_device_train_batch_size=1`.
*   **Precision:** Use `fp16` or `bf16` (3060 supports bf16 which is more stable).

---

## Final Strategy Recommendation
1.  **Phase 1 (Training):** Use the **A10** to fine-tune **Qwen2.5-7B-Instruct** using the `sft_qa` subset of Fineweb-Edu-Chinese-V2.2.
2.  **Phase 2 (Export):** Quantize the resulting model to **4-bit GGUF**.
3.  **Phase 3 (Deployment):** Run the quantized model on the **RTX 3060** for the final Terminal Interface.
