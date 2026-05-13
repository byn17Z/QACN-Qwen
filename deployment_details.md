# Deployment Implementation Details

## Overview

The deployment module serves the merged model via a single server process exposing both a FastAPI REST API and a Gradio web UI. It loads the model once at startup with 4-bit BitsAndBytes NF4 quantization (configurable), and optionally loads RAG components (ChaDB, embedder, reranker) for retrieval-augmented inference.

## Architecture

```
deploy.py (entry point)
    │
    ├── load_config()              # from src/utils.py
    ├── resolve_model_path()       # override > current_model > base_model
    ├── load_engine()              # model + tokenizer + RAG → InferenceEngine
    │       │
    │       ├── AutoTokenizer.from_pretrained()
    │       ├── AutoModelForCausalLM.from_pretrained()  (4-bit or full precision)
    │       ├── rag.rag_db.load_knowledge_db()           (if rag_inference_deploy)
    │       ├── rag.rag_db.load_embedder()               (if rag_inference_deploy)
    │       └── rag.rag_retrieve.load_reranker()         (if rag_inference_deploy)
    │
    ├── create_api_router()        # from src/api.py
    ├── create_gradio_app()        # from src/gradio_app.py
    └── uvicorn.run()
```

## Components

### 1. Entry Point (`deploy.py`)

Responsible for CLI argument parsing, config loading, component initialization, and server startup.

**CLI arguments:**
- `--config` — path to config YAML (default: `configs/config.yaml`)
- `--host` — override config host
- `--port` — override config port

**Model path resolution** (same logic as `test_eval.py`):
1. If `deploy.model_path_override` is non-empty, use it
2. Else if `model.current_model_path` exists and is non-empty, use it
3. Else fall back to `model.base_model_path`

**Environment variables** (matching `main.py` conventions):
- `HF_ENDPOINT=https://hf-mirror.com` — HuggingFace mirror for Chinese network
- `PYTHONUTF8=1` — force UTF-8 encoding on Windows

### 2. Inference Engine (`src/inference_engine.py`)

Singleton-style class that holds all loaded components and provides a single `generate()` method.

**Constructor parameters:**
- `model` — loaded `AutoModelForCausalLM`
- `tokenizer` — loaded `AutoTokenizer`
- `collection` — ChromaDB collection (or `None` if RAG disabled)
- `embedder` — SentenceTransformer embedder (or `None`)
- `reranker` — CrossEncoder reranker (or `None`)
- `deploy_config` — deploy section of config
- `rag_config` — rag section of config

**`generate()` method flow:**

1. **Validate input** — raise `ValueError` if instruction is empty
2. **Resolve generation params** — request overrides take precedence over config defaults; all use `is not None` checks to preserve explicit zero/false values
3. **RAG retrieval** (if `use_rag=True` and components loaded):
   - Calls `rag.rag_retrieve.retrieve()` with `top_k` and `top_n` from rag config
   - On failure: logs warning, proceeds without context (graceful degradation)
4. **Build prompt** — matches the format used during training:
   - With RAG context: `Context: {doc1}\n{doc2}\n...\nInstruction: {q}\nInput: {inp}\nOutput:`
   - Without RAG: `Instruction: {q}\nInput: {inp}\nOutput:`
5. **Tokenize** — `tokenizer(prompt, return_tensors="pt")`, move to model device
6. **Generate** — `model.generate()` inside `torch.inference_mode()` context manager
7. **Decode** — decode only generated tokens (exclude prompt), strip whitespace
8. **Return** — `{"response": str, "context_used": List[str] | None, "generation_params": dict}`

### 3. API Router (`src/api.py`)

FastAPI router factory that closes over the engine instance. No global state.

**Endpoints:**

| Method | Path | Request | Response |
|--------|------|---------|----------|
| `GET` | `/health` | — | `{"status", "model_path", "rag_enabled", "model_family"}` |
| `GET` | `/v1/models` | — | `{"model_path", "model_family", "model_size", "quantization", "rag_available", "deploy_config"}` |
| `POST` | `/v1/chat/completions` | `ChatRequest` | `ChatResponse` |

**`ChatRequest` schema:**
```json
{
  "instruction": "什么是光合作用?",   // required, min_length=1
  "input": "",                        // optional
  "use_rag": false,                   // toggle RAG per-request
  "max_new_tokens": null,             // null = use config default
  "temperature": null,
  "top_p": null,
  "top_k": null,
  "repetition_penalty": null,
  "do_sample": null
}
```

**`ChatResponse` schema:**
```json
{
  "response": "光合作用是...",
  "context_used": ["doc1", "doc2"],   // null if RAG not used
  "generation_params": {"max_new_tokens": 512, "temperature": 0.7, ...}
}
```

**Error handling:**
- Empty instruction → `400 Bad Request`
- CUDA OOM → `500` with suggestion to reduce input/max_new_tokens, clears CUDA cache
- Other exceptions → `500` with error message

### 4. Gradio UI (`src/gradio_app.py`)

Browser-based chat interface mounted at `/gradio` on the FastAPI app.

**Layout:**
- Chat window (`gr.Chatbot`)
- Text input (`gr.Textbox`)
- RAG toggle (`gr.Checkbox`) — default from `deploy.rag_enabled_default`
- Max tokens slider (`gr.Slider`, 64–2048, default from config)
- Temperature slider (`gr.Slider`, 0.0–2.0, default from config)
- Submit and Clear buttons

**Behavior:**
- Submit triggers `engine.generate()` with current slider/toggle values
- If RAG was used, appends truncated source documents below the response
- Clear resets chat history and input field

## Configuration

All deployment settings live under the `deploy:` section in `configs/config.yaml`:

```yaml
deploy:
  host: "0.0.0.0"                # server bind address
  port: 8000                     # server port
  quantization: "4bit"           # "4bit" (BitsAndBytes NF4) or "none" (bf16/fp16)
  model_path_override: ""        # non-empty overrides current_model_path
  rag_enabled_default: false     # default RAG toggle for Gradio UI
  max_new_tokens: 512            # default max tokens to generate
  temperature: 0.7               # default sampling temperature
  top_p: 0.9                     # default nucleus sampling threshold
  top_k: 50                      # default top-k sampling
  repetition_penalty: 1.1        # default repetition penalty
  do_sample: true                # default sampling mode (false = greedy)
```

RAG integration is controlled by `rag.rag_inference_deploy` in the `rag:` section. When `true`, the server loads ChromaDB, embedder, and reranker at startup. The RAG toggle in Gradio and the `use_rag` parameter in the API allow per-request control.

## Quantization

**4-bit (default):** Uses BitsAndBytes NF4 with double quantization, matching the quantization used during QLoRA training. Config:
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # or float16
)
```

**Full precision:** Loads model in bfloat16 (or float16 if bf16 not supported). No quantization config needed.

## Prompt Format

The prompt format must exactly match what the model was trained on. The model was SFT'd with the `Instruction:/Input:/Output:` format (and optionally `Context:` prefix), as defined in `sft_train.py` and `test_eval.py`.

**With RAG context:**
```
Context: {retrieved document 1}
{retrieved document 2}
...
Instruction: {user question}
Input: {optional input}
Output:
```

**Without context:**
```
Instruction: {user question}
Input: {optional input}
Output:
```

The model generates text after `Output:`. The engine decodes only the generated tokens (excluding the prompt) and strips trailing whitespace.

## Reused Existing Code

| Component | Source | Function |
|-----------|--------|----------|
| Config loading | `src/utils.py` | `load_config()` |
| RAG DB loading | `rag/rag_db.py` | `load_knowledge_db()`, `load_embedder()` |
| RAG retrieval | `rag/rag_retrieve.py` | `load_reranker()`, `retrieve()` |
| Model loading pattern | `test_eval.py` | Lines 87-103 |
| Prompt format | `sft_train.py`, `test_eval.py` | `formatting_prompts_func` / `calculate_perplexity` |
| BitsAndBytes config | `sft_train.py` | Lines 57-62 |

## Dependencies

Added to `requirements.txt`:
```
fastapi>=0.115.0
gradio>=5.0.0
```

Already available:
- `uvicorn==0.46.0` (transitive dependency of chromadb)
- `pydantic==2.12.5` (transitive dependency of huggingface_hub)
- `transformers`, `bitsandbytes`, `torch`, `sentence-transformers`, `chromadb`

## Running

```bash
# Default (reads configs/config.yaml)
python deploy.py

# Custom config
python deploy.py --config configs/config.yaml

# Override host/port
python deploy.py --host 127.0.0.1 --port 9000
```

Server starts at `http://{host}:{port}`:
- API: `http://localhost:8000/v1/chat/completions`
- Gradio UI: `http://localhost:8000/gradio`
- Health: `http://localhost:8000/health`

## Testing

**API test (non-RAG):**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"instruction": "什么是光合作用?", "use_rag": false}'
```

**API test (RAG):**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"instruction": "什么是光合作用?", "use_rag": true}'
```

**Health check:**
```bash
curl http://localhost:8000/health
```
