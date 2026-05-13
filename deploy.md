# Model Deployment Implementation Plan

## Context

The QEdpediaCN-Qwen project has a complete training pipeline (preprocess → train → merge → evaluate) but no deployment layer. The `rag_inference_deploy` config flag exists as a placeholder with no consumer. This plan adds a FastAPI + Gradio deployment server with optional RAG integration, using 4-bit BitsAndBytes NF4 quantization for efficient serving.

## Files to Create

| File | Responsibility |
|---|---|
| `deploy.py` | Entry point. CLI args, config loading, model/RAG initialization, FastAPI + Gradio mounting, uvicorn startup. |
| `src/inference_engine.py` | `InferenceEngine` class holding model, tokenizer, RAG components. Single `generate()` method for prompt construction and text generation. |
| `src/api.py` | FastAPI router: `/health`, `/v1/models`, `/v1/chat/completions`. Pydantic request/response schemas. |
| `src/gradio_app.py` | Gradio chat interface with RAG toggle, max_tokens slider, temperature slider. `create_gradio_app()` returns `gr.Blocks`. |

## Files to Modify

| File | Change |
|---|---|
| `configs/config.yaml` | Add `deploy:` section (host, port, quantization, generation defaults). |
| `requirements.txt` | Add `fastapi>=0.115.0`, `gradio>=5.0.0`. |
| `CLAUDE.md` | Add deployment documentation section. |

## Config Addition (`configs/config.yaml`)

```yaml
deploy:
  host: "0.0.0.0"
  port: 8000
  quantization: "4bit"              # "4bit" or "none"
  model_path_override: ""           # non-empty overrides current_model_path
  rag_enabled_default: false        # default RAG toggle for Gradio UI
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  repetition_penalty: 1.1
  do_sample: true
```

## Architecture

### Startup Sequence (`deploy.py`)

1. `load_config()` → resolve model path (override > current_model > base_model)
2. Load tokenizer + model (4-bit NF4 via `BitsAndBytesConfig` if `quantization=="4bit"`, else bf16/fp16)
3. If `rag.rag_inference_deploy`: load ChromaDB collection, embedder, reranker via existing `rag/` module
4. Construct `InferenceEngine(model, tokenizer, rag_components, config)`
5. Create FastAPI app, include API router, mount Gradio at `/gradio`
6. `uvicorn.run()`

### InferenceEngine (`src/inference_engine.py`)

```python
class InferenceEngine:
    def __init__(self, model, tokenizer, collection, embedder, reranker, deploy_config, rag_config): ...
    def generate(self, instruction, input_text="", use_rag=False, **gen_kwargs) -> dict:
        # 1. RAG retrieval if use_rag (reuses rag.retrieval.retrieve())
        # 2. Prompt construction matching training format
        # 3. model.generate() with torch.inference_mode()
        # 4. Decode, extract after "Output:", return response + metadata
```

Prompt format (matches sft_train.py / test_eval.py):
- With RAG: `Context: {docs}\nInstruction: {q}\nInput: {inp}\nOutput:`
- Without RAG: `Instruction: {q}\nInput: {inp}\nOutput:`

### API Endpoints (`src/api.py`)

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check — model path, RAG status |
| `GET` | `/v1/models` | Model metadata (family, size, quantization) |
| `POST` | `/v1/chat/completions` | Main inference — `ChatRequest` → `ChatResponse` |
| `GET` | `/` | Redirect to `/gradio` |

`ChatRequest`: instruction (required), input, use_rag, generation param overrides
`ChatResponse`: response text, context_used (if RAG), generation_params used

### Gradio UI (`src/gradio_app.py`)

Layout: chat window + textbox + RAG checkbox + max_tokens slider + temperature slider + submit/clear buttons. Mounted at `/gradio` via `gr.mount_gradio_app()`.

## Reused Existing Code

- `src/utils.py:load_config()` — config loading
- `rag/rag_db.py:load_knowledge_db()`, `load_embedder()` — RAG DB and embedder loading
- `rag/rag_retrieve.py:load_reranker()`, `retrieve()` — two-stage retrieval pipeline
- Model loading pattern from `test_eval.py` lines 87-103
- Prompt format from `sft_train.py` and `test_eval.py`

## Run Command

```bash
python deploy.py                              # defaults
python deploy.py --config configs/config.yaml # explicit config
python deploy.py --host 127.0.0.1 --port 9000 # override host/port
```

## Error Handling

- **Startup**: model path not found → exit 1; RAG DB missing when required → exit 1; CUDA unavailable → warning, proceed on CPU
- **Runtime**: empty instruction → 400; RAG retrieval failure → log warning, proceed without context; CUDA OOM → `torch.cuda.empty_cache()`, return 500
- **API**: structured error responses `{"error": str, "detail": str}`

## Dependencies

Add to `requirements.txt`:
```
fastapi>=0.115.0
gradio>=5.0.0
```

`uvicorn==0.46.0` and `pydantic` already available as transitive dependencies.

## Verification

1. **Static**: imports resolve, config parses, Pydantic schemas valid
2. **Smoke test**: `curl localhost:8000/health`, POST to `/v1/chat/completions` with and without RAG
3. **Gradio**: open `localhost:8000/gradio`, test chat with RAG toggle
4. **Error cases**: empty instruction → 400, OOM handling
5. **Integration**: run `main.py` pipeline, then `deploy.py`, verify merged model is served
