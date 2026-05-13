# RAG Implementation Plan

## Components

* Embedder: BAAI/bge-small-zh-v1.5

* Reranker: BAAI/bge-reranker-base

* Database: ChromaDB

## Directory Structure

In subdirectory `./rag/`

* `./rag/__init__.py`: package init

* `./rag/rag_db.py`: construct and load knowledge DB for RAG

* `./rag/rag_retrieve.py`: retrieval + reranking logic

* `./rag/rag_test.py`: test RAG retrieval quality and compare context modes

* `./rag/db/`: subdirectory to store ChromaDB (gitignored)

* `./rag/models/`: subdirectory to store embedder and reranker models (gitignored)

## Data Format

Raw data format: `{"instruction": "...", "input": "", "output": "...", "raw_content": "..."}`

`raw_content` is the labeled knowledge context used as the RAG knowledge base.

## Config Parameters (`configs/config.yaml`)

New `rag` section:

```yaml
rag:
  # Paths
  embedder_path: "rag/models/bge-small-zh-v1.5"
  reranker_path: "rag/models/bge-reranker-base"
  db_path: "rag/db"

  # Retrieval params
  retrieve_top_k: 20         # candidates from embedding search
  retrieve_top_n: 5          # final results after reranking

  # Pipeline flags
  build_rag_db: true         # true = rebuild DB from sft_data.jsonl; false = load existing DB
  rag_inference_deploy: false # integrate RAG when deploying
  context_mask_test: false    # hide context during testing
  rag_inference_test: false   # true = use RAG docs; false = use labeled context; ignored if context_mask_test=true
  context_mask_train: false   # hide context during training
  rag_inference_train: false  # true = use RAG docs; false = use labeled context; ignored if context_mask_train=true
```

## Implementation Plan

### Step 1: Create `rag/` package

**`rag/__init__.py`** — exports main functions.

**`rag/rag_db.py`** — Knowledge base construction and loading.

Functions:
- `load_embedder(model_path) -> SentenceTransformer`
- `build_knowledge_db(data_path, db_path, embedder_path) -> Collection`
- `load_knowledge_db(db_path, embedder_path) -> Collection`

Logic:
1. Load all entries from `sft_data.jsonl`
2. Extract `raw_content` from each entry
3. Encode with `bge-small-zh-v1.5`
4. Store in ChromaDB with IDs and metadata
5. Persist to `rag/db/`

**`rag/rag_retrieve.py`** — Retrieval and reranking.

Functions:
- `load_reranker(model_path) -> CrossEncoder`
- `retrieve(query, collection, reranker, embedder, top_k=20, top_n=5) -> list[str]`

Logic:
1. Encode query with embedder
2. Query ChromaDB for top_k candidates
3. Rerank with `bge-reranker-base`
4. Return top_n context strings

**`rag/rag_test.py`** — RAG performance testing.

Test 1 — Retrieval Quality:
- For each QA pair in test set, retrieve top-n contexts using instruction as query
- Check if labeled `raw_content` appears in retrieved results
- Compute Precision@k, Recall@k, MRR

Test 2 — Context Comparison (Perplexity):
- Run evaluation under three conditions:
  1. `context_mask_test=true` — no context (baseline)
  2. `context_mask_test=false, rag_inference_test=false` — labeled context
  3. `context_mask_test=false, rag_inference_test=true` — RAG-retrieved context
- Compare perplexity across modes

Usage: `python rag/rag_test.py --config configs/config.yaml --mode validation --timestamp <ts>`

Output: prints retrieval metrics + saves to `outputs/rag_test_stats.json`

### Step 2: Update `.gitignore`

Add:
```
rag/db/
rag/models/
```

### Step 3: Update `configs/config.yaml`

Add the `rag` section as shown above.

### Step 4: Update `requirements.txt`

Add:
```
sentence-transformers
chromadb
```

### Step 5: Update `data_preprocess.py`

Preserve `raw_content` field in `format_dataset_entry()`:

```python
def format_dataset_entry(entry: Dict[str, Any]) -> Dict[str, str]:
    return {
        "instruction": entry.get("instruction", ""),
        "input": entry.get("input", ""),
        "output": entry.get("output", ""),
        "raw_content": entry.get("raw_content", ""),
    }
```

### Step 6: Update `sft_train.py`

Add context-aware formatting in `formatting_prompts_func`:

| `context_mask_train` | `rag_inference_train` | Format |
|---|---|---|
| `true` | (ignored) | `Instruction: ...\nOutput: ...` |
| `false` | `false` | `Context: {raw_content}\nInstruction: ...\nOutput: ...` |
| `false` | `true` | `Context: {retrieve(instruction)}\nInstruction: ...\nOutput: ...` |

If `rag_inference_train` is true, load RAG components at script start and call `retrieve()` per example.

### Step 7: Update `test_eval.py`

Same context logic as training:

| `context_mask_test` | `rag_inference_test` | Format |
|---|---|---|
| `true` | (ignored) | No context |
| `false` | `false` | Labeled context from `raw_content` |
| `false` | `true` | RAG-retrieved context |

Update `calculate_perplexity()` to accept optional RAG components and build `full_text` with context based on config flags.

### Step 8: Update `main.py`

Add Stage 0 before preprocessing:

```python
rag_config = config.get("rag", {})
if rag_config.get("build_rag_db", False):
    print("=== Stage 0: Building RAG Knowledge Base ===")
    from rag.rag_db import build_knowledge_db
    build_knowledge_db(...)
else:
    print("=== Stage 0: Loading Existing RAG Knowledge Base ===")
    from rag.rag_db import load_knowledge_db
    load_knowledge_db(...)
```

When `build_rag_db=true`: rebuild DB from `sft_data.jsonl` (overwrites existing).
When `build_rag_db=false`: load existing DB from `rag/db/`.

### Step 9: Update `CLAUDE.md`

Document the RAG module, new config flags, and new commands.

## Execution Order

| Step | File | Action |
|------|------|--------|
| 1 | `rag/__init__.py` | Create package init |
| 2 | `rag/rag_db.py` | Implement DB construction + loading |
| 3 | `rag/rag_retrieve.py` | Implement retrieval + reranking |
| 4 | `rag/rag_test.py` | Implement RAG testing |
| 5 | `.gitignore` | Add `rag/db/`, `rag/models/` |
| 6 | `configs/config.yaml` | Add `rag` section |
| 7 | `requirements.txt` | Add dependencies |
| 8 | `data_preprocess.py` | Preserve `raw_content` field |
| 9 | `sft_train.py` | Add context-aware formatting |
| 10 | `test_eval.py` | Add context-aware formatting |
| 11 | `main.py` | Add Stage 0 (RAG DB build/load) |
| 12 | `CLAUDE.md` | Update documentation |
