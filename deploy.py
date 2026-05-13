"""
Description: Deployment server for QEdpediaCN-Qwen model.
Loads the merged model with optional RAG integration, serves via FastAPI + Gradio.
Usage:
    python deploy.py
    python deploy.py --config configs/config.yaml
    python deploy.py --host 127.0.0.1 --port 9000
Dependencies: transformers, torch, bitsandbytes, fastapi, gradio, uvicorn, sentence-transformers, chromadb
"""

import argparse
import os
import sys
from typing import Optional

import torch
import uvicorn
import gradio as gr
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.api import create_api_router
from src.gradio_app import create_gradio_app
from src.inference_engine import InferenceEngine
from src.utils import load_config


def resolve_model_path(model_config: dict, deploy_config: dict) -> str:
    """
    Resolve model path: override > current_model > base_model.

    Args:
        model_config: The model section of config.
        deploy_config: The deploy section of config.

    Returns:
        str: Path to the model directory.
    """
    override = deploy_config.get("model_path_override", "")
    if override:
        return override

    current_path = model_config.get("current_model_path", "")
    if current_path and os.path.exists(current_path) and os.listdir(current_path):
        return current_path

    return model_config.get("base_model_path", "")


def load_engine(config: dict) -> InferenceEngine:
    """
    Load model, tokenizer, and RAG components. Construct InferenceEngine.

    Args:
        config: Full config dictionary.

    Returns:
        InferenceEngine instance.
    """
    model_config = config["model"]
    deploy_config = config.get("deploy", {})
    rag_config = config.get("rag", {})

    # Resolve model path
    model_path = resolve_model_path(model_config, deploy_config)
    if not os.path.exists(model_path) or not os.listdir(model_path):
        print(f"Error: Model path {model_path} not found or empty.")
        sys.exit(1)

    print(f"Loading model from: {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization
    quantization = deploy_config.get("quantization", "4bit")
    if quantization == "4bit":
        print("Loading with 4-bit BitsAndBytes NF4 quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        print("Loading in full precision (bf16/fp16)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    # Load RAG components if enabled
    collection = None
    embedder = None
    reranker = None
    rag_available = False

    if rag_config.get("rag_inference_deploy", False):
        print("Loading RAG components...")
        from rag.rag_db import load_knowledge_db, load_embedder
        from rag.rag_retrieve import load_reranker

        db_path = rag_config.get("db_path", "rag/db")
        if not os.path.exists(db_path):
            print(f"Error: RAG DB not found at {db_path}. Run with build_rag_db=true first.")
            sys.exit(1)

        collection = load_knowledge_db(db_path, rag_config.get("embedder_path", ""))
        embedder = load_embedder(rag_config.get("embedder_path", ""))
        reranker = load_reranker(rag_config.get("reranker_path", ""))
        rag_available = True
        print("RAG components loaded.")
    else:
        print("RAG disabled for deployment.")

    engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        collection=collection,
        embedder=embedder,
        reranker=reranker,
        deploy_config=deploy_config,
        rag_config=rag_config,
    )

    print("Inference engine ready.")
    return engine


def main(config_path: str, host_override: Optional[str], port_override: Optional[int]) -> None:
    """
    Main entry point for the deployment server.

    Args:
        config_path: Path to config YAML file.
        host_override: Override config host.
        port_override: Override config port.
    """
    # Set environment variables matching main.py conventions
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("PYTHONUTF8", "1")

    config = load_config(config_path)
    deploy_config = config.get("deploy", {})
    model_config = config["model"]

    # Load engine
    engine = load_engine(config)

    # Create FastAPI app
    app = FastAPI(title="QEdpediaCN-Qwen", version="1.0.0")

    # Mount API routes
    rag_available = engine.collection is not None
    router = create_api_router(engine, model_config, deploy_config, rag_available)
    app.include_router(router)

    # Mount Gradio
    demo = create_gradio_app(engine, deploy_config)
    app = gr.mount_gradio_app(app, demo, path="/gradio")

    # Start server
    host = host_override or deploy_config.get("host", "0.0.0.0")
    port = port_override or int(deploy_config.get("port", 8000))
    print(f"Starting server at http://{host}:{port}")
    print(f"  API: http://{host}:{port}/v1/chat/completions")
    print(f"  Gradio UI: http://{host}:{port}/gradio")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy QEdpediaCN-Qwen model server")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--host", type=str, default=None, help="Override config host")
    parser.add_argument("--port", type=int, default=None, help="Override config port")
    args = parser.parse_args()

    main(args.config, args.host, args.port)
