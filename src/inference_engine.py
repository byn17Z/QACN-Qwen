"""
Description: Inference engine for model deployment.
Holds the loaded model, tokenizer, and RAG components. Provides a single generate() method.
Usage: from src.inference_engine import InferenceEngine
Dependencies: transformers, torch, bitsandbytes, sentence-transformers, chromadb
"""

from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag.rag_retrieve import retrieve


class InferenceEngine:
    """
    Singleton-style engine holding model, tokenizer, and optional RAG components.
    Provides generate() for prompt construction and text generation.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        collection: Optional[Any],
        embedder: Optional[Any],
        reranker: Optional[Any],
        deploy_config: Dict[str, Any],
        rag_config: Dict[str, Any],
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.collection = collection
        self.embedder = embedder
        self.reranker = reranker
        self.deploy_config = deploy_config
        self.rag_config = rag_config

    def generate(
        self,
        instruction: str,
        input_text: str = "",
        use_rag: bool = False,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Build prompt, run generation, return response + metadata.

        Args:
            instruction: The question/query (required).
            input_text: Additional input context.
            use_rag: Whether to use RAG retrieval for context.
            max_new_tokens: Override default max tokens to generate.
            temperature: Override default temperature.
            top_p: Override default top_p.
            top_k: Override default top_k.
            repetition_penalty: Override default repetition penalty.
            do_sample: Override default sampling mode.

        Returns:
            Dict with keys: response (str), context_used (List[str] or None), generation_params (dict).
        """
        if not instruction.strip():
            raise ValueError("instruction must not be empty")

        # Resolve generation params (request overrides > config defaults)
        gen_params = {
            "max_new_tokens": max_new_tokens if max_new_tokens is not None else self.deploy_config.get("max_new_tokens", 512),
            "temperature": temperature if temperature is not None else self.deploy_config.get("temperature", 0.7),
            "top_p": top_p if top_p is not None else self.deploy_config.get("top_p", 0.9),
            "top_k": top_k if top_k is not None else self.deploy_config.get("top_k", 50),
            "repetition_penalty": repetition_penalty if repetition_penalty is not None else self.deploy_config.get("repetition_penalty", 1.1),
            "do_sample": do_sample if do_sample is not None else self.deploy_config.get("do_sample", True),
        }

        # RAG retrieval
        context_docs: Optional[List[str]] = None
        if use_rag and self.collection is not None:
            try:
                context_docs = retrieve(
                    query=instruction,
                    collection=self.collection,
                    reranker=self.reranker,
                    embedder=self.embedder,
                    top_k=int(self.rag_config.get("retrieve_top_k", 20)),
                    top_n=int(self.rag_config.get("retrieve_top_n", 5)),
                )
            except Exception as e:
                print(f"RAG retrieval failed, proceeding without context: {e}")
                context_docs = None

        # Build prompt matching training format
        if use_rag and context_docs:
            context = "\n".join(context_docs)
            prompt = f"Context: {context}\nInstruction: {instruction}\nInput: {input_text}\nOutput:"
        else:
            prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        # Generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(gen_params["max_new_tokens"]),
                temperature=float(gen_params["temperature"]),
                top_p=float(gen_params["top_p"]),
                top_k=int(gen_params["top_k"]),
                repetition_penalty=float(gen_params["repetition_penalty"]),
                do_sample=bool(gen_params["do_sample"]),
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated tokens (exclude prompt)
        generated_ids = output_ids[0][input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return {
            "response": response,
            "context_used": context_docs if use_rag else None,
            "generation_params": gen_params,
        }
