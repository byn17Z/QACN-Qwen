"""
Description: FastAPI router for model deployment.
Provides /health, /v1/models, and /v1/chat/completions endpoints.
Usage: from src.api import create_api_router
Dependencies: fastapi, pydantic
"""

from typing import Any, Dict, List, Optional

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request schema for chat completion."""

    instruction: str = Field(..., min_length=1, description="The question or query")
    input: str = Field(default="", description="Additional input context")
    use_rag: bool = Field(default=False, description="Enable RAG retrieval for context")
    max_new_tokens: Optional[int] = Field(default=None, ge=1, le=4096)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1, le=200)
    repetition_penalty: Optional[float] = Field(default=None, ge=1.0, le=2.0)
    do_sample: Optional[bool] = None


class ChatResponse(BaseModel):
    """Response schema for chat completion."""

    response: str
    context_used: Optional[List[str]] = None
    generation_params: Dict[str, Any]


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str
    model_path: str
    rag_enabled: bool
    model_family: str


class ModelInfoResponse(BaseModel):
    """Response schema for model info."""

    model_path: str
    model_family: str
    model_size: str
    quantization: str
    rag_available: bool
    deploy_config: Dict[str, Any]


def create_api_router(
    engine: Any,
    model_config: Dict[str, Any],
    deploy_config: Dict[str, Any],
    rag_available: bool,
) -> APIRouter:
    """
    Create FastAPI router with all endpoints.

    Args:
        engine: The InferenceEngine instance.
        model_config: The model section of config.
        deploy_config: The deploy section of config.
        rag_available: Whether RAG components are loaded.

    Returns:
        APIRouter with all endpoints registered.
    """
    router = APIRouter()

    @router.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(
            status="ok",
            model_path=model_config.get("current_model_path", ""),
            rag_enabled=rag_available,
            model_family=model_config.get("model_family", ""),
        )

    @router.get("/v1/models", response_model=ModelInfoResponse)
    async def model_info() -> ModelInfoResponse:
        """Model metadata endpoint."""
        return ModelInfoResponse(
            model_path=model_config.get("current_model_path", ""),
            model_family=model_config.get("model_family", ""),
            model_size=model_config.get("size", ""),
            quantization=deploy_config.get("quantization", "none"),
            rag_available=rag_available,
            deploy_config=deploy_config,
        )

    @router.post("/v1/chat/completions", response_model=ChatResponse)
    async def chat_completions(request: ChatRequest) -> ChatResponse:
        """Main inference endpoint."""
        try:
            result = engine.generate(
                instruction=request.instruction,
                input_text=request.input,
                use_rag=request.use_rag,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                do_sample=request.do_sample,
            )
            return ChatResponse(
                response=result["response"],
                context_used=result["context_used"],
                generation_params=result["generation_params"],
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise HTTPException(
                status_code=500,
                detail="CUDA out of memory. Try shorter input or smaller max_new_tokens.",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router
