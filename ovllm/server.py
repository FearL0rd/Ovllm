"""
FastAPI server for Ovllm with OpenAI-compatible API.

Provides endpoints for chat completions, model management, and OpenWebUI integration.
"""

import asyncio
import json
import time
import uuid
from typing import Optional, List, Dict, Any, AsyncGenerator
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from .config import OvllmConfig
from .models import ModelManager, ModelInfo
from .engine import AsyncEngine, SamplingParams


# ============== Request/Response Models ==============


class ChatMessage(BaseModel):
    """Chat message."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 256
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    n: Optional[int] = 1


class CompletionRequest(BaseModel):
    """OpenAI-compatible completion request."""

    model: str
    prompt: str
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 256
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class ModelResponse(BaseModel):
    """Model information response."""

    id: str
    object: str = "model"
    created: int
    owned_by: str = "ovllm"


class ChatCompletionResponse(BaseModel):
    """Chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


# ============== Server Application ==============


def create_app(
    config: Optional[OvllmConfig] = None,
    model_manager: Optional[ModelManager] = None,
    engine: Optional[AsyncEngine] = None,
) -> FastAPI:
    """Create the FastAPI application."""

    config = config or OvllmConfig()
    model_manager = model_manager or ModelManager(config)
    engine = engine or AsyncEngine(config, model_manager)

    app = FastAPI(
        title="Ovllm API",
        description="Ollama-like API for vLLM",
        version="0.1.0",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ============== OpenAI-Compatible Endpoints ==============

    @app.post("/v1/chat/completions")
    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        """OpenAI-compatible chat completions endpoint."""

        if not engine.is_loaded:
            # Auto-load model
            try:
                engine.load_model(request.model)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        sampling_params = SamplingParams(
            temperature=request.temperature or 0.7,
            top_p=request.top_p or 0.95,
            max_tokens=request.max_tokens or 256,
            stop=request.stop,
            n=request.n or 1,
        )

        # Format messages
        messages = [msg.dict() for msg in request.messages]

        if request.stream:
            return StreamingResponse(
                stream_chat_completion(
                    messages, sampling_params, request.model
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            # Non-streaming completion
            result = await generate_chat(messages, sampling_params)

            return {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        """OpenAI-compatible completions endpoint."""

        if not engine.is_loaded:
            try:
                engine.load_model(request.model)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        sampling_params = SamplingParams(
            temperature=request.temperature or 0.7,
            top_p=request.top_p or 0.95,
            max_tokens=request.max_tokens or 256,
            stop=request.stop,
        )

        if request.stream:
            return StreamingResponse(
                stream_completion(request.prompt, sampling_params),
                media_type="text/event-stream",
            )
        else:
            result = await generate_text(request.prompt, sampling_params)
            return {
                "id": f"cmpl-{uuid.uuid4()}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "text": result,
                        "finish_reason": "stop",
                    }
                ],
            }

    @app.get("/v1/models")
    @app.get("/models")
    async def list_models():
        """List available models (OpenAI-compatible)."""
        models = model_manager.list_models()
        return {
            "object": "list",
            "data": [
                {
                    "id": m.model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "ovllm",
                }
                for m in models
            ],
        }

    # ============== Ollama-Compatible Endpoints ==============

    @app.post("/api/pull")
    async def pull_model(request: Request):
        """Pull (download) a model from HuggingFace."""
        data = await request.json()
        model_id = data.get("name") or data.get("model")

        if not model_id:
            raise HTTPException(status_code=400, detail="Model name required")

        try:
            # Run download in background
            asyncio.create_task(
                asyncio.to_thread(
                    model_manager.download, model_id
                )
            )
            return {"status": "downloading", "model": model_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/tags")
    @app.post("/api/tags")
    async def list_tags():
        """List local models (Ollama-compatible)."""
        models = model_manager.list_models()
        return {
            "models": [
                {
                    "name": m.model_id,
                    "model": m.model_id,
                    "modified_at": m.downloaded_at,
                    "size": m.size_bytes,
                }
                for m in models
            ]
        }

    @app.get("/api/ps")
    async def list_running_models():
        """List running models (Ollama-compatible)."""
        # Return currently loaded model if any
        if engine.current_model:
            return {
                "models": [
                    {
                        "name": engine.current_model,
                        "model": engine.current_model,
                        "size": 0,  # Size not available
                        "digest": "",
                    }
                ]
            }
        return {"models": []}

    @app.get("/api/version")
    async def get_version():
        """Get version info (Ollama-compatible)."""
        return {
            "version": "0.1.0",
        }

    @app.post("/api/generate")
    async def api_generate(request: Request):
        """Generate completion (Ollama-compatible)."""
        data = await request.json()
        model_id = data.get("model")
        prompt = data.get("prompt", "")
        stream = data.get("stream", False)

        if not model_id:
            # Use loaded model or first available
            if engine.current_model:
                model_id = engine.current_model
            else:
                models = model_manager.list_models()
                if models:
                    model_id = models[0].model_id
                else:
                    raise HTTPException(
                        status_code=400, detail="No model specified"
                    )

        sampling_params = SamplingParams(
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 0.95),
            max_tokens=data.get("max_tokens", 256),
        )

        if stream:
            return StreamingResponse(
                stream_generate(model_id, prompt, sampling_params),
                media_type="application/x-ndjson",
            )
        else:
            result = await generate_text(prompt, sampling_params)
            return {
                "model": model_id,
                "response": result,
                "done": True,
            }

    @app.post("/api/chat")
    async def api_chat(request: Request):
        """Chat completion (Ollama-compatible)."""
        data = await request.json()
        model_id = data.get("model")
        messages = data.get("messages", [])
        stream = data.get("stream", False)

        if not model_id:
            if engine.current_model:
                model_id = engine.current_model
            else:
                models = model_manager.list_models()
                if models:
                    model_id = models[0].model_id
                else:
                    raise HTTPException(
                        status_code=400, detail="No model specified"
                    )

        sampling_params = SamplingParams(
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 0.95),
            max_tokens=data.get("max_tokens", 256),
        )

        if stream:
            return StreamingResponse(
                stream_chat(model_id, messages, sampling_params),
                media_type="application/x-ndjson",
            )
        else:
            result = await generate_chat(messages, sampling_params)
            return {
                "model": model_id,
                "message": {
                    "role": "assistant",
                    "content": result,
                },
                "done": True,
            }

    @app.post("/api/delete")
    async def delete_model(request: Request):
        """Delete a model."""
        data = await request.json()
        model_id = data.get("name") or data.get("model")

        if not model_id:
            raise HTTPException(status_code=400, detail="Model name required")

        removed = model_manager.remove(model_id)
        return {"removed": removed}

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "service": "ovllm",
            "version": "0.1.0",
            "status": "running",
        }

    @app.get("/health")
    async def health():
        """Health check."""
        return {"status": "healthy"}

    # ============== Helper Functions ==============

    async def generate_text(
        prompt: str, sampling_params: SamplingParams
    ) -> str:
        """Generate text completion."""
        if not engine.is_loaded:
            raise HTTPException(status_code=400, detail="No model loaded")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: engine._llm.generate(
                [prompt], sampling_params.to_vllm()
            ),
        )

        return result[0].outputs[0].text if result else ""

    async def generate_chat(
        messages: List[Dict], sampling_params: SamplingParams
    ) -> str:
        """Generate chat response."""
        # Format chat messages
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += f"{role.capitalize()}: {content}\n"
        formatted += "Assistant:"

        # Add stop sequences to prevent model from continuing the conversation
        if sampling_params.stop is None:
            sampling_params.stop = ["\n\nUser:", "\nUser:", "User:"]
        else:
            sampling_params.stop = sampling_params.stop + ["\n\nUser:", "\nUser:", "User:"]

        result = await generate_text(formatted, sampling_params)

        # Clean up response - remove any trailing "User:" or "Assistant:" that might cause loops
        result = result.strip()
        # Remove any prefix that looks like our formatting
        if result.startswith("Assistant:"):
            result = result[len("Assistant:"):].strip()
        if result.startswith("User:"):
            result = result[len("User:"):].strip()
        # Remove any trailing User: pattern that might have slipped through
        if "\nUser:" in result:
            result = result.split("\nUser:")[0].strip()

        return result

    async def stream_generate(
        model_id: str, prompt: str, sampling_params: SamplingParams
    ) -> AsyncGenerator[str, None]:
        """Stream generation output."""
        if not engine.is_loaded:
            engine.load_model(model_id)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: engine._llm.generate(
                [prompt], sampling_params.to_vllm()
            ),
        )

        if result:
            text = result[0].outputs[0].text
            chunk = json.dumps({"response": text, "done": False}) + "\n"
            yield chunk
            yield json.dumps({"done": True}) + "\n"

    async def stream_chat(
        model_id: str,
        messages: List[Dict],
        sampling_params: SamplingParams,
    ) -> AsyncGenerator[str, None]:
        """Stream chat response."""
        if not engine.is_loaded:
            engine.load_model(model_id)

        # Format chat
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += f"{role.capitalize()}: {content}\n"
        formatted += "Assistant:"

        # Add stop sequences to prevent model from continuing the conversation
        if sampling_params.stop is None:
            sampling_params.stop = ["\n\nUser:", "\nUser:", "User:"]
        else:
            sampling_params.stop = sampling_params.stop + ["\n\nUser:", "\nUser:", "User:"]

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: engine._llm.generate(
                [formatted], sampling_params.to_vllm()
            ),
        )

        if result:
            text = result[0].outputs[0].text
            # Clean up response
            text = text.strip()
            if text.startswith("Assistant:"):
                text = text[len("Assistant:"):].strip()
            if text.startswith("User:"):
                text = text[len("User:"):].strip()
            if "\nUser:" in text:
                text = text.split("\nUser:")[0].strip()

            chunk = json.dumps({
                "message": {"role": "assistant", "content": text},
                "done": False,
            }) + "\n"
            yield chunk
            yield json.dumps({"done": True}) + "\n"

    async def stream_chat_completion(
        messages: List[Dict],
        sampling_params: SamplingParams,
        model: str,
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion (OpenAI format)."""
        if not engine.is_loaded:
            engine.load_model(model)

        # Format chat
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += f"{role.capitalize()}: {content}\n"
        formatted += "Assistant:"

        # Add stop sequences to prevent model from continuing the conversation
        if sampling_params.stop is None:
            sampling_params.stop = ["\n\nUser:", "\nUser:", "User:"]
        else:
            sampling_params.stop = sampling_params.stop + ["\n\nUser:", "\nUser:", "User:"]

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: engine._llm.generate(
                [formatted], sampling_params.to_vllm()
            ),
        )

        if result:
            text = result[0].outputs[0].text
            # Clean up response
            text = text.strip()
            if text.startswith("Assistant:"):
                text = text[len("Assistant:"):].strip()
            if text.startswith("User:"):
                text = text[len("User:"):].strip()
            if "\nUser:" in text:
                text = text.split("\nUser:")[0].strip()

            chunk_data = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk_data)}\n\n"

            yield "data: [DONE]\n\n"

    async def stream_completion(
        prompt: str, sampling_params: SamplingParams
    ) -> AsyncGenerator[str, None]:
        """Stream text completion (OpenAI format)."""
        if not engine.is_loaded:
            raise HTTPException(status_code=400, detail="No model loaded")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: engine._llm.generate(
                [prompt], sampling_params.to_vllm()
            ),
        )

        if result:
            text = result[0].outputs[0].text
            chunk_data = {
                "id": f"cmpl-{uuid.uuid4()}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": engine.current_model,
                "choices": [
                    {
                        "index": 0,
                        "text": text,
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(chunk_data)}\n\n"

            yield "data: [DONE]\n\n"

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 11434,
    config: Optional[OvllmConfig] = None,
) -> None:
    """Run the Ovllm server."""
    config = config or OvllmConfig()
    app = create_app(config)

    print(f"Starting Ovllm server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
