"""
FastAPI gateway for Triton Inference Server (Llama 3).
Exposes an OpenAI-compatible /v1/chat/completions endpoint.
"""

import os
import json
import asyncio
import logging
from typing import AsyncIterator

import httpx
import tritonclient.http.aio as triton_http
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=os.getenv("LOG_LEVEL", "info").upper())
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
TRITON_HOST = os.getenv("TRITON_HOST", "localhost")
TRITON_HTTP_PORT = int(os.getenv("TRITON_HTTP_PORT", 8000))
TRITON_GRPC_PORT = int(os.getenv("TRITON_GRPC_PORT", 8001))
API_KEY = os.getenv("API_KEY", "changeme")
MODEL_NAME = "llama3"

# ── Auth ─────────────────────────────────────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return key

# ── Schemas ──────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[Message]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    stream: bool = False
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)

class ChatChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    choices: list[ChatChoice]

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Llama 3 LLM Gateway",
    description="OpenAI-compatible API backed by Triton Inference Server + vLLM",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def messages_to_prompt(messages: list[Message]) -> str:
    """Convert chat messages to Llama 3 instruct format."""
    parts = ["<|begin_of_text|>"]
    for msg in messages:
        parts.append(f"<|start_header_id|>{msg.role}<|end_header_id|>\n\n{msg.content}<|eot_id|>")
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


async def call_triton(prompt: str, sampling_params: dict, stream: bool) -> str:
    """Send inference request to Triton and return the response text."""
    client = triton_http.InferenceServerClient(
        url=f"{TRITON_HOST}:{TRITON_HTTP_PORT}"
    )

    text_input = triton_http.InferInput("text_input", [1], "BYTES")
    text_input.set_data_from_numpy(np.array([prompt.encode("utf-8")], dtype=object))

    stream_input = triton_http.InferInput("stream", [1], "BOOL")
    stream_input.set_data_from_numpy(np.array([stream], dtype=bool))

    params_input = triton_http.InferInput("sampling_parameters", [1], "BYTES")
    params_input.set_data_from_numpy(
        np.array([json.dumps(sampling_params).encode("utf-8")], dtype=object)
    )

    output = triton_http.InferRequestedOutput("text_output")

    result = await client.infer(
        model_name=MODEL_NAME,
        inputs=[text_input, stream_input, params_input],
        outputs=[output],
    )

    response_bytes = result.as_numpy("text_output")
    return response_bytes[0].decode("utf-8")


async def stream_triton(prompt: str, sampling_params: dict) -> AsyncIterator[str]:
    """Stream tokens from Triton using the decoupled (streaming) protocol."""
    client = triton_http.InferenceServerClient(
        url=f"{TRITON_HOST}:{TRITON_HTTP_PORT}"
    )

    text_input = triton_http.InferInput("text_input", [1], "BYTES")
    text_input.set_data_from_numpy(np.array([prompt.encode("utf-8")], dtype=object))

    stream_input = triton_http.InferInput("stream", [1], "BOOL")
    stream_input.set_data_from_numpy(np.array([True], dtype=bool))

    params_input = triton_http.InferInput("sampling_parameters", [1], "BYTES")
    params_input.set_data_from_numpy(
        np.array([json.dumps(sampling_params).encode("utf-8")], dtype=object)
    )

    async with client.stream_infer(
        model_name=MODEL_NAME,
        inputs=[text_input, stream_input, params_input],
    ) as stream:
        async for result in stream:
            token = result.as_numpy("text_output")[0].decode("utf-8")
            chunk = {
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"content": token}, "index": 0, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

    yield "data: [DONE]\n\n"


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models():
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}


@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(req: ChatRequest):
    prompt = messages_to_prompt(req.messages)
    sampling_params = {
        "temperature": req.temperature,
        "top_p": req.top_p,
        "max_tokens": req.max_tokens,
    }

    if req.stream:
        return StreamingResponse(
            stream_triton(prompt, sampling_params),
            media_type="text/event-stream",
        )

    try:
        text = await call_triton(prompt, sampling_params, stream=False)
    except Exception as e:
        logger.exception("Triton inference error")
        raise HTTPException(status_code=502, detail=f"Triton error: {e}")

    return ChatResponse(
        id="chatcmpl-triton",
        model=req.model,
        choices=[
            ChatChoice(
                index=0,
                message=Message(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
    )
