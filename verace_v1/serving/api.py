"""
Verace V1 Production FastAPI Serving Server
Provides OpenAI-compatible REST API endpoints for completions, chat completions, and health checks.

Configuration (environment variables):
  VERACE_CHECKPOINT_PATH   Path to a checkpoint saved by verace_v1.training.pretrain.save_checkpoint.
                           If unset, the model runs with random initialization -- fine for
                           exercising the serving pipeline, useless for real completions --
                           and a warning is logged loudly at startup, not hidden.
  VERACE_API_KEY           If set, all /v1/* endpoints require `Authorization: Bearer <key>`.
                           If unset, the server runs with no auth (fine for local dev) and
                           logs a warning at startup so this is never silently the case in
                           a real deployment.
  VERACE_CORS_ORIGINS      Comma-separated allowed origins (e.g. "https://app.example.com").
                           Defaults to none (no cross-origin requests allowed). Credentials
                           are never combined with a wildcard origin -- that combination is
                           invalid and insecure.
  VERACE_RATE_LIMIT_PER_MINUTE
                           Per-client request cap on /v1/* endpoints (default 30). Basic,
                           in-process, per-IP protection only -- not a substitute for a real
                           gateway/WAF in front of a multi-instance deployment.
  VERACE_VOCAB_SIZE, VERACE_HIDDEN_DIM, VERACE_NUM_LAYERS, VERACE_NUM_HEADS, VERACE_HEAD_DIM
                           Served model dimensions (defaults: 163840, 2048, 12, 16, 128 --
                           sized for a data-center GPU). Override to fit smaller hardware;
                           must match whatever VERACE_CHECKPOINT_PATH was trained with.
"""

import asyncio
import logging
import os
import secrets
import time
from functools import partial
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from verace_v1.config import VeraceV1Config
from verace_v1.modules.backbone import VeraceV1Model
from verace_v1.serving.hyper_generate import VeraceV1Generator
from verace_v1.tokenizer import VeraceTokenizer
from verace_v1.training.pretrain import load_checkpoint

logger = logging.getLogger("verace_v1.serving.api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Verace V1 Inference Server", version="1.0.0")

_cors_origins = [o.strip() for o in os.environ.get("VERACE_CORS_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,  # empty by default -- no cross-origin requests allowed
    allow_credentials=False,      # never combine credentials with a wildcard origin
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# Global model state
global_model: Optional[VeraceV1Model] = None
global_config: Optional[VeraceV1Config] = None
global_generator: Optional[VeraceV1Generator] = None
global_tokenizer: Optional[VeraceTokenizer] = None

# Basic per-client sliding-window rate limiting (in-process only -- resets on restart,
# does not coordinate across multiple server instances).
_RATE_LIMIT_PER_MINUTE = int(os.environ.get("VERACE_RATE_LIMIT_PER_MINUTE", "30"))
_rate_limit_state: Dict[str, List[float]] = {}


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    use_tree_search: bool = True
    num_branches: int = 4

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    use_tree_search: bool = True


def verify_api_key(request: Request):
    expected_key = os.environ.get("VERACE_API_KEY")
    if not expected_key:
        return  # auth disabled -- warned about loudly at startup, see initialize_engine()
    auth_header = request.headers.get("authorization", "")
    provided = auth_header[7:] if auth_header.lower().startswith("bearer ") else ""
    if not provided or not secrets.compare_digest(provided, expected_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def check_rate_limit(request: Request):
    client_id = request.client.host if request.client else "unknown"
    now = time.time()
    window_start = now - 60.0
    recent = [t for t in _rate_limit_state.get(client_id, []) if t > window_start]
    if len(recent) >= _RATE_LIMIT_PER_MINUTE:
        _rate_limit_state[client_id] = recent
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    recent.append(now)
    _rate_limit_state[client_id] = recent


def initialize_engine():
    global global_model, global_config, global_generator, global_tokenizer
    if global_model is not None:
        return

    global_config = VeraceV1Config(
        vocab_size=int(os.environ.get("VERACE_VOCAB_SIZE", "163840")),
        hidden_dim=int(os.environ.get("VERACE_HIDDEN_DIM", "2048")),
        num_layers=int(os.environ.get("VERACE_NUM_LAYERS", "12")),
        num_heads=int(os.environ.get("VERACE_NUM_HEADS", "16")),
        head_dim=int(os.environ.get("VERACE_HEAD_DIM", "128"))
    )
    global_tokenizer = VeraceTokenizer(vocab_size=global_config.vocab_size)
    global_model = VeraceV1Model(global_config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    global_model.to(device)

    checkpoint_path = os.environ.get("VERACE_CHECKPOINT_PATH")
    if checkpoint_path:
        load_checkpoint(global_model, None, checkpoint_path, map_location=device)
    else:
        logger.warning(
            "VERACE_CHECKPOINT_PATH is not set -- serving a randomly initialized model. "
            "Completions will not be meaningful. Set VERACE_CHECKPOINT_PATH to a checkpoint "
            "from verace_v1.training.pretrain.save_checkpoint for real inference."
        )

    if not os.environ.get("VERACE_API_KEY"):
        logger.warning(
            "VERACE_API_KEY is not set -- this server is accepting unauthenticated requests. "
            "Set VERACE_API_KEY before exposing this server beyond local development."
        )

    global_model.eval()
    global_generator = VeraceV1Generator(global_model, global_config, tokenizer=global_tokenizer)


@app.on_event("startup")
async def startup_event():
    initialize_engine()

@app.get("/health")
async def health_check():
    if global_model is None:
        return {"status": "initializing", "model": "Verace-V1"}
    return {"status": "ok", "model": "Verace-V1", "device": str(next(global_model.parameters()).device)}

@app.post("/v1/completions", dependencies=[Depends(verify_api_key), Depends(check_rate_limit)])
async def generate_completion(req: CompletionRequest):
    if global_generator is None:
        initialize_engine()

    start_time = time.time()
    try:
        # generate() is a long-running, synchronous, GPU/CPU-bound call -- running it
        # directly in this coroutine would block the event loop (and every other
        # request, including /health) until it finishes. Offloading to the default
        # executor lets FastAPI keep serving other requests concurrently.
        loop = asyncio.get_event_loop()
        completion_text = await loop.run_in_executor(
            None,
            partial(
                global_generator.generate,
                prompt_text=req.prompt,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                use_tree_search=req.use_tree_search,
                num_branches=req.num_branches
            )
        )
    except Exception:
        logger.exception("Completion request failed")
        raise HTTPException(status_code=500, detail="Internal error while generating completion")

    duration = time.time() - start_time

    return {
        "id": f"cmpl-{int(time.time())}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": "Verace-V1",
        "choices": [
            {
                "text": completion_text,
                "index": 0,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(global_tokenizer.encode(req.prompt)),
            "completion_tokens": len(global_tokenizer.encode(completion_text)),
            "total_tokens": len(global_tokenizer.encode(req.prompt + completion_text)),
            "latency_seconds": round(duration, 3)
        }
    }

@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key), Depends(check_rate_limit)])
async def generate_chat_completion(req: ChatCompletionRequest):
    if global_generator is None:
        initialize_engine()

    formatted_prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in req.messages]) + "\nassistant:"
    start_time = time.time()

    try:
        loop = asyncio.get_event_loop()
        completion_text = await loop.run_in_executor(
            None,
            partial(
                global_generator.generate,
                prompt_text=formatted_prompt,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                use_tree_search=req.use_tree_search
            )
        )
    except Exception:
        logger.exception("Chat completion request failed")
        raise HTTPException(status_code=500, detail="Internal error while generating completion")

    duration = time.time() - start_time

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "Verace-V1",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": completion_text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "latency_seconds": round(duration, 3)
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
