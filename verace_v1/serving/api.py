"""
Verace V1 Production FastAPI Serving Server
Provides OpenAI-compatible REST API endpoints for completions, chat completions, and health checks.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import time
import torch

from verace_v1.config import VeraceV1Config
from verace_v1.modules.backbone import VeraceV1Model
from verace_v1.serving.hyper_generate import VeraceV1Generator
from verace_v1.tokenizer import VeraceTokenizer

app = FastAPI(title="Verace V1 Inference Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
global_model: Optional[VeraceV1Model] = None
global_config: Optional[VeraceV1Config] = None
global_generator: Optional[VeraceV1Generator] = None
global_tokenizer: Optional[VeraceTokenizer] = None

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

def initialize_engine():
    global global_model, global_config, global_generator, global_tokenizer
    if global_model is None:
        global_config = VeraceV1Config(
            vocab_size=163840,
            hidden_dim=2048,
            num_layers=12,
            num_heads=16,
            head_dim=128
        )
        global_tokenizer = VeraceTokenizer(vocab_size=global_config.vocab_size)
        global_model = VeraceV1Model(global_config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        global_model.to(device)
        global_model.eval()
        global_generator = VeraceV1Generator(global_model, global_config, tokenizer=global_tokenizer)

@app.on_event("startup")
async def startup_event():
    initialize_engine()

@app.get("/health")
async def health_check():
    return {"status": "ok", "model": "Verace-V1", "device": str(next(global_model.parameters()).device)}

@app.post("/v1/completions")
async def generate_completion(req: CompletionRequest):
    if global_generator is None:
        initialize_engine()
    
    start_time = time.time()
    try:
        completion_text = global_generator.generate(
            prompt_text=req.prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            use_tree_search=req.use_tree_search,
            num_branches=req.num_branches
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
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

@app.post("/v1/chat/completions")
async def generate_chat_completion(req: ChatCompletionRequest):
    if global_generator is None:
        initialize_engine()
        
    formatted_prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in req.messages]) + "\nassistant:"
    start_time = time.time()
    
    try:
        completion_text = global_generator.generate(
            prompt_text=formatted_prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            use_tree_search=req.use_tree_search
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
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
