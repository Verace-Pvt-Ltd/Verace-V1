import sys, time
import torch
from dataclasses import dataclass
from verace_v1.config import VeraceV1Config
from verace_v1.modules.backbone import VeraceV1Model
from verace_v1.optimizer.unitary_muon import build_hybrid_optimizer

@dataclass
class TinyVision:
    embed_dim: int = 16
    num_layers: int = 1
    num_heads: int = 1
    patch_size: int = 14

CONFIGS = {
    "1M": dict(hidden_dim=50, num_heads=2, head_dim=25, spectral_dim=50, chams_holographic_dim=32, mcmoe_rank=8, mcmoe_num_components=25),
    "2.5M": dict(hidden_dim=80, num_heads=2, head_dim=40, spectral_dim=80, chams_holographic_dim=48, mcmoe_rank=13, mcmoe_num_components=40),
    "8.3M": dict(hidden_dim=136, num_heads=2, head_dim=68, spectral_dim=136, chams_holographic_dim=64, mcmoe_rank=22, mcmoe_num_components=68),
}

name = sys.argv[1]
batch = int(sys.argv[2])
ctx = int(sys.argv[3])
device = "cuda"

cfg = VeraceV1Config(
    vocab_size=10000, num_layers=8, max_cognitive_depth=8, min_cognitive_depth=1,
    vision_config=TinyVision(), **CONFIGS[name]
)
model = VeraceV1Model(cfg).to(device)
optimizer = build_hybrid_optimizer(model)
n = sum(p.numel() for p in model.parameters())
print(f"{name}: {n/1e6:.3f}M params, batch={batch} ctx={ctx}")

torch.cuda.reset_peak_memory_stats()
times = []
for step in range(15):
    input_ids = torch.randint(0, cfg.vocab_size, (batch, ctx), device=device)
    labels = input_ids.clone()
    t0 = time.time()
    optimizer.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, depth_counts, hidden = model(input_ids, use_adaptive_depth=True, return_hidden=True)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    times.append(time.time() - t0)

peak = torch.cuda.max_memory_allocated() / 1e9
avg_ms = sum(times[5:]) / len(times[5:]) * 1000
tokens_per_step = batch * ctx
tokens_per_sec = tokens_per_step / (avg_ms / 1000)
print(f"peak_mem={peak:.2f}GB avg_step={avg_ms:.0f}ms tokens/step={tokens_per_step} tokens/sec={tokens_per_sec:.0f}")
