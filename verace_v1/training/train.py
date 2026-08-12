"""
Verace V1 Pretraining Pilot Driver
Runnable single-GPU pretraining loop: builds a pilot-scale model + hybrid
Muon/AdamW optimizer (see verace_v1/optimizer/unitary_muon.py), trains on a
local text corpus (or a loudly-flagged synthetic fallback if none is given),
logs loss/depth/throughput, and checkpoints periodically with resume support.

Model defaults are sized for a single ~6GB-VRAM GPU, not for a
production-scale run -- override every dimension via CLI flags once the
pilot proves the architecture trains, to scale up on bigger hardware.
"""
import argparse
import json
import os
import time
from dataclasses import dataclass

import torch

from verace_v1.config import VeraceV1Config
from verace_v1.dataset.dataset import create_pretrain_dataloader
from verace_v1.modules.backbone import VeraceV1Model
from verace_v1.optimizer.unitary_muon import build_hybrid_optimizer
from verace_v1.tokenizer import VeraceTokenizer
from verace_v1.training.diagnostics import CHAMInvariantProbe
from verace_v1.training.pretrain import (
    get_cosine_schedule_with_warmup,
    load_checkpoint,
    save_checkpoint,
    train_pretrain_step,
)


@dataclass
class PilotVisionConfig:
    """Tiny vision encoder so a text-only pilot run doesn't sink most of its
    VRAM budget into an encoder it never calls (images=None means the vision
    encoder's weights exist but never run a forward pass)."""
    embed_dim: int = 128
    num_layers: int = 2
    num_heads: int = 2
    patch_size: int = 14


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verace V1 pretraining pilot driver")

    # Data
    p.add_argument("--data_path", type=str, default=None,
                    help="Path to a .txt/.jsonl file or a directory of them. "
                         "Omit to use the synthetic fallback (NOT real training -- "
                         "only useful for smoke-testing the pipeline).")
    p.add_argument("--context_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_tokens", type=int, default=None,
                    help="Cap on tokens read from --data_path (stops reading/tokenizing "
                         "once reached). Omit to use the whole corpus.")

    # Model -- pilot-scale defaults sized for a single ~6GB GPU
    p.add_argument("--vocab_size", type=int, default=8192)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument("--spectral_dim", type=int, default=64)
    p.add_argument("--chams_holographic_dim", type=int, default=128)
    p.add_argument("--mcmoe_rank", type=int, default=16)
    p.add_argument("--mcmoe_num_components", type=int, default=16)
    p.add_argument("--max_cognitive_depth", type=int, default=6)
    p.add_argument("--min_cognitive_depth", type=int, default=1)

    # Optimization
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--muon_lr", type=float, default=0.03)
    p.add_argument("--muon_momentum", type=float, default=0.98)
    p.add_argument("--muon_weight_decay", type=float, default=0.05)
    p.add_argument("--adamw_lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=20)
    p.add_argument("--depth_penalty_weight", type=float, default=0.001)
    p.add_argument("--energy_penalty_weight", type=float, default=0.01)
    p.add_argument("--amp", dest="use_amp", action="store_true", default=True)
    p.add_argument("--no_amp", dest="use_amp", action="store_false")

    # Logging / checkpointing
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None,
                    help="Path to a checkpoint .pt file (from save_checkpoint) to resume from.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)

    # Experiment tracking -- a local JSONL log is always written (no extra dependency);
    # wandb is opt-in and degrades to a warning if the package isn't installed.
    p.add_argument("--wandb", action="store_true", default=False,
                    help="Also log metrics to Weights & Biases (requires `pip install wandb`).")
    p.add_argument("--wandb_project", type=str, default="verace-v1-pretrain")
    p.add_argument("--wandb_run_name", type=str, default=None)

    return p.parse_args()


def estimate_param_and_optimizer_memory_gb(model: torch.nn.Module) -> tuple:
    """
    Rough upper-bound estimate (fp32, worst case 2x-state optimizer for every
    param) of params+grad+optimizer-state memory. Excludes activations, which
    scale with batch_size * context_length and dominate at longer sequences --
    this is a floor, not a full VRAM prediction.
    """
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bytes_per_param = 4  # fp32
    slots = 1 + 1 + 2  # param + grad + up-to-2 optimizer state slots (AdamW worst case)
    return n_params, (n_params * bytes_per_param * slots) / 1e9


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    config = VeraceV1Config(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        spectral_dim=args.spectral_dim,
        chams_holographic_dim=args.chams_holographic_dim,
        mcmoe_rank=args.mcmoe_rank,
        mcmoe_num_components=args.mcmoe_num_components,
        max_cognitive_depth=args.max_cognitive_depth,
        min_cognitive_depth=args.min_cognitive_depth,
        vision_config=PilotVisionConfig(),
    )

    tokenizer = VeraceTokenizer(vocab_size=config.vocab_size)

    if args.data_path is None:
        print("[Verace V1 Pretrain] WARNING: no --data_path given -- training on the "
              "synthetic fallback corpus. This proves the pipeline runs, NOT that the "
              "model learns anything. Pass --data_path to a real .txt/.jsonl file or "
              "directory for an actual pretraining pilot.")

    dataloader = create_pretrain_dataloader(
        data_path=args.data_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        context_length=args.context_length,
        num_workers=args.num_workers,
        max_tokens=args.max_tokens,
    )

    model = VeraceV1Model(config).to(args.device)
    n_params, est_mem_gb = estimate_param_and_optimizer_memory_gb(model)
    print(f"[Verace V1 Pretrain] Model: {n_params / 1e6:.2f}M params on {args.device}. "
          f"Rough params+grad+optimizer-state memory floor: {est_mem_gb:.2f} GB "
          f"(excludes activations -- reduce --batch_size / --context_length if you OOM).")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(args.checkpoint_dir, "train_log.jsonl")
    cham_probe = CHAMInvariantProbe(model)

    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
        except ImportError:
            print("[Verace V1 Pretrain] WARNING: --wandb passed but the `wandb` package isn't "
                  "installed -- skipping. `pip install wandb` to enable it.")

    optimizer = build_hybrid_optimizer(
        model,
        muon_lr=args.muon_lr,
        muon_momentum=args.muon_momentum,
        muon_weight_decay=args.muon_weight_decay,
        adamw_lr=args.adamw_lr,
    )
    # Two schedulers (one per inner optimizer) because HybridMuonAdamW composes two
    # real torch.optim.Optimizer instances rather than being one itself.
    muon_scheduler = get_cosine_schedule_with_warmup(optimizer.muon_optimizer, args.warmup_steps, args.steps)
    adamw_scheduler = get_cosine_schedule_with_warmup(optimizer.adamw_optimizer, args.warmup_steps, args.steps)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(model, optimizer, args.resume, map_location=args.device)
        # Fast-forward both LR schedules to the resumed step -- otherwise they'd
        # silently restart from warmup, spiking the LR right after resume.
        for _ in range(start_step):
            muon_scheduler.step()
            adamw_scheduler.step()

    data_iter = iter(dataloader)
    recent_step_times = []

    try:
        for step in range(start_step, args.steps):
            t0 = time.time()
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            batch = {k: v.to(args.device) for k, v in batch.items()}

            diagnostics = {}
            ce_loss, mean_depth = train_pretrain_step(
                model, optimizer, batch,
                depth_penalty_weight=args.depth_penalty_weight,
                energy_penalty_weight=args.energy_penalty_weight,
                use_amp=args.use_amp and args.device == "cuda",
                diagnostics=diagnostics,
            )
            cham_deviation = cham_probe.pop_mean_deviation()
            muon_scheduler.step()
            adamw_scheduler.step()
            recent_step_times.append(time.time() - t0)

            if step % args.log_every == 0 or step == args.steps - 1:
                window = recent_step_times[-args.log_every:]
                avg_ms = (sum(window) / len(window)) * 1000
                muon_lr = optimizer.muon_optimizer.param_groups[0]["lr"]
                adamw_lr = optimizer.adamw_optimizer.param_groups[0]["lr"]
                cham_str = f"{cham_deviation:.2e}" if cham_deviation is not None else "n/a"

                print(f"[step {step}/{args.steps}] loss={ce_loss:.4f} "
                      f"depth={diagnostics.get('depth_mean', mean_depth):.2f}"
                      f"(std={diagnostics.get('depth_std', 0.0):.2f}, "
                      f"range=[{diagnostics.get('depth_min', 0):.0f},{diagnostics.get('depth_max', 0):.0f}])"
                      f"/{config.num_layers} cham_dev={cham_str} "
                      f"muon_lr={muon_lr:.2e} adamw_lr={adamw_lr:.2e} {avg_ms:.0f}ms/step")

                log_record = {
                    "step": step, "loss": ce_loss, "cham_deviation": cham_deviation,
                    "muon_lr": muon_lr, "adamw_lr": adamw_lr, "ms_per_step": avg_ms,
                    **diagnostics,
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(log_record) + "\n")
                if wandb_run is not None:
                    wandb_run.log(log_record, step=step)

            if (step + 1) % args.save_every == 0:
                save_checkpoint(model, optimizer, step + 1, args.checkpoint_dir)

    except KeyboardInterrupt:
        print("[Verace V1 Pretrain] Interrupted -- saving a checkpoint before exiting.")
        save_checkpoint(model, optimizer, step, args.checkpoint_dir)
        cham_probe.remove()
        raise

    cham_probe.remove()
    save_checkpoint(model, optimizer, args.steps, args.checkpoint_dir)
    if wandb_run is not None:
        wandb_run.finish()
    print("[Verace V1 Pretrain] Pilot run complete.")


if __name__ == "__main__":
    main()
