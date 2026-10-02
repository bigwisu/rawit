# Copyright 2026 Wisu Suntoyo — Apache-2.0
"""Supervised distillation training script for Rawit.

Trains the decision head + top N backbone layers on JSONL triplets using
Soft Cross-Entropy against teacher soft distributions. Uses PyTorch DDP for
multi-GPU and Accelerate for mixed precision / gradient accumulation.

Optimised for GCP NVIDIA L4 (24 GB VRAM, CUDA 12.x, Ada Lovelace BF16
tensor cores).  Key flags for L4:

  --grad_checkpoint   Enable gradient checkpointing (saves ~8 GB VRAM, costs
                      ~30% step time).  Always use when backbone layers are
                      unfrozen.
  --compile           Wrap the model with torch.compile(mode='reduce-overhead')
                      for CUDA graph capture — cuts ~15-20% step time after the
                      2-min warmup.  Worth enabling for runs > 500 steps.

Usage (GCP L4 — recommended):
    python -m rawit.training.train_distill \\
        --train_file data/train.jsonl \\
        --val_file   data/val.jsonl \\
        --output_dir checkpoints/rawit-300m \\
        --backbone   aisingapore/SEA-LION-ModernBERT-300M \\
        --epochs     3 \\
        --batch_size 32 \\
        --max_len    1024 \\
        --lr         2e-5 \\
        --unfreeze_backbone_layers 4 \\
        --num_workers 4 \\
        --grad_checkpoint \\
        --compile \\
        --device cuda

Usage (single GPU / MPS, debug):
    python -m rawit.training.train_distill \\
        --train_file data/train.jsonl \\
        --val_file data/val.jsonl \\
        --output_dir checkpoints/rawit-300m \\
        --backbone aisingapore/SEA-LION-ModernBERT-300M \\
        --epochs 3 \\
        --batch_size 32 \\
        --unfreeze_backbone_layers 4

Distributed (torchrun):
    torchrun --nproc_per_node=4 -m rawit.training.train_distill [args]
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..configuration_rawit import RawitConfig
from ..modeling_rawit import RawitModel
from ..tokenization_rawit import load_tokenizer
from .dataset import TripletDataset, collate_fn
from .losses import decision_loss

_log = logging.getLogger("rawit.train_distill")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _freeze_backbone(model: RawitModel, unfreeze_top_n: int = 4):
    """Freeze all backbone layers except the top N transformer blocks."""
    if model.encoder is None:
        return
    # Freeze everything first
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    # Unfreeze the top N encoder layers (ModernBERT uses .layers attribute)
    layers = None
    for attr in ("layers", "encoder.layers"):
        try:
            layers = model.encoder
            for part in attr.split("."):
                layers = getattr(layers, part)
            break
        except AttributeError:
            layers = None
    if layers is not None and unfreeze_top_n > 0:
        for layer in list(layers)[-unfreeze_top_n:]:
            for p in layer.parameters():
                p.requires_grad_(True)
    # Always train the head
    for p in model.head.parameters():
        p.requires_grad_(True)


def _enable_gradient_checkpointing(model: RawitModel):
    """Enable gradient checkpointing on the encoder backbone if supported.

    ModernBERT / most HuggingFace models expose gradient_checkpointing_enable().
    Falls back silently if the encoder doesn't support it.
    """
    if model.encoder is None:
        return
    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
        _log.info("Gradient checkpointing enabled on encoder backbone.")
    else:
        _log.warning(
            "Encoder does not expose gradient_checkpointing_enable(); "
            "gradient checkpointing not applied."
        )


def train(args):
    config = RawitConfig(backbone_name=args.backbone)
    model = RawitModel.from_pretrained_backbone(config)
    _freeze_backbone(model, unfreeze_top_n=args.unfreeze_backbone_layers)

    # ── Gradient checkpointing (L4: saves ~8 GB VRAM) ─────────────────────────
    if args.grad_checkpoint:
        _enable_gradient_checkpointing(model)

    tok = load_tokenizer(args.backbone)
    # Resize encoder embeddings to include the new [OPT_k] / [RUBRIC] tokens
    model.encoder.resize_token_embeddings(len(tok))

    train_ds = TripletDataset(args.train_file, tok, max_len=args.max_len)
    val_ds = TripletDataset(args.val_file, tok, max_len=args.max_len) if args.val_file else None

    pad_id = tok.pad_token_id or 0
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, pad_id=pad_id),
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(args.device or (
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    ))
    model = model.to(device)

    # ── torch.compile (L4: ~15-20% step speedup via CUDA graph capture) ───────
    # compile must happen after .to(device) and before the training loop.
    # Requires Triton + a working GCC/libcuda linkage in the active venv.
    # If the backend compiler fails (CalledProcessError from gcc/triton) the
    # error surfaces on the first forward pass, not at torch.compile() time,
    # so we suppress dynamo errors globally and fall back to eager execution.
    if args.compile:
        if device.type != "cuda":
            _log.warning("--compile is only effective on CUDA devices; skipping.")
        else:
            try:
                import torch._dynamo
                torch._dynamo.config.suppress_errors = True
                model = torch.compile(model, mode="reduce-overhead")
                _log.info("torch.compile enabled (mode=reduce-overhead); "
                          "will fall back to eager if Triton linkage fails.")
            except Exception as e:
                _log.warning("torch.compile failed (%s); continuing without.", e)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    # L4 Ada Lovelace: use bfloat16 (native BF16 tensor cores; avoids float16 NaN spikes)
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=amp_dtype):
                logits, _esc = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    option_marker_positions=batch["option_marker_positions"],
                    option_marker_mask=batch["option_marker_mask"],
                    qtype=batch["qtype"],
                )
                loss = decision_loss(logits, batch["target"], batch["option_marker_mask"], batch["qtype"])

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            global_step += 1
            if step % 100 == 0:
                _log.info("epoch=%d step=%d loss=%.4f", epoch + 1, step, loss.item())

        avg_loss = epoch_loss / max(1, len(train_loader))
        _log.info("epoch=%d avg_loss=%.4f", epoch + 1, avg_loss)

        # Save checkpoint after each epoch
        ckpt_path = output_dir / f"checkpoint-epoch{epoch + 1}"
        ckpt_path.mkdir(exist_ok=True)
        torch.save(model.state_dict(), ckpt_path / "model.pt")
        config.save_pretrained(str(ckpt_path))
        tok.save_pretrained(str(ckpt_path))
        _log.info("Saved checkpoint: %s", ckpt_path)

    _log.info("Training complete. Final checkpoint in: %s", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Rawit distillation training")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--val_file", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--backbone", default="aisingapore/SEA-LION-ModernBERT-300M")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="L4 recommendation: 32 (top-4 unfrozen) or 64 (head-only)")
    parser.add_argument("--max_len", type=int, default=1024,
                        help="Use 512 for fast iteration, 1024 for final runs")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="2e-5 for head+top-4; use 5e-6 for full backbone")
    parser.add_argument("--unfreeze_backbone_layers", type=int, default=4,
                        help="0 = head-only; 4 = standard; 22 = full backbone")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    # ── L4 / CUDA flags ───────────────────────────────────────────────────────
    parser.add_argument(
        "--grad_checkpoint", action="store_true",
        help="Enable gradient checkpointing on the encoder (saves ~8 GB VRAM "
             "at ~30%% step time cost). Recommended when backbone layers are unfrozen.",
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="Wrap model with torch.compile(mode='reduce-overhead') for CUDA graph "
             "capture (~15-20%% step speedup on L4). Adds ~2 min warmup; not worth "
             "it for < 500 steps.",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
