# Copyright 2025 Wisu Suntoyo — Apache-2.0
"""Supervised distillation training script for Rawit.

Trains the decision head + top N backbone layers on JSONL triplets using
Soft Cross-Entropy against teacher soft distributions. Uses PyTorch DDP for
multi-GPU and Accelerate for mixed precision / gradient accumulation.

Usage (single GPU / MPS):
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


def train(args):
    config = RawitConfig(backbone_name=args.backbone)
    model = RawitModel.from_pretrained_backbone(config)
    _freeze_backbone(model, unfreeze_top_n=args.unfreeze_backbone_layers)

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

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
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
            with torch.cuda.amp.autocast(enabled=device.type == "cuda", dtype=torch.bfloat16):
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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--unfreeze_backbone_layers", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
