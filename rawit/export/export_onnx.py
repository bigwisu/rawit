# Copyright 2026 Wisu Suntoyo — Apache-2.0
"""Export a Rawit checkpoint to ONNX with dynamic axes.

Usage:
    python -m rawit.export.export_onnx \\
        --checkpoint checkpoints/rawit-300m \\
        --output rawit.onnx \\
        --fp16

Dynamic axes: batch_size, sequence_length, num_options
"""

import argparse
import logging
from pathlib import Path

import torch

from ..configuration_rawit import RawitConfig
from ..modeling_rawit import RawitModel
from ..tokenization_rawit import load_tokenizer

_log = logging.getLogger("rawit.export_onnx")


class _OnnxWrapper(torch.nn.Module):
    """Thin wrapper exposing a fixed-signature forward for the ONNX exporter.

    The exporter needs concrete input names that map to the dynamic axes dict.
    """

    def __init__(self, model: RawitModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, option_marker_positions, option_marker_mask, qtype):
        logits, esc_logit = self.model(
            input_ids, attention_mask, option_marker_positions, option_marker_mask, qtype
        )
        return logits, torch.sigmoid(esc_logit)


def export(checkpoint: str, output: str, fp16: bool = False, opset: int = 17):
    _log.info("Loading checkpoint: %s", checkpoint)
    from transformers import AutoConfig
    cfg_raw = AutoConfig.from_pretrained(checkpoint)
    config = RawitConfig(**{k: getattr(cfg_raw, k) for k in RawitConfig().__dict__ if hasattr(cfg_raw, k)})
    model = RawitModel.from_pretrained_backbone(config, no_init=True)
    state = torch.load(Path(checkpoint) / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    if fp16:
        model = model.half()

    wrapper = _OnnxWrapper(model)

    # Dummy inputs (batch=1, seq=64, k=4)
    B, L, K = 1, 64, 4
    dummy = (
        torch.zeros(B, L, dtype=torch.long),     # input_ids
        torch.ones(B, L, dtype=torch.long),      # attention_mask
        torch.zeros(B, K, dtype=torch.long),     # option_marker_positions
        torch.ones(B, K, dtype=torch.bool),      # option_marker_mask
        torch.zeros(B, dtype=torch.long),        # qtype
    )

    dynamic_axes = {
        "input_ids":               {0: "batch_size", 1: "sequence_length"},
        "attention_mask":          {0: "batch_size", 1: "sequence_length"},
        "option_marker_positions": {0: "batch_size", 1: "num_options"},
        "option_marker_mask":      {0: "batch_size", 1: "num_options"},
        "qtype":                   {0: "batch_size"},
        "option_logits":           {0: "batch_size", 1: "num_options"},
        "escalation_prob":         {0: "batch_size"},
    }

    _log.info("Exporting to ONNX: %s (opset=%d fp16=%s)", output, opset, fp16)
    torch.onnx.export(
        wrapper,
        dummy,
        output,
        input_names=["input_ids", "attention_mask", "option_marker_positions",
                     "option_marker_mask", "qtype"],
        output_names=["option_logits", "escalation_prob"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    _log.info("Exported: %s", output)

    if fp16:
        try:
            from onnxruntime.transformers.optimizer import optimize_model
            opt = optimize_model(output, model_type="bert", use_gpu=True)
            opt.convert_float_to_float16()
            opt_path = output.replace(".onnx", "-fp16-opt.onnx")
            opt.save_model_to_file(opt_path)
            _log.info("FP16-optimized model saved: %s", opt_path)
        except ImportError:
            _log.warning("onnxruntime-gpu not available; skipping FP16 graph optimization.")


def main():
    parser = argparse.ArgumentParser(description="Export Rawit to ONNX")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="rawit.onnx")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    export(args.checkpoint, args.output, fp16=args.fp16, opset=args.opset)


if __name__ == "__main__":
    main()
