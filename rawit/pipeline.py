# Copyright 2026 Wisu Suntoyo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RawitPipeline — high-level Python inference API.

Usage:
    from rawit import RawitPipeline

    pipe = RawitPipeline.from_pretrained("bigwisu/rawit-300m")

    result = pipe.decide(
        context="Saldo rekening saya terpotong tapi uang tidak keluar.",
        rubric="Klasifikasikan kategori keluhan nasabah.",
        options=[
            {"id": "transaksi_gagal", "label": "Kegagalan Transaksi ATM"},
            {"id": "kartu_tertelan",  "label": "Kartu Tertelan / Rusak"},
            {"id": "informasi_umum",  "label": "Pertanyaan Layanan Umum"},
            {"id": "indikasi_fraud",  "label": "Laporan Penipuan / Pembobolan"},
        ],
        escalate_threshold=0.65,
    )
    # result.decision, result.confidence, result.probabilities, result.escalate
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from .configuration_rawit import RawitConfig
from .modeling_rawit import QTYPES, RawitModel
from .tokenization_rawit import encode_decision, load_tokenizer


@dataclass
class DecisionResult:
    """Output of a single RawitPipeline.decide() call."""

    decision: str
    """ID of the highest-probability option."""

    confidence: float
    """Normalized Shannon entropy confidence ∈ [0, 1]. 1 = maximally certain."""

    probabilities: Dict[str, float]
    """Calibrated probability for every option id."""

    escalate: bool
    """True when escalation_score >= escalate_threshold."""

    escalation_score: float
    """P(escalate) ∈ [0, 1]."""

    execution_time_ms: float
    """Wall-clock time for this call in milliseconds."""

    model: str = "rawit-300m"


class RawitPipeline:
    """Single-pass decision pipeline wrapping RawitModel + tokenizer.

    Args:
        model: Loaded RawitModel.
        tokenizer: Loaded tokenizer with Rawit special tokens registered.
        device: Torch device to run inference on.
        max_len: Maximum total sequence length.
        head_max_len: Maximum token budget for the rubric + option section.
    """

    def __init__(
        self,
        model: RawitModel,
        tokenizer,
        device: Optional[Union[str, torch.device]] = None,
        max_len: int = 1024,
        head_max_len: int = 256,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.head_max_len = head_max_len

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.model.to(self.device).eval()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        device: Optional[str] = None,
        revision: Optional[str] = None,
        **kwargs,
    ) -> "RawitPipeline":
        """Load a Rawit checkpoint from HuggingFace Hub or a local directory."""
        from transformers import AutoConfig
        cfg_raw = AutoConfig.from_pretrained(model_id, revision=revision)
        # Rebuild as RawitConfig in case it was saved as a generic config
        config = RawitConfig(**{
            k: getattr(cfg_raw, k)
            for k in RawitConfig().__dict__
            if hasattr(cfg_raw, k)
        })
        model = RawitModel.from_pretrained_backbone(config, revision=revision, no_init=True)
        # Load the full checkpoint (encoder + head + temperature buffers)
        from huggingface_hub import hf_hub_download
        import safetensors.torch
        weights_path = hf_hub_download(model_id, "model.safetensors", revision=revision)
        safetensors.torch.load_model(model, weights_path)
        tok = load_tokenizer(config.backbone_name, revision=revision)
        return cls(model, tok, device=device, **kwargs)

    @torch.inference_mode()
    def decide(
        self,
        context: str,
        rubric: str,
        options: List[Dict[str, str]],
        qtype: str = "choice",
        escalate_threshold: float = 0.65,
        truncate_context_left: bool = False,
    ) -> DecisionResult:
        """Run a single decision forward pass.

        Args:
            context: The state / document the decision is about.
            rubric: The question or classification instruction.
            options: List of {"id": ..., "label": ...} dicts.
            qtype: "choice" | "score" | "noul".
            escalate_threshold: P(escalate) threshold above which escalate=True.
            truncate_context_left: Truncate from the left (for conversation turns).

        Returns:
            DecisionResult with calibrated probabilities and escalation flag.
        """
        t0 = time.perf_counter()

        option_labels = [o["label"] for o in options]
        option_ids = [o["id"] for o in options]
        k = len(option_labels)

        input_ids, marker_positions = encode_decision(
            self.tokenizer,
            context=context,
            rubric=rubric,
            options=option_labels,
            max_len=self.max_len,
            head_max_len=self.head_max_len,
            truncate_context_left=truncate_context_left,
        )

        # Pad to batch size 1
        L = len(input_ids)
        pad_id = self.tokenizer.pad_token_id or 0
        ids_t = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        att_t = torch.ones((1, L), dtype=torch.long, device=self.device)

        # Build marker tensors (B=1, K)
        kmax = max(k, 1)
        mpos = torch.zeros((1, kmax), dtype=torch.long, device=self.device)
        mmask = torch.zeros((1, kmax), dtype=torch.bool, device=self.device)
        for i, pos in enumerate(marker_positions[:k]):
            mpos[0, i] = pos
            mmask[0, i] = True

        qtype_t = torch.tensor(
            [QTYPES.get(qtype, 0)], dtype=torch.long, device=self.device
        )

        logits, esc_logit = self.model(ids_t, att_t, mpos, mmask, qtype_t)

        # Calibrated probabilities
        probs_t = self.model.calibrated_probs(logits, qtype_t, mmask)
        probs = probs_t[0, :k].cpu().float().numpy()

        escalation_score = float(torch.sigmoid(esc_logit[0]).item())
        best_idx = int(np.argmax(probs))

        # Confidence: normalized Shannon entropy
        p = probs[:k]
        if k >= 2:
            ent = -float(np.sum(p * np.log(np.clip(p, 1e-12, 1.0))))
            confidence = float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))
        else:
            confidence = float(p[0])

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return DecisionResult(
            decision=option_ids[best_idx],
            confidence=confidence,
            probabilities={oid: float(p[i]) for i, oid in enumerate(option_ids)},
            escalate=escalation_score >= escalate_threshold,
            escalation_score=escalation_score,
            execution_time_ms=round(elapsed_ms, 2),
        )
