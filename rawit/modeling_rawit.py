# Copyright 2025 Wisu Suntoyo
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
#
# Decision head architecture derived from Laya (Apache-2.0):
# https://github.com/convaiinnovations/laya

"""RawitModel — SEA-LION encoder backbone + Rawit typed decision head.

Key difference from Laya: option representations are read exclusively from
abstract [OPT_k] marker positions, not from [MASK] tokens placed at literal
option words. This neutralises polarity bias (the Sun & Xu failure mode) where
a model trained on ya/tidak learns to score lexical priors rather than the rubric.

Sequence format:
  [BOS] context ... [SEP] rubric / question ... [OPT_0] desc_0 [OPT_1] desc_1 ... [EOS]

The decision head gathers hidden states at the [OPT_k] positions, projects them
through a two-layer scorer, and emits per-option logits + an escalation scalar.
"""

import math
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_rawit import RawitConfig

# Map primitive name -> integer index (shared with tokenizer and training code)
QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}

# Temperature clamp: a value below TEMP_MIN sharpens logits more than any honest
# calibration requires and should be treated as a broken file.
TEMP_MIN = 0.5
TEMP_MAX = 5.0


def clamp_temperature(t, lo: float = TEMP_MIN, hi: float = TEMP_MAX) -> float:
    """Clamp a temperature scalar to [lo, hi], returning 1.0 for non-numeric input."""
    if isinstance(t, bool):
        return 1.0
    try:
        t = float(t)
    except (TypeError, ValueError):
        return 1.0
    if t != t or t in (float("inf"), float("-inf")):
        return 1.0
    return min(hi, max(lo, t))


class _DynamicMultiheadAttention(nn.MultiheadAttention):
    """nn.MultiheadAttention with shape-dynamic ONNX export support.

    The stock module bakes the traced sequence length as a constant during ONNX
    export. This subclass uses chunk/unflatten/flatten so all shapes remain
    symbolic, enabling dynamic-axes export across arbitrary sequence lengths.

    Derived from Laya's _DynamicMultiheadAttention (Apache-2.0).
    """

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    ):
        # Fall back to stock for paths not on the traced route.
        if (
            need_weights
            or self.in_proj_weight is None
            or self.bias_k is not None
            or self.bias_v is not None
            or (attn_mask is not None and attn_mask.dtype != torch.bool)
        ):
            return super().forward(
                query, key, value,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        if self.batch_first:
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        # (T, B, E) → attention on (B, H, T, D)
        if query is key is value:
            q, k, v = (
                part.unflatten(-1, (self.num_heads, self.head_dim)).permute(1, 2, 0, 3)
                for part in F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
            )
        else:
            embed_dim = query.shape[-1]
            wq, wk, wv = self.in_proj_weight.split(embed_dim, dim=0)
            bq, bk, bv = (
                (None, None, None)
                if self.in_proj_bias is None
                else self.in_proj_bias.split(embed_dim, dim=0)
            )
            q, k, v = (
                F.linear(t, w, b).unflatten(-1, (self.num_heads, self.head_dim)).permute(1, 2, 0, 3)
                for t, w, b in ((query, wq, bq), (key, wk, bk), (value, wv, bv))
            )
        mask = None
        if attn_mask is not None:
            mask = ~attn_mask
        if key_padding_mask is not None:
            keep = key_padding_mask[:, None, None, :] == 0
            mask = keep if mask is None else mask & keep
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            is_causal=is_causal and mask is None,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn = attn.permute(2, 0, 1, 3).flatten(-2)
        if self.batch_first:
            attn = attn.transpose(0, 1)
        return self.out_proj(attn), None


class RawitDecisionHead(nn.Module):
    """Typed decision head: 2x Pre-LN Transformer + option scorer + escalation subhead.

    Reads representations exclusively from [OPT_k] marker positions gathered via
    option_marker_positions. This is the core architectural fix over Laya: no lexical
    signal from option text reaches the scoring network directly.
    """

    def __init__(self, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        nhead = max(1, hidden_dim // 64)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN
        )
        # Swap attention module for ONNX-exportable variant
        layer.self_attn = _DynamicMultiheadAttention(
            hidden_dim, nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        # Per-primitive type embedding so the head knows which scoring regime applies
        self.type_emb = nn.Embedding(len(QTYPES), hidden_dim)
        # Option scorer: LN → Linear → GELU → Linear → scalar logit
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # Escalation subhead: reads [CLS] pooled representation + decision features
        # Features: [top1_prob, top1-top2_margin, entropy, k/255]  (4 scalars, from Laya)
        self.escalation_head = nn.Sequential(
            nn.Linear(hidden_dim + 4, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,          # (B, L, D)
        attention_mask: torch.Tensor,          # (B, L)  1=real, 0=pad
        option_marker_positions: torch.Tensor, # (B, K)  token indices of [OPT_k] markers
        option_marker_mask: torch.Tensor,      # (B, K)  True where marker is valid
        qtype: torch.Tensor,                   # (B,)    primitive index
    ):
        # Inject primitive-type signal into every position
        h = hidden_states + self.type_emb(qtype)[:, None, :]

        # Run through Pre-LN Transformer layers
        pad_mask = ~attention_mask.bool()
        for layer in self.transformer.layers:
            h = layer(h, src_key_padding_mask=pad_mask)

        # Gather [OPT_k] marker representations  →  (B, K, D)
        idx = option_marker_positions.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        marker_h = torch.gather(h, 1, idx)

        # Score each option
        logits = self.scorer(marker_h).squeeze(-1).float()          # (B, K)
        logits = logits.masked_fill(~option_marker_mask, -1e4)

        # Build decision features for the escalation subhead
        p = torch.softmax(logits.detach(), dim=-1)
        k = option_marker_mask.sum(-1).clamp(min=2).float()         # (B,)
        entropy = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        if p.size(-1) >= 2:
            top2 = p.topk(2, dim=-1).values
        else:
            top1 = p.topk(1, dim=-1).values
            top2 = torch.cat([top1, torch.zeros_like(top1)], dim=-1)
        feats = torch.stack(
            [top2[:, 0], top2[:, 0] - top2[:, 1], entropy, k / 255.0], dim=-1
        )                                                            # (B, 4)

        # Escalation logit from [CLS] (position 0) + decision features
        cls_h = h[:, 0].float()                                      # (B, D)
        esc_logit = self.escalation_head(
            torch.cat([cls_h, feats], dim=-1)
        ).squeeze(-1)                                                # (B,)

        return logits, esc_logit


class RawitModel(nn.Module):
    """SEA-LION-ModernBERT-300M encoder + RawitDecisionHead.

    Usage:
        config = RawitConfig()
        model = RawitModel(config)

        # For training from a fresh pretrained encoder:
        model = RawitModel.from_pretrained_backbone(config)
    """

    def __init__(self, config: RawitConfig):
        super().__init__()
        self.config = config
        # Encoder is populated by from_pretrained_backbone() or load_state_dict().
        # Constructed empty here so the module tree is consistent.
        self.encoder = None
        self.head = RawitDecisionHead(
            hidden_dim=config.head_hidden_dim,
            num_layers=config.num_head_layers,
            dropout=config.dropout_prob,
        )
        self.register_buffer(
            "temperature", torch.tensor(config.temperature, dtype=torch.float32)
        )

    @classmethod
    def from_pretrained_backbone(
        cls,
        config: RawitConfig,
        revision: Optional[str] = None,
        no_init: bool = False,
    ) -> "RawitModel":
        """Build model loading the SEA-LION encoder from HuggingFace Hub.

        Args:
            config: RawitConfig specifying backbone_name and head hyperparameters.
            revision: Optional git revision for the backbone checkpoint.
            no_init: If True, skip head weight initialisation (use when loading a
                     full Rawit checkpoint immediately after construction).
        """
        from transformers import AutoConfig, AutoModel

        model = cls(config)

        if no_init:
            ecfg = AutoConfig.from_pretrained(config.backbone_name)
            with _no_init_weights():
                enc = AutoModel.from_config(ecfg, attn_implementation="sdpa")
        else:
            kw = {"attn_implementation": "sdpa"}
            if revision:
                kw["revision"] = revision
            enc = AutoModel.from_pretrained(config.backbone_name, **kw)

        model.encoder = enc
        return model

    def forward(
        self,
        input_ids: torch.Tensor,               # (B, L)
        attention_mask: torch.Tensor,           # (B, L)
        option_marker_positions: torch.Tensor,  # (B, K)
        option_marker_mask: torch.Tensor,       # (B, K)
        qtype: torch.Tensor,                    # (B,)
        detach_encoder: bool = False,
    ):
        """Single forward pass returning (option_logits, escalation_logit).

        option_logits:    (B, K) — raw logits before temperature scaling
        escalation_logit: (B,)  — raw logit before sigmoid
        """
        assert self.encoder is not None, (
            "encoder is not initialised; call RawitModel.from_pretrained_backbone() "
            "or load a full checkpoint."
        )
        h = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state                                         # (B, L, D)

        if detach_encoder:
            h = h.detach()

        return self.head(h, attention_mask, option_marker_positions, option_marker_mask, qtype)

    def calibrated_probs(
        self,
        option_logits: torch.Tensor,   # (B, K)
        qtype: torch.Tensor,           # (B,)
        option_marker_mask: torch.Tensor,  # (B, K)
    ) -> torch.Tensor:
        """Apply per-primitive temperature and return calibrated probabilities."""
        tau = self.temperature[qtype][:, None]                      # (B, 1)
        logits = option_logits / tau.clamp(min=TEMP_MIN)
        logits = logits.masked_fill(~option_marker_mask, -1e4)
        return torch.softmax(logits, dim=-1)


def _no_init_weights():
    """Locate no_init_weights across transformers versions."""
    try:
        from transformers.initialization import no_init_weights
    except ImportError:
        from transformers.modeling_utils import no_init_weights
    return no_init_weights()
