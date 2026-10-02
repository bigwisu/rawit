# Copyright 2025 Wisu Suntoyo — Apache-2.0
"""Proper scoring rule losses for Rawit training.

Three losses are implemented:
    soft_cross_entropy  — Soft-CE against teacher distribution q
    ranked_probability_score (RPS) — Ordinal score primitive
    brier_score         — Quadratic scoring rule

The composite training loss combines Soft-CE + RPS (for score primitives).
"""

import torch
import torch.nn.functional as F

from ..modeling_rawit import QTYPES


def soft_cross_entropy(
    logits: torch.Tensor,      # (B, K)
    target: torch.Tensor,      # (B, K) soft distribution
    mask: torch.Tensor,        # (B, K) True where option is valid
    temperature: float = 1.0,
) -> torch.Tensor:
    """Cross-entropy loss against a soft teacher distribution.

    L_CE = -1/B * sum_i sum_k q_{i,k} * log p_{i,k}
    """
    logits = logits.masked_fill(~mask, -1e4)
    log_p = F.log_softmax(logits / max(temperature, 1e-3), dim=-1)
    loss = -(target * log_p)
    # Only sum over valid option slots
    loss = (loss * mask.float()).sum(dim=-1)
    return loss.mean()


def ranked_probability_score(
    probs: torch.Tensor,   # (B, K) predicted distribution
    target: torch.Tensor,  # (B, K) target distribution (soft or one-hot)
    mask: torch.Tensor,    # (B, K)
) -> torch.Tensor:
    """Ranked Probability Score for ordinal (score) primitives.

    RPS = 1/(K-1) * sum_{m=1}^{K-1} (CDF_p[m] - CDF_q[m])^2
    """
    k = mask.sum(dim=-1).clamp(min=2).float()  # (B,)
    cdf_p = torch.cumsum(probs * mask.float(), dim=-1)
    cdf_t = torch.cumsum(target * mask.float(), dim=-1)
    rps = ((cdf_p - cdf_t) ** 2 * mask.float()).sum(dim=-1) / (k - 1)
    return rps.mean()


def brier_score(
    probs: torch.Tensor,   # (B, K)
    target: torch.Tensor,  # (B, K)
    mask: torch.Tensor,    # (B, K)
) -> torch.Tensor:
    """Brier Score: mean squared error between predicted and target distributions."""
    k = mask.sum(dim=-1).clamp(min=1).float()
    bs = ((probs - target) ** 2 * mask.float()).sum(dim=-1) / k
    return bs.mean()


def decision_loss(
    logits: torch.Tensor,   # (B, K) raw option logits
    target: torch.Tensor,   # (B, K) soft target distribution
    mask: torch.Tensor,     # (B, K)
    qtype: torch.Tensor,    # (B,)
    w_rps: float = 0.5,
) -> torch.Tensor:
    """Composite training loss: Soft-CE + optional RPS for score primitives."""
    probs = F.softmax(logits.masked_fill(~mask, -1e4), dim=-1)
    ce = soft_cross_entropy(logits, target, mask)

    is_score = (qtype == QTYPES["score"]).float()
    if is_score.sum() > 0:
        rps = ranked_probability_score(probs, target, mask)
        return ce + w_rps * (rps * is_score).mean()
    return ce


def escalation_loss(
    esc_logit: torch.Tensor,  # (B,)
    esc_target: torch.Tensor, # (B,) binary 0/1
) -> torch.Tensor:
    """Binary cross-entropy for the escalation subhead."""
    return F.binary_cross_entropy_with_logits(esc_logit, esc_target.float())
