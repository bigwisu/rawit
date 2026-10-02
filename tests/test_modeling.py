# Copyright 2026 Wisu Suntoyo — Apache-2.0
"""Tests for RawitModel forward pass, logit shapes, and marker geometry."""

import torch
import pytest

from rawit.configuration_rawit import RawitConfig
from rawit.modeling_rawit import RawitModel, QTYPES


@pytest.fixture
def tiny_config():
    """A minimal config using a tiny encoder stub for fast unit tests."""
    return RawitConfig(
        backbone_name="bert-base-uncased",  # swap for SEA-LION in integration tests
        num_head_layers=1,
        head_hidden_dim=768,
        dropout_prob=0.0,
    )


def test_config_defaults():
    cfg = RawitConfig()
    assert cfg.backbone_name == "aisingapore/SEA-LION-ModernBERT-300M"
    assert cfg.num_head_layers == 2
    assert cfg.head_hidden_dim == 768
    assert cfg.temperature == [1.0, 1.0, 1.0]
    assert cfg.max_options == 32


def test_decision_head_output_shape():
    """Decision head produces (B, K) logits and (B,) escalation logit."""
    from rawit.modeling_rawit import RawitDecisionHead
    B, L, K, D = 2, 32, 4, 768
    head = RawitDecisionHead(hidden_dim=D, num_layers=1, dropout=0.0)
    head.eval()

    h = torch.randn(B, L, D)
    att = torch.ones(B, L, dtype=torch.long)
    mpos = torch.randint(0, L, (B, K))
    mmask = torch.ones(B, K, dtype=torch.bool)
    qtype = torch.zeros(B, dtype=torch.long)

    with torch.no_grad():
        logits, esc = head(h, att, mpos, mmask, qtype)

    assert logits.shape == (B, K), f"expected ({B}, {K}), got {logits.shape}"
    assert esc.shape == (B,), f"expected ({B},), got {esc.shape}"


def test_masked_options_fill_neg_inf():
    """Masked option positions should receive -1e4 fill, not contribute to softmax."""
    from rawit.modeling_rawit import RawitDecisionHead
    B, L, K, D = 1, 32, 4, 768
    head = RawitDecisionHead(hidden_dim=D, num_layers=1, dropout=0.0)
    head.eval()

    h = torch.randn(B, L, D)
    att = torch.ones(B, L, dtype=torch.long)
    mpos = torch.zeros(B, K, dtype=torch.long)
    # Only first 2 options are valid
    mmask = torch.tensor([[True, True, False, False]])
    qtype = torch.zeros(B, dtype=torch.long)

    with torch.no_grad():
        logits, _ = head(h, att, mpos, mmask, qtype)

    assert logits[0, 2].item() < -999, "masked option should be near -1e4"
    assert logits[0, 3].item() < -999, "masked option should be near -1e4"


def test_dynamic_option_count():
    """Forward pass should work for any K from 2 to MAX_OPTIONS."""
    from rawit.modeling_rawit import RawitDecisionHead
    D, L = 768, 32
    head = RawitDecisionHead(hidden_dim=D, num_layers=1, dropout=0.0)
    head.eval()
    for K in [2, 4, 8, 16, 32]:
        h = torch.randn(1, L, D)
        att = torch.ones(1, L, dtype=torch.long)
        mpos = torch.randint(0, L, (1, K))
        mmask = torch.ones(1, K, dtype=torch.bool)
        qtype = torch.zeros(1, dtype=torch.long)
        with torch.no_grad():
            logits, _ = head(h, att, mpos, mmask, qtype)
        assert logits.shape == (1, K)
