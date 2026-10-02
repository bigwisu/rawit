# Copyright 2026 Wisu Suntoyo — Apache-2.0
"""Sun & Xu polarity-bias audit test.

Verifies that swapping option labels (e.g. "ya" ↔ "tidak") does not flip
the model's decision more than 8% of the time, validating the neutral
[OPT_k] marker fix against lexical polarity bias.

Reference: Sun & Xu (2023) — label reversal / polarity inversion evaluation.
"""

import pytest


def test_polarity_inversion_rate_below_threshold():
    """
    Placeholder: load a checkpoint, run inference on a polar test set, measure
    the fraction of cases where swapping "ya"/"tidak" (or "yes"/"no") labels
    flips the predicted decision.

    Success criterion: inversion_rate < 0.08  (< 8%).

    TODO: implement once a trained checkpoint is available.
    """
    pytest.skip("Requires a trained checkpoint — implement in Phase 3.")


def test_neutral_marker_tokens_are_registered(tmp_path):
    """[OPT_0] .. [OPT_31] and [RUBRIC] must be registered as special tokens."""
    from rawit.tokenization_rawit import load_tokenizer, option_tokens, RUBRIC_TOKEN, MAX_OPTIONS

    # Use a lightweight backbone available on CI without downloading SEA-LION
    try:
        tok = load_tokenizer("bert-base-uncased")
    except Exception:
        pytest.skip("bert-base-uncased not available in this environment.")

    vocab = tok.get_vocab()
    for t in option_tokens(MAX_OPTIONS) + [RUBRIC_TOKEN]:
        assert t in vocab, f"Expected special token {t!r} in vocabulary"


def test_marker_positions_are_distinct(tmp_path):
    """Each [OPT_k] token must occupy a distinct position in the encoded sequence."""
    from rawit.tokenization_rawit import load_tokenizer, encode_decision

    try:
        tok = load_tokenizer("bert-base-uncased")
    except Exception:
        pytest.skip("bert-base-uncased not available in this environment.")

    input_ids, markers = encode_decision(
        tok,
        context="Pelanggan mengeluh saldo terpotong.",
        rubric="Pilih kategori keluhan.",
        options=["Transaksi gagal", "Kartu tertelan", "Pertanyaan umum"],
        max_len=256,
    )

    assert len(markers) == 3, "Should return one marker position per option."
    assert len(set(markers)) == len(markers), "Marker positions must be distinct."
    # Each marker must be a valid index into input_ids
    for pos in markers:
        assert 0 <= pos < len(input_ids), f"Marker position {pos} out of bounds."
