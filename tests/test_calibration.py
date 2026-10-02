# Copyright 2025 Wisu Suntoyo — Apache-2.0
"""Tests for temperature calibration logic."""

import numpy as np
import pytest

from rawit.calibration import (
    ece_score,
    fit_one_temperature,
    fit_temperature_map,
    temp_bucket,
    MIN_TYPE_N,
)
from rawit.modeling_rawit import QTYPES


def _make_records(n, qtype=0, k=4, temperature=2.0):
    """Generate synthetic records where logits are scaled by `temperature`."""
    rng = np.random.RandomState(42)
    records = []
    for _ in range(n):
        logits = rng.randn(k).astype(np.float32)
        # Soft target: softmax of logits / temperature (the "true" distribution)
        z = logits / temperature
        z -= z.max()
        p = np.exp(z)
        target = p / p.sum()
        records.append((qtype, logits, target, k))
    return records


def test_ece_score_perfect():
    """Perfectly calibrated model should have ECE close to 0."""
    conf = np.linspace(0.0, 1.0, 100)
    correct = (np.random.RandomState(0).rand(100) < conf).astype(float)
    ece = ece_score(conf, correct)
    assert ece < 0.15, f"ECE {ece:.3f} too high for roughly calibrated model"


def test_ece_score_empty():
    assert np.isnan(ece_score(np.array([]), np.array([])))


def test_temp_bucket():
    assert temp_bucket(QTYPES["choice"], 2) == "choice:2"
    assert temp_bucket(QTYPES["choice"], 5) == "choice:3-5"
    assert temp_bucket(QTYPES["choice"], 10) == "choice:6-10"
    assert temp_bucket(QTYPES["choice"], 11) == "choice:11+"
    assert temp_bucket(QTYPES["noul"], 2) == "noul:2"


def test_fit_one_temperature_recovers_scale():
    """Temperature fitting should approximately recover the generating temperature."""
    records = _make_records(n=5000, qtype=QTYPES["choice"], k=4, temperature=2.0)
    pairs = [(z, t) for _, z, t, _ in records]
    fitted = fit_one_temperature(pairs, min_n=100)
    assert 1.5 <= fitted <= 2.5, f"Expected ~2.0, got {fitted:.3f}"


def test_fit_one_temperature_returns_one_on_small_n():
    pairs = [(np.array([1.0, -1.0]), np.array([0.7, 0.3]))]
    assert fit_one_temperature(pairs, min_n=10) == 1.0


def test_fit_temperature_map_structure():
    records = _make_records(n=500, qtype=QTYPES["choice"], k=4, temperature=1.5)
    result = fit_temperature_map(records, compute_ece=False)
    assert "temperature" in result
    assert len(result["temperature"]) == 3
    assert "temperature_by_options" in result
    assert "n_by_bucket" in result


def test_fit_temperature_map_with_ece():
    """Verify that fit_temperature_map returns the expected report keys when compute_ece=True.

    Note: ECE improvement is only guaranteed on real model logits, not synthetic random
    data. This test checks structure and that fitting completes without error.
    """
    records = _make_records(n=5000, qtype=QTYPES["choice"], k=4, temperature=2.0)
    result = fit_temperature_map(records, compute_ece=True, seed=0)
    assert "report" in result
    report = result["report"]
    assert "ece_before" in report
    assert "ece_after" in report
    assert "n" in report
    assert "n_eval" in report
    assert report["n"] == 5000.0
    assert report["n_eval"] > 0
