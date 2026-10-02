# Copyright 2025 Wisu Suntoyo — Apache-2.0
"""Tests for the FastAPI /v1/systemone server."""

import pytest


@pytest.fixture
def client():
    """Create a test client with a mocked pipeline."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")

    from unittest.mock import MagicMock, patch
    from fastapi.testclient import TestClient
    from rawit.server.app import app
    from rawit.pipeline import DecisionResult

    mock_result = DecisionResult(
        decision="transaksi_gagal",
        confidence=0.94,
        probabilities={"transaksi_gagal": 0.958, "kartu_tertelan": 0.042},
        escalate=False,
        escalation_score=0.06,
        execution_time_ms=28.5,
    )

    mock_pipeline = MagicMock()
    mock_pipeline.decide.return_value = mock_result

    with patch("rawit.server.app._get_pipeline", return_value=mock_pipeline):
        with patch("rawit.server.app._pipeline", mock_pipeline):
            yield TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_systemone_valid_request(client):
    payload = {
        "context": "Saldo rekening saya terpotong tapi uang tidak keluar.",
        "question": "Klasifikasikan kategori keluhan.",
        "type": "choice",
        "options": [
            {"id": "transaksi_gagal", "label": "Kegagalan Transaksi ATM"},
            {"id": "kartu_tertelan",  "label": "Kartu Tertelan / Rusak"},
        ],
        "escalate_threshold": 0.65,
    }
    r = client.post("/v1/systemone", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "decision" in body
    assert "confidence" in body
    assert "probabilities" in body
    assert "escalate" in body
    assert "escalation_score" in body
    assert "execution_time_ms" in body


def test_systemone_missing_options_rejected(client):
    payload = {
        "context": "test",
        "question": "test",
        "type": "choice",
        "options": [{"id": "a", "label": "A"}],  # only 1, min is 2
    }
    r = client.post("/v1/systemone", json=payload)
    assert r.status_code == 422


def test_systemone_missing_questions_rejected(client):
    r = client.post("/v1/systemone", json={"context": "test"})
    assert r.status_code in (400, 422)
