.PHONY: install install-dev test lint smoke-test serve export-onnx quantize bench clean

# ── Installation ──────────────────────────────────────────────────────────────

install:
	pip install -e .

install-dev:
	pip install -e ".[serve,onnx,train,dev]"

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v --tb=short

# Run only the weight-free unit tests (no checkpoint download required)
test-unit:
	pytest tests/test_modeling.py tests/test_calibration.py tests/test_polarity_bias.py -v --tb=short

# ── Smoke test: forward pass on MPS / CPU with random weights ─────────────────

smoke-test:
	python - <<'EOF'
import torch
from rawit.configuration_rawit import RawitConfig
from rawit.modeling_rawit import RawitDecisionHead, QTYPES

D, L, K, B = 768, 64, 4, 2
head = RawitDecisionHead(hidden_dim=D, num_layers=1, dropout=0.0).eval()
device = "mps" if torch.backends.mps.is_available() else "cpu"
head = head.to(device)
h    = torch.randn(B, L, D, device=device)
att  = torch.ones(B, L, dtype=torch.long, device=device)
mpos = torch.randint(0, L, (B, K), device=device)
mmask = torch.ones(B, K, dtype=torch.bool, device=device)
qt   = torch.zeros(B, dtype=torch.long, device=device)
with torch.no_grad():
    logits, esc = head(h, att, mpos, mmask, qt)
print(f"OK  logits={logits.shape}  esc={esc.shape}  device={device}")
EOF

# ── Serving ───────────────────────────────────────────────────────────────────

serve:
	RAWIT_MODEL=${RAWIT_MODEL:-bigwisu/rawit-300m} \
	uvicorn rawit.server.app:app --host 0.0.0.0 --port 8080 --log-level info

# ── Export ────────────────────────────────────────────────────────────────────

CHECKPOINT ?= checkpoints/rawit-300m
ONNX_OUT   ?= rawit.onnx
INT8_OUT   ?= rawit-int8.onnx

export-onnx:
	python -m rawit.export.export_onnx \
		--checkpoint $(CHECKPOINT) \
		--output $(ONNX_OUT)

export-onnx-fp16:
	python -m rawit.export.export_onnx \
		--checkpoint $(CHECKPOINT) \
		--output $(ONNX_OUT) \
		--fp16

quantize:
	python -m rawit.export.quantize \
		--input $(ONNX_OUT) \
		--output $(INT8_OUT)

# ── Benchmarks ────────────────────────────────────────────────────────────────

bench-latency:
	python benchmarks/benchmark_latency.py \
		--checkpoint ${RAWIT_MODEL:-bigwisu/rawit-300m} \
		--n_runs 200

bench-nlu:
	python benchmarks/evaluate_indonesian_nlu.py \
		--checkpoint ${RAWIT_MODEL:-bigwisu/rawit-300m} \
		--dataset indonlp/nusax_senti \
		--lang ind \
		--split test

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf dist/ build/ *.egg-info .pytest_cache
