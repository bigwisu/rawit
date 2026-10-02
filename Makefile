.PHONY: install install-dev test lint smoke-test serve export-onnx quantize bench clean \
        train-l4 train-l4-head-only label-teacher bench-nlu-colloquial cuda-check

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
	python -c "\
import torch; \
from rawit.configuration_rawit import RawitConfig; \
from rawit.modeling_rawit import RawitDecisionHead, QTYPES; \
D, L, K, B = 768, 64, 4, 2; \
head = RawitDecisionHead(hidden_dim=D, num_layers=1, dropout=0.0).eval(); \
device = 'cuda' if torch.cuda.is_available() else 'cpu'; \
head = head.to(device); \
h = torch.randn(B, L, D, device=device); \
att = torch.ones(B, L, dtype=torch.long, device=device); \
mpos = torch.randint(0, L, (B, K), device=device); \
mmask = torch.ones(B, K, dtype=torch.bool, device=device); \
qt = torch.zeros(B, dtype=torch.long, device=device); \
logits, esc = head(h, att, mpos, mmask, qt); \
print(f'OK  logits={logits.shape}  esc={esc.shape}  device={device}') \
"

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

# ── GCP / CUDA L4 Training ────────────────────────────────────────────────────
# Prerequisites: VM with NVIDIA L4, CUDA 12.x, rawit installed with [train] extras.
# Run `make cuda-check` first to confirm the GPU is visible.

TRAIN_FILE   ?= data/train.jsonl
VAL_FILE     ?= data/val.jsonl
TRAIN_OUT    ?= checkpoints/rawit-300m
BACKBONE     ?= aisingapore/SEA-LION-ModernBERT-300M

cuda-check:
	python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('GPU:', torch.cuda.get_device_name(0)); print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB'); print('CUDA:', torch.version.cuda)"

# Standard L4 training run: head + top-4 backbone layers, grad checkpointing
# ~30 min for 5,000 samples × 3 epochs
# Note: --compile is omitted by default; add it manually once Triton/GCC
#       linkage is confirmed working (run make cuda-check first).
train-l4:
	python -m rawit.training.train_distill \
		--train_file $(TRAIN_FILE) \
		--val_file   $(VAL_FILE) \
		--output_dir $(TRAIN_OUT) \
		--backbone   $(BACKBONE) \
		--epochs     3 \
		--batch_size 32 \
		--max_len    1024 \
		--lr         2e-5 \
		--unfreeze_backbone_layers 4 \
		--num_workers 4 \
		--grad_checkpoint \
		--device cuda \
		2>&1 | tee training.log

# Head-only run: fastest iteration, no backbone gradient — good for first sanity check
# ~4 min for 5,000 samples × 3 epochs
train-l4-head-only:
	python -m rawit.training.train_distill \
		--train_file $(TRAIN_FILE) \
		--val_file   $(VAL_FILE) \
		--output_dir $(TRAIN_OUT)-head-only \
		--backbone   $(BACKBONE) \
		--epochs     3 \
		--batch_size 64 \
		--max_len    1024 \
		--lr         2e-5 \
		--unfreeze_backbone_layers 0 \
		--num_workers 4 \
		--compile \
		--device cuda \
		2>&1 | tee training-head-only.log

# Offline IndoBERTweet teacher labelling (run once before training colloquial data)
LABEL_INPUT  ?= data/raw_colloquial.jsonl
LABEL_OUTPUT ?= data/train_colloquial_labelled.jsonl

label-teacher:
	python -m rawit.training.label_teacher \
		--input  $(LABEL_INPUT) \
		--output $(LABEL_OUTPUT) \
		--main_teacher $(BACKBONE) \
		--tweet_weight 0.4 \
		--device cuda

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

# Colloquial benchmark: Rawit vs IndoBERTweet ceiling on tweet-sourced SmSA split
bench-nlu-colloquial:
	python benchmarks/evaluate_indonesian_nlu.py \
		--checkpoint ${RAWIT_MODEL:-bigwisu/rawit-300m} \
		--dataset indonlp/indonlu \
		--lang smsa \
		--split test \
		--reference_model indolem/indobertweet-base-uncased

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf dist/ build/ *.egg-info .pytest_cache
