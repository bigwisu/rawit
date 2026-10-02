# Rawit — System 1 Decision Engine for Southeast Asian Languages

> *Cabe rawit* is the smallest chilli in the Indonesian kitchen — and the hottest.  
> *Layah* (Javanese: *leyeh*) is the stone mortar base where sambal is ground: the foundation that holds and transforms.  
> Rawit is both: small, fast, grounded in the region, and built to cut through.

**Repository:** https://github.com/bigwisu/rawit  
**Base Checkpoint:** aisingapore/SEA-LION-ModernBERT-300M  
**License:** Apache-2.0  
**Target Domain:** Bahasa Indonesia & Bahasa Melayu (with Southeast Asian code-switching support)  
**System Profile:** Non-autoregressive, single-pass System 1 decision engine — sub-35 ms on commodity GPUs.

---

## The Name

The name carries two layers.

**Rawit** (*cabe rawit*) is the bird's eye chilli — tiny, unassuming, and disproportionately powerful. It is a fixture of Indonesian and Malay kitchens across the archipelago.

**Layah** is the Javanese word for the flat stone base of a *cobek* (mortar) — the surface on which *cabe rawit* is ground into sambal. In Javanese, *layah* also carries the sense of *to crush*, *to reduce to essence*. It is the foundation that does the real work.

Rawit is a conscious twist on *Laya*, the upstream model this project forks and re-architects. Where Laya is the base, Rawit is what it produces: something concentrated, regional, and ready to use. The name is also a quiet nod to the people and kitchens of the archipelago — a model that speaks the language because it was made by someone who grew up in it.

---

## What Rawit Does

Most enterprise AI pipelines route decisions through large autoregressive LLMs — models that generate token by token and are architecturally unsuited to latency-sensitive System 1 tasks. Rawit takes a different position.

Given a **context**, a **question or rubric**, and a declared set of **options**, Rawit outputs:

- A calibrated probability distribution across options in a single forward pass
- A confidence score derived from normalized Shannon entropy
- An escalation flag for cases that exceed a configurable uncertainty threshold

This makes it a drop-in decision primitive for customer support routing, compliance classification, triage workflows, and any enterprise task where the answer is a choice, not a paragraph.

---

## Architecture

```
Input Tokens: [BOS] Context ... [SEP] Rubric / Question ... [OPT_0] Desc 0 [OPT_1] Desc 1 ... [EOS]
                                  │
          ┌───────────────────────┴───────────────────────┐
          │  Backbone: SEA-LION-ModernBERT-300M (Frozen / Tuned)  │
          └───────────────────────┬───────────────────────┘
                                  │ Hidden States (d = 768)
          ┌───────────────────────┴───────────────────────┐
          │             Rawit Typed Decision Head         │
          │   - 2x Pre-LN Transformer Blocks (d = 768)    │
          │   - Option Marker Pooling / Readout Geometry  │
          │   - Epistemic Uncertainty & Escalation Subhead│
          └───────────────┬───────────────────────┬───────┘
                          │                       │
                 Scalar Logits (z_k)       Escalation Logit
                          │                       │
               Softmax / Temp (tau)            Sigmoid
                          │                       │
                 Calibrated P(y=k)           P(escalate)
```

### Encoder Backbone

- **Source:** aisingapore/SEA-LION-ModernBERT-300M
- **Base Licence:** MIT (fully redistributable under Apache-2.0 downstream)
- **Architecture:** 22 Transformer layers, hidden dim $d_{\text{model}} = 768$, 12 attention heads, RoPE, unpadded FlashAttention
- **Tokenizer:** Gemma 3 regional tokenizer, 262,144 vocabulary — eliminates token bloat on Bahasa affixes and morphology
- **Sequence Window:** 8,192 tokens native; defaults to 1,024 for sub-35 ms System 1 gating; up to 4,096 for document triage

### Typed Decision Head (~18M–22M parameters)

1. **Intermediate Transformer Layers:** 2 bidirectional Pre-LN Transformer layers, dropout $\rho = 0.1$
2. **Readout Projection:** Gathers hidden states $\mathbf{h}_{k} \in \mathbb{R}^{768}$ at each `[OPT_k]` marker position
3. **Scoring Network:**  
   $$\hat{z}_{k} = \mathbf{W}_{2} \, \text{GELU}(\mathbf{W}_{1} \, \text{LN}(\mathbf{h}_{k}) + \mathbf{b}_{1}) + b_{2}$$
4. **Escalation Subhead:** Auxiliary linear projection over pooled `[CLS]` representation:  
   $$P(\text{escalate}) = \sigma(\hat{z}_{\text{esc}})$$

### Decision Primitives

- **Choice:** Nominal categorization across $K$ candidates via temperature-scaled softmax
- **Noul:** Binary proposition ($K = 2$), $P(\text{true}) + P(\text{false}) = 1.0$
- **Score:** Bounded ordinal regression over structured rubric levels
- **Calibrated Confidence:**  
  $$\text{Confidence} = 1.0 - \left( -\frac{1}{\ln K} \sum_{k=1}^{K} P(y = k) \ln P(y = k) \right)$$

---

## Structural Fixes Over the Upstream Architecture

| Known Vulnerability | Root Cause | Fix in Rawit |
| :---- | :---- | :---- |
| **Option Polarity Bias** | Scoring literal words (e.g. ya/tidak) at `[MASK]` tokens causes lexical priors to override rubric instructions | **Neutral Marker Injection:** Abstract `[OPT_0]`, `[OPT_1]`, ..., `[OPT_K-1]` tokens injected before each option; head reads exclusively from these invariant positions |
| **Colloquial Bahasa Fragility** | Encoders trained on formal web text fail on Bahasa Gaul, SMS/WhatsApp abbreviations | **Slang-Augmented Corpus:** Training data blends formal Indonesian/Malay with conversational datasets (*bgt*, *klo*, *sdh*, *tlg*, *gmn*) |
| **Overconfidence (ECE > 0.40)** | Raw masked LM logits produce sharp, uncalibrated peaks | **Post-Hoc Temperature Scaling:** Per-primitive temperature $\tau$ fitted on held-out validation via L-BFGS NLL minimization |
| **Option Budget Contention** | 512-token context limit truncates schemas beyond ~20 options | **Dynamic Token Budgeting:** SEA-LION's native 8k window; 512 tokens reserved for schema, supporting up to 60 options |

---

## API

### `POST /v1/systemone`

**Request:**

```json
{
  "context": "Pengguna: Saldo rekening saya terpotong tetapi uang tidak keluar dari ATM BCA kemarin sore.",
  "question": "Klasifikasikan kategori keluhan nasabah untuk routing tiket.",
  "type": "choice",
  "options": [
    {"id": "transaksi_gagal", "label": "Kegagalan Transaksi ATM / Saldo Terpotong"},
    {"id": "kartu_tertelan", "label": "Kartu Tertelan / Rusak"},
    {"id": "informasi_umum", "label": "Pertanyaan Biaya Admin / Layanan Umum"},
    {"id": "indikasi_fraud", "label": "Laporan Penipuan / Pembobolan Akun"}
  ],
  "escalate_threshold": 0.65
}
```

**Response:**

```json
{
  "decision": "transaksi_gagal",
  "confidence": 0.9421,
  "probabilities": {
    "transaksi_gagal": 0.9580,
    "kartu_tertelan": 0.0215,
    "informasi_umum": 0.0120,
    "indikasi_fraud": 0.0085
  },
  "escalate": false,
  "escalation_score": 0.0579,
  "execution_time_ms": 31.4,
  "model": "rawit-300m-v1"
}
```

---

## Repository Structure

```
rawit/
├── LICENSE                          # Apache License 2.0
├── README.md
├── pyproject.toml                   # Poetry/UV build configuration
├── Makefile
├── rawit/
│   ├── __init__.py
│   ├── configuration_rawit.py       # PretrainedConfig implementation
│   ├── modeling_rawit.py            # PyTorch implementation (Backbone + Head)
│   ├── tokenization_rawit.py        # SEA-LION Gemma3 tokenizer wrapper
│   ├── pipeline.py                  # High-level Python inference API
│   ├── calibration.py               # Temperature & Platt scaling routines
│   ├── server/
│   │   ├── __init__.py
│   │   ├── app.py                   # FastAPI application
│   │   └── schemas.py               # Pydantic models for /v1/systemone
│   ├── training/
│   │   ├── dataset.py               # JSONL triplet dataset loader
│   │   ├── losses.py                # Proper scoring rules (Brier, RPS, Soft-CE)
│   │   ├── train_distill.py         # Supervised teacher distillation
│   │   └── train_rlcd.py            # RLCD distributed training loop
│   └── export/
│       ├── export_onnx.py           # Dynamic axes ONNX export
│       └── quantize.py              # INT8 dynamic quantization script
├── tests/
│   ├── test_modeling.py
│   ├── test_polarity_bias.py        # Sun & Xu label reversal audit test
│   ├── test_calibration.py
│   └── test_server.py
└── benchmarks/
    ├── benchmark_latency.py         # Latency/throughput profiler
    └── evaluate_indonesian_nlu.py   # IndoNLU / NusaX evaluation suite
```

---

## Sprint Plan (Two Weeks)

### Phase 1 — Local Scaffolding & Core Re-Architecture (Days 1–3)
*Mac M4, local.*

1. **Fork and prune the Laya chassis** — reuse forward-pass harnesses, loss calculations, length-grouped batching, and server schemas. Remove the dual-model router; Rawit runs a single regional backbone.
2. **Inject the SEA-LION backbone** — wire `aisingapore/SEA-LION-ModernBERT-300M` as the encoder; bind the Gemma 3 regional tokenizer (262k vocab).
3. **Neutralize marker geometry** — refactor the prompt builder to inject abstract `[OPT_k]` markers ahead of option strings; gather representations exclusively from those positions.
4. **MPS smoke test** — verify forward pass on M4 via `torch.device("mps")` with random dummy weights; confirm batching, pooling shapes, and logit output dimensions.

### Phase 2 — Data Assembly & Triplet Generation (Days 4–6)
*Target: 30,000–50,000 structured decision triplets.*

- **60% Bahasa Indonesia** — formal (banking, compliance, e-commerce) + conversational Bahasa Gaul (*bgt*, *klo*, *sdh*, *tlg*, *gmn*)
- **15% Bahasa Melayu** — standard Malaysian business and customer-support terminology
- **25% English** — enterprise workflows and safety guardrails to preserve bilingual stability

**Sources:**
- Convert IndoNLU (SmSA/EmoT) and NusaX benchmark sets into choice triplets
- Synthetic soft targets via SeaLLM-7B / SEA-LION v4.8 / Llama-3.3-70B across ASEAN enterprise workflows (fintech dispute routing, BPJS health classification, logistics dispatch, customer support triage)
- Discard any teacher-generated triplet where output entropy > 0.9 × ln K — those are noise, not signal

### Phase 3 — Cloud Training & Calibration (Day 7)
*Single cloud burst: GCP asia-southeast1/2 L4, or RunPod/AWS RTX 4090 / A10G. ~2–3 hours, under $5.*

1. Train 2-layer Pre-LN decision head + top 4 ModernBERT layers via Soft Cross-Entropy:  
   $$\mathcal{L}_{\text{CE}} = - \frac{1}{B} \sum_{i=1}^{B} \sum_{k=1}^{K} q_{i,k} \ln p_{i,k}$$  
   AdamW, BF16 mixed precision, sequence length 1,024, batch size 32, 3 epochs.
2. Post-hoc calibration: L-BFGS on 1,000 held-out Indonesian decision cases to fit $\tau > 1$, targeting ECE ≤ 0.065.
3. Push checkpoint directly to Hugging Face Hub (`your-org/rawit-300m`), then terminate the instance.

### Phase 4 — Local Runtime, ONNX Export & M4 Profiling (Days 8–10)
*Mac M4, local.*

1. Export with dynamic axes (`batch_size`, `sequence_length`, `num_options`); produce `rawit-int8.onnx`.
2. Benchmark p95 single-query latency across three engines:
   - PyTorch MPS (M4 GPU)
   - ONNX Runtime CPU INT8 (M4 performance cores)
   - CoreML / Neural Engine *(stretch goal — op support not guaranteed; defer to v1.1 if blocked)*
3. Target: p95 latency **10–25 ms** on M4, **≤ 35 ms** on NVIDIA T4 GPU.
4. Spin up FastAPI server locally; confirm `POST /v1/systemone` emits correct confidence, distribution, and escalation payloads.

### Phase 5 — Packaging & Open-Source Launch (Days 11–14)

1. Apply Apache-2.0 headers across codebase; add attribution notices for SEA-LION and Laya upstream concepts.
2. Upload weights, tokenizer configs, and ONNX bundles to Hugging Face Hub.
3. Write model card covering: architecture spec, calibration curves, temperature settings, and evaluation metrics across Indonesian intent and classification suites.
4. Community positioning: **sovereign, sub-35 ms, UU PDP-compliant System 1 decision engine for Southeast Asian languages** — runs locally on commodity hardware or Mac M4 without trans-pacific cloud latency.

---

## Benchmarks & Success Criteria

| Benchmark / Dataset | Target Domain | Metric | Target |
| :---- | :---- | :---- | :---- |
| IndoNLU / NusaX (Sentimen & Intent) | Customer chat & intent classification | Macro F1 | > 92.5% |
| BPJS / Indo-Banking Triage | Enterprise service routing | 10-Class Top-1 Accuracy | > 84.0% |
| Sun & Xu Label Inversion Test | Polarity bias resilience | Decision Inversion Rate | < 8.0% |
| Post-Hoc Calibration ECE | Probability reliability | Expected Calibration Error | ≤ 0.065 |
| Latency Benchmark | Real-time SLA | p95 Latency, batch=1, T4 GPU | ≤ 35 ms |

---

## Acknowledgements

Rawit builds on and is architecturally indebted to:

- **AI Singapore / SEA-LION** — for `SEA-LION-ModernBERT-300M` and the Gemma 3 regional tokenizer (MIT License)
- **Laya** — for the forward-pass harness, training scaffolding, and serving schemas (Apache-2.0)

*The name Rawit honours the kitchens and language of the archipelago.*  
*Layah holds the cobek. Rawit is what it makes.*
