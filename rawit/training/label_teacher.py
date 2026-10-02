# Copyright 2026 Wisu Suntoyo — Apache-2.0
"""Offline blended teacher labelling with IndoBERTweet.

Reads a JSONL file whose records contain 'context', 'rubric', 'options', and
'qtype' but no 'target', and writes a new JSONL file with 'target' filled in
as a blended soft distribution from two teachers:

  - Main teacher (default: aisingapore/SEA-LION-ModernBERT-300M) handles
    formal Bahasa Indonesia / Melayu and general-domain samples.
  - IndoBERTweet teacher (indolem/indobertweet-base-uncased) specialises in
    informal register — Bahasa Gaul, abbreviations (bgt, klo, sdh, tlg, gak,
    aja, yg), SMS/WhatsApp style, and Twitter slang.

For each record the [CLS] hidden state of the main teacher at each option's
[OPT_k] token position is compared to a reference informal-register probe to
produce a colloquiality weight alpha ∈ [0, 1].  The final soft label is:

    target = (1 - alpha) * p_main + alpha * p_tweet

where alpha defaults to a fixed `--tweet_weight` (0.4) unless
`--auto_alpha` is passed, in which case it is estimated per-sample from the
average subword-fertility ratio (IndoBERTweet tokens / main-teacher tokens).
A high fertility ratio signals that the main tokeniser is fragmenting informal
words, meaning IndoBERTweet should receive more weight.

Both teachers run frozen (no gradients).  Options are encoded as plain text
fed to each teacher's own tokeniser without Rawit [OPT_k] markers, since
IndoBERTweet does not know those tokens.  The teachers are used only to
produce per-option likelihood scores via the [CLS] representation similarity
to each option string — a standard zero-shot NLI-style scoring approach.

Usage:
    python -m rawit.training.label_teacher \\
        --input  data/raw_colloquial.jsonl \\
        --output data/train_colloquial_labelled.jsonl \\
        --main_teacher aisingapore/SEA-LION-ModernBERT-300M \\
        --tweet_weight 0.4 \\
        --batch_size 16 \\
        --device cuda
"""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

_log = logging.getLogger("rawit.label_teacher")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

INDOBERTWEET_ID = "indolem/indobertweet-base-uncased"


# ── Teacher helpers ────────────────────────────────────────────────────────────

def _load_teacher(model_id: str, device: torch.device):
    """Load a frozen encoder teacher and its tokeniser."""
    _log.info("Loading teacher: %s", model_id)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


@torch.inference_mode()
def _score_options(
    tok,
    model,
    context: str,
    rubric: str,
    options: List[str],
    device: torch.device,
    max_len: int = 512,
) -> List[float]:
    """Return a normalised probability distribution over options.

    Each option is scored by the cosine similarity between the [CLS]
    representation of the full (context + rubric) prompt and the [CLS]
    representation of that option string alone.  Softmax over cosine sims
    produces a calibration-free soft distribution suitable as a teacher signal.
    """
    # Encode context + rubric once
    prompt = f"{context} {rubric}"
    enc_prompt = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        padding=False,
    ).to(device)
    cls_prompt = model(**enc_prompt).last_hidden_state[:, 0, :]  # (1, H)

    # Encode each option independently
    cls_opts = []
    for opt in options:
        enc_opt = tok(
            opt,
            return_tensors="pt",
            truncation=True,
            max_length=64,
            padding=False,
        ).to(device)
        cls_opt = model(**enc_opt).last_hidden_state[:, 0, :]  # (1, H)
        cls_opts.append(cls_opt)

    # Stack → (K, H), compute cosine sims with prompt cls
    cls_stack = torch.cat(cls_opts, dim=0)  # (K, H)
    sims = F.cosine_similarity(cls_prompt, cls_stack, dim=-1)  # (K,)
    probs = F.softmax(sims, dim=0).cpu().tolist()
    return probs


def _fertility_alpha(
    main_tok,
    tweet_tok,
    text: str,
    base_alpha: float = 0.4,
    lo: float = 0.1,
    hi: float = 0.7,
) -> float:
    """Estimate per-sample IndoBERTweet blend weight from subword fertility.

    Fertility = IndoBERTweet token count / main-teacher token count.
    A ratio < 1 means IndoBERTweet handles the text more efficiently
    (fewer fragments), so we raise alpha toward `hi`.
    A ratio > 1 means the main teacher is more efficient → lower alpha.
    The result is clamped to [lo, hi] and centred around base_alpha.
    """
    n_main = len(main_tok.encode(text, add_special_tokens=False))
    n_tweet = len(tweet_tok.encode(text, add_special_tokens=False))
    if n_main == 0:
        return base_alpha
    ratio = n_tweet / n_main  # < 1 → tweet better → more weight
    # Map ratio ∈ (0, 2] to alpha: ratio=1 → base_alpha, ratio→0 → hi, ratio→2 → lo
    alpha = base_alpha + (1.0 - ratio) * (hi - base_alpha)
    return float(max(lo, min(hi, alpha)))


# ── Main pipeline ──────────────────────────────────────────────────────────────

def label_file(args):
    device = torch.device(args.device or (
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    ))

    main_tok, main_model = _load_teacher(args.main_teacher, device)
    tweet_tok, tweet_model = _load_teacher(INDOBERTWEET_ID, device)

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    _log.info("Labelling %d records from %s", len(records), input_path)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, rec in enumerate(records):
            context = rec["context"]
            rubric = rec.get("rubric", "")
            options = rec["options"]

            p_main = _score_options(
                main_tok, main_model, context, rubric, options, device
            )
            p_tweet = _score_options(
                tweet_tok, tweet_model, context, rubric, options, device
            )

            if args.auto_alpha:
                alpha = _fertility_alpha(main_tok, tweet_tok, context)
            else:
                alpha = args.tweet_weight

            # Blend: (1 - alpha) * p_main + alpha * p_tweet
            target = [
                (1.0 - alpha) * pm + alpha * pt
                for pm, pt in zip(p_main, p_tweet)
            ]
            # Re-normalise for floating-point safety
            s = sum(target)
            target = [t / s for t in target]

            out_rec = dict(rec)
            out_rec["target"] = target
            out_rec["_label_alpha"] = round(alpha, 4)
            out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

            if (i + 1) % 100 == 0:
                _log.info("  %d / %d labelled …", i + 1, len(records))

    _log.info("Done. Labelled records written to %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Offline blended teacher labelling with IndoBERTweet"
    )
    parser.add_argument("--input",  required=True,
                        help="Input JSONL (context/rubric/options/qtype, no target)")
    parser.add_argument("--output", required=True,
                        help="Output JSONL with blended soft 'target' added")
    parser.add_argument("--main_teacher", default="aisingapore/SEA-LION-ModernBERT-300M",
                        help="HuggingFace model id for the main teacher encoder")
    parser.add_argument("--tweet_weight", type=float, default=0.4,
                        help="Fixed IndoBERTweet blend weight alpha ∈ [0, 1]")
    parser.add_argument("--auto_alpha", action="store_true",
                        help="Estimate alpha per-sample from subword fertility ratio")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    label_file(args)


if __name__ == "__main__":
    main()
