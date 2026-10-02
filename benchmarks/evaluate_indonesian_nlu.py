# Copyright 2025 Wisu Suntoyo — Apache-2.0
"""IndoNLU / NusaX evaluation suite for Rawit.

Evaluates accuracy and macro-F1 on Indonesian NLU benchmarks using the
Rawit pipeline as a zero-shot or few-shot classifier.

Supported datasets (via HuggingFace datasets):
    - indonlp/indonlu  (SmSA sentiment, EmoT emotion)
    - indonlp/nusax_senti  (NusaX multilingual sentiment)

Usage:
    python benchmarks/evaluate_indonesian_nlu.py \\
        --checkpoint bigwisu/rawit-300m \\
        --dataset indonlp/nusax_senti \\
        --lang ind \\
        --split test

    # With IndoBERTweet ceiling baseline:
    python benchmarks/evaluate_indonesian_nlu.py \\
        --checkpoint bigwisu/rawit-300m \\
        --dataset indonlp/indonlu \\
        --lang smsa \\
        --split test \\
        --reference_model indolem/indobertweet-base-uncased

Success criterion: Macro F1 >= 92.5% on NusaX sentiment (Indonesian).
"""

import argparse
import collections
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def _macro_f1(y_true: List[int], y_pred: List[int], n_classes: int) -> float:
    tp = collections.Counter()
    fp = collections.Counter()
    fn = collections.Counter()
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1
    f1s = []
    for c in range(n_classes):
        precision = tp[c] / max(1, tp[c] + fp[c])
        recall = tp[c] / max(1, tp[c] + fn[c])
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


# ── IndoBERTweet reference classifier ─────────────────────────────────────────

@torch.inference_mode()
def _indobertweet_predict(
    model,
    tok,
    context: str,
    rubric: str,
    label_names: List[str],
    device: torch.device,
) -> int:
    """Zero-shot classify via IndoBERTweet [CLS] cosine similarity.

    Scores each label by the cosine similarity between the [CLS] embedding of
    (context + rubric) and the [CLS] embedding of the label name alone.
    Returns the index of the highest-scoring label.
    """
    prompt = f"{context} {rubric}"
    enc_p = tok(prompt, return_tensors="pt", truncation=True,
                max_length=512, padding=False).to(device)
    cls_p = model(**enc_p).last_hidden_state[:, 0, :]  # (1, H)

    sims = []
    for name in label_names:
        enc_l = tok(name, return_tensors="pt", truncation=True,
                    max_length=32, padding=False).to(device)
        cls_l = model(**enc_l).last_hidden_state[:, 0, :]  # (1, H)
        sims.append(float(F.cosine_similarity(cls_p, cls_l, dim=-1).item()))

    return int(sims.index(max(sims)))


def _load_reference_model(model_id: str, device: torch.device):
    """Load a frozen reference encoder and its tokeniser."""
    print(f"Loading reference model: {model_id}")
    ref_tok = AutoTokenizer.from_pretrained(model_id)
    ref_model = AutoModel.from_pretrained(model_id).eval().to(device)
    for p in ref_model.parameters():
        p.requires_grad_(False)
    return ref_tok, ref_model


# ── Main evaluation loop ───────────────────────────────────────────────────────

def run(args):
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets to run NLU evaluation.")

    from rawit.pipeline import RawitPipeline

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"Loading model: {args.checkpoint}")
    pipe = RawitPipeline.from_pretrained(args.checkpoint)

    # Optionally load the IndoBERTweet ceiling baseline
    ref_model, ref_tok = None, None
    if args.reference_model:
        ref_tok, ref_model = _load_reference_model(args.reference_model, device)

    print(f"Loading dataset: {args.dataset}  lang={args.lang}  split={args.split}")
    ds = load_dataset(args.dataset, args.lang, split=args.split)

    # Heuristic: discover label names from the dataset features
    label_col = "label" if "label" in ds.features else "sentiment"
    text_col = "text" if "text" in ds.features else "sentence"
    label_names: List[str] = ds.features[label_col].names

    options = [{"id": str(i), "label": name} for i, name in enumerate(label_names)]
    rubric = f"Classify the sentiment of the following text. Options: {', '.join(label_names)}."

    y_true, y_pred, y_ref = [], [], []
    for i, row in enumerate(ds):
        result = pipe.decide(
            context=row[text_col],
            rubric=rubric,
            options=options,
        )
        y_pred.append(int(result.decision))
        y_true.append(int(row[label_col]))

        if ref_model is not None:
            y_ref.append(_indobertweet_predict(
                ref_model, ref_tok,
                context=row[text_col],
                rubric=rubric,
                label_names=label_names,
                device=device,
            ))

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(ds)} evaluated …")

    correct = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = correct / len(y_true)
    f1 = _macro_f1(y_true, y_pred, n_classes=len(label_names))

    print(f"\nResults on {args.dataset} [{args.lang}] split={args.split}")
    print(f"  Rawit — Accuracy : {accuracy * 100:.2f}%")
    print(f"  Rawit — Macro F1 : {f1 * 100:.2f}%")

    if y_ref:
        ref_correct = sum(t == p for t, p in zip(y_true, y_ref))
        ref_acc = ref_correct / len(y_true)
        ref_f1 = _macro_f1(y_true, y_ref, n_classes=len(label_names))
        print(f"\n  {args.reference_model} (ceiling baseline)")
        print(f"  Reference — Accuracy : {ref_acc * 100:.2f}%")
        print(f"  Reference — Macro F1 : {ref_f1 * 100:.2f}%")
        gap = (ref_f1 - f1) * 100
        sign = "+" if gap >= 0 else ""
        print(f"  Colloquial gap (reference − Rawit): {sign}{gap:.2f} pp F1")

    target_f1 = 0.925
    status = "✓ PASS" if f1 >= target_f1 else "✗ FAIL"
    print(f"\nMacro F1 target (>= {target_f1 * 100:.1f}%): {status}  ({f1 * 100:.2f}%)")
    return f1 >= target_f1


def main():
    parser = argparse.ArgumentParser(description="IndoNLU / NusaX evaluation")
    parser.add_argument("--checkpoint", default="bigwisu/rawit-300m")
    parser.add_argument("--dataset", default="indonlp/nusax_senti")
    parser.add_argument("--lang", default="ind")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--reference_model", default=None,
        help=(
            "Optional HuggingFace model id to run as a ceiling baseline alongside "
            "Rawit.  Use 'indolem/indobertweet-base-uncased' to measure the "
            "colloquial Bahasa gap."
        ),
    )
    args = parser.parse_args()
    ok = run(args)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
