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

Success criterion: Macro F1 >= 92.5% on NusaX sentiment (Indonesian).
"""

import argparse
import collections
from typing import Dict, List


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


def run(args):
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets to run NLU evaluation.")

    from rawit.pipeline import RawitPipeline

    print(f"Loading model: {args.checkpoint}")
    pipe = RawitPipeline.from_pretrained(args.checkpoint)

    print(f"Loading dataset: {args.dataset}  lang={args.lang}  split={args.split}")
    ds = load_dataset(args.dataset, args.lang, split=args.split)

    # Heuristic: discover label names from the dataset features
    label_col = "label" if "label" in ds.features else "sentiment"
    text_col = "text" if "text" in ds.features else "sentence"
    label_names: List[str] = ds.features[label_col].names

    options = [{"id": str(i), "label": name} for i, name in enumerate(label_names)]
    rubric = f"Classify the sentiment of the following text. Options: {', '.join(label_names)}."

    y_true, y_pred = [], []
    for i, row in enumerate(ds):
        result = pipe.decide(
            context=row[text_col],
            rubric=rubric,
            options=options,
        )
        pred = int(result.decision)
        y_pred.append(pred)
        y_true.append(int(row[label_col]))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(ds)} evaluated …")

    correct = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = correct / len(y_true)
    f1 = _macro_f1(y_true, y_pred, n_classes=len(label_names))

    print(f"\nResults on {args.dataset} [{args.lang}] split={args.split}")
    print(f"  Accuracy : {accuracy * 100:.2f}%")
    print(f"  Macro F1 : {f1 * 100:.2f}%")

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
    args = parser.parse_args()
    ok = run(args)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
