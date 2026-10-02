# Copyright 2025 Wisu Suntoyo — Apache-2.0
"""JSONL triplet dataset loader for Rawit distillation training.

Expected JSONL record format:
    {
        "context": "...",
        "rubric":  "...",
        "options": ["label_0", "label_1", ...],
        "qtype":   "choice" | "score" | "noul",
        "target":  [0.9, 0.05, 0.05]   // soft probability distribution from teacher
    }

Records produced by rawit.training.label_teacher carry an additional
'_label_alpha' field recording the IndoBERTweet blend weight used; it is
ignored during training but useful for corpus-curation audits.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from ..modeling_rawit import QTYPES
from ..tokenization_rawit import encode_decision


class TripletDataset(Dataset):
    """Torch Dataset over JSONL decision triplets.

    Args:
        path: Path to a .jsonl file.
        tokenizer: Loaded Rawit tokenizer.
        max_len: Maximum total sequence length.
        head_max_len: Maximum token budget for the rubric + option section.
    """

    def __init__(
        self,
        path: str,
        tokenizer,
        max_len: int = 1024,
        head_max_len: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.head_max_len = head_max_len
        self.records: List[Dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        rec = self.records[idx]
        input_ids, marker_positions = encode_decision(
            self.tokenizer,
            context=rec["context"],
            rubric=rec["rubric"],
            options=rec["options"],
            max_len=self.max_len,
            head_max_len=self.head_max_len,
        )
        k = len(rec["options"])
        target = rec["target"][:k]
        # Normalise in case teacher soft labels don't sum to 1
        s = sum(target)
        target = [t / s for t in target] if s > 0 else [1.0 / k] * k

        return {
            "input_ids": input_ids,
            "marker_positions": marker_positions,
            "qtype": QTYPES.get(rec.get("qtype", "choice"), 0),
            "target": target,
            "k": k,
        }


# ── Colloquiality curation utility ────────────────────────────────────────────

def colloquial_score(
    text: str,
    tweet_model,
    tweet_tok,
    reference_texts: Optional[List[str]] = None,
    device: str = "cpu",
) -> float:
    """Estimate how informal/colloquial a Bahasa Indonesia text is.

    Uses IndoBERTweet's [CLS] embedding to measure cosine similarity against a
    small set of known-informal reference sentences.  Returns a score ∈ [0, 1]
    where 1.0 is maximally informal.  Intended for **data curation only** — not
    called during training or inference.

    Args:
        text: The candidate text to score.
        tweet_model: Loaded IndoBERTweet AutoModel (frozen, on ``device``).
        tweet_tok: Corresponding IndoBERTweet AutoTokenizer.
        reference_texts: Optional override for the reference informal sentences.
            Defaults to a built-in set of Bahasa Gaul examples.
        device: Torch device string.

    Example::

        from transformers import AutoModel, AutoTokenizer
        tweet_tok   = AutoTokenizer.from_pretrained("indolem/indobertweet-base-uncased")
        tweet_model = AutoModel.from_pretrained("indolem/indobertweet-base-uncased").eval()

        score = colloquial_score("gak tau sih bgt capek bngt", tweet_model, tweet_tok)
        # → close to 1.0

        score = colloquial_score("Saldo rekening Anda telah berhasil dikreditkan.", tweet_model, tweet_tok)
        # → lower value
    """
    import torch
    import torch.nn.functional as F

    _DEFAULT_INFORMAL = [
        "gak tau sih bgt males bngt",        # heavy abbreviation
        "klo mau pergi tlg kasih tau ya",    # klo / tlg
        "udh makan blm? laper nih",          # udh / blm
        "yg bener aja deh sdh keterlaluan",  # yg / sdh
        "iya dong aku jg mau ikut aja",      # aja / jg
    ]
    refs = reference_texts if reference_texts else _DEFAULT_INFORMAL

    dev = torch.device(device)

    def _cls(t: str):
        enc = tweet_tok(
            t, return_tensors="pt", truncation=True, max_length=128, padding=False
        )
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.inference_mode():
            return tweet_model(**enc).last_hidden_state[:, 0, :]  # (1, H)

    cls_text = _cls(text)
    sims = [
        float(F.cosine_similarity(cls_text, _cls(r), dim=-1).item())
        for r in refs
    ]
    # Normalise from [-1, 1] cosine range to [0, 1]
    raw = sum(sims) / len(sims)
    return float(max(0.0, min(1.0, (raw + 1.0) / 2.0)))


def collate_fn(batch: List[Dict], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    """Pad a list of dataset items into a batch of tensors."""
    max_len = max(len(item["input_ids"]) for item in batch)
    max_k = max(item["k"] for item in batch)
    B = len(batch)

    ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    att = torch.zeros((B, max_len), dtype=torch.long)
    mpos = torch.zeros((B, max_k), dtype=torch.long)
    mmask = torch.zeros((B, max_k), dtype=torch.bool)
    target = torch.zeros((B, max_k), dtype=torch.float32)
    qtypes = torch.zeros(B, dtype=torch.long)

    for i, item in enumerate(batch):
        L = len(item["input_ids"])
        ids[i, :L] = torch.tensor(item["input_ids"])
        att[i, :L] = 1
        k = item["k"]
        for j, pos in enumerate(item["marker_positions"][:k]):
            mpos[i, j] = pos
            mmask[i, j] = True
        target[i, :k] = torch.tensor(item["target"][:k], dtype=torch.float32)
        qtypes[i] = item["qtype"]

    return {
        "input_ids": ids,
        "attention_mask": att,
        "option_marker_positions": mpos,
        "option_marker_mask": mmask,
        "target": target,
        "qtype": qtypes,
    }
