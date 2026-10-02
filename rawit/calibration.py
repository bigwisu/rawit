# Copyright 2025 Wisu Suntoyo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Temperature fitting logic derived from Laya (Apache-2.0):
# https://github.com/convaiinnovations/laya

"""Post-hoc temperature calibration for Rawit logits.

Fits per-primitive temperature scalars (and optionally per-option-count buckets)
by minimising NLL on held-out validation logits using L-BFGS.

Usage:
    from rawit.calibration import fit_temperature_map, ece_score

    # records: list of (qtype_int, logits_1d, target_1d, k_int)
    result = fit_temperature_map(records, compute_ece=True)
    # result["temperature"]         — [float, float, float]  (choice, score, noul)
    # result["temperature_by_options"] — {bucket_str: float}
    # result["report"]["ece_before"], result["report"]["ece_after"]
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .modeling_rawit import QTYPES, QTYPE_NAMES, TEMP_MIN, TEMP_MAX, clamp_temperature

# Per-bucket floor: buckets with fewer examples are skipped; type-level scalar covers them.
MIN_BUCKET_N = 2000
# Lower floor for the type-level scalar so small datasets still get calibrated.
MIN_TYPE_N = 10
# Fraction of each bucket held out when compute_ece=True.
ECE_HOLDOUT_FRAC = 0.2

N_QTYPES = len(QTYPES)

# (qtype, logits_1d, target_1d, k)
Record = Tuple[int, Any, Any, int]


def temp_bucket(qtype: int, k: int) -> str:
    """Return the temperature-bucket key for a given primitive type and option count."""
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


def ece_score(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Expected Calibration Error across equal-width confidence bins."""
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = (conf >= lo if i == 0 else conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


def fit_one_temperature(pairs: Sequence, min_n: Optional[int] = None) -> float:
    """Fit one scalar T by NLL + L-BFGS on log(T).

    Returns clamp_temperature(fitted_T) in [TEMP_MIN, TEMP_MAX], or 1.0 when
    fewer than min_n pairs are provided.
    """
    if min_n is None:
        min_n = MIN_BUCKET_N
    sel = list(pairs)
    if len(sel) < min_n:
        return 1.0

    kmax = max(len(_vec(z)) for z, _ in sel)
    z_mat = torch.full((len(sel), kmax), -1e4)
    t_mat = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        z = _vec(z)
        t = _vec(t)
        n = min(len(z), len(t))
        z_mat[i, :n] = torch.from_numpy(np.ascontiguousarray(z[:n]))
        t_mat[i, :n] = torch.from_numpy(np.ascontiguousarray(t[:n]))

    with torch.enable_grad():
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

        def closure():
            opt.zero_grad()
            loss = -(t_mat * torch.log_softmax(z_mat / log_t.exp(), dim=-1)).sum(-1).mean()
            loss.backward()
            return loss

        opt.step(closure)

    fitted = float(log_t.detach().exp().item())
    return clamp_temperature(fitted, TEMP_MIN, TEMP_MAX)


def fit_temperature_map(
    records: Iterable,
    compute_ece: bool = False,
    seed: int = 0,
) -> Dict[str, Any]:
    """Fit type-level scalars and per-bucket temperatures from validation records.

    Args:
        records: Iterable of (qtype, logits, target[, k]) tuples.
        compute_ece: If True, hold out ECE_HOLDOUT_FRAC of each bucket for evaluation.
        seed: Random seed for the holdout split (ignored when compute_ece=False).

    Returns dict with keys:
        temperature            — [float]*3  (choice, score, noul)
        temperature_by_options — {bucket: float}
        n_by_bucket            — {bucket: int}
        report                 — (only when compute_ece=True) ECE metrics
    """
    recs = _iter_records(records)
    all_by_type, all_buckets = _group_records(recs)
    n_by_bucket = {key: len(pairs) for key, pairs in all_buckets.items()}

    if compute_ece:
        fit_recs, eval_recs, excluded = _ece_split(recs, seed)
        by_type, by_bucket = _group_records(fit_recs)
    else:
        eval_recs = None
        excluded = []
        by_type, by_bucket = all_by_type, all_buckets

    temperature, temperature_by_options = _fit_groups(by_type, by_bucket)
    out: Dict[str, Any] = {
        "temperature": temperature,
        "temperature_by_options": temperature_by_options,
        "n_by_bucket": n_by_bucket,
    }
    if compute_ece:
        report = _ece_report(eval_recs, temperature, temperature_by_options)
        report["n"] = float(len(recs))
        report["n_eval"] = float(len(eval_recs))
        report["buckets_excluded_from_eval"] = excluded
        out["report"] = report
    return out


# ── internals ────────────────────────────────────────────────────────────────

def _vec(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32).reshape(-1)


def _iter_records(records: Iterable) -> List[Record]:
    out = []
    for rec in records:
        if len(rec) == 4:
            qtype, logits, target, k = rec
        elif len(rec) == 3:
            qtype, logits, target = rec
            k = len(_vec(logits))
        else:
            raise ValueError("record must be (qtype, logits, target[, k])")
        logits = _vec(logits)
        target = _vec(target)
        k = int(k)
        if k < 1:
            raise ValueError("k must be >= 1")
        out.append((int(qtype), logits[:k], target[:k], k))
    return out


def _group_records(recs):
    by_type = {qt: [] for qt in range(N_QTYPES)}
    by_bucket: Dict[str, list] = {}
    for qt, z, t, k in recs:
        if qt in by_type:
            by_type[qt].append((z, t))
        key = temp_bucket(qt, k)
        by_bucket.setdefault(key, []).append((z, t))
    return by_type, by_bucket


def _fit_groups(by_type, by_bucket):
    temperature = [1.0] * N_QTYPES
    for qt in range(N_QTYPES):
        pairs = by_type[qt]
        if len(pairs) >= MIN_TYPE_N:
            temperature[qt] = fit_one_temperature(pairs, min_n=MIN_TYPE_N)
    temperature_by_options = {}
    for key, pairs in by_bucket.items():
        if len(pairs) >= MIN_BUCKET_N:
            temperature_by_options[key] = fit_one_temperature(pairs)
    return temperature, temperature_by_options


def _softmax(z, t_scale: float) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / max(1e-3, float(t_scale))
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def _ece_report(recs, temperature, temperature_by_options) -> Dict[str, Any]:
    conf_b, cor_b, conf_a, cor_a = [], [], [], []
    for qt, z, t, k in recs:
        y = int(np.argmax(t[:k]))
        p0 = _softmax(z[:k], 1.0)
        conf_b.append(float(p0.max()))
        cor_b.append(float(int(p0.argmax()) == y))
        t_scale = temperature_by_options.get(temp_bucket(qt, k), temperature[qt])
        p1 = _softmax(z[:k], t_scale)
        conf_a.append(float(p1.max()))
        cor_a.append(float(int(p1.argmax()) == y))
    return {
        "ece_before": ece_score(np.asarray(conf_b), np.asarray(cor_b)),
        "ece_after": ece_score(np.asarray(conf_a), np.asarray(cor_a)),
        "n": float(len(recs)),
    }


def _ece_split(recs, seed):
    by_bucket: Dict[str, list] = {}
    for i, rec in enumerate(recs):
        qt, _z, _t, k = rec
        by_bucket.setdefault(temp_bucket(qt, k), []).append(i)
    rng = np.random.RandomState(seed)
    fit_idx, eval_idx, excluded = [], [], []
    for key in sorted(by_bucket):
        idxs = by_bucket[key]
        n = len(idxs)
        n_eval = int(round(n * ECE_HOLDOUT_FRAC))
        if n - n_eval < MIN_BUCKET_N:
            fit_idx.extend(idxs)
            excluded.append(key)
            continue
        perm = rng.permutation(n)
        for j in perm[:n_eval]:
            eval_idx.append(idxs[int(j)])
        for j in perm[n_eval:]:
            fit_idx.append(idxs[int(j)])
    fit_idx.sort()
    eval_idx.sort()
    return [recs[i] for i in fit_idx], [recs[i] for i in eval_idx], excluded
