# Copyright 2025 Wisu Suntoyo — Apache-2.0
"""Latency and throughput benchmark for Rawit.

Measures p50 / p95 / p99 single-query latency across PyTorch (MPS/CUDA/CPU)
and optionally ONNX Runtime INT8.

Usage:
    python benchmarks/benchmark_latency.py \\
        --checkpoint bigwisu/rawit-300m \\
        --n_runs 200 \\
        --max_len 1024

Target: p95 latency <= 35 ms on NVIDIA T4 GPU.
"""

import argparse
import statistics
import time
from typing import List


def _warmup_and_measure(pipe, context, rubric, options, n_runs: int) -> List[float]:
    # Warmup
    for _ in range(10):
        pipe.decide(context=context, rubric=rubric, options=options)
    # Measure
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        pipe.decide(context=context, rubric=rubric, options=options)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return latencies


def _percentile(data: List[float], p: float) -> float:
    data = sorted(data)
    idx = (len(data) - 1) * p / 100.0
    lo, hi = int(idx), min(int(idx) + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (idx - lo)


def run(args):
    print(f"Loading: {args.checkpoint}")
    from rawit.pipeline import RawitPipeline

    pipe = RawitPipeline.from_pretrained(args.checkpoint, max_len=args.max_len)

    context = (
        "Pengguna: Saldo rekening saya terpotong tetapi uang tidak keluar dari ATM "
        "BCA kemarin sore. Saya sudah mencoba menghubungi call center namun belum ada "
        "respons. Mohon bantuannya."
    )
    rubric = "Klasifikasikan kategori keluhan nasabah untuk routing tiket."
    options = [
        {"id": "transaksi_gagal", "label": "Kegagalan Transaksi ATM / Saldo Terpotong"},
        {"id": "kartu_tertelan",  "label": "Kartu Tertelan / Rusak"},
        {"id": "informasi_umum",  "label": "Pertanyaan Biaya Admin / Layanan Umum"},
        {"id": "indikasi_fraud",  "label": "Laporan Penipuan / Pembobolan Akun"},
    ]

    device = str(pipe.device)
    print(f"Device: {device}  |  n_runs: {args.n_runs}  |  max_len: {args.max_len}")
    latencies = _warmup_and_measure(pipe, context, rubric, options, args.n_runs)

    p50  = _percentile(latencies, 50)
    p95  = _percentile(latencies, 95)
    p99  = _percentile(latencies, 99)
    mean = statistics.mean(latencies)

    print(f"\n{'Metric':<12} {'ms':>8}")
    print("-" * 22)
    print(f"{'mean':<12} {mean:>8.2f}")
    print(f"{'p50':<12} {p50:>8.2f}")
    print(f"{'p95':<12} {p95:>8.2f}")
    print(f"{'p99':<12} {p99:>8.2f}")
    print(f"{'min':<12} {min(latencies):>8.2f}")
    print(f"{'max':<12} {max(latencies):>8.2f}")

    target_ms = 35.0
    status = "✓ PASS" if p95 <= target_ms else "✗ FAIL"
    print(f"\np95 target (<= {target_ms} ms): {status}  ({p95:.2f} ms)")
    return p95 <= target_ms


def main():
    parser = argparse.ArgumentParser(description="Rawit latency benchmark")
    parser.add_argument("--checkpoint", default="bigwisu/rawit-300m")
    parser.add_argument("--n_runs", type=int, default=200)
    parser.add_argument("--max_len", type=int, default=1024)
    args = parser.parse_args()
    ok = run(args)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
