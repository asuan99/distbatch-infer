#!/usr/bin/env python3
"""Phase 5: CSV -> matplotlib graphs (plotting only, never on the serving path).

Reads results/*.csv produced by run_experiments.sh and writes results/*.png.
Each plot is independent; a missing CSV is skipped with a note.
"""
import csv
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = sys.argv[1] if len(sys.argv) > 1 else "results"


def load(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        print(f"[plot] skip {name} (not found)")
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    # numeric coercion where possible
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (ValueError, TypeError):
                pass
    return rows


def tag_num(tag):
    m = re.search(r"(\d+)", tag)
    return int(m.group(1)) if m else 0


def save(fig, name):
    p = os.path.join(RES, name)
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {p}")


def plot_batch():
    rows = load("batch_scaling.csv")
    if not rows:
        return
    rows.sort(key=lambda r: r["batch"])
    b = [r["batch"] for r in rows]
    rps = [r["throughput_rps"] for r in rows]
    sps = [r["throughput_rps"] * r["batch"] for r in rows]  # samples/s
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(b, sps, "o-", color="tab:blue", label="samples/s")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("batch size")
    ax1.set_ylabel("throughput (samples/s)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(b, rps, "s--", color="tab:red", label="requests/s")
    ax2.set_ylabel("throughput (requests/s)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_title("GPU batch scaling (1 worker, S=32)")
    save(fig, "batch_scaling.png")


def plot_seq():
    rows = load("seqlen_scaling.csv")
    if not rows:
        return
    rows.sort(key=lambda r: r["seq_len"])
    s = [r["seq_len"] for r in rows]
    lat = [r["lat_mean_ms"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(s, lat, "o-", color="tab:green", label="measured")
    # O(S^2) reference anchored at the first point
    guide = [lat[0] * (x / s[0]) ** 2 for x in s]
    ax.plot(s, guide, "k--", alpha=0.4, label="O(S²) reference")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length")
    ax.set_ylabel("latency (ms)")
    ax.legend()
    ax.set_title("Sequence length scaling (1 worker, batch=1)")
    save(fig, "seqlen_scaling.png")


def plot_workers():
    rows = load("worker_scaling.csv")
    if not rows:
        return
    rows.sort(key=lambda r: tag_num(r["tag"]))
    n = [tag_num(r["tag"]) for r in rows]
    tp = [r["throughput_rps"] for r in rows]
    base = tp[0] if tp else 1.0
    speedup = [t / base for t in tp]
    x = list(range(len(n)))

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.bar(x, tp, color="tab:purple", alpha=0.65, label="throughput")
    ax1.set_xticks(x)
    ax1.set_xticklabels(n)
    ax1.set_xlabel("# workers (all sharing one physical GPU)")
    ax1.set_ylabel("throughput (req/s)", color="tab:purple")
    ax1.tick_params(axis="y", labelcolor="tab:purple")
    ax1.set_ylim(0, max(tp) * 1.18)
    for xi, t in zip(x, tp):
        ax1.text(xi, t + max(tp) * 0.02, f"{t:.0f}", ha="center", fontsize=8,
                 color="tab:purple")

    ax2 = ax1.twinx()
    ax2.plot(x, speedup, "o-", color="tab:red", lw=2, label="speedup")
    ax2.axhline(1.0, color="gray", ls=":", alpha=0.6)
    ax2.set_ylabel("speedup vs 1 worker", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(0.9, max(speedup) * 1.25)
    for xi, s in zip(x, speedup):
        ax2.annotate(f"{s:.2f}×", (xi, s), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=8, color="tab:red")

    ax1.set_title("Worker scaling — throughput peaks at 2–3 workers, then declines\n"
                  "(all workers time-share one GPU; ideal would be N×, off-chart)")
    save(fig, "worker_scaling.png")


def plot_blocksize():
    rows = load("blocksize.csv")
    if not rows:
        return
    rows.sort(key=lambda r: tag_num(r["tag"]))
    t = [tag_num(r["tag"]) for r in rows]
    lat = [r["lat_mean_ms"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(x) for x in t], lat, color="tab:orange")
    ax.set_xlabel("GEMM_TILE")
    ax.set_ylabel("latency (ms)")
    ax.set_title("Block size sweep (batch=16 S=128)")
    save(fig, "blocksize.png")


def plot_breakdown():
    rows = load("breakdown.csv")
    if not rows:
        return
    rows.sort(key=lambda r: r["req_bytes"])
    labels = [f'{r["tag"]}\n{int(r["req_bytes"])//1024}KB' for r in rows]
    q = [r["queue_mean_ms"] for r in rows]
    c = [r["compute_mean_ms"] for r in rows]
    o = [r["other_mean_ms"] for r in rows]
    x = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, q, label="queue", color="tab:red")
    ax.bar(x, c, bottom=q, label="GPU compute", color="tab:blue")
    bottom2 = [qi + ci for qi, ci in zip(q, c)]
    ax.bar(x, o, bottom=bottom2, label="serialize+transport+H2D/D2H", color="tab:gray")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("request config (batch/seq, payload)")
    ax.set_ylabel("latency (ms)")
    ax.set_title("Per-request latency breakdown vs payload (concurrency=1)\n"
                 "queue floor = 5ms batching window; 'other' = serialize+transport+H2D/D2H")
    ax.legend()
    save(fig, "breakdown.png")


def plot_roofline():
    rows = load("roofline.csv")
    if not rows:
        return
    # RTX 5060 Ti (Blackwell GB206) peaks, FP32:
    #   compute = 2 * 4608 cores * 3.09 GHz  (36 SM x 128 FP32/SM, max clock)
    #   bandwidth = 2 * 14.001 GHz * 16 B    (128-bit GDDR7, ~28 Gbps)
    PEAK_GFLOPS = 2 * 4608 * 3.09           # ~28476 GFLOP/s
    PEAK_BW = 2 * 14.001 * 16               # ~448 GB/s
    ridge = PEAK_GFLOPS / PEAK_BW           # FLOP/byte where roofs meet

    import numpy as np
    fig, ax = plt.subplots(figsize=(7, 5))
    ai = np.logspace(-2, 3, 200)
    roof = np.minimum(PEAK_BW * ai, PEAK_GFLOPS)
    ax.plot(ai, roof, "k-", lw=2, label="roofline")
    ax.axhline(PEAK_GFLOPS, color="gray", ls="--", alpha=0.6,
               label=f"FP32 peak {PEAK_GFLOPS/1000:.1f} TFLOP/s")
    ax.plot(ai, PEAK_BW * ai, color="tab:gray", ls=":", alpha=0.6,
            label=f"DRAM peak {PEAK_BW:.0f} GB/s")
    ax.axvline(ridge, color="k", ls=":", alpha=0.3)
    ax.text(ridge * 1.1, PEAK_GFLOPS * 0.25, f"ridge\n{ridge:.0f} FLOP/B",
            fontsize=8, alpha=0.7)

    colors = {"gemm": "tab:blue", "gelu": "tab:orange", "softmax": "tab:green"}
    for r in rows:
        x, y = r["ai_flop_per_byte"], r["gflops"]
        ax.scatter(x, y, s=110, color=colors.get(r["kernel"], "tab:red"), zorder=5)
        ax.annotate(f'{r["kernel"]}\n{y:.0f} GFLOP/s',
                    (x, y), textcoords="offset points", xytext=(8, -4), fontsize=9)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("performance (GFLOP/s)")
    ax.set_ylim(top=PEAK_GFLOPS * 2)
    ax.set_title("Roofline — RTX 5060 Ti (FP32, measured by ncu)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, which="both", alpha=0.2)
    save(fig, "roofline.png")


if __name__ == "__main__":
    plot_batch()
    plot_seq()
    plot_workers()
    plot_blocksize()
    plot_breakdown()
    plot_roofline()
    print("[plot] done")
