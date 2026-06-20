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
    ax.plot(s, lat, "o-", color="tab:green")
    ax.set_xlabel("sequence length")
    ax.set_ylabel("latency (ms)")
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
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(n, speedup, "o-", color="tab:purple", label="speedup")
    ax1.plot(n, n, "k:", alpha=0.5, label="ideal")
    ax1.set_xlabel("# workers (sharing one GPU)")
    ax1.set_ylabel("speedup vs 1 worker")
    ax1.set_xticks(n)
    ax1.legend(loc="upper left")
    ax1.set_title("Worker scaling (dispatcher, batch=8 S=256)")
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
    r = rows[-1]
    parts = [("queue", r["queue_mean_ms"]),
             ("GPU compute", r["compute_mean_ms"]),
             ("serialize+transport+H2D/D2H", r["other_mean_ms"])]
    fig, ax = plt.subplots(figsize=(5, 5))
    bottom = 0.0
    colors = ["tab:red", "tab:blue", "tab:gray"]
    for (label, val), c in zip(parts, colors):
        ax.bar(["e2e latency"], [val], bottom=[bottom], label=f"{label} ({val:.1f}ms)", color=c)
        bottom += val
    ax.set_ylabel("latency (ms)")
    ax.set_title(f"Latency breakdown (batch=8 S=256, {int(r['req_bytes'])}B)")
    ax.legend()
    save(fig, "breakdown.png")


def plot_roofline():
    rows = load("roofline.csv")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in rows:
        ax.scatter(r["dram_pct"], r["sm_pct"], s=80)
        ax.annotate(r["kernel"], (r["dram_pct"], r["sm_pct"]),
                    textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("DRAM throughput (% of peak)")
    ax.set_ylabel("SM/compute throughput (% of peak)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.plot([0, 100], [0, 100], "k:", alpha=0.3)
    ax.set_title("Kernel roofline position (ncu)")
    save(fig, "roofline.png")


if __name__ == "__main__":
    plot_batch()
    plot_seq()
    plot_workers()
    plot_blocksize()
    plot_breakdown()
    plot_roofline()
    print("[plot] done")
