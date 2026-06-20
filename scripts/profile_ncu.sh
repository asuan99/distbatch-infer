#!/usr/bin/env bash
# ncu roofline data for the three kernels -> results/roofline.csv
#
# Measured-roofline methodology (verified on Blackwell sm_120, ncu 2025.2):
#   FP32 FLOPs   = fadd + fmul + 2*ffma  (sm__sass_thread_inst_executed_op_*)
#   DRAM bytes   = dram__bytes.sum
#   duration     = gpu__time_duration.sum (ns)
#   => arithmetic intensity = FLOPs / DRAM bytes,  achieved = FLOPs / time
# If a future ncu renames these, run `ncu --query-metrics` and update below.
set -euo pipefail
cd "$(dirname "$0")/.."

export BUILD="${BUILD:-build}"
mkdir -p results
cmake --build "$BUILD" -j4 --target bench_kernels >/dev/null

if ! command -v ncu >/dev/null; then
  echo "[profile_ncu] ncu not found; skipping (roofline.csv not generated)" >&2
  exit 0
fi

python3 - <<'PY'
import csv, io, os, subprocess
M = {
  "fadd": "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
  "fmul": "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
  "ffma": "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
  "bytes": "dram__bytes.sum",
  "time": "gpu__time_duration.sum",
}
build = os.environ.get("BUILD", "build")
out = subprocess.run(
    ["ncu","--csv","--metrics",",".join(M.values()),
     "--kernel-name","regex:gemm_tiled_kernel|fused_bias_gelu_kernel|softmax_kernel",
     "--launch-count","3", f"{build}/bench_kernels"],
    capture_output=True, text=True)
lines = out.stdout.splitlines()
start = next((i for i,l in enumerate(lines) if l.startswith('"ID"')), None)
if start is None:
    print("[profile_ncu] ncu produced no CSV (perf-counter permission? try sudo).")
    print(out.stderr[-400:]); raise SystemExit(0)
rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
friendly = {"gemm_tiled_kernel":"gemm","fused_bias_gelu_kernel":"gelu","softmax_kernel":"softmax"}
name2metric = {v:k for k,v in M.items()}
agg = {}
for r in rows:
    k = r.get("Kernel Name","")
    name = next((f for key,f in friendly.items() if key in k), None)
    if not name: continue
    metric = name2metric.get(r.get("Metric Name",""))
    if not metric: continue
    try: val = float(str(r.get("Metric Value","")).replace(",",""))
    except ValueError: continue
    agg.setdefault(name, {})[metric] = val

with open("results/roofline.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["kernel","time_ms","flops","dram_bytes","gflops","ai_flop_per_byte","bw_gbs"])
    for name in ("gemm","gelu","softmax"):
        a = agg.get(name)
        if not a: continue
        flops = a.get("fadd",0) + a.get("fmul",0) + 2*a.get("ffma",0)
        b = a.get("bytes",0); t = a.get("time",0)*1e-9
        if t<=0 or b<=0: continue
        gflops = flops/1e9/t; ai = flops/b; bw = b/1e9/t
        w.writerow([name, f"{t*1e3:.4f}", int(flops), int(b),
                    f"{gflops:.2f}", f"{ai:.3f}", f"{bw:.2f}"])
        print(f"[profile_ncu] {name:8s} AI={ai:7.3f} FLOP/B  perf={gflops:8.1f} GFLOP/s  BW={bw:6.1f} GB/s")
print("[profile_ncu] wrote results/roofline.csv")
PY
