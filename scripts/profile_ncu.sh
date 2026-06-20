#!/usr/bin/env bash
# ncu roofline data for the three kernels -> results/roofline.csv
#
# Metric names verified on this machine (Blackwell sm_120, ncu 2025.2):
#   sm__throughput.avg.pct_of_peak_sustained_elapsed     (compute %)
#   dram__throughput.avg.pct_of_peak_sustained_elapsed   (memory %)
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
sm   = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
dram = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
build = os.environ.get("BUILD", "build")
out = subprocess.run(
    ["ncu","--csv","--metrics",f"{sm},{dram}",
     "--kernel-name","regex:gemm_tiled_kernel|fused_bias_gelu_kernel|softmax_kernel",
     "--launch-count","3", f"{build}/bench_kernels"],
    capture_output=True, text=True)
raw = out.stdout
lines = raw.splitlines()
start = next((i for i,l in enumerate(lines) if l.startswith('"ID"')), None)
if start is None:
    print("[profile_ncu] ncu produced no CSV (perf-counter permission? try sudo).")
    print(out.stderr[-400:])
    raise SystemExit(0)
rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
friendly = {"gemm_tiled_kernel":"gemm","fused_bias_gelu_kernel":"gelu","softmax_kernel":"softmax"}
agg = {}
for r in rows:
    k = r.get("Kernel Name","")
    name = next((f for key,f in friendly.items() if key in k), None)
    if not name: continue
    m, v = r.get("Metric Name",""), r.get("Metric Value","")
    try: v = float(str(v).replace(",",""))
    except ValueError: continue
    agg.setdefault(name, {})
    if m == sm:   agg[name]["sm"]   = v
    if m == dram: agg[name]["dram"] = v
with open("results/roofline.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["kernel","sm_pct","dram_pct"])
    for name in ("gemm","gelu","softmax"):
        if name in agg:
            w.writerow([name, agg[name].get("sm",0), agg[name].get("dram",0)])
            print(f"[profile_ncu] {name}: sm={agg[name].get('sm',0):.1f}% dram={agg[name].get('dram',0):.1f}%")
print("[profile_ncu] wrote results/roofline.csv")
PY
