#!/usr/bin/env bash
# Phase 5 experiment sweeps -> results/*.csv
#
#   GPU batch scaling   : throughput vs batch        (direct to 1 worker)
#   Block size sweep    : latency vs GEMM_TILE        (separate builds)
#   Worker scaling      : throughput/speedup vs #workers (dispatcher)
#   Seq length scaling  : latency vs seq_len          (direct to 1 worker)
#   Bottleneck breakdown: queue/compute/other         (dispatcher + 1 worker)
#
# All workers share the single physical GPU, so worker-scaling speedup reflects
# stream/process overlap on one GPU, not independent devices (see README).
set -euo pipefail
cd "$(dirname "$0")/.."

BUILD="${BUILD:-build}"
FIX="${FIX:-fixtures}"
RES="results"
REQUESTS="${REQUESTS:-400}"
CONC="${CONC:-16}"
WORKER_BASE=50061
DISP_PORT=50050
mkdir -p "$RES"

# --- ensure build + fixtures ---
[ -d "$BUILD" ] || cmake -B "$BUILD" -S . >/dev/null
cmake --build "$BUILD" -j4 --target worker dispatcher client >/dev/null
python3 tests/ref_block.py --out "$FIX" >/dev/null

PIDS=()
cleanup() {
  for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
  [ -f /tmp/distbatch_workers.pids ] && kill $(cat /tmp/distbatch_workers.pids) 2>/dev/null || true
  PIDS=()
}
trap cleanup EXIT

start_workers() {  # N base build -> echo csv
  bash scripts/launch_workers.sh "$1" "$2" "$FIX" "${3:-$BUILD}" 2>/dev/null
}
start_dispatcher() {  # csv maxbatch -> sets DPID
  : > /tmp/disp.log
  "$BUILD/dispatcher" --port "$DISP_PORT" --workers "$1" \
    --window-ms 5 --max-batch "${2:-32}" --routing ll >/tmp/disp.log 2>&1 &
  DPID=$!; PIDS+=("$DPID")
  for _ in $(seq 1 100); do grep -q listening /tmp/disp.log && break; sleep 0.05; done
}

echo "=================================================================="
echo " 1) GPU batch scaling (direct to 1 worker)"
echo "=================================================================="
rm -f "$RES/batch_scaling.csv"
CSV=$(start_workers 1 $WORKER_BASE)
for b in 1 2 4 8 16 32 64 128 256; do
  "$BUILD/client" --target "$CSV" --requests "$REQUESTS" --concurrency "$CONC" \
    --batch "$b" --seq_len 32 --hidden_dim 128 \
    --csv "$RES/batch_scaling.csv" --tag "batch$b" | sed -n '2p'
done
cleanup

echo "=================================================================="
echo " 2) Seq length scaling (direct to 1 worker)"
echo "=================================================================="
rm -f "$RES/seqlen_scaling.csv"
CSV=$(start_workers 1 $WORKER_BASE)
for s in 128 512 1024 2048; do
  "$BUILD/client" --target "$CSV" --requests "$((REQUESTS/2))" --concurrency "$CONC" \
    --batch 1 --seq_len "$s" --hidden_dim 128 \
    --csv "$RES/seqlen_scaling.csv" --tag "seq$s" | sed -n '2p'
done
cleanup

echo "=================================================================="
echo " 3) Worker scaling (dispatcher, big requests so compute matters)"
echo "=================================================================="
rm -f "$RES/worker_scaling.csv"
for n in 1 2 4; do
  CSV=$(start_workers "$n" $WORKER_BASE)
  start_dispatcher "$CSV" 1            # max-batch 1: one micro-batch per request
  "$BUILD/client" --target "localhost:$DISP_PORT" --requests "$REQUESTS" \
    --concurrency "$((CONC*2))" --batch 8 --seq_len 256 --hidden_dim 128 \
    --csv "$RES/worker_scaling.csv" --tag "workers$n" | sed -n '2p'
  cleanup
done

echo "=================================================================="
echo " 4) Bottleneck breakdown (dispatcher + 1 worker)"
echo "=================================================================="
rm -f "$RES/breakdown.csv"
CSV=$(start_workers 1 $WORKER_BASE)
start_dispatcher "$CSV" 32
"$BUILD/client" --target "localhost:$DISP_PORT" --requests "$REQUESTS" \
  --concurrency "$CONC" --batch 8 --seq_len 256 --hidden_dim 128 \
  --csv "$RES/breakdown.csv" --tag "breakdown" | sed -n '3p'
cleanup

echo "=================================================================="
echo " 5) Block size sweep (GEMM_TILE rebuild, direct to 1 worker)"
echo "=================================================================="
rm -f "$RES/blocksize.csv"
for TILE in 8 16 32; do
  bdir="build_tile$TILE"
  cmake -B "$bdir" -S . -DGEMM_TILE="$TILE" >/dev/null
  cmake --build "$bdir" -j4 --target worker client >/dev/null
  CSV=$(start_workers 1 $WORKER_BASE "$bdir")
  "$bdir/client" --target "$CSV" --requests "$REQUESTS" --concurrency "$CONC" \
    --batch 16 --seq_len 128 --hidden_dim 128 \
    --csv "$RES/blocksize.csv" --tag "tile$TILE" | sed -n '2p'
  cleanup
done

echo "=================================================================="
echo " 6) ncu roofline + plots"
echo "=================================================================="
bash scripts/profile_ncu.sh || echo "[run_experiments] ncu step skipped"
python3 scripts/plot.py || echo "[run_experiments] plot step skipped"

echo
echo "All sweeps complete. CSVs + PNGs in $RES/:"
ls -1 "$RES"/*.csv "$RES"/*.png 2>/dev/null
