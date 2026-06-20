#!/usr/bin/env bash
# Launch N workers on consecutive ports. Prints the worker CSV to stdout
# (for --workers), logs + PIDs to stderr / a pidfile.
#
# usage: launch_workers.sh N [base_port] [fixtures_dir] [build_dir]
#   stop with: kill $(cat /tmp/distbatch_workers.pids)
set -euo pipefail

N="${1:-2}"
BASE_PORT="${2:-50051}"
FIX="${3:-fixtures}"
BUILD="${4:-build}"
PIDFILE="/tmp/distbatch_workers.pids"
: > "$PIDFILE"

csv=""
for i in $(seq 0 $((N - 1))); do
  port=$((BASE_PORT + i))
  "$BUILD/worker" --port "$port" --id "$i" \
    --weights "$FIX/weights.bin" --dims "$FIX/dims.txt" \
    >"/tmp/distbatch_worker_${i}.log" 2>&1 &
  echo "$!" >> "$PIDFILE"
  echo "[launch_workers] worker $i pid=$! port=$port log=/tmp/distbatch_worker_${i}.log" >&2
  csv+="localhost:${port}"
  [ "$i" -lt $((N - 1)) ] && csv+=","
done

# wait until each worker is listening
for i in $(seq 0 $((N - 1))); do
  for _ in $(seq 1 100); do
    grep -q listening "/tmp/distbatch_worker_${i}.log" 2>/dev/null && break
    sleep 0.05
  done
done

echo "$csv"
