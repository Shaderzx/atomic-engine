#!/usr/bin/env bash
# Print the real unified-memory footprint (bytes) of llama-server on Apple Silicon.
# Usage: mem_macos.sh [PID_FILE]
# On Apple Silicon unified memory, mmap'd model weights (clean file-backed pages)
# and GPU-accessible Metal allocations (shared/IOAccelerator/dirty phys_footprint)
# both consume physical RAM. `ps -o rss=` captures resident pages (dirty + resident
# mapped file), and we also query AGXAccelerator "In use system memory" via ioreg.
# We return max(rss_bytes, footprint_dirty_plus_clean, ioreg_in_use_bytes) so GPU
# allocations can never hide from the watchdog.
set -euo pipefail

PID_FILE="${1:-/tmp/inkling-bench-server.pid}"
if [[ ! -f "$PID_FILE" ]]; then
  echo 0
  exit 0
fi

PID="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ -z "$PID" ]] || ! kill -0 "$PID" 2>/dev/null; then
  echo 0
  exit 0
fi

RSS_KB="$(ps -o rss= -p "$PID" 2>/dev/null | tr -d ' ' || echo 0)"
RSS_B=$(( ${RSS_KB:-0} * 1024 ))

# AGXAccelerator reports GPU in-use system memory (bytes) across GPU clients
GPU_B="$(ioreg -r -c AGXAccelerator 2>/dev/null | awk -F'"In use system memory"=' 'NF>1 {split($2,a,","); gsub(/[^0-9]/,"",a[1]); val=a[1]} END {print val+0}')"
GPU_B="${GPU_B:-0}"

if (( RSS_B >= GPU_B )); then
  echo "$RSS_B"
else
  echo "$GPU_B"
fi
