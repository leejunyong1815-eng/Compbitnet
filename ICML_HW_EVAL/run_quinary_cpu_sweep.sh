#!/usr/bin/env bash
set -euo pipefail

SRC="quinary_cpu_table.cpp"
BIN="quinary_cpu_table"
OUT_CSV="cpu_sweep_results.csv"

if [[ ! -f "$SRC" ]]; then
  echo "[ERROR] $SRC not found in current directory"
  echo "Run this script from the directory containing $SRC, or edit SRC/BIN paths."
  exit 1
fi

g++ -O3 -mavx2 -mfma -fopenmp -std=c++17 "$SRC" -o "$BIN"

./"$BIN" \
  --iters 30 \
  --warmup 5 \
  --tile-j 64 \
  --verify-n 128 \
  --n 1024 \
  --zero-prob 0.2,0.4,0.6,0.8 \
  --threads 1,8,16 \
  --csv "$OUT_CSV"

echo "[OK] wrote $OUT_CSV"
