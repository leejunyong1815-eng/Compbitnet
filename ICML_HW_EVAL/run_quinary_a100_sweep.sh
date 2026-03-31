#!/usr/bin/env bash
set -euo pipefail

SRC="quinary_a100_eval.cu"
BIN="quinary_a100_eval"
OUT_CSV="a100_sweep_results.csv"
NVCC_BIN=${NVCC:-nvcc}

"${NVCC_BIN}" -O3 -std=c++17 -arch=sm_80 "${SRC}" -lcublas -lnvidia-ml -o "${BIN}"

./"${BIN}" \
  --iters 1000 \
  --warmup 100 \
  --sample-ms 5 \
  --modes bf16,int8,dense_cuda_int8,opcode_adddbl,opcode_shift \
  --sweep-n 1024 \
  --sweep-zero-prob 0.2,0.4,0.6,0.8 \
  --csv "${OUT_CSV}"

echo "[OK] wrote ${OUT_CSV}"
