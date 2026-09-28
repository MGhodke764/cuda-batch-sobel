#!/usr/bin/env bash
set -euo pipefail

IMAGE_COUNT="${1:-500}"
BLOCK_SIZE="${2:-16}"

python3 scripts/generate_dataset.py \
  --output data/images \
  --count "${IMAGE_COUNT}" \
  --width 256 \
  --height 256

make

./cuda_sobel \
  --input data/images \
  --output results/edges \
  --limit "${IMAGE_COUNT}" \
  --block "${BLOCK_SIZE}" \
  | tee results/execution_log.txt

echo
echo "Artifacts written to results/"
