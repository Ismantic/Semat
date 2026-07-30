#!/bin/bash
# Batch segmentation using Wapic.
# Usage: cut.sh <input> <output> [threads] [model]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

INPUT=${1:?usage: cut.sh <input> <output> [threads] [model]}
OUTPUT=${2:?usage: cut.sh <input> <output> [threads] [model]}
THREADS=${3:-$(nproc)}
MODEL=${4:-}
PYTHON=${PYTHON:-python3}

export OMP_NUM_THREADS="$THREADS"
mkdir -p "$(dirname "$OUTPUT")"

if [[ -n "$MODEL" ]]; then
    "$PYTHON" "$SCRIPT_DIR/segment_wapic.py" "$INPUT" "$OUTPUT" --model "$MODEL"
else
    "$PYTHON" "$SCRIPT_DIR/segment_wapic.py" "$INPUT" "$OUTPUT"
fi
