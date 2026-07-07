#!/bin/bash
# Reproduce the torch CPU baseline.
#
# Usage:   bash benchmarks/torch_baseline/run_cpu.sh
#
# Requires the venv at /tmp/opencode/venv_torch (or set VENV env var).
# Uses the official `qwen_asr` Python package, which is the Qwen team's
# reference implementation and matches the local model.safetensors'
# `thinker.audio_tower.*` key naming.

set -euo pipefail

VENV="${VENV:-/tmp/opencode/venv_torch}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL_DIR="${QWEN3_ASR_MODEL_DIR:-$REPO_ROOT/models/Qwen--Qwen3-ASR-0.6B}"
AUDIO_DIR="$REPO_ROOT/tests/fixtures/audio"
RESULTS_DIR="$REPO_ROOT/benchmarks/results"

if [ ! -d "$VENV" ]; then
  echo "ERROR: venv not found at $VENV. Create it with:"
  echo "  python3 -m venv $VENV"
  echo "  $VENV/bin/pip install -r $REPO_ROOT/benchmarks/torch_baseline/requirements.txt"
  exit 1
fi

source "$VENV/bin/activate"
python "$REPO_ROOT/benchmarks/torch_baseline/transcribe.py" \
    --model-dir "$MODEL_DIR" \
    --audio-dir "$AUDIO_DIR" \
    --runs 10 --warmup 1 \
    --device cpu \
    --label torch-cpu \
    --json-out "$RESULTS_DIR/torch-cpu.json"
