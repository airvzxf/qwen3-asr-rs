#!/bin/bash
# Reproduce all Rust CPU benchmarks in sequence.
#
# Usage:   bash benchmarks/run_all.sh
#
# This script is intentionally simple and serial. Each step writes its
# JSON + Markdown output to benchmarks/results/. The torch baseline is
# a separate script in benchmarks/torch_baseline/run_cpu.sh because it
# needs the Python venv activated.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${QWEN3_ASR_MODEL_DIR:-$REPO_ROOT/models/Qwen--Qwen3-ASR-0.6B}"
AUDIO_DIR="$REPO_ROOT/tests/fixtures/audio"
RESULTS_DIR="$REPO_ROOT/benchmarks/results"
BIN="$REPO_ROOT/target/release/benchmark"

mkdir -p "$RESULTS_DIR"

# Build the benchmark binary (CPU, no CUDA/Metal)
echo ">>> Building target/release/benchmark (CPU, no default features, hub) ..."
(cd "$REPO_ROOT" && cargo build --release --bin benchmark --no-default-features --features hub)

if [ ! -x "$BIN" ]; then
  echo "ERROR: build did not produce $BIN"
  exit 1
fi

run_scenario() {
  local label="$1"
  local mode="$2"
  local extra="${3:-}"
  echo
  echo "=========================================="
  echo "  Scenario: $label ($mode)"
  echo "=========================================="
  $BIN \
      --model-dir "$MODEL_DIR" \
      --audio-dir "$AUDIO_DIR" \
      --runs 10 --warmup 1 \
      --mode "$mode" \
      --label "$label" \
      $extra \
      --json-out "$RESULTS_DIR/${label}.json" \
      --markdown-out "$RESULTS_DIR/${label}.md" \
      2>&1 | tail -25
}

# 3.1: Rust CPU batch
run_scenario "rust-cpu-batch" batch

# 3.3: Rust CPU streaming (2-second chunks, default)
run_scenario "rust-cpu-streaming" streaming

echo
echo "=========================================="
echo "  Cold start: 1 single-file transcribe"
echo "=========================================="
$BIN \
    --model-dir "$MODEL_DIR" \
    --audio-dir "$AUDIO_DIR" \
    --runs 1 --warmup 0 \
    --label "rust-cpu-coldstart" \
    --json-out "$RESULTS_DIR/rust-cpu-coldstart.json" \
    --markdown-out "$RESULTS_DIR/rust-cpu-coldstart.md" \
    2>&1 | tail -20

echo
echo "All Rust scenarios complete. JSON + MD in $RESULTS_DIR"
ls -la "$RESULTS_DIR"/*.json "$RESULTS_DIR"/*.md 2>/dev/null | tail -20
