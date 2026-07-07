#!/bin/bash
# Bootstrap the qwen3-asr-rs CUDA benchmark on a vast.ai instance.
#
# Tested base image: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
# GPU target:       RTX 3090 (sm_86, 24 GB VRAM) or better
# Expected time:     ~30-40 min total (10 min setup + 15-20 min benchmarks)
#
# What it produces, in /root/bench-results.tgz:
#   benchmarks/results/rust-cpu.json       (Rust CPU baseline, 5 audios × 10 runs)
#   benchmarks/results/rust-cuda-f32.json  (Rust CUDA, BF16→F32 conversion, sm<80 path)
#   benchmarks/results/rust-cuda-bf16.json(Rust CUDA, native BF16, sm_80+ path)
#   benchmarks/results/torch-cpu.json      (torch + qwen_asr CPU baseline)
#   benchmarks/results/torch-cuda.json     (torch + qwen_asr CUDA baseline)
#
# To download after completion:
#   scp -P <remote-port> root@<vast-ai-host>:/root/bench-results.tgz .

set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

REPO_DIR="$HOME/qwen3-asr-rs"
MODEL_DIR="$HOME/models/Qwen--Qwen3-ASR-0.6B"
RESULTS_DIR="$REPO_DIR/benchmarks/results"
REPO_URL="https://github.com/airvzxf/qwen3-asr-rs.git"
BRANCH="bench/compare-rust-vs-torch"

# The vastai/pytorch image ships a preinstalled PyTorch venv at /venv/main
# (with torch + torchvision + huggingface-hub already in it). Use it as the
# source of truth for Python and pip — installing into system site-packages
# (--break-system-packages) is a hack that breaks the principle.
source /venv/main/bin/activate
PYTHON="python"        # now points to /venv/main/bin/python (with torch)
PIP="pip"              # venv's pip
PIP_FLAGS=()            # no --break-system-packages needed inside a venv

# If running inside a vast.ai Docker container, CUDA_COMPUTE_CAP is auto-detected
# from the GPU. We override here as a safety net (the env var is read by
# bindgen_cuda 0.1.6 before it falls back to nvidia-smi).
export CUDA_COMPUTE_CAP="${CUDA_COMPUTE_CAP:-86}"

NPROC=$(nproc)
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  qwen3-asr-rs CUDA benchmark — bootstrap"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  Host      : $(hostname)"
echo "  GPU       : $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
echo "  CUDA cap  : $CUDA_COMPUTE_CAP (sm_$CUDA_COMPUTE_CAP)"
echo "  CPUs      : $NPROC"
echo "  PyTorch   : $($PYTHON -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda)')"
echo "  Working in: $REPO_DIR"
echo

# ── 1. Install Rust (no sudo needed, user-mode install) ───────────────────────
if ! command -v cargo >/dev/null 2>&1; then
    echo ">>> Installing Rust toolchain (user-mode) ..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
        sh -s -- -y --default-toolchain stable --profile minimal
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
    rustc --version
    cargo --version
else
    echo ">>> Rust already present: $(rustc --version)"
fi

# ── 2. Clone the bench branch ────────────────────────────────────────────────
if [ ! -d "$REPO_DIR" ]; then
    echo ">>> Cloning $REPO_URL @ $BRANCH ..."
    git clone --depth 1 -b "$BRANCH" "$REPO_URL" "$REPO_DIR"
else
    echo ">>> Repo already cloned at $REPO_DIR; pulling latest ..."
    cd "$REPO_DIR" && git pull --ff-only origin "$BRANCH" || true
fi
cd "$REPO_DIR"
git rev-parse HEAD
git log --pretty="%h %G? %s" -1

# ── 3. Install Python deps for the torch baseline ──────────────────────────
echo ">>> Installing qwen_asr + huggingface_hub into /venv/main ..."
$PIP install --quiet "${PIP_FLAGS[@]}" qwen_asr huggingface_hub
$PYTHON -c "import qwen_asr, torch, huggingface_hub; print('qwen_asr OK | torch', torch.__version__, 'cuda', torch.version.cuda)"

# ── 4. Download the Qwen3-ASR-0.6B model ──────────────────────────────────
mkdir -p "$(dirname "$MODEL_DIR")"
if [ ! -f "$MODEL_DIR/model.safetensors" ] || [ ! -f "$MODEL_DIR/tokenizer.json" ]; then
    echo ">>> Downloading Qwen/Qwen3-ASR-0.6B (1.7 GB) ..."
    $PYTHON -c "
from huggingface_hub import snapshot_download
# Be explicit: 'tokenizer.json' is a critical file but huggingface_hub's
# '*.json' pattern can drop it on this model snapshot. List everything we need.
snapshot_download('Qwen/Qwen3-ASR-0.6B', local_dir='$MODEL_DIR',
                  allow_patterns=[
                      'config.json', 'preprocessor_config.json',
                      'tokenizer.json', 'tokenizer_config.json',
                      'vocab.json', 'merges.txt', 'chat_template.json',
                      'generation_config.json', 'model.safetensors',
                  ])
"
else
    echo ">>> Model already present at $MODEL_DIR (with tokenizer.json)"
fi
ls -lh "$MODEL_DIR" | head

# ── 5. Build the Rust benchmark binary with CUDA ───────────────────────────
echo ">>> Building benchmark binary (CUDA, sm_$CUDA_COMPUTE_CAP) ..."
echo "    This takes ~5-10 min on a clean build."
time cargo build --release \
    --bin benchmark \
    --no-default-features \
    --features cuda \
    -j "$NPROC"
ls -lh target/release/benchmark
file target/release/benchmark

# ── 6. Define the run helper ────────────────────────────────────────────────
run_rust() {
    local label="$1"
    local native_bf16="${2:-0}"
    echo
    echo "=========================================="
    echo "  Rust CUDA benchmark: $label"
    echo "=========================================="
    local start=$(date +%s)
    # env vars (e.g. QWEN3_ASR_CUDA_NATIVE_BF16) are passed via env(1) to
    # avoid bash parsing ambiguity with `time VAR=val command`.
    if [ "$native_bf16" = "1" ]; then
        env QWEN3_ASR_CUDA_NATIVE_BF16=1 time ./target/release/benchmark \
            --model-dir "$MODEL_DIR" \
            --audio-dir tests/fixtures/audio \
            --runs 10 --warmup 1 \
            --label "$label" \
            --json-out "$RESULTS_DIR/$label.json" \
            --markdown-out "$RESULTS_DIR/$label.md"
    else
        time ./target/release/benchmark \
            --model-dir "$MODEL_DIR" \
            --audio-dir tests/fixtures/audio \
            --runs 10 --warmup 1 \
            --label "$label" \
            --json-out "$RESULTS_DIR/$label.json" \
            --markdown-out "$RESULTS_DIR/$label.md"
    fi
    local end=$(date +%s)
    echo "  -> $label.json written ($((end - start))s wall)"
}

run_torch() {
    local label="$1"
    local device="$2"
    echo
    echo "=========================================="
    echo "  torch baseline: $label ($device)"
    echo "=========================================="
    local start=$(date +%s)
    time $PYTHON benchmarks/torch_baseline/transcribe.py \
        --model-dir "$MODEL_DIR" \
        --audio-dir tests/fixtures/audio \
        --runs 10 --warmup 1 \
        --device "$device" \
        --label "$label" \
        --json-out "$RESULTS_DIR/$label.json" \
        > "$RESULTS_DIR/$label.log" 2>&1
    local rc=$?
    local end=$(date +%s)
    if [ $rc -ne 0 ]; then
        echo "  !! $label FAILED (exit $rc). Last 20 lines of log:"
        tail -20 "$RESULTS_DIR/$label.log"
    else
        echo "  -> $label.json written ($((end - start))s wall)"
    fi
    return $rc
}

# ── 7. Run the Rust CPU baseline (sanity check, fast) ──────────────────────
run_rust "rust-cpu-batch"

# ── 8. Run the Rust CUDA scenarios ─────────────────────────────────────────
# 8a. F32 path: BF16→F32 conversion on (the original unrefined patch behavior).
#     This is the path the user's patch was originally designed for.
run_rust "rust-cuda-f32" 0

# 8b. BF16 native path: env var skips the conversion. Represents what candle
#     would do on sm_80+ without the workaround.
run_rust "rust-cuda-bf16" 1

# ── 9. Run the torch baselines ─────────────────────────────────────────────
run_torch "torch-cpu" "cpu"
run_torch "torch-cuda" "cuda" || true   # don't fail the whole script if torch CUDA fails (sm<75)

# ── 10. Print summary ───────────────────────────────────────────────────────
echo
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  All runs done. Summary:"
echo "═══════════════════════════════════════════════════════════════════════════════"
ls -lh "$RESULTS_DIR"/*.json 2>/dev/null
echo
echo "  Hardware that produced these numbers:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo
echo "  CPU: $(lscpu | grep 'Model name' | awk -F: '{print $2}' | xargs)"
echo "  RAM: $(free -h | grep Mem | awk '{print $2}')"
echo "  PyTorch: $($PYTHON -c 'import torch; print(torch.__version__)')"
echo "  Rust:    $(rustc --version)"
echo

# ── 11. Package the results ────────────────────────────────────────────────
cd "$HOME"
tar -czf bench-results.tgz \
    qwen3-asr-rs/benchmarks/results \
    qwen3-asr-rs/benchmarks/RESULTS.md
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  Results packaged:"
echo "═══════════════════════════════════════════════════════════════════════════════"
ls -lh bench-results.tgz
echo
echo "Download with:"
echo "  scp -P <ssh-port> root@<vast-ai-host>:/root/bench-results.tgz ."
echo
echo "DESTROY THE INSTANCE NOW (vast.ai charges by the hour, currently ~\$0.30-0.50/hr)"
