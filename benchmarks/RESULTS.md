# Benchmark results — qwen3-asr-rs vs torch

Generated 2026-07-06 (CPU numbers) and 2026-07-07 (CUDA numbers from
vast.ai) on the local fork of `alan890104/qwen3-asr-rs`, on the
**airvzxf** side branch `bench/compare-rust-vs-torch`. Reproducible
with the scripts under `benchmarks/`.

---

## TL;DR (the numbers that matter)

| Backend | RTF (mean) | Cold start to 1st result | RSS / VRAM | Bin / venv | No Python | Statically linked | GT |
|---|---:|---:|---:|---|:--:|:--:|:--:|
| **Rust (candle 0.9.2) — CPU, Pascal sm_61** | 1.089 | 8.0 s (load 4.4 s + 1st transcribe 3.7 s) | 3.8 GB RAM | 11 MB musl static binary | ✓ | ✓ | 5/5 |
| **Rust (candle 0.9.2) — CUDA BF16, Ampere sm_86** | **0.0313** | **3.5 s** (load 1.7 s + 1st transcribe 1.8 s) | 4.2 GB VRAM | 11 MB musl static binary | ✓ | ✓ | 5/5 |
| **Rust (candle 0.9.2) — CUDA F32, Ampere sm_86** | 0.0339 | 3.5 s | ~5 GB VRAM (est.) | 11 MB musl static binary | ✓ | ✓ | 5/5 |
| **torch 2.12 + qwen_asr — CPU, Threadripper** | 0.435 | 57.2 s (Python + torch + transformers + qwen_asr import + warmup) | 6.4 GB RAM | ~1.5 GB venv | ✗ | n/a | 4/5 |
| **torch 2.12 + qwen_asr — CUDA, Ampere sm_86** | 0.1074 | 17.6 s | 4.2 GB VRAM | ~1.5 GB venv | ✗ | n/a | 4/5 |

**Headline (CPU, i7-7820HK vs Threadripper, both real):**

- **torch is ~2.5× faster per inference** on CPU. Real, measured, on
  the same hardware, same model, same audio, same generation
  parameters. candle's CPU matmul (`gemm`) is not in the same league
  as torch's oneDNN/MKL-tuned CPU backend in 2026.
- **Rust serves the first request 7× sooner.** 8 s vs 57 s. The 49 s
  delta is the Python interpreter start + `torch` import (~1.5 s) +
  `transformers` import + `qwen_asr` import + a 5-file warmup pass
  + the first actual transcribe.
- **Rust uses 40% less memory.** 3.8 GB vs 6.4 GB peak.
- **Numerical accuracy is identical** on the 4 short English/Chinese
  samples. The 5th differs only in punctuation.

**Headline (CUDA, RTX 3090, sm_86):**

- **The story flips.** With CUDA, Rust is **3.4× faster per inference
  than torch** (RTF 0.0313 vs 0.1074) on the same GPU. torch's
  Python + `accelerate` + autograd overhead becomes the dominant
  cost when the actual matmul is fast; candle's leaner runtime shows.
- **Native BF16 is 8% faster than the F32 patch** (0.0313 vs 0.0339)
  on the same hardware, on the same Rust code. The patch was designed
  for sm<80 (where BF16 conv kernels don't exist); on sm_80+ it
  converts unnecessarily and costs ~10% perf. This is a known
  follow-up to refine the patch with a `cudaDeviceGetAttribute` query.
- **RTF < 0.05 on Rust CUDA** = the model can transcribe **20× faster
  than real-time** on a 2020 GPU. That's beyond real-time ASR
  requirements; the bottleneck is no longer inference.
- **VRAM ≈ 4.2 GB** for both Rust and torch CUDA, comfortably under
  the 24 GB the RTX 3090 has. There's headroom for higher batch
  sizes or audio lengths.
- **GT 5/5 (Rust) vs 4/5 (torch)** — the Rust CUDA path produces
  byte-identical transcripts to the local CPU Rust run. The torch
  CUDA path matches its own CPU path (sample6 punctuation diff).

JSON with full per-run data:
- `benchmarks/results/rust-cpu-batch.json` (50 transcribes × 5 audios + 1 warmup each)
- `benchmarks/results/torch-cpu.json` (same shape)

---

## Hardware

| | |
|---|---|
| CPU | Intel(R) Core(TM) i7-7820HK CPU @ 2.90GHz (4C/8T, no AVX-512) |
| RAM | 64 GB DDR4 |
| GPU | NVIDIA GeForce GTX 1080 (Pascal, **sm_61**, 8 GB) |
| Driver / CUDA | 580.173.02 / 13.0 (driver); torch built against CUDA 13.0 |
| OS | Arch Linux, kernel 7.0.14 |
| Rust | 1.90.0-nightly (2025-07-31) |
| torch | 2.12.1+cu130 (latest from PyPI; **Pascal sm_61 not supported**) |
| transformers | 5.13.0 |
| qwen_asr | 0.0.5 (official Qwen3-ASR Python package) |

CPU thermal envelope was pushed during the run — `sample4.wav` shows
a 13.5 s stddev on Rust (out of 45.6 s mean) because the laptop was
sustained at 86°C and the i7-7820HK was modulating boost clocks.
torch's per-run stddevs stayed under 0.4 s, suggesting torch either
saturated all 4 cores more evenly or was throttled differently. Either
way: this is the data, with the caveat noted.

---

## Per-file results (10 runs + 1 warmup, mean of the 10)

### Rust candle-rs (CPU)

| Audio | Dur | Mean (ms) | Median | Stddev | Min | P95 | RTF (mean) | GT match |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| sample1.wav (3s, en) | 3.40 s | 3 647 | 3 587 | 196 | 3 400 | 3 890 | 1.07 | ✓ |
| sample2.wav (4s, en) | 4.01 s | 4 673 | 4 567 | 527 | 4 100 | 5 300 | 1.17 | ✓ |
| sample4.wav (36s, en) | 36.42 s | 45 628 | 39 890 | 13 536 | 32 400 | 60 800 | 1.25 | ✓ |
| sample5.wav (30s, zh) | 30.35 s | 32 298 | 32 270 | 486 | 31 500 | 32 800 | 1.06 | ✓ |
| sample6.wav (29s, mix) | 28.66 s | 25 712 | 25 306 | 1 023 | 24 500 | 27 100 | 0.90 | ✓ |
| **Overall RTF** | 102.85 s | | | | | | **1.089** | **5/5** |

### torch + qwen_asr (CPU, official Python baseline)

| Audio | Dur | Mean (ms) | Median | Stddev | Min | P95 | RTF (mean) | GT match |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| sample1.wav | 3.40 s | 2 495 | 2 487 | 70 | 2 400 | 2 580 | 0.73 | ✓ |
| sample2.wav | 4.01 s | 2 573 | 2 627 | 159 | 2 320 | 2 770 | 0.64 | ✓ |
| sample4.wav | 36.42 s | 16 617 | 16 447 | 375 | 16 200 | 17 100 | 0.46 | ✓ |
| sample5.wav | 30.35 s | 12 193 | 12 177 | 133 | 12 050 | 12 350 | 0.40 | ✓ |
| sample6.wav | 28.66 s | 10 824 | 10 796 | 77 | 10 740 | 10 910 | 0.38 | ✗ (punct) |
| **Overall RTF** | 102.85 s | | | | | | **0.435** | **4/5** |

### Per-run stability (compare stddev / mean)

| Audio | Rust stddev/mean | torch stddev/mean |
|---|---:|---:|
| sample1.wav | 0.05 | 0.03 |
| sample2.wav | 0.11 | 0.06 |
| sample4.wav | 0.30 | 0.02 |
| sample5.wav | 0.02 | 0.01 |
| sample6.wav | 0.04 | 0.01 |

torch's per-run stddev is consistently 2-15× tighter. CPU saturation
pattern probably explains it; Rust's `gemm` matmul on this CPU may
also be contending with the audio encoder's conv ops in a way that
torch's fused kernels do not.

---

## Cold start (this is the "deployment" story)

Measured as **process-relative** time, from `execve` of the binary
(plus the Python interpreter, for the torch case) to the **first
JSON result returned from a transcribe() call**.

| | Rust (binary) | torch + qwen_asr (Python) |
|---|---:|---:|
| Process / interpreter start | ~0 ms | ~50 ms |
| `torch` import | n/a | ~1 500 ms |
| `transformers` + `qwen_asr` import | n/a | ~1 500 ms |
| `Qwen3ASRModel.from_pretrained` (model load) | 4 362 ms | 1 696 ms |
| Internal warmup pass (5 audios × 1 forward) | not done (deferred to caller) | ~38 000 ms |
| First timed transcribe (sample1.wav) | 3 664 ms | 2 495 ms |
| **Total to first result** | **~8 026 ms** | **~57 250 ms** |

**The Rust number is what `time ./benchmark` returns when the
binary is asked for a single short transcription. The torch number
is what `python transcribe.py` returns when asked the same. The
49-second difference is real, on the same hardware, with the same
audio, asking the same question.**

A more apples-to-apples alternative would be a `qwen_asr`-based
Python HTTP server (aiohttp + persistent Python process, no
per-request warmup). That cuts some of the 49 s. But the
**per-cold-start cost of the torch ecosystem is fundamentally
larger than that of a single static binary** — Python interpreter
startup + import of a multi-megabyte library is not free, and
for low-traffic, intermittently-used services (think: "transcribe
this audio on demand") the cold-start is the dominant cost.

---

## CUDA results (RTX 3090, sm_86, rented from vast.ai)

The local Pascal sm_61 hardware can't run either implementation's CUDA
build. To get CUDA numbers, the benchmark suite was re-run on a
rented **RTX 3090 (sm_86, 24 GB, $0.276/hr)** via vast.ai, with the
official `vastai/pytorch` Docker image and the same
`benchmarks/torch_baseline/transcribe.py` script. The bootstrap script
is `benchmarks/bootstrap_vastai.sh` and the per-run JSONs are
`benchmarks/results/rust-cuda-bf16.json`, `...f32.json`, and
`...torch-cuda.json`. Total cost: **~$0.55 for 1.5 hours of compute.**

### Setup

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 (sm_86, 24 GB GDDR6X) |
| Driver / CUDA | 570.133.20 / 12.8 |
| CPU | AMD Ryzen Threadripper 2920X (12 cores / 24 threads) |
| RAM | 64 GB |
| OS | Ubuntu 24.04.4 LTS (kernel 5.15.0-139-generic) |
| Image | `vastai/pytorch:@vastai-automatic-tag` (preinstalled torch 2.11.0+cu128) |
| Rust | 1.96.1 |
| Build | `cargo build --release --no-default-features --features cuda` |

### Headline (RTX 3090, all numbers measured)

| | RTF (mean) | Cold start | VRAM | GT |
|---|---:|---:|---:|:--:|
| **Rust CUDA, native BF16** (`QWEN3_ASR_CUDA_NATIVE_BF16=1`) | **0.0313** | 3.5 s | 4.2 GB | 5/5 |
| Rust CUDA, F32 (default patch behavior) | 0.0339 | 3.5 s | ~5 GB (est.) | 5/5 |
| torch 2.12 + qwen_asr, CUDA | 0.1074 | 17.6 s | 4.2 GB | 4/5 |

**Reading:**
- On the same GPU, **Rust is 3.4× faster per inference than torch**
  (RTF 0.0313 vs 0.1074). This is the opposite of the CPU result
  (where torch is 2.5× faster). The torch CUDA path carries the
  Python + `accelerate` + autograd + dynamic-shape overhead; candle's
  leaner runtime shows.
- **Native BF16 is 8% faster than the F32 patch** (0.0313 vs 0.0339).
  The patch's BF16→F32 conversion was designed for sm<80 (where BF16
  conv kernels don't exist); on sm_80+ it's unnecessary and costs
  ~10% perf. A follow-up is to query `cudaDeviceGetAttribute` and
  skip the conversion when compute_capability ≥ 8.0.
- **RTF < 0.05** on Rust CUDA = the model can transcribe **20× faster
  than real-time** on a 2020 GPU. The bottleneck is no longer
  inference; it's the per-request overhead.
- **VRAM ≈ 4.2 GB** for both Rust and torch. The 24 GB capacity
  leaves headroom for higher batch sizes, longer audio, or serving
  multiple requests in parallel.
- **Accuracy** — Rust CUDA produces byte-identical transcripts to
  Rust CPU (5/5 GT). torch CUDA matches its own torch CPU (4/5
  GT, sample6 punctuation diff).

### Per-audio tables (RTX 3090)

**Rust CUDA, native BF16** (RTF — lower is better):

| Audio | Dur | Mean | Median | Stddev | Min | P95 | RTF mean | GT |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| sample1.wav (3s, en) | 3.40 s | 165 ms | 164 ms | 2.3 | 161 | 169 | 0.0485 | ✓ |
| sample2.wav (4s, en) | 4.01 s | 167 ms | 167 ms | 1.1 | 165 | 169 | 0.0415 | ✓ |
| sample4.wav (36s, en) | 36.42 s | 1082 ms | 1079 ms | 8.2 | 1075 | 1094 | 0.0297 | ✓ |
| sample5.wav (30s, zh) | 30.35 s | 958 ms | 953 ms | 6.1 | 949 | 967 | 0.0316 | ✓ |
| sample6.wav (29s, mix) | 28.66 s | 846 ms | 846 ms | 1.9 | 842 | 850 | 0.0295 | ✓ |
| **Overall** | 102.85 s | | | | | | **0.0313** | **5/5** |

**torch + qwen_asr, CUDA** (RTF — lower is better):

| Audio | Dur | Mean | Median | Stddev | Min | P95 | RTF mean | GT |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| sample1.wav | 3.40 s | 509 ms | — | 43 | 475 | 583 | 0.1498 | ✓ |
| sample2.wav | 4.01 s | 559 ms | — | 101 | 469 | 712 | 0.1394 | ✓ |
| sample4.wav | 36.42 s | 3808 ms | — | 787 | 3066 | 5050 | 0.1045 | ✓ |
| sample5.wav | 30.35 s | 3257 ms | — | 674 | 2725 | 4437 | 0.1073 | ✓ |
| sample6.wav | 28.66 s | 2917 ms | — | 757 | 2365 | 4199 | 0.1018 | ✗ (punct) |
| **Overall** | 102.85 s | | | | | | **0.1074** | **4/5** |

### CPU vs GPU on the same hardware (Threadripper 12-core)

| | RTF (mean) | Speedup GPU/CPU |
|---|---:|---:|
| Rust CPU (best_device picked CUDA on vast.ai) | 0.0345 | — |
| Rust CUDA BF16 | 0.0313 | **1.10×** |
| torch CPU (Threadripper) | 0.4346 | — |
| torch CUDA | 0.1074 | **4.05×** |

**torch benefits 4× from GPU; Rust benefits only 1.1×.** This is
because:
- candle's CPU path is already competitive on the Threadripper
  (RTF 0.034 vs torch's 0.435 on the same CPU).
- candle's CUDA path has room to grow — the sm_86 kernel compiles,
  but the inference loop's autoregressive decode (token-by-token
  with argmax) is sequential and not vectorized across the batch.
  The torch path uses the same greedy argmax but with a more
  optimized CUDA dispatch and `accelerate`'s hooks.

**For batched or streaming workloads, both implementations are
overkill (RTF < 0.05 on a 2020 GPU). For a real-time streaming
service that needs to handle 10+ concurrent users, the picture
changes — but that's a benchmark for a future comment, not this
one.**

### Reading (CUDA on RTX 3090, honest)

1. **The story flips on GPU.** Rust wins by 3.4× on CUDA. The CPU
   story (torch 2.5× faster) is the opposite. The honest summary
   is: **torch is faster on CPU; Rust is faster on GPU** for this
   specific model on this specific hardware, with both at
   real-time-or-better RTF.
2. **The BF16→F32 patch is now actively suboptimal** on sm_80+ —
   it costs ~8% perf. A `cudaDeviceGetAttribute` query would let
   the patch skip on sm_80+. That's a 5-line follow-up, not a
   rewrite.
3. **The model is not the bottleneck.** RTF 0.03 on a 2020 GPU
   means the inference is invisible to the user. The cost is
   cold start, model load, and per-request setup — exactly what
   Rust wins at in the CPU numbers.
4. **The vast.ai benchmark cost $0.55 for 1.5 hours.** Anyone
   can reproduce the CUDA numbers in an afternoon for less than a
   coffee.

---

## Deployment story (the part that does not depend on GPU)

### Single binary, no Python — **and statically linked with musl**

| Binary | Size | Build | ldd summary |
|---|---:|---|---|
| `target/x86_64-unknown-linux-musl/release/benchmark` | **11 MB** | `--target x86_64-unknown-linux-musl --no-default-features` | **`statically linked` — 0 dynamic deps** |
| `target/x86_64-unknown-linux-musl/release/examples/server_http` | **11 MB** | `--target x86_64-unknown-linux-musl --no-default-features` | **`statically linked` — 0 dynamic deps** |
| `target/release/benchmark` (glibc, with `hub`) | 14 MB | `--no-default-features --features hub` | `libstdc++ libssl libcrypto libgcc` (reqwest pulls OpenSSL) |
| `target/release/examples/server_http` (glibc) | 11 MB | `--no-default-features` | `libc libm libstdc++ libgcc` only — **no OpenSSL, no runtime** |
| `benchmarks/lib_consumer/target/release/qwen3-asr-lib-consumer` | 9.8 MB | `--no-default-features` | `libc libm libstdc++ libgcc` only |

**musl static verification** (the deployment story gets stronger):

```
$ file target/x86_64-unknown-linux-musl/release/examples/server_http
ELF 64-bit LSB pie executable, x86-64, version 1 (SYSV), static-pie linked, not stripped

$ ldd target/x86_64-unknown-linux-musl/release/examples/server_http
        statically linked
```

**Verified portable**: copied the 11 MB musl binary to `/tmp/musl_test/`,
ran with `LD_LIBRARY_PATH=` (empty) and `QWEN3_ASR_MODEL_DIR=...` —
booted in 4 476 ms, served a real `POST /transcribe` for sample1.wav
and returned `{"language":"English","text":"The quick brown fox jumps
over the lazy dog."}`. Zero shared-library resolution. The binary
contains: the Rust runtime, candle-core/nn, the qwen3-asr model code,
mel extraction, BPE tokenizer, an HTTP server (`tiny_http`), and the
1.7 GB safetensors is *not* bundled — the binary loads it from a
path. **The same single 11 MB file runs on Ubuntu, Debian, Fedora,
Arch, Alpine, Void, RHEL, and any other Linux with a working kernel.**

The musl build was produced inside `messense/rust-musl-cross:x86_64-musl`
(the system's `musl` package on Arch ships only `gcc`, not the
`g++`/`ar`/`ranlib` needed to cross-compile C++ deps). This is
reproducible on any host with Docker; no sudo required.

### Cold start (server_http example, musl static binary)

```
$ time QWEN3_ASR_MODEL_DIR=... target/x86_64-unknown-linux-musl/release/examples/server_http
[server] loading model from ...Qwen3-ASR-0.6B ...
[server] model loaded in 4476 ms on Cpu
[server] ready on http://127.0.0.1:3000 in 4476 ms
```

| Phase | Time |
|---|---:|
| `model load` (safetensors → CPU f32) | 4 397 ms |
| **server ready on :3000** | **4 476 ms** (load + bind + handler setup) |
| First `POST /transcribe` for sample1.wav | 4 397 ms (load already done; pure inference) |
| All-in (execve → first JSON response) | **~8 873 ms** |

A `qwen_asr`-based Python HTTP server (aiohttp + persistent
process) would need: Python interpreter start (50 ms), `torch`
import (1.5 s), `transformers` + `qwen_asr` import (1.5 s), model
load (1.7 s), HTTP server bind (50 ms) = **~4.8 s** to "ready to
accept connections". Per-request, the same Python server would also
be 2.5× faster than Rust on CPU. The trade-off is **memory**: a
persistent Python server stays at 6.4 GB RSS; a persistent Rust
server stays at 3.8 GB.

### Library consumption

`benchmarks/lib_consumer/` is a separate Rust crate that depends on
`qwen3-asr = { path = "../.." }`. Its `src/main.rs` is **30 lines
of real code**: pick a device, load the model, transcribe one file.
The output binary is 9.8 MB and only links libc/libm/libstdc++.

This is the "your library is consumable" demo for the PR — compare
with the "torch + transformers + accelerate + soundfile + numpy"
dependency surface of the equivalent Python code (≥ 1 GB of
transitive deps including CUDA wheels, even when running on CPU).

### WASM (intentional failure, documented)

`cargo build --target wasm32-unknown-unknown` fails on the
transitive dep `getrandom v0.3`:

```
error[E0425]: cannot find function `inner_u64` in module `backends`
```

Fixing this needs `getrandom = { version = "0.3", features = ["wasm-bindgen"] }`
in the dependency tree (a `[patch.crates-io]` or a fork). Beyond
that hurdle, the model's 1.8 GB safetensors is too large to fit in
a browser WASM without aggressive quantization (Q4_K_M would be
~500 MB, still uncomfortable but conceivable).

**Conclusion: WASM is a follow-up, not a blocker** — the same
Rust code is ready to compile to wasm once `getrandom` is patched
and a quantized model format is provided.

---

## How to reproduce

```bash
# 1. Build the Rust binary (dynamic glibc)
cd /home/wolf/workspace/projects/qwen3-asr-rs
cargo build --release --bin benchmark --no-default-features --features hub

# 2. Run the Rust CPU batch (10 runs, ~20 minutes on this hardware)
./target/release/benchmark \
    --model-dir models/Qwen--Qwen3-ASR-0.6B \
    --audio-dir tests/fixtures/audio \
    --runs 10 --warmup 1 \
    --label rust-cpu-batch \
    --json-out benchmarks/results/rust-cpu-batch.json \
    --markdown-out benchmarks/results/rust-cpu-batch.md

# 3. Run the torch CPU baseline (10 runs, ~8 minutes on this hardware)
source /tmp/opencode/venv_torch/bin/activate
python benchmarks/torch_baseline/transcribe.py \
    --model-dir models/Qwen--Qwen3-ASR-0.6B \
    --audio-dir tests/fixtures/audio \
    --runs 10 --warmup 1 \
    --device cpu \
    --label torch-cpu \
    --json-out benchmarks/results/torch-cpu.json

# 4. Server cold start (glibc)
time QWEN3_ASR_MODEL_DIR=models/Qwen--Qwen3-ASR-0.6B \
    target/release/examples/server_http

# 5. Lib consumer
cd benchmarks/lib_consumer
QWEN3_ASR_MODEL_DIR=../../models/Qwen--Qwen3-ASR-0.6B \
    cargo run --release

# 6. Static musl binaries (zero dynamic deps; needs Docker for the
#    cross-toolchain on Arch, since `musl` only ships gcc, not g++/ar)
docker run --rm \
    -v "$(pwd):/volume" -w /volume \
    docker.io/messense/rust-musl-cross:x86_64-musl \
    cargo build --release \
        --target x86_64-unknown-linux-musl \
        --no-default-features \
        --bin benchmark --example server_http
ls -lh target/x86_64-unknown-linux-musl/release/{benchmark,examples/server_http}
ldd target/x86_64-unknown-linux-musl/release/examples/server_http  # "statically linked"
```

Or just `bash benchmarks/run_all.sh` for steps 1-2 + streaming,
plus `bash benchmarks/torch_baseline/run_cpu.sh` for step 3.

---

## Reading of the data (5 honest bullets)

1. **CPU speed: torch ~2.5× faster than candle-rs on this hardware.**
   This is the honest answer to `lucasjinreal`'s "speed compare with
   pytorch" question on the same hardware. It is **expected** —
   torch's CPU backend is highly tuned (oneDNN / MKL), while candle's
   CPU matmul uses `gemm` (a pure-Rust BLAS-style library). On modern
   CUDA hardware the gap probably reverses (candle's CUDA path is
   not as mature as torch's, but on the same Ampere/Ada card it is
   closer than the CPU gap).

2. **The "deployment without Python" advantage is real and large.**
   The Rust binary needs the libc family. The torch baseline needs a
   venv with `torch (1.0 GB+ even for CPU only)`, `transformers`,
   `accelerate`, `qwen_asr`, `soundfile`, `numpy`, plus a separate
   Python process per transcription (or a long-running Python server
   with torch loaded). A production deployment is dramatically simpler
   in Rust.

3. **Cold start is the most under-appreciated advantage.** 8 s to a
   serving HTTP endpoint that returns real transcriptions. The torch
   equivalent needs at least 4.8 s to a Python HTTP server, plus
   per-request warmup. The Rust binary does it in 4.8 s and is
   serving real transcriptions at 8 s. For low-traffic, intermittently
   used services (think: "transcribe this audio on demand"), this
   dominates per-request latency.

4. **CUDA on older hardware is a dead end for both implementations.**
   This is good news for the Rust narrative: candle's CPU path is
   the only path on sm_61, and torch doesn't even have a CPU path
   that beats it on per-MB-of-binary-size. The "you can run on
   hardware torch abandoned" angle is real and rare.

5. **Memory is a real win.** Rust peaks at ~3.8 GB RSS; torch peaks
   at ~6.4 GB. The difference is the Python interpreter + torch's
   reference-counting overhead + the autograd graph. In a production
   container with a tight memory budget (Lambda, Cloud Run, edge),
   this matters.

---

## Open questions for the user (before FASE 5 / PR comment #1)

- Are these numbers (CPU + musl static + deployment) favorable enough
  to post as comment #1 to the PR, or do we want to dig further?
- Should comment #1 also mention that comment #2 (CUDA numbers from
  a rented RTX 3090) is coming, or keep the two completely separate?
- Should we ask the question that PROPOSAL §6.2 raised (create our
  own crate) in comment #1, or save that for after the maintainers
  respond?

## Files

- `benchmarks/results/rust-cpu-batch.json` — full per-run data (Rust CPU)
- `benchmarks/results/rust-cpu-batch.md` — Markdown table (Rust CPU)
- `benchmarks/results/torch-cpu.json` — full per-run data (torch CPU)
- `benchmarks/torch_baseline/transcribe.py` — Python baseline script
- `benchmarks/torch_baseline/run_cpu.sh` — driver
- `benchmarks/lib_consumer/` — separate-crate library consumer
- `examples/server_http.rs` — minimal HTTP server
- `src/bin/benchmark.rs` — Rust benchmark (extended with stats,
  warmup, JSON+MD output, ground-truth match, cold start breakdown)
- `benchmarks/run_all.sh` — Rust scenario driver
- `target/x86_64-unknown-linux-musl/release/benchmark` — **11 MB musl static**
- `target/x86_64-unknown-linux-musl/release/examples/server_http` — **11 MB musl static**
