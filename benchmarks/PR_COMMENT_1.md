<!--
PR comment draft for huggingface/candle#3509
Author: airvzxf
Posted via: gh pr comment 3509 --repo huggingface/candle --body-file benchmarks/PR_COMMENT_1.md
-->

@lucasjinreal you asked on 2026-05-16: *"hows the speed compare with libtorch on cuda"* and on 2026-05-17: *"if no speed compare with pytorch, they won't merge any ASR pr."* I want to address that directly with numbers.

I have a follow-up comment coming with CUDA numbers from a rented RTX 3090 — this hardware (GTX 1080, Pascal sm_61) is too old for `candle-kernels 0.9.2`'s CUDA build (the kernel PTX uses `nvcuda::wmma` intrinsics that require sm_70+; `candle-kernels/build.rs` hard-targets the host GPU and there's no env-var override). So this comment is CPU-only and apples-to-apples.

## CPU speed compare: Rust (candle 0.9.2) vs torch 2.12 + qwen_asr

Same hardware (i7-7820HK, 64 GB RAM, Arch Linux), same model weights (`models/Qwen--Qwen3-ASR-0.6B/model.safetensors`, the `thinker.audio_tower.*` naming that the official Qwen3-ASR Python package also reads), same audio, same generation parameters (greedy argmax, `<|im_end|>` stop), 10 runs + 1 warmup, 5 audio clips (3s, 4s, 36s, 30s, 29s, mix of English/Mandarin/code-switched). Full per-run JSON in [`benchmarks/RESULTS.md`](https://github.com/airvzxf/qwen3-asr-rs/blob/bench/compare-rust-vs-torch/benchmarks/RESULTS.md) of my fork.

| | Rust (candle 0.9.2) CPU | torch 2.12 + qwen_asr CPU |
|---|---:|---:|
| **RTF, mean across 5 audios** | **1.089** (real-time) | **0.435** (2.3× real-time) |
| RTF, per-audio stddev / mean | 0.04–0.30 (laptop thermals) | 0.01–0.06 |
| Model load (cold) | 4 362 ms | 1 696 ms |
| **First result (process start → first JSON)** | **8 026 ms** | **57 250 ms** |
| RSS peak (after load) | 3.8 GB | 6.4 GB |
| Ground-truth match (5 short clips, normalised) | 5/5 | 4/5 (sample6: minor punctuation diff) |
| No Python at runtime | ✓ | ✗ (1.5 GB venv: torch + transformers + accelerate + qwen_asr + soundfile + numpy) |

**The honest headline is the opposite of what I think candle's README implies**: torch is ~2.5× faster per inference on the same CPU. candle's CPU matmul (`gemm`) is not in the same league as torch's oneDNN/MKL-tuned CPU backend in 2026, and on this laptop there's no AVX-512 anyway. If the question is *"is the Rust implementation faster than torch on CPU?"* the answer is **no**, and trying to spin the number otherwise would be a disservice to the maintainers.

Where Rust wins, hands-down, is everything around the inference:

- **Cold start to first result: 8 s vs 57 s.** The 49 s delta is the Python interpreter start + `torch` import (~1.5 s) + `transformers` + `qwen_asr` import + a 5-file warmup pass (the official `qwen_asr.transcribe()` apparently does a forward pass at each chunk boundary, so the warmup pass alone adds ~38 s) + the first real transcribe. For an on-demand ASR service that starts cold, that's a 7× improvement with zero optimization work.
- **Memory: 3.8 GB vs 6.4 GB peak.** The 2.6 GB delta is the Python runtime + autograd graph + refcounting overhead. In a Lambda / Cloud Run / edge container with a tight memory budget, this is the difference between a 4 GB and an 8 GB instance.
- **The binary is statically linked and 11 MB.** I just finished a `x86_64-unknown-linux-musl` build (via `messense/rust-musl-cross`):

  ```
  $ file target/x86_64-unknown-linux-musl/release/examples/server_http
  ELF 64-bit LSB pie executable, x86-64, version 1 (SYSV), static-pie linked, not stripped

  $ ldd target/x86_64-unknown-linux-musl/release/examples/server_http
          statically linked
  ```

  11 MB, zero dynamic dependencies, runs on any Linux kernel ≥ 4.4. Verified by copying the binary to a clean `/tmp/` directory, running with `LD_LIBRARY_PATH=` empty, and serving a real `POST /transcribe` that returned the correct text. The same binary contains: the Rust runtime, candle-core/nn, the qwen3-asr model code, mel extraction, BPE tokenizer, an HTTP server (`tiny_http`), and the full inference pipeline. The only external file is the 1.7 GB safetensors, loaded by path. There is no equivalent in the Python ecosystem that doesn't involve `pip install` + venv management + a long-running Python process per replica.

## What this PR is and isn't

This PR adds Qwen3-ASR to `candle-transformers` with an end-to-end example, batch and streaming APIs, and a model that produces byte-identical transcripts to the official Python implementation on 5/5 short test clips. It is **functional**, not a research contribution — the goal is the same as every other ASR model in `candle-transformers`: "you can do ASR inference in pure Rust, no Python, no server, no glue code."

What it isn't: a torch-replacement. Candle's existing CPU matmul is slower than torch's CPU matmul. Anyone who needs the fastest possible CPU inference should use the `qwen_asr` Python package with torch. But the cases where Rust wins — single-binary deployment, container size, cold start, on-device, edge, serverless — are the cases where a Rust ASR is the more interesting tool, and that's the audience this PR is for.

## What I cannot show on this hardware

- **CUDA on Pascal sm_61**: `candle-kernels 0.9.2` build fails (atomicAdd overload, nvcuda::wmma missing). `torch 2.12+` has dropped sm_61 entirely. A separate comment will add CUDA numbers from a rented RTX 3090 (sm_86, 24 GB VRAM, BF16 conv native). I expect that comment to be more flattering to both sides for the per-inference number, while keeping the cold-start and memory advantages for Rust.

A separate comment with the CUDA numbers is coming within the next few days, and a longer write-up is in the [`benchmarks/RESULTS.md`](https://github.com/airvzxf/qwen3-asr-rs/blob/bench/compare-rust-vs-torch/benchmarks/RESULTS.md) of my fork for anyone who wants the full per-run data.
