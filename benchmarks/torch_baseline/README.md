# torch / qwen_asr baseline

Baseline for the qwen3-asr-rs benchmark suite. Loads the same
Qwen3-ASR-0.6B weights used by the Rust crate via the official
[`qwen_asr`](https://pypi.org/project/qwen-asr/) Python package (the
reference implementation from the Qwen team), and transcribes the same
5 audio files used by `src/bin/benchmark.rs` in `tests/fixtures/audio/`.

## Why this is a fair baseline

| Aspect | Rust crate | torch baseline |
|---|---|---|
| Model weights | `models/Qwen--Qwen3-ASR-0.6B/model.safetensors` | same file |
| Weight naming | `thinker.audio_tower.*` (read by alan890104's fork) | same file, read by `qwen_asr` |
| Audio input | 16 kHz f32 mono | 16 kHz f32 mono |
| Tokenizer | `tokenizer.json` | same file |
| Generation | greedy argmax with `<|im_end|>` stop | greedy argmax with same stop |
| Output text | normalized (lowercase, whitespace-collapsed) | same normalization |

The two implementations should produce **byte-identical** transcripts on
the same audio, and the only difference being measured is the inference
time.

## Hardware caveat (important)

`qwen_asr` requires `torch` with `transformers >= 5.0` and
`accelerate`. The latest `torch` wheels (2.12+) only support **sm_75 and
newer** GPUs. On a **GTX 1080 (Pascal, sm_61)**, `torch.cuda.is_available()`
returns `True` but every CUDA op fails with `cudaErrorNoKernelImageForDevice`.

This means: on this hardware we can only run the torch baseline on
**CPU**. The CPU vs CPU comparison is therefore the apples-to-apples
one. Documented in `benchmarks/RESULTS.md`.

## How to run

From the project root, with the model already at
`models/Qwen--Qwen3-ASR-0.6B/`:

```bash
# one-time: create the venv
python3 -m venv benchmarks/.venv
source benchmarks/.venv/bin/activate
pip install -r benchmarks/torch_baseline/requirements.txt

# run CPU baseline
python benchmarks/torch_baseline/transcribe.py \
    --model-dir models/Qwen--Qwen3-ASR-0.6B \
    --audio-dir tests/fixtures/audio \
    --runs 10 --warmup 1 \
    --device cpu \
    --label torch-cpu \
    --json-out benchmarks/results/torch-cpu.json

# try CUDA (will fall back to CPU on sm_61 with current torch)
python benchmarks/torch_baseline/transcribe.py \
    --device cuda \
    --label torch-cuda-attempt \
    --json-out benchmarks/results/torch-cuda-attempt.json
```

## Output

`--json-out` produces the same JSON schema as the Rust `benchmark` binary
(see `src/bin/benchmark.rs::Report`), so the two can be diffed with the
same downstream tools.
