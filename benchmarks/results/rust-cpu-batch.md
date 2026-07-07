# Benchmark — batch-cpu

- **Label**: rust-cpu-batch
- **Generated**: 2026-07-07T02:40:14Z
- **Hardware**: Intel(R) Core(TM) i7-7820HK CPU @ 2.90GHz | 8 cores | 64249 MiB RAM
- **GPU**: NVIDIA GeForce GTX 1080 | 8192 | 580.173.02
- **OS**: Arch Linux (kernel 7.0.14-arch1-1)
- **Build**: x86_64-linux | features: hub
- **Model**: models/Qwen--Qwen3-ASR-0.6B (1789 MiB)
- **Mode**: batch | warmup: 1 | runs/audio: 10
- **Load**: 4362 ms | RSS after: 3738 MiB | VmPeak: 3755 MiB

## Per-file results

| File | Dur (s) | Mean (ms) | Med (ms) | Std | Min | P95 | P99 | Max | RTF (mean) | GT |
|------|---------:|----------:|---------:|----:|----:|----:|----:|----:|-----------:|:--:|
| sample1.wav | 3.40 | 3646 | 3587 | 196 | 3510 | 3974 | 4134 | 4174 | 1.072 | ✓ |
| sample2.wav | 4.01 | 4673 | 4567 | 527 | 3839 | 5406 | 5587 | 5633 | 1.165 | ✓ |
| sample4.wav | 36.42 | 45628 | 39890 | 13536 | 38610 | 69924 | 78796 | 81014 | 1.253 | ✓ |
| sample5.wav | 30.35 | 32298 | 32270 | 486 | 31560 | 32904 | 32994 | 33016 | 1.064 | ✓ |
| sample6.wav | 28.66 | 25712 | 25306 | 1023 | 24498 | 27383 | 27409 | 27415 | 0.897 | ✓ |

**Overall RTF**: 1.089 | total audio: 102.8s | total inference: 112.0s | GT matched: 5/5

**Cold start (process-relative)**: load 4362 ms | first inference 8026 ms | total 1228909 ms
