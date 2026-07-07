# Benchmark — batch-cuda

- **Label**: rust-cuda-f32
- **Generated**: 2026-07-07T06:14:02Z
- **Hardware**: AMD Ryzen Threadripper 2920X 12-Core Processor | 24 cores | 64140 MiB RAM
- **GPU**: NVIDIA GeForce RTX 3090 | 24576 | 570.133.20
- **OS**: Ubuntu 24.04.4 LTS (kernel 5.15.0-139-generic)
- **Build**: x86_64-linux | features: cuda
- **Model**: /root/models/Qwen--Qwen3-ASR-0.6B (1789 MiB)
- **Mode**: batch | warmup: 1 | runs/audio: 10
- **Load**: 1675 ms | RSS after: 2058 MiB | VmPeak: 56037 MiB

## Per-file results

| File | Dur (s) | Mean (ms) | Med (ms) | Std | Min | P95 | P99 | Max | RTF (mean) | GT |
|------|---------:|----------:|---------:|----:|----:|----:|----:|----:|-----------:|:--:|
| sample1.wav | 3.40 | 176 | 175 | 1 | 175 | 178 | 178 | 178 | 0.052 | ✓ |
| sample2.wav | 4.01 | 177 | 176 | 1 | 176 | 179 | 179 | 179 | 0.044 | ✓ |
| sample4.wav | 36.42 | 1186 | 1183 | 10 | 1177 | 1203 | 1204 | 1204 | 0.033 | ✓ |
| sample5.wav | 30.35 | 1022 | 1019 | 8 | 1013 | 1036 | 1037 | 1037 | 0.034 | ✓ |
| sample6.wav | 28.66 | 922 | 923 | 5 | 915 | 929 | 930 | 930 | 0.032 | ✓ |

**Overall RTF**: 0.034 | total audio: 102.8s | total inference: 3.5s | GT matched: 5/5

**Cold start (process-relative)**: load 1675 ms | first inference 2214 ms | total 38414 ms
