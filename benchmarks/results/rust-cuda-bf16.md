# Benchmark — batch-cuda

- **Label**: rust-cuda-bf16
- **Generated**: 2026-07-07T06:15:17Z
- **Hardware**: AMD Ryzen Threadripper 2920X 12-Core Processor | 24 cores | 64140 MiB RAM
- **GPU**: NVIDIA GeForce RTX 3090 | 24576 | 570.133.20
- **OS**: Ubuntu 24.04.4 LTS (kernel 5.15.0-139-generic)
- **Build**: x86_64-linux | features: cuda
- **Model**: /root/models/Qwen--Qwen3-ASR-0.6B (1789 MiB)
- **Mode**: batch | warmup: 1 | runs/audio: 10
- **Load**: 1656 ms | RSS after: 2057 MiB | VmPeak: 56037 MiB

## Per-file results

| File | Dur (s) | Mean (ms) | Med (ms) | Std | Min | P95 | P99 | Max | RTF (mean) | GT |
|------|---------:|----------:|---------:|----:|----:|----:|----:|----:|-----------:|:--:|
| sample1.wav | 3.40 | 165 | 166 | 2 | 161 | 167 | 167 | 167 | 0.048 | ✓ |
| sample2.wav | 4.01 | 166 | 165 | 3 | 164 | 171 | 172 | 173 | 0.041 | ✓ |
| sample4.wav | 36.42 | 1080 | 1080 | 6 | 1073 | 1089 | 1092 | 1093 | 0.030 | ✓ |
| sample5.wav | 30.35 | 958 | 953 | 11 | 946 | 974 | 975 | 975 | 0.032 | ✓ |
| sample6.wav | 28.66 | 847 | 846 | 2 | 844 | 850 | 850 | 850 | 0.030 | ✓ |

**Overall RTF**: 0.031 | total audio: 102.8s | total inference: 3.2s | GT matched: 5/5

**Cold start (process-relative)**: load 1656 ms | first inference 2161 ms | total 35667 ms
