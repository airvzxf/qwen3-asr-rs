# Benchmark — batch-cuda

- **Label**: rust-cpu-batch
- **Generated**: 2026-07-07T06:05:13Z
- **Hardware**: AMD Ryzen Threadripper 2920X 12-Core Processor | 24 cores | 64140 MiB RAM
- **GPU**: NVIDIA GeForce RTX 3090 | 24576 | 570.133.20
- **OS**: Ubuntu 24.04.4 LTS (kernel 5.15.0-139-generic)
- **Build**: x86_64-linux | features: cuda
- **Model**: /root/models/Qwen--Qwen3-ASR-0.6B (1789 MiB)
- **Mode**: batch | warmup: 1 | runs/audio: 10
- **Load**: 1688 ms | RSS after: 2057 MiB | VmPeak: 56037 MiB

## Per-file results

| File | Dur (s) | Mean (ms) | Med (ms) | Std | Min | P95 | P99 | Max | RTF (mean) | GT |
|------|---------:|----------:|---------:|----:|----:|----:|----:|----:|-----------:|:--:|
| sample1.wav | 3.40 | 178 | 178 | 2 | 175 | 181 | 181 | 181 | 0.052 | ✓ |
| sample2.wav | 4.01 | 178 | 178 | 2 | 176 | 181 | 181 | 181 | 0.044 | ✓ |
| sample4.wav | 36.42 | 1202 | 1196 | 17 | 1187 | 1230 | 1238 | 1240 | 0.033 | ✓ |
| sample5.wav | 30.35 | 1049 | 1041 | 24 | 1030 | 1088 | 1104 | 1108 | 0.035 | ✓ |
| sample6.wav | 28.66 | 943 | 936 | 19 | 930 | 976 | 989 | 992 | 0.033 | ✓ |

**Overall RTF**: 0.035 | total audio: 102.8s | total inference: 3.6s | GT matched: 5/5

**Cold start (process-relative)**: load 1688 ms | first inference 2252 ms | total 39148 ms
