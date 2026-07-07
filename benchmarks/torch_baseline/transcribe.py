#!/usr/bin/env python3
"""
torch / qwen_asr baseline for the qwen3-asr-rs benchmark suite.

Mirrors the Rust benchmark binary's JSON schema (benchmarks/RESULTS.md §"JSON
output schema") so that the Rust and Python results can be diffed with the
same scripts.

Pipeline: load Qwen3-ASR-0.6B via the official `qwen_asr` package (which is
the Qwen team's reference implementation and matches the local
model.safetensors' `thinker.audio_tower.*` key naming), transcribe the same
5 audio files used by the Rust benchmark, repeat N times with 1 warmup,
compute per-file stats, and write a JSON report.

Usage:
    python transcribe.py --model-dir models/Qwen--Qwen3-ASR-0.6B \
        --audio-dir tests/fixtures/audio --runs 10 --warmup 1 \
        --label torch-cpu --json-out benchmarks/results/torch-cpu.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

# Force offline mode: we have the model on disk, no need to hit the Hub.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from qwen_asr import Qwen3ASRModel  # noqa: E402


# ─── stats (matches Rust compute_stats in src/bin/benchmark.rs) ───────────────

def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    rank = p / 100.0 * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    if lo == hi:
        return sorted_vals[lo]
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def compute_stats(values: list[float]) -> dict[str, float]:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": 0.0, "median": 0.0, "stddev": 0.0,
                "min": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    mean = float(np.mean(values))
    stddev = float(np.std(values, ddof=1)) if n > 1 else 0.0
    s = sorted(values)
    return {
        "n": n,
        "mean": mean,
        "median": percentile(s, 50.0),
        "stddev": stddev,
        "min": float(s[0]),
        "p95": percentile(s, 95.0),
        "p99": percentile(s, 99.0),
        "max": float(s[-1]),
    }


# ─── hardware / system info (matches Rust detect_hardware) ────────────────────

def read_text(path: Path) -> str:
    try:
        return path.read_text()
    except FileNotFoundError:
        return ""


def detect_hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "os": "linux",
        "kernel": "",
        "cpu": "",
        "cores": 1,
        "ram_total_mib": 0,
        "gpu": "no GPU detected",
    }
    os_release = read_text(Path("/etc/os-release"))
    for line in os_release.splitlines():
        if line.startswith("PRETTY_NAME="):
            info["os"] = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("ID="):
            info["os_id"] = line.split("=", 1)[1].strip().strip('"')
    version = read_text(Path("/proc/version"))
    if version:
        info["kernel"] = version.split()[2] if len(version.split()) > 2 else ""
    cpuinfo = read_text(Path("/proc/cpuinfo"))
    for line in cpuinfo.splitlines():
        if line.startswith("model name") and not info["cpu"]:
            info["cpu"] = line.split(":", 1)[1].strip()
    info["cores"] = sum(1 for l in cpuinfo.splitlines() if l.startswith("processor"))
    meminfo = read_text(Path("/proc/meminfo"))
    for line in meminfo.splitlines():
        if line.startswith("MemTotal:"):
            kb = line.split()[1]
            info["ram_total_mib"] = int(kb) // 1024
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
            mem_mib = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            info["gpu"] = f"{name} | {mem_mib} | CUDA built={torch.version.cuda}"
        except Exception as e:
            info["gpu"] = f"CUDA available but query failed: {e}"
    return info


def detect_build() -> dict[str, Any]:
    return {
        "target": f"{os.uname().machine}-{os.uname().sysname}".lower(),
        "features": ["qwen_asr", f"torch=={torch.__version__}"],
        "rust_version": "",  # not applicable to Python baseline
        "binary": sys.executable,
    }


# ─── text normalization (matches Rust normalize_text) ─────────────────────────

_WS_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    return _WS_RE.sub(" ", s.lower()).strip()


# ─── ground truth loading ─────────────────────────────────────────────────────

def read_ground_truth(wav: Path) -> str | None:
    txt = wav.with_suffix(".txt")
    if not txt.exists():
        return None
    return txt.read_text().strip()


def wav_duration(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


# ─── report schema (mirrors Rust Report) ──────────────────────────────────────

@dataclass
class RunReport:
    elapsed_ms: float
    rtf: float
    text: str


@dataclass
class AudioReport:
    file: str
    duration_s: float
    expected_text: str | None
    matched_ground_truth: bool
    runs: list[RunReport] = field(default_factory=list)
    stats_ms: dict[str, float] = field(default_factory=dict)
    stats_rtf: dict[str, float] = field(default_factory=dict)


@dataclass
class Aggregate:
    total_audio_s: float = 0.0
    total_inference_s: float = 0.0
    overall_rtf: float = 0.0
    transcripts_matched: int = 0
    transcripts_total: int = 0
    all_texts_matched: bool = False
    cold_start_ms: float = 0.0
    cold_start_first_inference_ms: float = 0.0
    cold_start_total_ms: float = 0.0


# ─── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path,
                   default=Path("models/Qwen--Qwen3-ASR-0.6B"),
                   help="Path to the local Qwen3-ASR-0.6B directory.")
    p.add_argument("--audio-dir", type=Path,
                   default=Path("tests/fixtures/audio"),
                   help="Directory with .wav files + paired .txt ground truth.")
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--label", type=str, default="torch-baseline")
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto",
                   help="Inference device. cuda requires GPU compute capability the "
                        "installed torch wheel supports (sm_75+ for torch 2.12).")
    return p.parse_args()


def pick_device(choice: str) -> str:
    if choice == "cpu":
        return "cpu"
    if choice == "cuda":
        if not torch.cuda.is_available():
            print("ERROR: --device cuda requested but torch.cuda.is_available()=False", file=sys.stderr)
            sys.exit(1)
        return "cuda"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> int:
    args = parse_args()
    if not args.model_dir.exists():
        print(f"ERROR: model dir not found: {args.model_dir}", file=sys.stderr)
        return 1
    if not args.audio_dir.exists():
        print(f"ERROR: audio dir not found: {args.audio_dir}", file=sys.stderr)
        return 1

    device = pick_device(args.device)
    scenario = f"batch-{device}"

    print("=" * 79)
    print(f"  qwen3-asr baseline (qwen_asr package) — {args.label}")
    print("=" * 79)
    hw = detect_hardware()
    bi = detect_build()
    print(f"  Device      : {device} (torch {torch.__version__}, CUDA built {torch.version.cuda})")
    print(f"  OS          : {hw['os']} (kernel {hw['kernel']})")
    print(f"  CPU         : {hw['cpu']} ({hw['cores']} cores)")
    print(f"  RAM total   : {hw['ram_total_mib']} MiB")
    print(f"  GPU         : {hw['gpu']}")
    print(f"  Audio dir   : {args.audio_dir}")
    print(f"  Warmup      : {args.warmup}")
    print(f"  Runs/audio  : {args.runs}")
    print()

    process_t0 = time.perf_counter()

    # ── Load model ───────────────────────────────────────────────────────────
    t_load = time.perf_counter()
    dtype = torch.float32
    try:
        model = Qwen3ASRModel.from_pretrained(
            str(args.model_dir),
            dtype=dtype,
            device_map=device,
        )
    except Exception as e:
        print(f"ERROR: failed to load model on {device}: {e}", file=sys.stderr)
        if device == "cuda":
            print("Falling back to CPU. This hardware's GPU compute capability may not be "
                  "supported by the installed torch wheel.", file=sys.stderr)
            device = "cpu"
            model = Qwen3ASRModel.from_pretrained(
                str(args.model_dir),
                dtype=dtype,
                device_map="cpu",
            )
        else:
            raise
    load_time = time.perf_counter() - t_load

    # rss after load
    rss_after = 0
    try:
        import resource
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_after = rss_kb // 1024  # macOS: bytes; Linux: KiB. Different scales but ok as indicator.
    except ImportError:
        pass

    vm_peak = 0
    if Path("/proc/self/status").exists():
        for line in read_text(Path("/proc/self/status")).splitlines():
            if line.startswith("VmPeak:"):
                vm_peak = int(line.split()[1]) // 1024
                break

    safetensors = args.model_dir / "model.safetensors"
    size_mib = safetensors.stat().st_size / (1024 * 1024) if safetensors.exists() else 0

    print(f"  ── Load ({device}) ──")
    print(f"  Elapsed     : {load_time * 1000:.0f} ms")
    print(f"  RSS peak    : {rss_after} MiB (high-water; units differ by OS)")
    print(f"  VmPeak      : {vm_peak} MiB (Linux)")
    print(f"  Model size  : {size_mib:.0f} MiB")
    print()

    # ── Collect audio files ──────────────────────────────────────────────────
    audio_files = sorted(args.audio_dir.glob("*.wav"))
    if not audio_files:
        print(f"No .wav files in {args.audio_dir}", file=sys.stderr)
        return 1
    print(f"  Found {len(audio_files)} audio files")
    print()

    # ── Warmup pass (separate from timed runs) ──────────────────────────────
    for w in range(args.warmup):
        for wav in audio_files:
            samples, sr = sf.read(str(wav), dtype="float32")
            _ = model.transcribe(audio=(samples, int(sr)))

    # ── Timed runs ──────────────────────────────────────────────────────────
    audio_reports: list[AudioReport] = []
    total_audio_s = 0.0
    total_inference_s = 0.0
    matched = 0
    cold_start_first_inference_ms = 0.0
    first_inference_set = False

    for wav in audio_files:
        samples, sr = sf.read(str(wav), dtype="float32")
        duration = wav_duration(wav)
        expected = read_ground_truth(wav)
        per_run: list[RunReport] = []
        last_text = ""

        for run_idx in range(args.runs):
            t = time.perf_counter()
            results = model.transcribe(audio=(samples, int(sr)))
            elapsed = time.perf_counter() - t
            rtf = elapsed / duration if duration > 0 else 0.0
            text = results[0].text.strip() if isinstance(results, list) and results else ""
            per_run.append(RunReport(elapsed_ms=elapsed * 1000, rtf=rtf, text=text))
            last_text = text
            if not first_inference_set:
                cold_start_first_inference_ms = (time.perf_counter() - process_t0) * 1000
                first_inference_set = True

        stats_ms = compute_stats([r.elapsed_ms for r in per_run])
        stats_rtf = compute_stats([r.rtf for r in per_run])
        gt_match = (expected is not None and normalize_text(last_text) == normalize_text(expected))
        if gt_match:
            matched += 1

        short = wav.name
        text_disp = last_text if len(last_text) <= 50 else last_text[:47] + "…"
        print(f"{short:<20}  mean={stats_ms['mean']:>7.0f}ms  med={stats_ms['median']:>7.0f}ms  "
              f"std={stats_ms['stddev']:>6.0f}  RTF={stats_rtf['mean']:.3f}  GT={'✓' if gt_match else '?'}  text={text_disp}")

        audio_reports.append(AudioReport(
            file=short, duration_s=duration, expected_text=expected,
            matched_ground_truth=gt_match, runs=per_run,
            stats_ms=stats_ms, stats_rtf=stats_rtf,
        ))
        total_audio_s += duration
        total_inference_s += stats_ms["mean"] / 1000.0

    overall_rtf = total_inference_s / total_audio_s if total_audio_s > 0 else 0.0
    cold_start_total_ms = (time.perf_counter() - process_t0) * 1000

    print()
    print("  ── Aggregate ──")
    print(f"  Overall RTF : {overall_rtf:.3f}  ({total_audio_s:.2f} audio-sec / {total_inference_s:.2f} inference-sec)")
    print(f"  GT matched  : {matched}/{len(audio_reports)}")
    print()
    print("  ── Cold start (process-relative) ──")
    print(f"  Model load     : {load_time * 1000:>7.0f} ms")
    print(f"  1st inference  : {cold_start_first_inference_ms:>7.0f} ms")
    print(f"  Total process  : {cold_start_total_ms:>7.0f} ms")

    # ── Build report ────────────────────────────────────────────────────────
    aggregate = Aggregate(
        total_audio_s=total_audio_s,
        total_inference_s=total_inference_s,
        overall_rtf=overall_rtf,
        transcripts_matched=matched,
        transcripts_total=len(audio_reports),
        all_texts_matched=(matched == len(audio_reports)),
        cold_start_ms=load_time * 1000,
        cold_start_first_inference_ms=cold_start_first_inference_ms,
        cold_start_total_ms=cold_start_total_ms,
    )

    report = {
        "schema_version": 1,
        "scenario": scenario,
        "label": args.label,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hardware": hw,
        "build": bi,
        "model": {"path": str(args.model_dir), "size_mib": size_mib},
        "load": {
            "elapsed_ms": load_time * 1000,
            "rss_mib_after": float(rss_after),
            "rss_current_mib": float(rss_after),
            "vm_peak_mib": float(vm_peak),
            "phys_footprint_mib": 0.0,
        },
        "mode": "batch",
        "warmup": args.warmup,
        "runs_per_audio": args.runs,
        "audio_dir": str(args.audio_dir),
        "audio": [asdict(a) for a in audio_reports],
        "aggregate": asdict(aggregate),
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"  Wrote: {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
