/// Benchmark binary for qwen3-asr.
///
/// Usage:
///   benchmark [--model-dir dir] [--runs N] [--audio-dir dir] [--warmup N]
///             [--mode batch|streaming] [--chunk-sec N]
///             [--json-out path] [--markdown-out path]
///             [--label text] [--skip-load]
///
/// Reports:
///   - Load time + RSS after load
///   - Cold start breakdown (load / first inference / total)
///   - Per-file transcription time (mean / median / stddev / min / p95 / p99 / max)
///   - Real-Time Factor (RTF) per file and overall
///   - Ground-truth match (case-insensitive, whitespace-collapsed)
///   - JSON + Markdown output for downstream analysis
///
/// Output to stdout: human-readable summary
/// Output to --json-out: machine-readable JSON (full per-run data)
/// Output to --markdown-out: Markdown table for the PR

use anyhow::Result;
use qwen3_asr::{AsrInference, StreamingOptions, TranscribeOptions};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

// ─── CLI ──────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Batch,
    Streaming,
}

struct Cli {
    model_dir: Option<PathBuf>,
    #[cfg(feature = "hub")]
    model_id: Option<String>,
    #[cfg(feature = "hub")]
    cache_dir: PathBuf,
    audio_dir: PathBuf,
    runs: usize,
    warmup: usize,
    mode: Mode,
    chunk_sec: f32,
    label: String,
    json_out: Option<PathBuf>,
    markdown_out: Option<PathBuf>,
    skip_load: bool,
}

fn parse_args() -> Cli {
    let args: Vec<String> = std::env::args().collect();
    let mut model_dir: Option<PathBuf> = None;
    #[cfg(feature = "hub")]
    let mut model_id: Option<String> = None;
    #[cfg(feature = "hub")]
    let mut cache_dir = PathBuf::from("models");
    let mut audio_dir = PathBuf::from("tests/fixtures/audio");
    let mut runs: usize = 10;
    let mut warmup: usize = 1;
    let mut mode = Mode::Batch;
    let mut chunk_sec: f32 = 2.0;
    let mut label = String::new();
    let mut json_out: Option<PathBuf> = None;
    let mut markdown_out: Option<PathBuf> = None;
    let mut skip_load = false;

    let mut i = 1;
    while i < args.len() {
        let arg = args[i].as_str();
        match arg {
            "--model-dir" => {
                model_dir = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--model-id" => {
                #[cfg(feature = "hub")]
                { model_id = Some(args[i + 1].clone()); }
                i += 2;
            }
            "--cache-dir" => {
                #[cfg(feature = "hub")]
                { cache_dir = PathBuf::from(&args[i + 1]); }
                i += 2;
            }
            "--audio-dir" => {
                audio_dir = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--runs" => {
                runs = args[i + 1].parse().unwrap_or(10);
                i += 2;
            }
            "--warmup" => {
                warmup = args[i + 1].parse().unwrap_or(1);
                i += 2;
            }
            "--mode" => {
                mode = match args[i + 1].as_str() {
                    "streaming" => Mode::Streaming,
                    _ => Mode::Batch,
                };
                i += 2;
            }
            "--chunk-sec" => {
                chunk_sec = args[i + 1].parse().unwrap_or(2.0);
                i += 2;
            }
            "--label" => {
                label = args[i + 1].clone();
                i += 2;
            }
            "--json-out" => {
                json_out = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--markdown-out" => {
                markdown_out = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--skip-load" => {
                skip_load = true;
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    if label.is_empty() {
        #[cfg(feature = "hub")]
        if let Some(id) = model_id.as_ref() {
            label = format!("hub({})", id);
        }
        if label.is_empty() {
            label = if let Some(ref d) = model_dir {
                format!("safetensors({})", d.display())
            } else {
                "safetensors(models/)".to_string()
            };
        }
    }

    Cli {
        model_dir,
        #[cfg(feature = "hub")]
        model_id,
        #[cfg(feature = "hub")]
        cache_dir,
        audio_dir,
        runs,
        warmup,
        mode,
        chunk_sec,
        label,
        json_out,
        markdown_out,
        skip_load,
    }
}

#[cfg(not(feature = "hub"))]
fn _cli_anchor(_x: &str) {}

// ─── Stats ────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Clone, Copy)]
struct Stats {
    n: usize,
    mean: f64,
    median: f64,
    stddev: f64,
    min: f64,
    p95: f64,
    p99: f64,
    max: f64,
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let n = sorted.len();
    let rank = p / 100.0 * (n - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = rank - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

fn compute_stats(values: &[f64]) -> Stats {
    let n = values.len();
    if n == 0 {
        return Stats { n: 0, mean: 0.0, median: 0.0, stddev: 0.0, min: 0.0, p95: 0.0, p99: 0.0, max: 0.0 };
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let variance = if n > 1 {
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64
    } else {
        0.0
    };
    let stddev = variance.sqrt();
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Stats {
        n,
        mean,
        median: percentile(&sorted, 50.0),
        stddev,
        min: sorted[0],
        p95: percentile(&sorted, 95.0),
        p99: percentile(&sorted, 99.0),
        max: sorted[n - 1],
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

// ─── Memory metrics (macOS + Linux) ───────────────────────────────────────────

/// Peak RSS in MiB from getrusage (macOS) or /proc/self/status (Linux).
/// On macOS ru_maxrss is bytes; on Linux it is kibibytes.
fn peak_rss_mib() -> f64 {
    #[cfg(target_os = "macos")]
    {
        extern "C" {
            fn getrusage(who: i32, usage: *mut Rusage) -> i32;
        }
        #[repr(C)]
        struct Rusage {
            ru_utime: [i64; 2],
            ru_stime: [i64; 2],
            ru_maxrss: i64,
            _pad: [i64; 13],
        }
        let mut u = Rusage { ru_utime: [0; 2], ru_stime: [0; 2], ru_maxrss: 0, _pad: [0; 13] };
        unsafe { getrusage(0, &mut u) };
        u.ru_maxrss as f64 / 1024.0 / 1024.0
    }
    #[cfg(not(target_os = "macos"))]
    {
        read_proc_kib("VmHWM")
    }
}

fn read_proc_kib(field: &str) -> f64 {
    if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
        for line in s.lines() {
            if line.starts_with(field) {
                if let Some(kb) = line.split_whitespace().nth(1).and_then(|v| v.parse::<f64>().ok()) {
                    return kb / 1024.0; // KiB → MiB
                }
            }
        }
    }
    0.0
}

fn read_meminfo_kib(field: &str) -> f64 {
    if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
        for line in s.lines() {
            if line.starts_with(field) {
                if let Some(kb) = line.split_whitespace().nth(1).and_then(|v| v.parse::<f64>().ok()) {
                    return kb / 1024.0; // KiB → MiB
                }
            }
        }
    }
    0.0
}

/// Current RSS in MiB (VmRSS) and VmPeak from /proc/self/status (Linux only).
fn current_rss_mib() -> f64 { read_proc_kib("VmRSS") }
fn vm_peak_mib() -> f64 { read_proc_kib("VmPeak") }

/// Physical memory footprint in MiB from task_info TASK_VM_INFO (macOS only).
fn phys_footprint_mib() -> f64 {
    #[cfg(target_os = "macos")]
    {
        extern "C" {
            fn task_info(target_task: u32, flavor: u32, task_info_out: *mut u64, task_info_outCnt: *mut u32) -> i32;
            fn mach_task_self() -> u32;
        }
        const TASK_VM_INFO: u32 = 22;
        let mut buf = [0u64; 40];
        let mut count: u32 = (buf.len() * 2) as u32;
        let ret = unsafe { task_info(mach_task_self(), TASK_VM_INFO, buf.as_mut_ptr(), &mut count) };
        if ret != 0 { return 0.0; }
        buf[18] as f64 / 1024.0 / 1024.0
    }
    #[cfg(not(target_os = "macos"))]
    { 0.0 }
}

// ─── Audio duration ───────────────────────────────────────────────────────────

fn wav_duration_secs(path: &Path) -> f64 {
    match hound::WavReader::open(path) {
        Ok(reader) => {
            let spec = reader.spec();
            reader.duration() as f64 / spec.sample_rate as f64
        }
        Err(_) => 0.0,
    }
}

fn load_samples(path: &Path) -> Vec<f32> {
    match hound::WavReader::open(path) {
        Ok(mut reader) => {
            let spec = reader.spec();
            if spec.bits_per_sample == 16 {
                reader.samples::<i16>()
                    .filter_map(|s| s.ok())
                    .map(|s| s as f32 / 32768.0)
                    .collect()
            } else if spec.bits_per_sample == 32 {
                reader.samples::<f32>()
                    .filter_map(|s| s.ok())
                    .collect()
            } else {
                Vec::new()
            }
        }
        Err(_) => Vec::new(),
    }
}

fn read_ground_truth(wav_path: &Path) -> Option<String> {
    let txt = wav_path.with_extension("txt");
    std::fs::read_to_string(&txt).ok().map(|s| s.trim().to_string())
}

fn normalize_text(s: &str) -> String {
    let lowered = s.to_lowercase();
    let mut out = String::with_capacity(lowered.len());
    let mut prev_ws = false;
    for c in lowered.chars() {
        if c.is_whitespace() {
            if !prev_ws && !out.is_empty() { out.push(' '); }
            prev_ws = true;
        } else {
            out.push(c);
            prev_ws = false;
        }
    }
    out.trim().to_string()
}

// ─── Hardware detection ───────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct Hardware {
    os: String,
    kernel: String,
    cpu: String,
    cores: usize,
    ram_total_mib: u64,
    gpu: String,
}

fn detect_hardware() -> Hardware {
    let os = std::fs::read_to_string("/etc/os-release")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("PRETTY_NAME="))
            .map(|l| l.trim_start_matches("PRETTY_NAME=").trim_matches('"').to_string()))
        .unwrap_or_else(|| std::env::consts::OS.to_string());
    let kernel = std::fs::read_to_string("/proc/version")
        .ok()
        .and_then(|s| s.split_whitespace().nth(2).map(|s| s.to_string()))
        .unwrap_or_default();
    let cpu = std::fs::read_to_string("/proc/cpuinfo").ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("model name"))
            .map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string()))
        .unwrap_or_default();
    let cores = std::fs::read_to_string("/proc/cpuinfo").ok()
        .map(|s| s.lines().filter(|l| l.starts_with("processor")).count())
        .unwrap_or(1);
    let ram_total_mib = read_meminfo_kib("MemTotal") as u64;
    let gpu = detect_gpu();
    Hardware { os, kernel, cpu, cores, ram_total_mib, gpu }
}

fn detect_gpu() -> String {
    // Try nvidia-smi first
    if let Ok(out) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
        .output()
    {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !s.is_empty() { return s.replace(", ", " | "); }
        }
    }
    "no GPU detected".to_string()
}

fn detect_features() -> Vec<String> {
    let mut f = Vec::new();
    #[cfg(feature = "cuda")] f.push("cuda".to_string());
    #[cfg(feature = "metal")] f.push("metal".to_string());
    #[cfg(feature = "hub")] f.push("hub".to_string());
    if f.is_empty() { f.push("(none)".to_string()); }
    f
}

// ─── JSON output schema ──────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    scenario: String,
    label: String,
    timestamp_utc: String,
    hardware: Hardware,
    build: BuildInfo,
    model: ModelInfo,
    load: LoadInfo,
    mode: String,
    warmup: usize,
    runs_per_audio: usize,
    audio_dir: String,
    audio: Vec<AudioReport>,
    aggregate: Aggregate,
}

#[derive(Debug, Serialize)]
struct BuildInfo {
    target: String,
    features: Vec<String>,
    rust_version: String,
    binary: String,
}

#[derive(Debug, Serialize)]
struct ModelInfo {
    path: String,
    size_mib: f64,
}

#[derive(Debug, Serialize)]
struct LoadInfo {
    elapsed_ms: f64,
    rss_mib_after: f64,
    rss_current_mib: f64,
    vm_peak_mib: f64,
    phys_footprint_mib: f64,
}

#[derive(Debug, Serialize)]
struct AudioReport {
    file: String,
    duration_s: f64,
    expected_text: Option<String>,
    matched_ground_truth: bool,
    runs: Vec<RunReport>,
    stats_ms: Stats,
    stats_rtf: Stats,
}

#[derive(Debug, Serialize)]
struct RunReport {
    elapsed_ms: f64,
    rtf: f64,
    text: String,
}

#[derive(Debug, Serialize)]
struct Aggregate {
    total_audio_s: f64,
    total_inference_s: f64,
    overall_rtf: f64,
    transcripts_matched: usize,
    transcripts_total: usize,
    all_texts_matched: bool,
    cold_start_ms: f64,
    cold_start_first_inference_ms: f64,
    cold_start_total_ms: f64,
}

fn file_size_mib(path: &Path) -> f64 {
    if let Ok(meta) = std::fs::metadata(path) {
        meta.len() as f64 / 1024.0 / 1024.0
    } else { 0.0 }
}

fn build_info() -> BuildInfo {
    let target = std::env::consts::ARCH.to_string() + "-" + std::env::consts::OS;
    let rust_version = std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let binary = std::env::args().next().unwrap_or_default();
    BuildInfo { target, features: detect_features(), rust_version, binary }
}

fn iso8601_utc_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    // Civil-from-days algorithm by Howard Hinnant (public domain).
    // https://howardhinnant.github.io/date_algorithms.html
    let z = (secs / 86400) as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;            // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;  // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);  // [0, 365]
    let mp = (5 * doy + 2) / 153;                    // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1;            // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 };   // [1, 12]
    let year = y + (if m <= 2 { 1 } else { 0 });
    let secs_today = secs % 86400;
    let h = secs_today / 3600;
    let mi = (secs_today % 3600) / 60;
    let s = secs_today % 60;
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", year, m, d, h, mi, s)
}

// month_day helper removed (replaced by Howard Hinnant civil-from-days above).

// ─── Markdown output ─────────────────────────────────────────────────────────

fn write_markdown(path: &Path, report: &Report) -> Result<()> {
    let mut s = String::new();
    s.push_str(&format!("# Benchmark — {}\n\n", report.scenario));
    s.push_str(&format!("- **Label**: {}\n", report.label));
    s.push_str(&format!("- **Generated**: {}\n", report.timestamp_utc));
    s.push_str(&format!("- **Hardware**: {} | {} cores | {} MiB RAM\n", report.hardware.cpu, report.hardware.cores, report.hardware.ram_total_mib));
    s.push_str(&format!("- **GPU**: {}\n", report.hardware.gpu));
    s.push_str(&format!("- **OS**: {} (kernel {})\n", report.hardware.os, report.hardware.kernel));
    s.push_str(&format!("- **Build**: {} | features: {}\n", report.build.target, report.build.features.join(",")));
    s.push_str(&format!("- **Model**: {} ({:.0} MiB)\n", report.model.path, report.model.size_mib));
    s.push_str(&format!("- **Mode**: {} | warmup: {} | runs/audio: {}\n", report.mode, report.warmup, report.runs_per_audio));
    s.push_str(&format!("- **Load**: {:.0} ms | RSS after: {:.0} MiB | VmPeak: {:.0} MiB\n\n",
        report.load.elapsed_ms, report.load.rss_mib_after, report.load.vm_peak_mib));

    s.push_str("## Per-file results\n\n");
    s.push_str("| File | Dur (s) | Mean (ms) | Med (ms) | Std | Min | P95 | P99 | Max | RTF (mean) | GT |\n");
    s.push_str("|------|---------:|----------:|---------:|----:|----:|----:|----:|----:|-----------:|:--:|\n");
    for a in &report.audio {
        let gt = if a.matched_ground_truth { "✓" } else { "✗" };
        s.push_str(&format!("| {} | {:.2} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} | {:.3} | {} |\n",
            a.file, a.duration_s,
            a.stats_ms.mean, a.stats_ms.median, a.stats_ms.stddev,
            a.stats_ms.min, a.stats_ms.p95, a.stats_ms.p99, a.stats_ms.max,
            a.stats_rtf.mean, gt));
    }
    s.push_str(&format!("\n**Overall RTF**: {:.3} | total audio: {:.1}s | total inference: {:.1}s | GT matched: {}/{}\n",
        report.aggregate.overall_rtf, report.aggregate.total_audio_s, report.aggregate.total_inference_s,
        report.aggregate.transcripts_matched, report.aggregate.transcripts_total));
    s.push_str(&format!("\n**Cold start (process-relative)**: load {:.0} ms | first inference {:.0} ms | total {:.0} ms\n",
        report.aggregate.cold_start_ms, report.aggregate.cold_start_first_inference_ms, report.aggregate.cold_start_total_ms));

    std::fs::write(path, s)?;
    Ok(())
}

// ─── Benchmark logic ─────────────────────────────────────────────────────────

fn collect_audio_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut v: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("wav"))
        .collect();
    v.sort();
    Ok(v)
}

fn truncate_for_display(s: &str, max_chars: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() > max_chars {
        format!("{}…", chars[..max_chars].iter().collect::<String>())
    } else {
        s.to_string()
    }
}

fn run_batch(engine: &AsrInference, cli: &Cli, audio_files: &[PathBuf], first_iter_global: &mut bool, process_t0: Instant) -> Result<(Vec<AudioReport>, Aggregate, Duration)> {
    let mut audio_reports = Vec::new();
    let mut total_audio_secs = 0.0;
    let mut total_infer_secs = 0.0;
    let mut matched_count = 0usize;
    let mut cold_start_first_inference_ms = 0.0f64;
    let cold_start_total_t0 = Instant::now();
    let mut first_result_at: Option<Instant> = None;

    for wav in audio_files {
        let wav_str = wav.to_str().unwrap_or_default();
        let audio_dur = wav_duration_secs(wav);
        let expected = read_ground_truth(wav);

        let mut per_run: Vec<RunReport> = Vec::new();
        let mut last_text = String::new();

        // Warmup
        for _ in 0..cli.warmup {
            let _ = engine.transcribe(wav_str, TranscribeOptions::default())?;
        }

        // Timed runs
        for _ in 0..cli.runs {
            let t = Instant::now();
            let result = engine.transcribe(wav_str, TranscribeOptions::default())?;
            let elapsed = t.elapsed();
            let rtf = if audio_dur > 0.0 { elapsed.as_secs_f64() / audio_dur } else { 0.0 };
            let text = result.text.trim().to_string();
            if *first_iter_global {
                let e = t.elapsed();
                cold_start_first_inference_ms = e.as_secs_f64() * 1000.0;
                first_result_at = Some(cold_start_total_t0 + e);
                *first_iter_global = false;
            }
            per_run.push(RunReport { elapsed_ms: ms(elapsed), rtf, text: text.clone() });
            last_text = text;
        }

        let stats_ms = compute_stats(&per_run.iter().map(|r| r.elapsed_ms).collect::<Vec<_>>());
        let stats_rtf = compute_stats(&per_run.iter().map(|r| r.rtf).collect::<Vec<_>>());
        let matched = if let Some(ref exp) = expected {
            normalize_text(&last_text) == normalize_text(exp)
        } else { false };
        if matched { matched_count += 1; }

        let short_name = wav.file_name().and_then(|s| s.to_str()).unwrap_or(wav_str).to_string();
        eprintln!("{:<20}  mean={:>7.1}ms  med={:>7.1}ms  std={:>6.1}  RTF={:.3}  GT={}  text={}",
            short_name, stats_ms.mean, stats_ms.median, stats_ms.stddev, stats_rtf.mean,
            if matched { "✓" } else { "?" },
            truncate_for_display(&last_text, 50));

        total_audio_secs += audio_dur;
        total_infer_secs += stats_ms.mean / 1000.0; // use mean per file

        audio_reports.push(AudioReport {
            file: short_name,
            duration_s: audio_dur,
            expected_text: expected,
            matched_ground_truth: matched,
            runs: per_run,
            stats_ms,
            stats_rtf,
        });
    }

    let overall_rtf = if total_audio_secs > 0.0 { total_infer_secs / total_audio_secs } else { 0.0 };
    let cold_start_total_ms = cold_start_total_t0.elapsed().as_secs_f64() * 1000.0;
    let total_files = audio_reports.len();
    let all_texts_matched = matched_count == total_files;

    // First result time relative to process start.
    let first_result_from_process_ms = first_result_at
        .map(|t| t.duration_since(process_t0).as_secs_f64() * 1000.0)
        .unwrap_or_else(|| cold_start_total_t0.elapsed().as_secs_f64() * 1000.0);

    Ok((audio_reports, Aggregate {
        total_audio_s: total_audio_secs,
        total_inference_s: total_infer_secs,
        overall_rtf,
        transcripts_matched: matched_count,
        transcripts_total: total_files,
        all_texts_matched,
        cold_start_ms: 0.0, // filled in by caller
        cold_start_first_inference_ms: first_result_from_process_ms,
        cold_start_total_ms,
    }, cold_start_total_t0.elapsed()))
}

fn run_streaming(engine: &AsrInference, cli: &Cli, audio_files: &[PathBuf], first_iter_global: &mut bool, process_t0: Instant) -> Result<(Vec<AudioReport>, Aggregate, Duration)> {
    let mut audio_reports = Vec::new();
    let mut total_audio_secs = 0.0;
    let mut total_infer_secs = 0.0;
    let mut matched_count = 0usize;
    let mut cold_start_first_inference_ms = 0.0f64;
    let cold_start_total_t0 = Instant::now();
    let mut first_result_at: Option<Instant> = None;

    for wav in audio_files {
        let audio_dur = wav_duration_secs(wav);
        let samples = load_samples(wav);
        let expected = read_ground_truth(wav);

        if samples.is_empty() {
            eprintln!("Skipping {} (could not load samples)", wav.display());
            continue;
        }

        let chunk_samples = (cli.chunk_sec * 16000.0) as usize;
        if chunk_samples == 0 {
            anyhow::bail!("--chunk-sec must produce a positive sample count (got {})", cli.chunk_sec);
        }

        // Warmup: one full streaming pass
        for _ in 0..cli.warmup {
            let mut state = engine.init_streaming(StreamingOptions::default().with_chunk_size_sec(cli.chunk_sec));
            for chunk in samples.chunks(chunk_samples) {
                let _ = engine.feed_audio(&mut state, chunk)?;
            }
            let _ = engine.finish_streaming(&mut state)?;
        }

        let mut per_run: Vec<RunReport> = Vec::new();
        let mut last_text = String::new();

        for _ in 0..cli.runs {
            let mut state = engine.init_streaming(StreamingOptions::default().with_chunk_size_sec(cli.chunk_sec));
            let t = Instant::now();
            let mut emitted = false;
            for chunk in samples.chunks(chunk_samples) {
                if let Some(_r) = engine.feed_audio(&mut state, chunk)? {
                    if *first_iter_global {
                        let e = t.elapsed();
                        cold_start_first_inference_ms = e.as_secs_f64() * 1000.0;
                        first_result_at = Some(cold_start_total_t0 + e);
                        *first_iter_global = false;
                    }
                    emitted = true;
                }
            }
            let final_result = engine.finish_streaming(&mut state)?;
            let elapsed = t.elapsed();
            let rtf = if audio_dur > 0.0 { elapsed.as_secs_f64() / audio_dur } else { 0.0 };
            let text = final_result.text.trim().to_string();
            if *first_iter_global && !emitted {
                let e = t.elapsed();
                cold_start_first_inference_ms = e.as_secs_f64() * 1000.0;
                first_result_at = Some(cold_start_total_t0 + e);
                *first_iter_global = false;
            }
            per_run.push(RunReport { elapsed_ms: ms(elapsed), rtf, text: text.clone() });
            last_text = text;
        }

        let stats_ms = compute_stats(&per_run.iter().map(|r| r.elapsed_ms).collect::<Vec<_>>());
        let stats_rtf = compute_stats(&per_run.iter().map(|r| r.rtf).collect::<Vec<_>>());
        let matched = if let Some(ref exp) = expected {
            normalize_text(&last_text) == normalize_text(exp)
        } else { false };
        if matched { matched_count += 1; }

        let short_name = wav.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string();
        eprintln!("{:<20}  mean={:>7.1}ms  med={:>7.1}ms  std={:>6.1}  RTF={:.3}  GT={}  text={}",
            short_name, stats_ms.mean, stats_ms.median, stats_ms.stddev, stats_rtf.mean,
            if matched { "✓" } else { "?" },
            truncate_for_display(&last_text, 50));

        total_audio_secs += audio_dur;
        total_infer_secs += stats_ms.mean / 1000.0;

        audio_reports.push(AudioReport {
            file: short_name,
            duration_s: audio_dur,
            expected_text: expected,
            matched_ground_truth: matched,
            runs: per_run,
            stats_ms,
            stats_rtf,
        });
    }

    let overall_rtf = if total_audio_secs > 0.0 { total_infer_secs / total_audio_secs } else { 0.0 };
    let cold_start_total_ms = cold_start_total_t0.elapsed().as_secs_f64() * 1000.0;
    let total_files = audio_reports.len();
    let first_result_from_process_ms = first_result_at
        .map(|t| t.duration_since(process_t0).as_secs_f64() * 1000.0)
        .unwrap_or_else(|| cold_start_total_t0.elapsed().as_secs_f64() * 1000.0);
    Ok((audio_reports, Aggregate {
        total_audio_s: total_audio_secs,
        total_inference_s: total_infer_secs,
        overall_rtf,
        transcripts_matched: matched_count,
        transcripts_total: total_files,
        all_texts_matched: matched_count == total_files,
        cold_start_ms: 0.0,
        cold_start_first_inference_ms: first_result_from_process_ms,
        cold_start_total_ms,
    }, cold_start_total_t0.elapsed()))
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = parse_args();
    let process_t0 = Instant::now();

    let device = qwen3_asr::best_device();

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("  qwen3-asr benchmark — {}", cli.label);
    println!("═══════════════════════════════════════════════════════════════════════════════");
    let hw = detect_hardware();
    let bi = build_info();
    println!("  OS          : {} (kernel {})", hw.os, hw.kernel);
    println!("  CPU         : {} ({} cores)", hw.cpu, hw.cores);
    println!("  RAM total   : {} MiB", hw.ram_total_mib);
    println!("  GPU         : {}", hw.gpu);
    println!("  Device      : {:?}", device);
    println!("  Target      : {}", bi.target);
    println!("  Features    : {}", bi.features.join(", "));
    println!("  Rust        : {}", bi.rust_version);
    println!("  Mode        : {:?}", cli.mode);
    if cli.mode == Mode::Streaming {
        println!("  Chunk size  : {:.1} s", cli.chunk_sec);
    }
    println!("  Audio dir   : {}", cli.audio_dir.display());
    println!("  Warmup      : {}", cli.warmup);
    println!("  Runs/audio  : {}", cli.runs);
    println!();

    // ── Load model ────────────────────────────────────────────────────────────
    let (engine, load_info, model_info, model_load_elapsed) = if cli.skip_load {
        eprintln!("--skip-load: skipping model load (engine unavailable)");
        anyhow::bail!("--skip-load requires a pre-loaded engine, not supported by this binary");
    } else {
        let t_load = Instant::now();
        #[cfg(feature = "hub")]
        let engine = if let Some(ref id) = cli.model_id {
            AsrInference::from_pretrained(id, &cli.cache_dir, device)?
        } else {
            let dir = cli.model_dir.as_deref().unwrap_or_else(|| Path::new("models"));
            AsrInference::load(dir, device)?
        };
        #[cfg(not(feature = "hub"))]
        let engine = {
            let dir = cli.model_dir.as_deref().unwrap_or_else(|| Path::new("models"));
            AsrInference::load(dir, device)?
        };
        let load_time = t_load.elapsed();
        let dir = cli.model_dir.as_deref().unwrap_or_else(|| Path::new("models"));
        let model_size = file_size_mib(&dir.join("model.safetensors"));
        let load_info = LoadInfo {
            elapsed_ms: ms(load_time),
            rss_mib_after: peak_rss_mib(),
            rss_current_mib: current_rss_mib(),
            vm_peak_mib: vm_peak_mib(),
            phys_footprint_mib: phys_footprint_mib(),
        };
        let model_info = ModelInfo {
            path: dir.display().to_string(),
            size_mib: model_size,
        };
        (engine, load_info, model_info, ms(load_time))
    };

    println!("  ── Load ──");
    println!("  Elapsed     : {:.0} ms", load_info.elapsed_ms);
    println!("  RSS peak    : {:.0} MiB (process high-water)", load_info.rss_mib_after);
    println!("  RSS current : {:.0} MiB", load_info.rss_current_mib);
    println!("  VmPeak      : {:.0} MiB (Linux)", load_info.vm_peak_mib);
    println!("  Phys footpt : {:.0} MiB (macOS only)", load_info.phys_footprint_mib);
    println!();

    // ── Audio files ──────────────────────────────────────────────────────────
    let audio_files = collect_audio_files(&cli.audio_dir)?;
    if audio_files.is_empty() {
        eprintln!("No .wav files found in {}", cli.audio_dir.display());
        return Ok(());
    }
    println!("  Found {} audio files", audio_files.len());
    println!();

    // ── Run benchmarks ───────────────────────────────────────────────────────
    let mode_str = match cli.mode { Mode::Batch => "batch", Mode::Streaming => "streaming" };
    let mut first_iter_global = true;
    let (audio_reports, mut aggregate, _run_batch_elapsed) = match cli.mode {
        Mode::Batch => run_batch(&engine, &cli, &audio_files, &mut first_iter_global, process_t0)?,
        Mode::Streaming => run_streaming(&engine, &cli, &audio_files, &mut first_iter_global, process_t0)?,
    };

    // cold_start_ms is the load time (from process_t0 to load complete).
    // cold_start_first_inference_ms is set inside run_batch relative to process_t0.
    // cold_start_total_ms is from process_t0 to all runs done (set inside run_batch).
    aggregate.cold_start_ms = model_load_elapsed;

    let overall_rtf = aggregate.overall_rtf;
    let total_audio_secs = aggregate.total_audio_s;
    let total_infer_secs = aggregate.total_inference_s;

    println!();
    println!("  ── Aggregate ──");
    println!("  Overall RTF : {:.3}  ({} audio-sec / {} inference-sec)",
        overall_rtf, total_audio_secs, total_infer_secs);
    println!("  GT matched  : {}/{}", aggregate.transcripts_matched, aggregate.transcripts_total);
    println!();
    println!("  ── Cold start (process-relative, this binary invocation) ──");
    println!("  Model load     : {:>7.0} ms", aggregate.cold_start_ms);
    println!("  1st inference  : {:>7.0} ms (load + first transcribe)", aggregate.cold_start_first_inference_ms);
    println!("  Total process  : {:>7.0} ms (load + all {} runs + cleanup-free exit)", aggregate.cold_start_total_ms, cli.runs);
    println!();
    println!("  RTF = inference_time / audio_duration   (lower is better)");
    println!("  GT  = ground truth match (case-insensitive, whitespace-collapsed)");

    // ── Build and write report ───────────────────────────────────────────────
    let scenario = format!("{}-{}", mode_str, if bi.features.contains(&"cuda".to_string()) { "cuda" } else { "cpu" });
    let report = Report {
        schema_version: 1,
        scenario,
        label: cli.label.clone(),
        timestamp_utc: iso8601_utc_now(),
        hardware: hw,
        build: bi,
        model: model_info,
        load: load_info,
        mode: mode_str.to_string(),
        warmup: cli.warmup,
        runs_per_audio: cli.runs,
        audio_dir: cli.audio_dir.display().to_string(),
        audio: audio_reports,
        aggregate,
    };

    if let Some(ref jp) = cli.json_out {
        if let Some(parent) = jp.parent() { std::fs::create_dir_all(parent)?; }
        std::fs::write(jp, serde_json::to_string_pretty(&report)?)?;
        eprintln!("  Wrote: {}", jp.display());
    }
    if let Some(ref mp) = cli.markdown_out {
        if let Some(parent) = mp.parent() { std::fs::create_dir_all(parent)?; }
        write_markdown(mp, &report)?;
        eprintln!("  Wrote: {}", mp.display());
    }

    eprintln!("  Total wall time: {:.0} ms", process_t0.elapsed().as_secs_f64() * 1000.0);
    Ok(())
}
