//! Demonstrates using `qwen3-asr` as a library dependency from a separate Rust
//! crate. This is the canonical "library consumer" pattern: a downstream
//! project depends on the crate, instantiates the engine, and calls the
//! public API.
//!
//! Build and run:
//!   cd benchmarks/lib_consumer
//!   QWEN3_ASR_MODEL_DIR=../../models/Qwen--Qwen3-ASR-0.6B \
//!       cargo run --release
//!
//! The result is intentionally minimal — 30 lines of real code — to make
//! the "library surface is small" point concrete for the PR. Compare to a
//! hypothetical Python equivalent which would need a virtualenv, a
//! requirements.txt with torch, transformers, accelerate, soundfile, plus
//! import boilerplate and dtype/device management.

use std::path::PathBuf;
use std::time::Instant;

use qwen3_asr::{AsrInference, TranscribeOptions};

fn main() -> anyhow::Result<()> {
    let t0 = Instant::now();

    // 1. Pick a device.
    let device = qwen3_asr::best_device();
    println!("[lib-consumer] device: {device:?}");

    // 2. Load the model.
    let model_dir = std::env::var("QWEN3_ASR_MODEL_DIR")
        .map(PathBuf::from)
        .expect("Set QWEN3_ASR_MODEL_DIR to the model directory");
    let engine = AsrInference::load(&model_dir, device)?;
    println!("[lib-consumer] loaded in {:.2}s", t0.elapsed().as_secs_f64());

    // 3. Transcribe a hard-coded sample.
    let wav = std::env::var("QWEN3_ASR_AUDIO")
        .unwrap_or_else(|_| "tests/fixtures/audio/sample1.wav".to_string());
    let t = Instant::now();
    let result = engine.transcribe(&wav, TranscribeOptions::default())?;
    let elapsed = t.elapsed();
    println!(
        "[lib-consumer] transcribed {} in {:.2}s",
        wav,
        elapsed.as_secs_f64()
    );
    println!("[lib-consumer] language: {}", result.language);
    println!("[lib-consumer] text:     {}", result.text.trim());

    Ok(())
}
