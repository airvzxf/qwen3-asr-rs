//! Real-time streaming transcription from raw PCM input (stdin).
//!
//! Reads 16-bit mono 16 kHz PCM samples from stdin and transcribes in
//! real-time. Use with arecord or a WAV file piped via ffmpeg/sox.
//!
//! Usage:
//!   # From microphone (ALSA):
//!   arecord -f S16_LE -r 16000 -c 1 | \
//!     QWEN3_ASR_MODEL_DIR=models/Qwen--Qwen3-ASR-0.6B \
//!     cargo run --example stdin_streaming --release --no-default-features --features cuda
//!
//!   # From WAV file (for testing):
//!   sox test.wav -t raw -r 16000 -c 1 -b 16 -e signed - | \
//!     QWEN3_ASR_MODEL_DIR=models/Qwen--Qwen3-ASR-0.6B \
//!     cargo run --example stdin_streaming --release --no-default-features --features cuda
//!
//! Environment:
//!   QWEN3_ASR_MODEL_DIR — path to the safetensors model directory
//!   CHUNK_SIZE_SEC      — audio chunk duration in seconds (default: 2.0)
//!
//! Press Ctrl+C to stop.

use anyhow::Result;
use qwen3_asr::StreamingOptions;
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<()> {
    let model_dir = std::env::var("QWEN3_ASR_MODEL_DIR")
        .map(PathBuf::from)
        .expect("Set QWEN3_ASR_MODEL_DIR to the model directory");

    let chunk_size_sec: f32 = std::env::var("CHUNK_SIZE_SEC")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2.0);

    eprintln!("Loading model from: {}", model_dir.display());
    let t0 = Instant::now();
    let device = qwen3_asr::best_device();
    let engine = qwen3_asr::AsrInference::load(&model_dir, device)?;
    eprintln!("Model loaded in {:.2}s", t0.elapsed().as_secs_f64());
    eprintln!("Chunk size: {:.1}s — stream is live, speak now!", chunk_size_sec);

    let stream_opts = StreamingOptions::default().with_chunk_size_sec(chunk_size_sec);
    let mut state = engine.init_streaming(stream_opts);

    let chunk_samples = (chunk_size_sec * 16000.0) as usize;
    let mut buf = vec![0u8; chunk_samples * 2]; // 16-bit samples = 2 bytes each
    let stdin = std::io::stdin();
    let mut handle = stdin.lock();
    let mut step = 0;
    let mut audio_secs = 0.0;
    let t_start = Instant::now();

    loop {
        match handle.read_exact(&mut buf) {
            Ok(()) => {
                // Convert i16 -> f32
                let samples: Vec<f32> = buf
                    .chunks_exact(2)
                    .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                    .collect();

                audio_secs += chunk_size_sec as f64;

                match engine.feed_audio(&mut state, &samples) {
                    Ok(Some(result)) => {
                        step += 1;
                        println!(
                            "[{:>2} | {:>5.1}s] {}",
                            step, audio_secs, result.text
                        );
                    }
                    Ok(None) => {
                        // Buffered, not enough for a chunk
                    }
                    Err(e) => {
                        eprintln!("ERROR at step {}: {}", step + 1, e);
                        break;
                    }
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                // End of input
                eprintln!("\n=== End of stream (EOF) ===");
                break;
            }
            Err(e) => {
                eprintln!("Read error: {e}");
                break;
            }
        }
    }

    // Final flush
    let final_t0 = Instant::now();
    let final_result = engine.finish_streaming(&mut state)?;
    let final_elapsed = final_t0.elapsed().as_secs_f64();
    let total_elapsed = t_start.elapsed().as_secs_f64();

    println!("\n═══ FINAL TRANSCRIPT ═══");
    println!("{}", final_result.text);
    println!("\nLanguage    : {}", final_result.language);
    println!("Audio       : {:.1}s", audio_secs);
    println!("Total time  : {:.2}s", total_elapsed);
    println!("Final flush : {:.3}s", final_elapsed);
    if total_elapsed > 0.0 && audio_secs > 0.0 {
        println!("RTF         : {:.3}x", total_elapsed / audio_secs);
    }

    Ok(())
}
