//! Minimal HTTP server wrapping the qwen3-asr library.
//!
//! Demonstrates the "no Python, no separate service" deployment story:
//! one static-ish binary that loads the model, opens a port, and serves
//! `POST /transcribe` requests with WAV bodies. The total memory footprint
//! is the model + the running binary; no Python interpreter, no torch
//! runtime, no extra orchestration.
//!
//! Usage:
//!   QWEN3_ASR_MODEL_DIR=models/Qwen--Qwen3-ASR-0.6B \
//!       cargo run --example server_http --release --no-default-features
//!
//! Then in another terminal:
//!   curl -X POST --data-binary @tests/fixtures/audio/sample1.wav \
//!        -H "Content-Type: audio/wav" \
//!        http://127.0.0.1:3000/transcribe
//!   # → {"language":"English","text":"The quick brown fox jumps over the lazy dog."}
//!
//! Cold start measurement (run with `time`):
//!   time QWEN3_ASR_MODEL_DIR=... cargo run --example server_http --release
//!   # → "ready on :3000" line in stderr marks the moment the server is up.

use std::io::Cursor;
use std::path::PathBuf;
use std::time::Instant;

use qwen3_asr::{AsrInference, TranscribeOptions};

fn main() -> anyhow::Result<()> {
    let process_t0 = Instant::now();

    let model_dir = std::env::var("QWEN3_ASR_MODEL_DIR")
        .map(PathBuf::from)
        .expect("Set QWEN3_ASR_MODEL_DIR to the model directory");
    let host = std::env::var("QWEN3_ASR_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port: u16 = std::env::var("QWEN3_ASR_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000);

    eprintln!("[server] loading model from {} ...", model_dir.display());
    let t_load = Instant::now();
    let device = qwen3_asr::best_device();
    let engine = AsrInference::load(&model_dir, device.clone())?;
    let load_elapsed_ms = t_load.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "[server] model loaded in {:.0} ms on {:?}",
        load_elapsed_ms, device
    );

    let server = tiny_http::Server::http((host.as_str(), port))
        .map_err(|e| anyhow::anyhow!("bind {}:{}: {}", host, port, e))?;
    let ready_at_ms = process_t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "[server] ready on http://{}:{} in {:.0} ms (load + bind + handler setup)",
        host, port, ready_at_ms
    );
    eprintln!("[server] POST a WAV file to /transcribe to get the transcription JSON");

    for mut request in server.incoming_requests() {
        match request.method().as_str() {
            "POST" if request.url() == "/transcribe" => {
                let t_req = Instant::now();
                let mut body = Vec::new();
                if let Err(e) = std::io::Read::read_to_end(
                    &mut request.as_reader(),
                    &mut body,
                ) {
                    let _ = request.respond(tiny_http::Response::from_string(
                        format!("read body failed: {e}"),
                    )
                    .with_status_code(tiny_http::StatusCode(400)));
                    continue;
                }

                // Parse WAV from in-memory bytes.
                let cursor = Cursor::new(&body);
                let mut reader = match hound::WavReader::new(cursor) {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = request.respond(
                            tiny_http::Response::from_string(format!("invalid WAV: {e}"))
                                .with_status_code(tiny_http::StatusCode(400)),
                        );
                        continue;
                    }
                };
                let spec = reader.spec();
                let samples: Vec<f32> = if spec.bits_per_sample == 16 {
                    reader
                        .samples::<i16>()
                        .filter_map(|s| s.ok())
                        .map(|s| s as f32 / 32768.0)
                        .collect()
                } else if spec.bits_per_sample == 32 {
                    reader.samples::<f32>().filter_map(|s| s.ok()).collect()
                } else {
                    let _ = request.respond(
                        tiny_http::Response::from_string(format!(
                            "unsupported bits_per_sample={}",
                            spec.bits_per_sample
                        ))
                        .with_status_code(tiny_http::StatusCode(415)),
                    );
                    continue;
                };

                let result = engine.transcribe_samples(&samples, TranscribeOptions::default());
                let elapsed_ms = t_req.elapsed().as_secs_f64() * 1000.0;
                match result {
                    Ok(r) => {
                        let body = serde_json::json!({
                            "language": r.language,
                            "text": r.text.trim(),
                            "elapsed_ms": elapsed_ms,
                            "audio_duration_s": samples.len() as f64 / spec.sample_rate as f64,
                        })
                        .to_string();
                        let _ = request.respond(
                            tiny_http::Response::from_string(body)
                                .with_header(
                                    "Content-Type: application/json\r\n"
                                        .parse::<tiny_http::Header>()
                                        .unwrap(),
                                ),
                        );
                    }
                    Err(e) => {
                        let _ = request.respond(
                            tiny_http::Response::from_string(format!("inference failed: {e}"))
                                .with_status_code(tiny_http::StatusCode(500)),
                        );
                    }
                }
            }
            "GET" if request.url() == "/health" => {
                let _ = request.respond(tiny_http::Response::from_string("ok"));
            }
            _ => {
                let _ = request.respond(
                    tiny_http::Response::from_string(
                        "POST a WAV file to /transcribe; GET /health for liveness",
                    )
                    .with_status_code(tiny_http::StatusCode(404)),
                );
            }
        }
    }
    Ok(())
}
