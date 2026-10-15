//! Model registry, lookup and download.
//!
//! omafind uses two Model2Vec static models for the fast tier, the EmbeddingGemma 2 ONNX
//! export (text, vision and audio encoders) for the deep tier and, for transcripts, a
//! whisper.cpp model with its voice-activity detector.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};

use crate::config;

pub struct ModelSpec {
    /// Directory name under a models dir.
    pub dir: &'static str,
    pub repo: &'static str,
    /// Pinned commit, so a download is always the files listed in `CHECKSUMS`.
    pub rev: &'static str,
    pub files: &'static [&'static str],
}

pub const STATIC_TEXT: ModelSpec = ModelSpec {
    dir: "potion-base-8M",
    repo: "minishlab/potion-base-8M",
    rev: "bf8b056651a2c21b8d2565580b8569da283cab23",
    files: &["config.json", "model.safetensors", "tokenizer.json"],
};

pub const STATIC_CODE: ModelSpec = ModelSpec {
    dir: "potion-code-16M-v2",
    repo: "minishlab/potion-code-16M-v2",
    rev: "e9d2a44ca6a05ac6685f3b23709ea57eb7352d5b",
    files: &["config.json", "model.safetensors", "tokenizer.json"],
};

pub const GEMMA_DIR: &str = "embeddinggemma-2";
pub const GEMMA_REPO: &str = "onnx-community/embeddinggemma-2-ONNX";
pub const GEMMA_REV: &str = "daa72c51243991dfcaf9f9137d2c573d8f7790c0";

pub const WHISPER_DIR: &str = "whisper";
pub const WHISPER_REPO: &str = "ggerganov/whisper.cpp";
pub const WHISPER_REV: &str = "5359861c739e955e79d9a303bcbc70fb988958b1";
pub const VAD_REPO: &str = "ggml-org/whisper-vad";
pub const VAD_REV: &str = "9ffd54a1e1ee413ddf265af9913beaf518d1639b";
pub const VAD_FILE: &str = "ggml-silero-v6.2.0.bin";

/// Size and SHA-256 of every file `fetch` may download, as `dir/file`. Model files are
/// executable input to ONNX Runtime, so nothing unverified is put in place.
pub const CHECKSUMS: &[(&str, u64, &str)] = &[
    ("potion-base-8M/config.json", 202, "2a6ac0e9aaa356a68a5688070db78fc3a464fefe85d2f06a1905ce3718687553"),
    ("potion-base-8M/model.safetensors", 30236760, "f65d0f325faadc1e121c319e2faa41170d3fa07d8c89abd48ca5358d9a223de2"),
    ("potion-base-8M/tokenizer.json", 683666, "e67e803f624fb4d67dea1c730d06e1067e1b14d830e2c2202569e3ef0f70bb50"),
    ("potion-code-16M-v2/config.json", 59, "148e5691a6fcc553437156859701fba017a1ba5d340b170f17e0f3668fb861a7"),
    (
        "potion-code-16M-v2/model.safetensors",
        32490072,
        "75cf7a6c2171b230ad19b1e7d8e0b1aee86da5a02af8e7cacedd9921d227623c",
    ),
    ("potion-code-16M-v2/tokenizer.json", 1024340, "107bbdcbad4bff1d299b7a4c3a2fb17c52890688b7dd0e4c9deab79d3c4f3d45"),
    ("embeddinggemma-2/tokenizer.json", 32170510, "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4"),
    ("embeddinggemma-2/config.json", 5031, "8d011bfe08b5e345bbe0b81e5c6fd02c381920b345b986047bc2a33ce7b90d1d"),
    ("embeddinggemma-2/onnx/model_q4.onnx", 490742, "f9eeba97acddf139b8ee2ddf04bc30dceafa88de93fadf74d7644e0d61a477a9"),
    (
        "embeddinggemma-2/onnx/model_q4.onnx_data",
        174028800,
        "c3975f2d1ab7a1878ae31a7d7a9b7804a827aff3800b60dfceafce21cac3df49",
    ),
    (
        "embeddinggemma-2/onnx/vision_encoder_q4.onnx",
        159400,
        "7ea284226d4938f0ad921ab091f1d80a9ca699aa802984ef5cd5eec4f4761d96",
    ),
    (
        "embeddinggemma-2/onnx/vision_encoder_q4.onnx_data",
        108957696,
        "0a9d6c927334f152a33dd90874f65d6ea5228999abe6a450d3f7813677fa704c",
    ),
    (
        "embeddinggemma-2/onnx/model_quantized.onnx",
        495165,
        "d06edd601f851c633a2519304cbeb8dc6170d7ceb61b436625c17fb9b6e74953",
    ),
    (
        "embeddinggemma-2/onnx/model_quantized.onnx_data",
        313724928,
        "278a7ff1248c3618e4bd11a607fc54f7bdc7778854230f3956d3f86bd9db4f3b",
    ),
    (
        "embeddinggemma-2/onnx/vision_encoder_quantized.onnx",
        162495,
        "bb0de2df53a2448a32dc7908a187c168c8afd514d4d6f674f7f46024875fa4e3",
    ),
    (
        "embeddinggemma-2/onnx/vision_encoder_quantized.onnx_data",
        195228672,
        "3dabd69c0a36e9a8771ad82030dde74daa5a0e02b7047a5d3f3382b1137bab89",
    ),
    ("embeddinggemma-2/onnx/model.onnx", 421421, "bc47de15f81208a5c99e5ab10f746d5e33b51ea228b7dc0bef9c133a94f1c1c3"),
    (
        "embeddinggemma-2/onnx/model.onnx_data",
        1084170240,
        "9fd452bfc6916e5a92f080f8dada35598ca4512b21c65897d4c5974d5f9630a2",
    ),
    (
        "embeddinggemma-2/onnx/vision_encoder.onnx",
        121253,
        "a3fb44f8b5b152cfee6afe16e2cd69e774b7e1d3b822d1db1190ddf7b08b575e",
    ),
    (
        "embeddinggemma-2/onnx/vision_encoder.onnx_data",
        671026176,
        "ca8d28c99feb8129618f141978a27adae25090e0dadb68a30c235e191c01e37e",
    ),
    (
        "embeddinggemma-2/onnx/audio_encoder_quantized.onnx",
        249330,
        "04a9a9094ba76fb169e4c69be45a4b621580188654d6be59eab860eadcd42d3c",
    ),
    (
        "embeddinggemma-2/onnx/audio_encoder_quantized.onnx_data",
        340058624,
        "aa6361d898e1f1d6303f8dd2b5fabf4fd3629e15fd709e9cb06ac5cb9416a030",
    ),
    (
        "embeddinggemma-2/onnx/audio_encoder.onnx",
        204577,
        "e0a62590c49949fe0b1410f5dc52c92a185abb4c4e69e2c50a4e316103fdbc09",
    ),
    (
        "embeddinggemma-2/onnx/audio_encoder.onnx_data",
        1172750848,
        "6943b910d1d1e9b5dffd4ccad63f61aa308ad10b70ed128bb61886b119da1e3a",
    ),
    ("whisper/ggml-base-q5_1.bin", 59707625, "422f1ae452ade6f30a004d7e5c6a43195e4433bc370bf23fac9cc591f01a8898"),
    ("whisper/ggml-small-q5_1.bin", 190085487, "ae85e4a935d7a567bd102fe55afc16bb595bdb618e11b2fc7591bc08120411bb"),
    (
        "whisper/ggml-large-v3-turbo-q5_0.bin",
        574041195,
        "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2",
    ),
    ("whisper/ggml-silero-v6.2.0.bin", 885098, "2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987"),
];

/// whisper.cpp model file for `indexing.speech_model`.
pub fn whisper_file(model: &str) -> Option<&'static str> {
    Some(match model {
        "base" => "ggml-base-q5_1.bin",
        "small" => "ggml-small-q5_1.bin",
        "large-v3-turbo" => "ggml-large-v3-turbo-q5_0.bin",
        _ => return None,
    })
}

/// The speech model and the voice-activity model, if downloaded.
pub fn find_whisper(model: &str) -> Option<(PathBuf, Option<PathBuf>)> {
    let file = whisper_file(model)?;
    let dir = find(WHISPER_DIR, &[file.to_string()])?;
    let vad = find(WHISPER_DIR, &[VAD_FILE.to_string()]).map(|d| d.join(VAD_FILE));
    Some((dir.join(file), vad))
}

fn checksum(dir: &str, file: &str) -> Option<(u64, &'static str)> {
    let key = format!("{dir}/{file}");
    CHECKSUMS.iter().find(|c| c.0 == key).map(|c| (c.1, c.2))
}

/// Gemma files for one precision: tokenizer plus the text and vision graphs.
pub fn gemma_files(precision: &str) -> Vec<String> {
    let suffix = precision_suffix(precision);
    let mut files = vec!["tokenizer.json".to_string(), "config.json".to_string()];
    for graph in ["model", "vision_encoder"] {
        files.push(format!("onnx/{graph}{suffix}.onnx"));
        files.push(format!("onnx/{graph}{suffix}.onnx_data"));
    }
    files
}

pub fn precision_suffix(precision: &str) -> &'static str {
    match precision {
        "q4" => "_q4",
        "quantized" => "_quantized",
        _ => "",
    }
}

/// Finds the first models dir that holds every listed file of a model.
pub fn find(dir: &str, files: &[String]) -> Option<PathBuf> {
    config::model_dirs().into_iter().map(|d| d.join(dir)).find(|d| files.iter().all(|f| d.join(f).is_file()))
}

pub fn find_static(spec: &ModelSpec) -> Result<PathBuf> {
    let files: Vec<String> = spec.files.iter().map(|s| s.to_string()).collect();
    find(spec.dir, &files).with_context(|| format!("static model {} not found; run `omafind models fetch`", spec.dir))
}

/// Audio encoder graph for a text-model precision. 4-bit costs audio the most quality, so
/// q4 installs pair with the 8-bit audio encoder.
pub fn gemma_audio_files(precision: &str) -> Vec<String> {
    let suffix = precision_suffix(if precision == "q4" { "quantized" } else { precision });
    vec![format!("onnx/audio_encoder{suffix}.onnx"), format!("onnx/audio_encoder{suffix}.onnx_data")]
}

/// The audio encoder's graph, if it was downloaded.
pub fn find_gemma_audio(precision: &str) -> Option<PathBuf> {
    let files = gemma_audio_files(precision);
    find(GEMMA_DIR, &files).map(|d| d.join(&files[0]))
}

pub fn find_gemma(precision: &str) -> Result<PathBuf> {
    find(GEMMA_DIR, &gemma_files(precision)).with_context(|| {
        format!("EmbeddingGemma 2 ({precision}) not found; run `omafind models fetch --precision {precision}`")
    })
}

/// What `fetch` downloads besides the static text model.
pub struct FetchOpts {
    pub code: bool,
    /// Gemma's audio encoder.
    pub audio: bool,
    /// whisper.cpp model name (`indexing.speech_model`).
    pub speech: Option<String>,
}

/// Downloads the given models into the user models dir, skipping files already present.
pub fn fetch(precisions: &[String], opts: &FetchOpts) -> Result<()> {
    let base = config::user_models_dir();
    // (repo, rev, dir, file)
    let mut jobs: Vec<(&str, &str, &str, String)> = Vec::new();
    let mut specs = vec![&STATIC_TEXT];
    if opts.code {
        specs.push(&STATIC_CODE);
    }
    for spec in specs {
        for f in spec.files {
            jobs.push((spec.repo, spec.rev, spec.dir, f.to_string()));
        }
    }
    for p in precisions {
        anyhow::ensure!(
            matches!(p.as_str(), "q4" | "quantized" | "fp32"),
            "unknown precision {p} (q4, quantized, fp32)"
        );
        let mut files = gemma_files(p);
        if opts.audio {
            files.extend(gemma_audio_files(p));
        }
        for f in files {
            jobs.push((GEMMA_REPO, GEMMA_REV, GEMMA_DIR, f));
        }
    }
    if let Some(model) = &opts.speech {
        let file = whisper_file(model).with_context(|| format!("unknown speech model {model}"))?;
        jobs.push((WHISPER_REPO, WHISPER_REV, WHISPER_DIR, file.to_string()));
        jobs.push((VAD_REPO, VAD_REV, WHISPER_DIR, VAD_FILE.to_string()));
    }
    jobs.sort();
    jobs.dedup();
    for (repo, rev, dir, file) in jobs {
        let dest = base.join(dir).join(&file);
        let (size, sha) = checksum(dir, &file).with_context(|| format!("no checksum for {dir}/{file}"))?;
        if dest.metadata().is_ok_and(|m| m.len() == size) {
            eprintln!("  have {}", dest.display());
            continue;
        }
        download(&format!("https://huggingface.co/{repo}/resolve/{rev}/{file}"), &dest, size, sha)?;
    }
    eprintln!("models are in {}", base.display());
    Ok(())
}

fn download(url: &str, dest: &Path, size: u64, sha256: &str) -> Result<()> {
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let part = dest.with_extension(format!(
        "{}.part",
        dest.extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default()
    ));
    eprintln!("  get  {url}");
    let resp = ureq::get(url).call().with_context(|| format!("GET {url}"))?;
    let total: Option<u64> =
        resp.headers().get("content-length").and_then(|v| v.to_str().ok()).and_then(|v| v.parse().ok());
    let mut reader = resp.into_body().into_reader();
    let mut out = std::fs::File::create(&part)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut done: u64 = 0;
    let mut last_report = 0u64;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        hasher.update(&buf[..n]);
        done += n as u64;
        if done - last_report > 32 << 20 {
            last_report = done;
            match total {
                Some(t) => eprint!("\r       {:>6.1} / {:.1} MB", done as f64 / 1e6, t as f64 / 1e6),
                None => eprint!("\r       {:>6.1} MB", done as f64 / 1e6),
            }
        }
    }
    if last_report > 0 {
        eprintln!();
    }
    out.sync_all()?;
    let digest: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();
    if done != size || digest != sha256 {
        let _ = std::fs::remove_file(&part);
        anyhow::bail!("{url}: got {done} bytes with SHA-256 {digest}, expected {size} bytes with {sha256}");
    }
    std::fs::rename(&part, dest)?;
    Ok(())
}

/// Human-readable status of every model.
pub fn describe() -> Vec<(String, Option<PathBuf>)> {
    let mut out = Vec::new();
    for spec in [&STATIC_TEXT, &STATIC_CODE] {
        let files: Vec<String> = spec.files.iter().map(|s| s.to_string()).collect();
        out.push((spec.dir.to_string(), find(spec.dir, &files)));
    }
    for p in ["q4", "quantized", "fp32"] {
        out.push((format!("{GEMMA_DIR} ({p})"), find(GEMMA_DIR, &gemma_files(p))));
    }
    for p in ["quantized", "fp32"] {
        out.push((format!("{GEMMA_DIR} audio ({p})"), find_gemma_audio(p)));
    }
    for m in ["base", "small", "large-v3-turbo"] {
        out.push((format!("whisper {m}"), find_whisper(m).map(|w| w.0)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_fetchable_file_has_a_checksum() {
        for spec in [&STATIC_TEXT, &STATIC_CODE] {
            for f in spec.files {
                assert!(checksum(spec.dir, f).is_some(), "{}/{f}", spec.dir);
            }
        }
        for p in ["q4", "quantized", "fp32"] {
            for f in gemma_files(p).into_iter().chain(gemma_audio_files(p)) {
                assert!(checksum(GEMMA_DIR, &f).is_some(), "{GEMMA_DIR}/{f}");
            }
        }
        for m in ["base", "small", "large-v3-turbo"] {
            assert!(checksum(WHISPER_DIR, whisper_file(m).unwrap()).is_some(), "{m}");
        }
        assert!(checksum(WHISPER_DIR, VAD_FILE).is_some());
    }
}
