//! Speech to text through whisper.cpp's `whisper-cli`, fed ten-minute WAV pieces on stdin
//! (nothing is written to disk) and read back as JSON.

use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;
use std::time::Duration;

use anyhow::{Context, Result};

use super::{media, tool};
use crate::chunk::Chunk;

/// Audio handed to whisper-cli per run. Short enough to stop between pieces when the
/// indexer has other work, long enough that model loading doesn't dominate.
const PIECE_SECS: f64 = 600.0;
const TIMEOUT: Duration = Duration::from_secs(1800);
/// Transcript chunks are about this long, like text chunks.
const CHUNK_CHARS: usize = 900;

pub fn available() -> bool {
    static FOUND: OnceLock<bool> = OnceLock::new();
    *FOUND.get_or_init(|| tool::which("whisper-cli").is_some())
}

#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub start_ms: i64,
    pub text: String,
}

#[derive(Debug, Default)]
pub struct Transcript {
    pub language: Option<String>,
    pub segments: Vec<Segment>,
}

pub struct Whisper<'a> {
    pub model: &'a Path,
    pub vad: Option<&'a Path>,
    pub threads: usize,
}

impl Whisper<'_> {
    /// Transcribes the first `max_secs` of a recording. `None` from `stop` between pieces
    /// abandons the file (it is retried later).
    pub fn transcribe(
        &self,
        path: &Path,
        duration: f64,
        max_secs: f64,
        stop: &dyn Fn() -> bool,
    ) -> Result<Option<Transcript>> {
        let mut out = Transcript::default();
        let end = duration.min(max_secs);
        let mut start = 0.0;
        while start < end {
            if stop() {
                return Ok(None);
            }
            let len = PIECE_SECS.min(end - start);
            let wav = media::wav(path, start, len)?;
            let mut cmd = Command::new("whisper-cli");
            cmd.arg("-m").arg(self.model).args(["-f", "-", "-np", "-oj", "-of", "-", "-l", "auto"]);
            cmd.args(["-t", &self.threads.to_string()]);
            if let Some(vad) = self.vad {
                cmd.arg("--vad").arg("-vm").arg(vad);
            }
            let json = tool::run(&mut cmd, Some(wav), TIMEOUT)?;
            let v: serde_json::Value = serde_json::from_slice(&json).context("parsing whisper-cli output")?;
            if out.language.is_none() {
                out.language = v["result"]["language"].as_str().map(str::to_string);
            }
            let offset = (start * 1000.0) as i64;
            for seg in v["transcription"].as_array().into_iter().flatten() {
                let text = seg["text"].as_str().unwrap_or("").trim().to_string();
                let from = seg["offsets"]["from"].as_i64().unwrap_or(0);
                out.segments.push(Segment { start_ms: offset + from, text });
            }
            start += len;
        }
        out.segments = clean(out.segments);
        Ok(Some(out))
    }
}

/// Drops what whisper writes for music and silence ("[Music]", "(upbeat music)", "♪") and
/// the loops it falls into, where one line repeats over and over.
fn clean(segments: Vec<Segment>) -> Vec<Segment> {
    let mut out: Vec<Segment> = Vec::new();
    let mut repeats = 0;
    for s in segments {
        let t = s.text.trim();
        let bracketed = (t.starts_with('[') && t.ends_with(']')) || (t.starts_with('(') && t.ends_with(')'));
        if t.is_empty() || bracketed || t.chars().all(|c| !c.is_alphanumeric()) {
            continue;
        }
        if out.last().is_some_and(|l| l.text == t) {
            repeats += 1;
            if repeats >= 2 {
                continue;
            }
        } else {
            repeats = 0;
        }
        out.push(Segment { start_ms: s.start_ms, text: t.to_string() });
    }
    // A few stray words over a whole recording are more likely noise than speech.
    let words: usize = out.iter().map(|s| s.text.split_whitespace().count()).sum();
    if words < 5 { Vec::new() } else { out }
}

/// Groups segments into chunks; each chunk's `line` is unused and its start is returned alongside.
pub fn chunks(segments: &[Segment]) -> Vec<(Chunk, i64)> {
    let mut out = Vec::new();
    let mut text = String::new();
    let mut start = 0;
    for s in segments {
        if text.is_empty() {
            start = s.start_ms;
        } else {
            text.push(' ');
        }
        text.push_str(&s.text);
        if text.len() >= CHUNK_CHARS {
            out.push((Chunk { text: std::mem::take(&mut text), line: 0 }, start));
        }
    }
    if !text.is_empty() {
        out.push((Chunk { text, line: 0 }, start));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(ms: i64, t: &str) -> Segment {
        Segment { start_ms: ms, text: t.into() }
    }

    #[test]
    fn drops_music_markers_and_loops() {
        let s = vec![
            seg(0, "[Music]"),
            seg(1000, "Welcome back to the show."),
            seg(2000, "♪ ♪"),
            seg(3000, "Thank you."),
            seg(4000, "Thank you."),
            seg(5000, "Thank you."),
            seg(6000, "Today we talk about sourdough."),
        ];
        let texts: Vec<String> = clean(s).into_iter().map(|s| s.text).collect();
        assert_eq!(texts, ["Welcome back to the show.", "Thank you.", "Thank you.", "Today we talk about sourdough."]);
        assert!(clean(vec![seg(0, "Okay.")]).is_empty());
    }

    #[test]
    fn groups_segments_into_chunks() {
        let long = "word ".repeat(200);
        let c = chunks(&[seg(0, &long), seg(5000, "tail")]);
        assert_eq!(c.len(), 2);
        assert_eq!((c[0].1, c[1].1), (0, 5000));
    }
}
