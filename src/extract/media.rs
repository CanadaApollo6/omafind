//! Audio and video through FFmpeg: details and tags (`ffprobe`), 16 kHz mono samples for the
//! audio encoder and speech recognition, frames for the vision encoder, and cover art.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;
use std::time::Duration;

use anyhow::{Context, Result};
use image::DynamicImage;
use serde::Serialize;

use super::tool;

pub const SAMPLE_RATE: u32 = 16_000;
const PROBE_TIMEOUT: Duration = Duration::from_secs(20);
const DECODE_TIMEOUT: Duration = Duration::from_secs(300);
/// Tags worth searching, in the order they're written into the tags chunk.
const TEXT_TAGS: &[&str] = &[
    "title",
    "artist",
    "album_artist",
    "album",
    "composer",
    "genre",
    "date",
    "show",
    "episode_id",
    "comment",
    "description",
    "synopsis",
];

pub fn available() -> bool {
    static FOUND: OnceLock<bool> = OnceLock::new();
    *FOUND.get_or_init(|| tool::which("ffprobe").is_some() && tool::which("ffmpeg").is_some())
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct MediaInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,
    /// Displayed size (rotation applied), for videos.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub codec: Option<String>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub has_audio: bool,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub has_video: bool,
    /// Embedded cover art (audio files).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub has_cover: bool,
    /// Lower-cased tag names.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub tags: BTreeMap<String, String>,
    /// Recording time from the container (phones write `creation_time`), Unix seconds.
    #[serde(skip)]
    pub created: Option<i64>,
}

impl MediaInfo {
    /// Music rather than a recording of speech: it carries album or artist tags.
    pub fn is_music(&self) -> bool {
        ["album", "artist", "album_artist"].iter().any(|t| self.tags.contains_key(*t))
    }

    /// Searchable text made of the tags ("Clair de Lune\nartist: Debussy\n..."), if any.
    pub fn tags_text(&self) -> Option<String> {
        let mut lines = Vec::new();
        for &t in TEXT_TAGS {
            if let Some(v) = self.tags.get(t) {
                let v = v.trim();
                if !v.is_empty() {
                    lines.push(if t == "title" { v.to_string() } else { format!("{}: {v}", t.replace('_', " ")) });
                }
            }
        }
        // A date on its own says nothing about what the recording is.
        (lines.iter().any(|l| !l.starts_with("date: "))).then(|| lines.join("\n"))
    }
}

pub fn probe(path: &Path) -> Result<MediaInfo> {
    let out = tool::run(
        Command::new("ffprobe")
            .args(["-v", "error", "-print_format", "json", "-show_format", "-show_streams", "-i"])
            .arg(path),
        None,
        PROBE_TIMEOUT,
    )?;
    let v: serde_json::Value = serde_json::from_slice(&out).context("parsing ffprobe output")?;
    Ok(parse_probe(&v))
}

fn parse_probe(v: &serde_json::Value) -> MediaInfo {
    let mut info = MediaInfo::default();
    let num = |x: &serde_json::Value| x.as_str().and_then(|s| s.parse::<f64>().ok()).or_else(|| x.as_f64());
    let format = &v["format"];
    info.duration = num(&format["duration"]).filter(|d| d.is_finite() && *d > 0.0);
    let mut add_tags = |tags: &serde_json::Value| {
        if let Some(map) = tags.as_object() {
            for (k, val) in map {
                if let Some(s) = val.as_str() {
                    info.tags.entry(k.to_ascii_lowercase()).or_insert_with(|| s.to_string());
                }
            }
        }
    };
    add_tags(&format["tags"]);
    for s in v["streams"].as_array().into_iter().flatten() {
        match s["codec_type"].as_str() {
            Some("video") if s["disposition"]["attached_pic"].as_i64() == Some(1) => info.has_cover = true,
            Some("video") if !info.has_video => {
                info.has_video = true;
                info.codec = s["codec_name"].as_str().map(str::to_string);
                let (w, h) = (s["width"].as_u64().unwrap_or(0) as u32, s["height"].as_u64().unwrap_or(0) as u32);
                let rotation = s["side_data_list"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .find_map(|d| d["rotation"].as_f64())
                    .or_else(|| num(&s["tags"]["rotate"]))
                    .unwrap_or(0.0);
                let quarter = (rotation.abs() as i64 / 90) % 2 == 1;
                if w > 0 && h > 0 {
                    (info.width, info.height) = if quarter { (Some(h), Some(w)) } else { (Some(w), Some(h)) };
                }
                if info.duration.is_none() {
                    info.duration = num(&s["duration"]);
                }
            }
            Some("audio") if !info.has_audio => {
                info.has_audio = true;
                if info.codec.is_none() {
                    info.codec = s["codec_name"].as_str().map(str::to_string);
                }
                if info.duration.is_none() {
                    info.duration = num(&s["duration"]);
                }
                add_tags(&s["tags"]);
            }
            _ => {}
        }
    }
    info.created = info
        .tags
        .get("creation_time")
        .and_then(|t| chrono::DateTime::parse_from_rfc3339(t).ok())
        .map(|t| t.timestamp())
        // Cameras without a clock write 1970 or 1904.
        .filter(|&t| t > 946_684_800);
    info
}

/// Mono 16 kHz samples of `[start, start + len)` seconds.
pub fn decode_audio(path: &Path, start: f64, len: f64) -> Result<Vec<f32>> {
    let out = tool::run(
        Command::new("ffmpeg")
            .args(["-nostdin", "-v", "error", "-ss", &format!("{start:.3}"), "-t", &format!("{len:.3}"), "-i"])
            .arg(path)
            .args(["-vn", "-ac", "1", "-ar", &SAMPLE_RATE.to_string(), "-f", "f32le", "-"]),
        None,
        DECODE_TIMEOUT,
    )?;
    Ok(out.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect())
}

/// Mono 16 kHz 16-bit WAV of `[start, start + len)` seconds, for speech recognition.
pub fn wav(path: &Path, start: f64, len: f64) -> Result<Vec<u8>> {
    tool::run(
        Command::new("ffmpeg")
            .args(["-nostdin", "-v", "error", "-ss", &format!("{start:.3}"), "-t", &format!("{len:.3}"), "-i"])
            .arg(path)
            .args(["-vn", "-ac", "1", "-ar", &SAMPLE_RATE.to_string(), "-c:a", "pcm_s16le", "-f", "wav", "-"]),
        None,
        DECODE_TIMEOUT,
    )
}

/// One frame at `at` seconds, scaled to fit `max_side` (rotation applied).
pub fn frame(path: &Path, at: f64, max_side: u32) -> Result<DynamicImage> {
    let out = tool::run(
        Command::new("ffmpeg").args(["-nostdin", "-v", "error", "-ss", &format!("{at:.3}"), "-i"]).arg(path).args([
            "-an",
            "-frames:v",
            "1",
            "-vf",
            &format!("scale={max_side}:{max_side}:force_original_aspect_ratio=decrease"),
            "-f",
            "image2pipe",
            "-c:v",
            "png",
            "-",
        ]),
        None,
        PROBE_TIMEOUT * 3,
    )?;
    anyhow::ensure!(!out.is_empty(), "no frame at {at:.1}s");
    Ok(image::load_from_memory(&out)?)
}

/// Embedded cover art of an audio file.
pub fn cover(path: &Path) -> Result<DynamicImage> {
    let out = tool::run(
        Command::new("ffmpeg").args(["-nostdin", "-v", "error", "-i"]).arg(path).args([
            "-an",
            "-map",
            "0:v:0",
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-c:v",
            "png",
            "-",
        ]),
        None,
        PROBE_TIMEOUT,
    )?;
    anyhow::ensure!(!out.is_empty(), "no cover art");
    Ok(image::load_from_memory(&out)?)
}

/// `n` timestamps spread evenly over `duration`, each in the middle of its slice.
pub fn spread(duration: f64, n: usize) -> Vec<f64> {
    (0..n).map(|i| duration * (i as f64 + 0.5) / n as f64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_probe_output() {
        let v: serde_json::Value = serde_json::from_str(
            r#"{"streams":[
                {"codec_type":"video","codec_name":"hevc","width":1920,"height":1080,
                 "side_data_list":[{"rotation":-90}]},
                {"codec_type":"audio","codec_name":"aac","tags":{"language":"eng"}}],
              "format":{"duration":"12.5","tags":{"creation_time":"2024-06-01T10:00:00.000000Z","TITLE":"Beach"}}}"#,
        )
        .unwrap();
        let info = parse_probe(&v);
        assert_eq!(info.duration, Some(12.5));
        assert_eq!((info.width, info.height), (Some(1080), Some(1920)));
        assert!(info.has_audio && info.has_video && !info.is_music());
        assert_eq!(info.created, Some(1_717_236_000));
        assert_eq!(info.tags_text().as_deref(), Some("Beach"));
    }

    #[test]
    fn music_tags_become_text() {
        let mut info = MediaInfo::default();
        for (k, v) in [("title", "Clair de Lune"), ("artist", "Debussy"), ("album", "Suite bergamasque")] {
            info.tags.insert(k.into(), v.into());
        }
        assert!(info.is_music());
        assert_eq!(info.tags_text().unwrap(), "Clair de Lune\nartist: Debussy\nalbum: Suite bergamasque");
        assert_eq!(spread(10.0, 2), vec![2.5, 7.5]);
    }
}
