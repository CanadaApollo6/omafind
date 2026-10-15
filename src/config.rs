//! User configuration (`~/.config/omafind/config.toml`) and well-known paths.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// Directories to index. `~` is expanded.
    pub roots: Vec<String>,
    /// Paths or glob patterns to skip, matched against absolute paths.
    pub exclude: Vec<String>,
    /// Index dotfiles and dot-directories.
    pub include_hidden: bool,
    /// Skip files ignored by `.gitignore` / `.ignore` files.
    pub respect_gitignore: bool,
    /// Index the contents of mounted encrypted folders (gocryptfs, CryFS, EncFS, eCryptfs,
    /// VeraCrypt). Off by default: the index is not encrypted, so it would hold a
    /// decrypted copy of their text.
    pub index_encrypted: bool,
    /// Largest text-like file whose content is read, in bytes.
    pub max_text_bytes: u64,
    /// Largest image that is decoded for the vision index, in bytes.
    pub max_image_bytes: u64,
    pub models: ModelConfig,
    pub indexing: IndexingConfig,
    pub search: SearchConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelConfig {
    /// EmbeddingGemma 2 ONNX weights: "auto", "q4", "quantized" (int8) or "fp32".
    /// "auto" picks q4 on the CPU (smallest memory) and fp32 on CUDA.
    pub precision: String,
    /// "auto", "cpu" or "cuda".
    pub device: String,
    /// Image soft-token budget: 70, 140, 280, 560 or 1120. 0 picks 280 on CUDA and 140 on the CPU.
    pub vision_tokens: u32,
    /// Matryoshka dimension stored for Gemma vectors: 128, 256, 512 or 768.
    pub dim: usize,
    /// ONNX Runtime intra-op threads. 0 picks half the cores, at most 8.
    pub threads: usize,
    /// Drop Gemma from memory after this many idle seconds (daemon only).
    pub unload_after_secs: u64,
    /// Use the code-specific static model for source files.
    pub static_code_model: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IndexingConfig {
    /// Embed images with the Gemma vision encoder.
    pub images: bool,
    /// Pre-compute Gemma vectors for every text chunk in the background (subject to the
    /// power budget), so deep search rarely has to embed on the query path. Costs about
    /// 1 KB per chunk at 256 dims and ~85 ms of CPU time per chunk.
    pub gemma_text: bool,
    /// What to do on battery: "full", "text" (defer Gemma work) or "pause".
    pub on_battery: String,
    /// Back off while the 1-minute load average per core is above this.
    pub max_load: f32,
    /// Worker threads for text extraction. 0 picks a quarter of the cores (at least 2).
    pub threads: usize,
    /// Write thumbnails for images into the cache directory.
    pub thumbnails: bool,
    /// Full rescan interval for the daemon, in hours (catches missed events).
    pub rescan_hours: u64,
    /// Read text in screenshots, photos of documents and scanned PDFs (needs `tesseract`).
    pub ocr: bool,
    /// Tesseract languages, such as `["eng", "deu"]`. Empty: English plus the locale's language.
    pub ocr_languages: Vec<String>,
    /// Embed recordings with Gemma's audio encoder (needs `ffmpeg`).
    pub audio: bool,
    /// Embed videos from frames sampled across them (needs `ffmpeg`).
    pub video: bool,
    /// Transcribe speech in recordings and videos (needs `whisper-cli` from whisper.cpp).
    pub speech: bool,
    /// Whisper model: "base", "small" or "large-v3-turbo".
    pub speech_model: String,
    /// Transcribe at most this many minutes of each recording.
    pub speech_max_minutes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SearchConfig {
    /// Text chunks Gemma re-scores after the fast stage.
    pub rescore_top: usize,
    /// Token cap for each chunk that is re-scored.
    pub rescore_max_tokens: usize,
    /// Time the deep stage may spend embedding shortlisted chunks that have no cached
    /// Gemma vector yet, in milliseconds. 0 = auto (600 on CPU, 3000 on CUDA).
    pub rescore_budget_ms: u64,
    /// Results kept per group.
    pub limit: usize,
    /// How far above a query's similarity to unrelated text a document must score to be
    /// shown on Gemma's evidence alone (keyword and name matches are always shown).
    pub text_margin: f64,
    /// The same for images.
    pub image_margin: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            roots: vec!["~".into()],
            exclude: vec![
                "~/.cache".into(),
                "~/.local/share/Trash".into(),
                "~/.local/share/Steam".into(),
                "~/.steam".into(),
                "~/.var".into(),
                "~/.mozilla".into(),
                "~/.cargo".into(),
                "~/.rustup".into(),
                "~/.npm".into(),
                "~/go/pkg".into(),
                "**/node_modules".into(),
                "**/.git".into(),
                "**/target/debug".into(),
                "**/target/release".into(),
                "**/__pycache__".into(),
                "**/.venv".into(),
                "**/venv".into(),
                // Keys and keystores: their text would end up in the (unencrypted) index.
                "**/*.pem".into(),
                "**/*.key".into(),
                "**/*.p12".into(),
                "**/*.pfx".into(),
                "**/*.kdbx".into(),
                "**/*.gpg".into(),
                "**/*.asc".into(),
                "**/id_rsa*".into(),
                "**/id_ecdsa*".into(),
                "**/id_ed25519*".into(),
            ],
            index_encrypted: false,
            include_hidden: false,
            respect_gitignore: true,
            max_text_bytes: 8 * 1024 * 1024,
            max_image_bytes: 200 * 1024 * 1024,
            models: ModelConfig::default(),
            indexing: IndexingConfig::default(),
            search: SearchConfig::default(),
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            precision: "auto".into(),
            device: "auto".into(),
            vision_tokens: 0,
            dim: 256,
            threads: 0,
            unload_after_secs: 300,
            static_code_model: true,
        }
    }
}

impl Default for IndexingConfig {
    fn default() -> Self {
        Self {
            images: true,
            gemma_text: true,
            on_battery: "text".into(),
            max_load: 0.75,
            threads: 0,
            thumbnails: true,
            rescan_hours: 12,
            ocr: true,
            ocr_languages: Vec::new(),
            audio: true,
            video: true,
            speech: true,
            speech_model: "base".into(),
            speech_max_minutes: 120,
        }
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            rescore_top: 100,
            rescore_max_tokens: 256,
            rescore_budget_ms: 0,
            limit: 8,
            text_margin: 0.13,
            image_margin: 0.075,
        }
    }
}

impl Config {
    /// Loads the config file, or the defaults when it doesn't exist.
    pub fn load(path: Option<&Path>) -> Result<Self> {
        let path = path.map(Path::to_path_buf).unwrap_or_else(config_path);
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        let cfg: Config = toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(matches!(self.models.dim, 128 | 256 | 512 | 768), "models.dim must be 128, 256, 512 or 768");
        anyhow::ensure!(
            matches!(self.models.vision_tokens, 0 | 70 | 140 | 280 | 560 | 1120),
            "models.vision_tokens must be 0, 70, 140, 280, 560 or 1120"
        );
        anyhow::ensure!(
            matches!(self.models.precision.as_str(), "auto" | "q4" | "quantized" | "fp32"),
            "models.precision must be auto, q4, quantized or fp32"
        );
        anyhow::ensure!(
            matches!(self.models.device.as_str(), "auto" | "cpu" | "cuda"),
            "models.device must be auto, cpu or cuda"
        );
        anyhow::ensure!(
            matches!(self.indexing.on_battery.as_str(), "full" | "text" | "pause"),
            "indexing.on_battery must be full, text or pause"
        );
        anyhow::ensure!(
            crate::models::whisper_file(&self.indexing.speech_model).is_some(),
            "indexing.speech_model must be base, small or large-v3-turbo"
        );
        Ok(())
    }

    pub fn root_paths(&self) -> Vec<PathBuf> {
        self.roots.iter().map(|r| expand_tilde(r)).collect()
    }

    pub fn index_threads(&self) -> usize {
        if self.indexing.threads > 0 {
            return self.indexing.threads;
        }
        (num_cpus() / 4).max(2)
    }
}

pub fn num_cpus() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

pub fn expand_tilde(p: &str) -> PathBuf {
    if p == "~" {
        return home_dir();
    }
    if let Some(rest) = p.strip_prefix("~/") {
        return home_dir().join(rest);
    }
    PathBuf::from(p)
}

pub fn home_dir() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/"))
}

fn xdg(var: &str, fallback: &str) -> PathBuf {
    match std::env::var_os(var) {
        Some(v) if !v.is_empty() => PathBuf::from(v),
        _ => home_dir().join(fallback),
    }
}

pub fn config_path() -> PathBuf {
    if let Some(p) = std::env::var_os("OMAFIND_CONFIG") {
        return PathBuf::from(p);
    }
    xdg("XDG_CONFIG_HOME", ".config").join("omafind/config.toml")
}

pub fn data_dir() -> PathBuf {
    xdg("XDG_DATA_HOME", ".local/share").join("omafind")
}

pub fn db_path() -> PathBuf {
    if let Some(p) = std::env::var_os("OMAFIND_DB") {
        return PathBuf::from(p);
    }
    data_dir().join("index.sqlite")
}

pub fn cache_dir() -> PathBuf {
    xdg("XDG_CACHE_HOME", ".cache").join("omafind")
}

pub fn thumbs_dir() -> PathBuf {
    cache_dir().join("thumbs")
}

/// Where `omafind models fetch` writes models.
pub fn user_models_dir() -> PathBuf {
    if let Some(p) = std::env::var_os("OMAFIND_MODELS") {
        return PathBuf::from(p);
    }
    data_dir().join("models")
}

/// Model directories, searched in order. Packages install into `/usr/share/omafind/models`.
pub fn model_dirs() -> Vec<PathBuf> {
    let mut dirs = vec![user_models_dir()];
    for d in ["/usr/share/omafind/models", "/usr/local/share/omafind/models"] {
        dirs.push(PathBuf::from(d));
    }
    dirs
}

pub fn socket_path() -> PathBuf {
    if let Some(p) = std::env::var_os("OMAFIND_SOCKET") {
        return PathBuf::from(p);
    }
    match std::env::var_os("XDG_RUNTIME_DIR") {
        Some(d) if !d.is_empty() => PathBuf::from(d).join("omafind.sock"),
        _ => std::env::temp_dir().join(format!("omafind-{}.sock", unsafe { libc::getuid() })),
    }
}

pub const EXAMPLE_CONFIG: &str = r#"# omafind configuration. Every key is optional; these are the defaults.

roots = ["~"]
exclude = [
  "~/.cache", "~/.local/share/Trash", "~/.local/share/Steam", "~/.steam", "~/.var",
  "~/.mozilla", "~/.cargo", "~/.rustup", "~/.npm", "~/go/pkg",
  "**/node_modules", "**/.git", "**/target/debug", "**/target/release",
  "**/__pycache__", "**/.venv", "**/venv",
  # keys and keystores
  "**/*.pem", "**/*.key", "**/*.p12", "**/*.pfx", "**/*.kdbx", "**/*.gpg", "**/*.asc",
  "**/id_rsa*", "**/id_ecdsa*", "**/id_ed25519*",
]
index_encrypted = false   # index inside mounted gocryptfs/CryFS/EncFS/eCryptfs/VeraCrypt folders
include_hidden = false
respect_gitignore = true
max_text_bytes = 8388608
max_image_bytes = 209715200

[models]
precision = "auto"        # auto | q4 | quantized | fp32
device = "auto"           # auto | cpu | cuda
vision_tokens = 0         # 0 = auto (280 on CUDA, 140 on CPU), or 70 | 140 | 280 | 560 | 1120
dim = 256                 # 128 | 256 | 512 | 768 (changing it requires `omafind reset --vectors`)
threads = 0               # 0 = auto
unload_after_secs = 300
static_code_model = true

[indexing]
images = true
gemma_text = true         # pre-embed text chunks with Gemma in the background (fast deep search)
on_battery = "text"       # full | text | pause
max_load = 0.75
threads = 0
thumbnails = true
rescan_hours = 12
ocr = true                # text in screenshots and scans (needs tesseract)
ocr_languages = []        # e.g. ["eng", "deu"]; empty = English plus your locale's language
audio = true              # recordings through Gemma's audio encoder (needs ffmpeg)
video = true              # videos through frames sampled across them (needs ffmpeg)
speech = true             # transcripts of speech (needs whisper-cli from whisper.cpp)
speech_model = "base"     # base | small | large-v3-turbo
speech_max_minutes = 120

[search]
rescore_top = 100
rescore_max_tokens = 256
rescore_budget_ms = 0     # 0 = auto: 600 on CPU, 3000 on CUDA
limit = 8
text_margin = 0.13        # raise to show fewer, surer results
image_margin = 0.075
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example_config_parses_to_defaults() {
        let cfg: Config = toml::from_str(EXAMPLE_CONFIG).unwrap();
        cfg.validate().unwrap();
        assert_eq!(toml::to_string(&cfg).unwrap(), toml::to_string(&Config::default()).unwrap());
    }
}
