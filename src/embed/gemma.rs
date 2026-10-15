//! Deep tier: EmbeddingGemma 2 through ONNX Runtime.
//!
//! The text model is always loaded; the vision and audio encoders are loaded when an image,
//! video or recording needs them and dropped again when the indexer runs out of work.
//! Images and video frames go through the vision encoder and recordings through the audio
//! encoder; their soft tokens are placed at `<|image|>`, `<|video|>` and `<|audio|>`
//! positions in the text model, so all vectors share one space and can be compared directly.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result, anyhow};
use image::DynamicImage;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::config::{self, Config};
use crate::models;

pub const QUERY_PREFIX: &str = "task: search result | query: ";
pub const CODE_QUERY_PREFIX: &str = "task: code retrieval | query: ";

const BOS: i64 = 2;
const EOS: i64 = 1;
const PAD: i64 = 0;
const BOI: i64 = 255_999;
const EOI: i64 = 258_882;
const IMAGE_TOKEN: i64 = 258_880;
const BOA: i64 = 256_000;
const AUDIO_TOKEN: i64 = 258_881;
const EOA: i64 = 258_883;
const VIDEO_TOKEN: i64 = 258_884;
/// Longest stretch of audio the encoder sees at once (the feature extractor's default cap).
pub const AUDIO_WINDOW_SECS: f64 = 30.0;
const HIDDEN: usize = 512;
const PATCH: usize = 16;
const POOL: usize = 3;
const BATCH: usize = 16;

pub struct Gemma {
    text: Mutex<Session>,
    vision: Mutex<Option<Session>>,
    audio: Mutex<Option<Session>>,
    tokenizer: Tokenizer,
    dir: PathBuf,
    suffix: &'static str,
    threads: usize,
    pub cuda: bool,
    pub precision: String,
    /// Matryoshka output dimension.
    pub dim: usize,
    /// Soft tokens per image.
    pub vision_tokens: u32,
    /// Soft tokens per video frame.
    pub video_tokens: u32,
    /// The audio encoder's graph, if it was downloaded.
    audio_path: Option<PathBuf>,
}

/// q4 is the fastest on CPU at a small quality cost; on a GPU fp32 is fast anyway.
fn precision_for(configured: &str, cuda: bool) -> String {
    match configured {
        "auto" if cuda => "fp32".to_string(),
        "auto" => "q4".to_string(),
        p => p.to_string(),
    }
}

/// The precision `models.precision` resolves to on this machine.
pub fn resolve_precision(cfg: &crate::config::ModelConfig) -> String {
    precision_for(&cfg.precision, want_cuda(&cfg.device).unwrap_or(false))
}

/// Formats a document chunk the way EmbeddingGemma expects.
pub fn document_input(title: Option<&str>, text: &str) -> String {
    format!("title: {} | text: {}", title.unwrap_or("none"), text)
}

impl Gemma {
    pub fn load(cfg: &Config) -> Result<Self> {
        let cuda = want_cuda(&cfg.models.device)?;
        let precision = precision_for(&cfg.models.precision, cuda);
        let dir = models::find_gemma(&precision)?;
        let suffix = models::precision_suffix(&precision);
        let threads = threads(cfg);
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("tokenizer: {e}"))?;
        let text = session(&dir.join(format!("onnx/model{suffix}.onnx")), threads, cuda)?;
        let vision_tokens = match cfg.models.vision_tokens {
            0 if cuda => 280,
            0 => 140,
            n => n,
        };
        Ok(Self {
            text: Mutex::new(text),
            vision: Mutex::new(None),
            audio: Mutex::new(None),
            tokenizer,
            audio_path: models::find_gemma_audio(&precision),
            dir,
            suffix,
            threads,
            cuda,
            precision,
            dim: cfg.models.dim,
            vision_tokens,
            video_tokens: if cuda { 140 } else { 70 },
        })
    }

    pub fn on_gpu(&self) -> bool {
        self.cuda
    }

    pub fn describe(&self) -> String {
        format!(
            "EmbeddingGemma 2 {} on {}, {} dims, {} image tokens",
            self.precision,
            if self.cuda { "CUDA" } else { "CPU" },
            self.dim,
            self.vision_tokens
        )
    }

    /// Embeds texts (already carrying their task prefix). Each input is cut to `max_tokens`.
    pub fn embed_texts(&self, texts: &[String], max_tokens: usize) -> Result<Vec<Vec<f32>>> {
        let mut encoded: Vec<(usize, Vec<i64>)> = Vec::with_capacity(texts.len());
        for (i, t) in texts.iter().enumerate() {
            let enc = self.tokenizer.encode(t.as_str(), true).map_err(|e| anyhow!("tokenize: {e}"))?;
            let mut ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
            if ids.len() > max_tokens.max(8) {
                ids.truncate(max_tokens.max(8) - 1);
                ids.push(EOS);
            }
            encoded.push((i, ids));
        }
        // Batch similar lengths together to waste less work on padding.
        encoded.sort_by_key(|(_, ids)| ids.len());
        let mut out = vec![Vec::new(); texts.len()];
        let mut session = self.text.lock().unwrap();
        for batch in encoded.chunks(BATCH) {
            let len = batch.iter().map(|(_, ids)| ids.len()).max().unwrap_or(1);
            let mut input_ids = Vec::with_capacity(batch.len() * len);
            let mut mask = Vec::with_capacity(batch.len() * len);
            for (_, ids) in batch {
                input_ids.extend_from_slice(ids);
                input_ids.extend(std::iter::repeat_n(PAD, len - ids.len()));
                mask.extend(std::iter::repeat_n(1i64, ids.len()));
                mask.extend(std::iter::repeat_n(0i64, len - ids.len()));
            }
            let vecs = run_text(&mut session, [batch.len(), len], input_ids, mask, Media::default())?;
            for ((i, _), v) in batch.iter().zip(vecs) {
                out[*i] = truncate_normalize(&v, self.dim);
            }
        }
        Ok(out)
    }

    pub fn embed_query(&self, query: &str, prefix: &str) -> Result<Vec<f32>> {
        Ok(self.embed_texts(&[format!("{prefix}{query}")], 256)?.remove(0))
    }

    /// Embeds one image into the shared space.
    pub fn embed_image(&self, img: &DynamicImage) -> Result<Vec<f32>> {
        let prep = preprocess(img, self.vision_tokens);
        let n = prep.positions.len() / 2;
        let features = self.encode_frames(1, n, prep.pixels, prep.positions)?;
        let soft = features.len() / HIDDEN;
        let mut ids = vec![BOS, BOI];
        ids.extend(std::iter::repeat_n(IMAGE_TOKEN, soft));
        ids.extend([EOI, EOS]);
        self.embed_media(ids, Media { image: features, ..Default::default() })
    }

    /// Embeds a video from frames sampled across it (about one per second, as in training).
    pub fn embed_video(&self, frames: &[DynamicImage]) -> Result<Vec<f32>> {
        anyhow::ensure!(!frames.is_empty(), "no frames");
        // Frames of one video share a size, so they go through the encoder as one batch.
        let mut pixels = Vec::new();
        let mut positions = Vec::new();
        let mut n = 0;
        let first = (frames[0].width(), frames[0].height());
        for f in frames {
            let f = if (f.width(), f.height()) == first {
                std::borrow::Cow::Borrowed(f)
            } else {
                std::borrow::Cow::Owned(f.resize_exact(first.0, first.1, image::imageops::FilterType::Triangle))
            };
            let prep = preprocess(&f, self.video_tokens);
            n = prep.positions.len() / 2;
            pixels.extend(prep.pixels);
            positions.extend(prep.positions);
        }
        let features = self.encode_frames(frames.len(), n, pixels, positions)?;
        let per_frame = features.len() / HIDDEN / frames.len();
        let mut ids = vec![BOS];
        for _ in frames {
            ids.push(BOI);
            ids.extend(std::iter::repeat_n(VIDEO_TOKEN, per_frame));
            ids.push(EOI);
        }
        ids.push(EOS);
        self.embed_media(ids, Media { video: features, ..Default::default() })
    }

    /// Whether recordings can be embedded (the audio encoder was downloaded).
    pub fn has_audio(&self) -> bool {
        self.audio_path.is_some()
    }

    /// Embeds up to `AUDIO_WINDOW_SECS` of 16 kHz mono audio.
    pub fn embed_audio(&self, samples: &[f32]) -> Result<Vec<f32>> {
        let max = (AUDIO_WINDOW_SECS * 16_000.0) as usize;
        let feats = super::audio::features(&samples[..samples.len().min(max)]);
        anyhow::ensure!(feats.tokens() > 0, "audio too short");
        let frames = feats.frames();
        let features = {
            let mut guard = self.audio.lock().unwrap();
            if guard.is_none() {
                let path = self.audio_path.as_ref().context("the audio encoder is not installed")?;
                *guard = Some(session(path, self.threads, self.cuda)?);
            }
            let audio = guard.as_mut().unwrap();
            let outputs = audio.run(ort::inputs! {
                "input_features" => Tensor::from_array(([1usize, frames, super::audio::MELS], feats.mel))?,
                "input_features_mask" => Tensor::from_array(([1usize, frames], feats.mask))?,
            })?;
            let (shape, data) = outputs["audio_features"].try_extract_tensor::<f32>()?;
            anyhow::ensure!(
                shape.len() == 2 && shape[1] as usize == HIDDEN,
                "unexpected audio_features shape {shape:?}"
            );
            data.to_vec()
        };
        let soft = features.len() / HIDDEN;
        let mut ids = vec![BOS, BOA];
        ids.extend(std::iter::repeat_n(AUDIO_TOKEN, soft));
        ids.extend([EOA, EOS]);
        self.embed_media(ids, Media { audio: features, ..Default::default() })
    }

    /// Runs the vision encoder over `frames` images of `n` patches each.
    fn encode_frames(&self, frames: usize, n: usize, pixels: Vec<f32>, positions: Vec<i64>) -> Result<Vec<f32>> {
        let mut guard = self.vision.lock().unwrap();
        if guard.is_none() {
            let path = self.dir.join(format!("onnx/vision_encoder{}.onnx", self.suffix));
            *guard = Some(session(&path, self.threads, self.cuda)?);
        }
        let vision = guard.as_mut().unwrap();
        let outputs = vision.run(ort::inputs! {
            "pixel_values" => Tensor::from_array(([frames, n, PATCH * PATCH * 3], pixels))?,
            "pixel_position_ids" => Tensor::from_array(([frames, n, 2], positions))?,
        })?;
        let (shape, data) = outputs["image_features"].try_extract_tensor::<f32>()?;
        anyhow::ensure!(shape.len() == 2 && shape[1] as usize == HIDDEN, "unexpected image_features shape {shape:?}");
        Ok(data.to_vec())
    }

    fn embed_media(&self, ids: Vec<i64>, media: Media) -> Result<Vec<f32>> {
        let len = ids.len();
        let mut session = self.text.lock().unwrap();
        let v = run_text(&mut session, [1, len], ids, vec![1; len], media)?;
        Ok(truncate_normalize(&v[0], self.dim))
    }

    /// Frees the vision and audio encoders (they are reloaded when needed).
    pub fn unload_encoders(&self) {
        let vision = self.vision.lock().unwrap().take().is_some();
        let audio = self.audio.lock().unwrap().take().is_some();
        if vision || audio {
            super::release_memory();
        }
    }
}

fn want_cuda(device: &str) -> Result<bool> {
    match device {
        "cpu" => Ok(false),
        "cuda" if !cfg!(feature = "cuda") => {
            anyhow::bail!("models.device = \"cuda\" needs a build with `--features cuda`")
        }
        "cuda" if !has_cudnn() => anyhow::bail!("models.device = \"cuda\" needs cuDNN 9 (Arch: pacman -S cudnn)"),
        "cuda" => Ok(true),
        _ => Ok(cuda_available()),
    }
}

#[cfg(feature = "cuda")]
fn cuda_available() -> bool {
    <ort::ep::CUDA as ort::ep::ExecutionProvider>::is_available().unwrap_or(false)
        && Path::new("/dev/nvidiactl").exists()
        && has_cudnn()
}

/// ONNX Runtime's CUDA provider only finds out cuDNN is missing at the first attention
/// node, so check up front and fall back to the CPU.
fn has_cudnn() -> bool {
    ["libcudnn.so.9", "libcudnn.so"].iter().any(|name| {
        let c = std::ffi::CString::new(*name).unwrap();
        let h = unsafe { libc::dlopen(c.as_ptr(), libc::RTLD_LAZY | libc::RTLD_LOCAL) };
        if !h.is_null() {
            unsafe { libc::dlclose(h) };
        }
        !h.is_null()
    })
}

#[cfg(not(feature = "cuda"))]
fn cuda_available() -> bool {
    false
}

fn environment() -> Result<&'static ort::environment::Environment> {
    static ENV: std::sync::OnceLock<ort::environment::Environment> = std::sync::OnceLock::new();
    if let Some(env) = ENV.get() {
        return Ok(env);
    }
    let env = ort::init().with_name("omafind").with_telemetry(false).build()?;
    Ok(ENV.get_or_init(|| env))
}

fn session(path: &Path, threads: usize, cuda: bool) -> Result<Session> {
    let mut builder = Session::builder(environment()?)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow!("{e}"))?
        .with_intra_threads(threads)
        .map_err(|e| anyhow!("{e}"))?
        .with_memory_pattern(false)
        .map_err(|e| anyhow!("{e}"))?;
    let mut eps = Vec::new();
    #[cfg(feature = "cuda")]
    if cuda {
        eps.push(ort::ep::CUDA::default().build());
    }
    let _ = cuda;
    // Without the arena, memory goes back to the system between runs.
    eps.push(ort::ep::CPU::default().with_arena_allocator(false).build());
    builder = builder.with_execution_providers(eps).map_err(|e| anyhow!("{e}"))?;
    builder.commit_from_file(path).with_context(|| format!("loading {}", path.display()))
}

/// Encoder outputs placed at the matching soft-token positions, each [tokens, HIDDEN].
#[derive(Default)]
struct Media {
    image: Vec<f32>,
    video: Vec<f32>,
    audio: Vec<f32>,
}

fn run_text(
    session: &mut Session,
    shape: [usize; 2],
    input_ids: Vec<i64>,
    mask: Vec<i64>,
    media: Media,
) -> Result<Vec<Vec<f32>>> {
    let rows = |v: &Vec<f32>| v.len() / HIDDEN;
    let outputs = session.run(ort::inputs! {
        "input_ids" => Tensor::from_array((shape, input_ids))?,
        "attention_mask" => Tensor::from_array((shape, mask))?,
        "image_features" => Tensor::from_array(([rows(&media.image), HIDDEN], media.image))?,
        "video_features" => Tensor::from_array(([rows(&media.video), HIDDEN], media.video))?,
        "audio_features" => Tensor::from_array(([rows(&media.audio), HIDDEN], media.audio))?,
    })?;
    let (out_shape, data) = outputs["sentence_embedding"].try_extract_tensor::<f32>()?;
    let width = *out_shape.last().context("empty output shape")? as usize;
    Ok(data.chunks(width).map(|c| c.to_vec()).collect())
}

/// Keeps the leading Matryoshka dimensions and re-normalizes.
pub fn truncate_normalize(v: &[f32], dim: usize) -> Vec<f32> {
    let mut out: Vec<f32> = v[..dim.min(v.len())].to_vec();
    let norm = out.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        out.iter_mut().for_each(|x| *x /= norm);
    }
    out
}

/// Threads for model inference: `models.threads`, or half the cores (at most 8).
pub fn threads(cfg: &Config) -> usize {
    if cfg.models.threads > 0 { cfg.models.threads } else { (config::num_cpus() / 2).clamp(1, 8) }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub struct Preprocessed {
    /// [patches, 16*16*3] in (row, col, channel) order within each patch.
    pub pixels: Vec<f32>,
    /// [patches, 2] as (x, y) patch coordinates.
    pub positions: Vec<i64>,
    pub width: u32,
    pub height: u32,
}

/// Port of transformers' `Gemma4ImageProcessor`: aspect-preserving resize to fit the patch
/// budget (sides divisible by 48), rescale to [0, 1], then 16x16 patches in raster order.
/// Padding patches are not added; the encoder only needs the real ones.
pub fn preprocess(img: &DynamicImage, max_soft_tokens: u32) -> Preprocessed {
    let rgb = img.to_rgb8();
    let max_patches = max_soft_tokens as usize * POOL * POOL;
    let (th, tw) = aspect_preserving_size(rgb.height() as usize, rgb.width() as usize, max_patches);
    let resized = if (th, tw) == (rgb.height() as usize, rgb.width() as usize) {
        rgb
    } else {
        image::imageops::resize(&rgb, tw as u32, th as u32, image::imageops::FilterType::CatmullRom)
    };
    let (ph, pw) = (th / PATCH, tw / PATCH);
    let mut pixels = Vec::with_capacity(ph * pw * PATCH * PATCH * 3);
    let mut positions = Vec::with_capacity(ph * pw * 2);
    for py in 0..ph {
        for px in 0..pw {
            for y in 0..PATCH {
                for x in 0..PATCH {
                    let p = resized.get_pixel((px * PATCH + x) as u32, (py * PATCH + y) as u32);
                    pixels.extend(p.0.iter().map(|&c| c as f32 / 255.0));
                }
            }
            positions.extend([px as i64, py as i64]);
        }
    }
    Preprocessed { pixels, positions, width: tw as u32, height: th as u32 }
}

/// Largest size within `max_patches` whose sides are multiples of pooling * patch size.
pub fn aspect_preserving_size(height: usize, width: usize, max_patches: usize) -> (usize, usize) {
    let side = POOL * PATCH;
    let target_px = (max_patches * PATCH * PATCH) as f64;
    let factor = (target_px / (height * width) as f64).sqrt();
    let mut th = ((factor * height as f64) / side as f64).floor() as usize * side;
    let mut tw = ((factor * width as f64) / side as f64).floor() as usize * side;
    let max_side = (max_patches / (POOL * POOL)) * side;
    if th == 0 && tw == 0 {
        return (side, side);
    }
    if th == 0 {
        th = side;
        tw = ((width / height) * side).min(max_side);
    } else if tw == 0 {
        tw = side;
        th = ((height / width) * side).min(max_side);
    }
    (th, tw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_match_transformers() {
        // Values computed with transformers' get_aspect_ratio_preserving_size.
        assert_eq!(aspect_preserving_size(1080, 1920, 280 * 9), (576, 1056));
        assert_eq!(aspect_preserving_size(4000, 3000, 140 * 9), (624, 480));
        assert_eq!(aspect_preserving_size(100, 100, 280 * 9), (768, 768));
        assert_eq!(aspect_preserving_size(10, 4000, 280 * 9), (48, 13440));
    }

    #[test]
    fn patches_are_raster_ordered() {
        // A square image already at its target size is not resampled.
        let (h, w) = aspect_preserving_size(48, 48, 70 * 9);
        assert_eq!(aspect_preserving_size(h, w, 70 * 9), (h, w));
        let mut img = image::RgbImage::new(w as u32, h as u32);
        img.put_pixel(16, 0, image::Rgb([255, 0, 0]));
        let p = preprocess(&DynamicImage::ImageRgb8(img), 70);
        assert_eq!((p.width as usize, p.height as usize), (w, h));
        assert_eq!(p.positions.len() / 2, (w / 16) * (h / 16));
        assert!(p.positions.len() / 2 <= 70 * 9);
        // Second patch (x=1, y=0) starts with the red pixel.
        assert_eq!(&p.positions[2..4], &[1, 0]);
        assert_eq!(p.pixels[768], 1.0);
        assert_eq!(p.pixels[769], 0.0);
    }

    #[test]
    fn truncation_renormalizes() {
        let v = truncate_normalize(&[3.0, 4.0, 100.0], 2);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
    }
}
