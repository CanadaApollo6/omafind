//! Larger pictures for the launcher's preview pane, made on demand and cached: a PDF page,
//! a video frame, a photo Qt can't decode (HEIC, camera RAW). Index thumbnails are too small.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime};

use anyhow::{Result, bail};
use image::DynamicImage;

use crate::config::{self, Config};
use crate::extract::{self, Kind, media, tool};

/// Longest side of a rendered preview, in pixels.
const SIDE: u32 = 1400;
/// Previews kept; the least recently shown go first.
const KEEP: usize = 200;
const PDF_TIMEOUT: Duration = Duration::from_secs(20);

pub fn previews_dir() -> PathBuf {
    config::cache_dir().join("previews")
}

/// Renders a preview of `path` (a PDF `page`, or a video frame `at_ms` in) and returns the
/// JPEG's path. Asking again for the same picture of an unchanged file reuses it.
pub fn render(cfg: &Config, path: &Path, page: Option<u32>, at_ms: Option<i64>) -> Result<PathBuf> {
    let meta = std::fs::metadata(path)?;
    let mtime = crate::index::mtime_secs(&meta);
    let key = format!("{}\0{mtime}\0{}\0{page:?}\0{at_ms:?}", path.display(), meta.len());
    let h = blake3::hash(key.as_bytes()).to_hex();
    let dir = previews_dir();
    let dest = dir.join(format!("{}.jpg", &h[..24]));
    if dest.exists() {
        // Mark it recently used so cleanup keeps it.
        if let Ok(f) = std::fs::File::options().append(true).open(&dest) {
            let _ = f.set_modified(SystemTime::now());
        }
        return Ok(dest);
    }
    std::fs::create_dir_all(&dir)?;
    let tmp = dest.with_extension("tmp");
    match extract::kind_of(path) {
        _ if extract::ext_of(path) == "pdf" => {
            let page = page.unwrap_or(1).max(1).to_string();
            let jpeg = tool::run(
                Command::new("pdftoppm")
                    .args(["-q", "-jpeg", "-jpegopt", "quality=85", "-scale-to", &SIDE.to_string()])
                    .args(["-f", &page, "-l", &page, "-singlefile"])
                    // No output root: the page goes to stdout.
                    .arg(path),
                None,
                PDF_TIMEOUT,
            )?;
            anyhow::ensure!(!jpeg.is_empty(), "no page {page}");
            std::fs::write(&tmp, jpeg)?;
        }
        Kind::Video => {
            let at = match at_ms {
                Some(ms) => ms as f64 / 1000.0,
                None => media::probe(path)?.duration.unwrap_or(0.0) / 2.0,
            };
            save(&media::frame(path, at, SIDE)?, &tmp)?;
        }
        Kind::Image => save(&extract::image::load(path, cfg.max_image_bytes)?, &tmp)?,
        _ => bail!("no preview for this kind of file"),
    }
    std::fs::rename(&tmp, &dest)?;
    prune(&dir);
    Ok(dest)
}

fn save(img: &DynamicImage, dest: &Path) -> Result<()> {
    let small;
    let img = if img.width().max(img.height()) > SIDE {
        small = img.thumbnail(SIDE, SIDE);
        &small
    } else {
        img
    };
    let mut out = std::io::BufWriter::new(std::fs::File::create(dest)?);
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 85).encode_image(&img.to_rgb8())?;
    Ok(())
}

/// Deletes all but the `KEEP` most recently used previews.
fn prune(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    let mut files: Vec<(SystemTime, PathBuf)> =
        entries.flatten().filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path()))).collect();
    if files.len() <= KEEP {
        return;
    }
    files.sort_by_key(|f| std::cmp::Reverse(f.0));
    for (_, p) in files.into_iter().skip(KEEP) {
        let _ = std::fs::remove_file(p);
    }
}
