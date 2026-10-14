//! Image decoding (common formats, HEIC/HEIF, camera RAW), capture dates and thumbnails.

use std::io::{BufReader, Read};
use std::path::Path;

use anyhow::{Context, Result, bail};
use image::metadata::Orientation;
use image::{DynamicImage, ImageDecoder, ImageReader};

use super::{RAW, ext_of};

/// Decodes an image with its EXIF orientation applied.
pub fn load(path: &Path, max_bytes: u64) -> Result<DynamicImage> {
    let size = std::fs::metadata(path)?.len();
    if size > max_bytes {
        bail!("image larger than max_image_bytes");
    }
    let ext = ext_of(path);
    match ext.as_str() {
        "heic" | "heif" | "hif" => load_heif(path),
        e if RAW.contains(&e) => load_raw_preview(path),
        "avif" | "jxl" => bail!("{ext} decoding is not supported yet"),
        _ => load_common(path),
    }
}

fn load_common(path: &Path) -> Result<DynamicImage> {
    let mut reader = ImageReader::open(path)?.with_guessed_format()?;
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(1 << 30);
    reader.limits(limits);
    let mut decoder = reader.into_decoder()?;
    let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
    let mut img = DynamicImage::from_decoder(decoder)?;
    img.apply_orientation(orientation);
    Ok(img)
}

#[cfg(feature = "heif")]
fn load_heif(path: &Path) -> Result<DynamicImage> {
    use libheif_rs::{ColorSpace, HeifContext, LibHeif, RgbChroma};
    let lib = LibHeif::new();
    let ctx = HeifContext::read_from_file(path.to_str().context("non-UTF-8 path")?)?;
    let handle = ctx.primary_image_handle()?;
    // libheif applies the irot/imir transforms itself.
    let img = lib.decode(&handle, ColorSpace::Rgb(RgbChroma::Rgb), None)?;
    let planes = img.planes();
    let plane = planes.interleaved.context("no interleaved plane")?;
    let (w, h) = (plane.width, plane.height);
    let mut buf = Vec::with_capacity((w * h * 3) as usize);
    for row in 0..h as usize {
        let start = row * plane.stride;
        buf.extend_from_slice(&plane.data[start..start + w as usize * 3]);
    }
    let rgb = image::RgbImage::from_raw(w, h, buf).context("bad HEIF plane")?;
    Ok(DynamicImage::ImageRgb8(rgb))
}

#[cfg(not(feature = "heif"))]
fn load_heif(_path: &Path) -> Result<DynamicImage> {
    bail!("built without HEIF support")
}

/// Camera RAW files embed a full-size or large JPEG preview. Decoding that preview is far
/// cheaper than demosaicing the sensor data and is all an embedding model needs.
fn load_raw_preview(path: &Path) -> Result<DynamicImage> {
    let data = std::fs::read(path)?;
    let (start, end) = largest_embedded_jpeg(&data).context("no embedded JPEG preview in RAW file")?;
    let mut decoder = image::codecs::jpeg::JpegDecoder::new(std::io::Cursor::new(&data[start..end]))?;
    let preview_orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
    let mut img = DynamicImage::from_decoder(decoder)?;
    // The preview usually carries no orientation of its own; the RAW's EXIF does.
    let orientation = if preview_orientation == Orientation::NoTransforms {
        exif_orientation(path).unwrap_or(Orientation::NoTransforms)
    } else {
        preview_orientation
    };
    img.apply_orientation(orientation);
    Ok(img)
}

/// Finds the embedded JPEG with the most pixels. Returns its byte range.
pub fn largest_embedded_jpeg(data: &[u8]) -> Option<(usize, usize)> {
    let mut best: Option<(u64, usize, usize)> = None;
    let mut i = 0;
    while i + 3 < data.len() {
        if data[i] == 0xFF
            && data[i + 1] == 0xD8
            && data[i + 2] == 0xFF
            && let Some((end, w, h)) = walk_jpeg(data, i)
        {
            let px = w as u64 * h as u64;
            if best.is_none_or(|b| px > b.0) {
                best = Some((px, i, end));
            }
            i = end;
            continue;
        }
        i += 1;
    }
    best.filter(|b| b.0 > 0).map(|b| (b.1, b.2))
}

/// Walks JPEG segments from an SOI marker. Returns (end offset, width, height).
fn walk_jpeg(d: &[u8], start: usize) -> Option<(usize, u16, u16)> {
    let be16 = |p: usize| -> Option<u16> { Some(u16::from_be_bytes([*d.get(p)?, *d.get(p + 1)?])) };
    let (mut w, mut h) = (0u16, 0u16);
    let mut p = start + 2;
    loop {
        if *d.get(p)? != 0xFF {
            return None;
        }
        while *d.get(p + 1)? == 0xFF {
            p += 1;
        }
        let m = *d.get(p + 1)?;
        match m {
            0xD9 => return Some((p + 2, w, h)),
            0x01 | 0xD0..=0xD7 => {
                p += 2;
                continue;
            }
            _ => {}
        }
        let len = be16(p + 2)? as usize;
        if len < 2 {
            return None;
        }
        if matches!(m, 0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF) {
            h = be16(p + 5)?;
            w = be16(p + 7)?;
        }
        p += 2 + len;
        if m == 0xDA {
            // Entropy-coded data runs until a marker that isn't stuffing or a restart.
            loop {
                if *d.get(p)? == 0xFF {
                    let n = *d.get(p + 1)?;
                    if n == 0x00 || (0xD0..=0xD7).contains(&n) || n == 0xFF {
                        p += if n == 0xFF { 1 } else { 2 };
                        continue;
                    }
                    break;
                }
                p += 1;
            }
        }
    }
}

fn exif_orientation(path: &Path) -> Option<Orientation> {
    let exif = read_exif(path)?;
    let field = exif.get_field(exif::Tag::Orientation, exif::In::PRIMARY)?;
    let v = field.value.get_uint(0)?;
    Orientation::from_exif(v as u8)
}

fn read_exif(path: &Path) -> Option<exif::Exif> {
    let file = std::fs::File::open(path).ok()?;
    let mut reader = BufReader::new(file.take(64 * 1024 * 1024));
    exif::Reader::new().read_from_container(&mut reader).ok()
}

/// Capture time from EXIF (DateTimeOriginal, falling back to DateTime), as Unix seconds.
/// EXIF times carry no zone; they're read as UTC, which is fine for date filters.
pub fn taken(path: &Path) -> Option<i64> {
    let exif = read_exif(path)?;
    let field = exif
        .get_field(exif::Tag::DateTimeOriginal, exif::In::PRIMARY)
        .or_else(|| exif.get_field(exif::Tag::DateTime, exif::In::PRIMARY))?;
    let exif::Value::Ascii(ref parts) = field.value else { return None };
    let dt = exif::DateTime::from_ascii(parts.first()?).ok()?;
    let date = chrono::NaiveDate::from_ymd_opt(dt.year as i32, dt.month as u32, dt.day as u32)?;
    let t = date.and_hms_opt(dt.hour as u32, dt.minute as u32, dt.second as u32)?;
    Some(t.and_utc().timestamp())
}

/// Writes a small JPEG thumbnail.
pub fn write_thumbnail(img: &DynamicImage, dest: &Path) -> Result<()> {
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let thumb = img.thumbnail(320, 320).to_rgb8();
    let tmp = dest.with_extension("tmp");
    let mut out = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 82).encode_image(&thumb)?;
    drop(out);
    std::fs::rename(&tmp, dest)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_jpeg(w: u32, h: u32) -> Vec<u8> {
        let img = image::RgbImage::from_pixel(w, h, image::Rgb([200, 30, 30]));
        let mut out = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 90).encode_image(&img).unwrap();
        out
    }

    #[test]
    fn finds_largest_embedded_jpeg() {
        let small = tiny_jpeg(16, 8);
        let big = tiny_jpeg(64, 48);
        let mut blob = b"II*\0 fake raw header".to_vec();
        blob.extend_from_slice(&small);
        blob.extend_from_slice(&[0u8, 1, 2, 0xFF, 0xD8, 0x00]);
        let at = blob.len();
        blob.extend_from_slice(&big);
        blob.extend_from_slice(b"sensor data");
        let (s, e) = largest_embedded_jpeg(&blob).unwrap();
        assert_eq!((s, e), (at, at + big.len()));
        let img = image::load_from_memory(&blob[s..e]).unwrap();
        assert_eq!((img.width(), img.height()), (64, 48));
    }
}
