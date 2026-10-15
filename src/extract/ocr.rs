//! Text in screenshots, photos of documents and scanned PDFs, through Tesseract.

use std::process::Command;
use std::sync::OnceLock;
use std::time::Duration;

use anyhow::Result;
use image::{DynamicImage, ImageEncoder};

use super::tool;

const TIMEOUT: Duration = Duration::from_secs(120);
/// Words Tesseract is less sure of than this are dropped: on photos they are mostly noise.
const MIN_WORD_CONF: f32 = 60.0;
/// A result with fewer real words than this is treated as "no text".
const MIN_WORDS: usize = 4;
/// Longest side handed to Tesseract.
const MAX_SIDE: u32 = 4000;

pub fn available() -> bool {
    static FOUND: OnceLock<bool> = OnceLock::new();
    *FOUND.get_or_init(|| tool::which("tesseract").is_some())
}

/// Tesseract language codes (`eng+deu`). Empty `configured` means English plus the
/// language of the user's locale, among those installed.
pub fn languages(configured: &[String]) -> String {
    if !configured.is_empty() {
        return configured.join("+");
    }
    static AUTO: OnceLock<String> = OnceLock::new();
    AUTO.get_or_init(|| {
        let installed = installed_languages();
        let locale = ["LC_ALL", "LC_MESSAGES", "LANG"]
            .iter()
            .filter_map(|v| std::env::var(v).ok())
            .find(|v| !v.is_empty() && v != "C" && v != "POSIX")
            .unwrap_or_default();
        pick_languages(&installed, &locale)
    })
    .clone()
}

fn installed_languages() -> Vec<String> {
    let Ok(out) = tool::run(Command::new("tesseract").arg("--list-langs"), None, Duration::from_secs(10)) else {
        return vec!["eng".into()];
    };
    // The first line is a heading ("List of available languages in ...").
    String::from_utf8_lossy(&out).lines().skip(1).map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect()
}

fn pick_languages(installed: &[String], locale: &str) -> String {
    let has = |l: &str| installed.iter().any(|i| i == l);
    let mut langs = Vec::new();
    if has("eng") {
        langs.push("eng".to_string());
    }
    let code = locale.split(['_', '.', '@']).next().unwrap_or("");
    if let Some(l) = tesseract_code(code)
        && has(l)
        && l != "eng"
    {
        langs.push(l.to_string());
    }
    if langs.is_empty()
        && let Some(first) = installed.iter().find(|l| *l != "osd" && *l != "equ")
    {
        langs.push(first.clone());
    }
    if langs.is_empty() { "eng".into() } else { langs.join("+") }
}

/// ISO 639-1 to Tesseract's ISO 639-2 names, for the languages people most often have installed.
fn tesseract_code(iso: &str) -> Option<&'static str> {
    Some(match iso {
        "en" => "eng",
        "de" => "deu",
        "fr" => "fra",
        "es" => "spa",
        "it" => "ita",
        "pt" => "por",
        "nl" => "nld",
        "sv" => "swe",
        "da" => "dan",
        "nb" | "nn" | "no" => "nor",
        "fi" => "fin",
        "pl" => "pol",
        "cs" => "ces",
        "sk" => "slk",
        "hu" => "hun",
        "ro" => "ron",
        "tr" => "tur",
        "el" => "ell",
        "ru" => "rus",
        "uk" => "ukr",
        "ja" => "jpn",
        "zh" => "chi_sim",
        "ko" => "kor",
        "ar" => "ara",
        "he" => "heb",
        "hi" => "hin",
        "th" => "tha",
        "vi" => "vie",
        "id" => "ind",
        _ => return None,
    })
}

/// Prepares an image for Tesseract: grayscale, large text scaled down to a sane size and
/// small crops scaled up (Tesseract wants letters at least ~20 px tall).
pub fn prepare_image(img: &DynamicImage) -> Result<Vec<u8>> {
    let gray = img.to_luma8();
    let (w, h) = gray.dimensions();
    let long = w.max(h);
    let gray = if long > MAX_SIDE {
        let s = MAX_SIDE as f32 / long as f32;
        image::imageops::resize(
            &gray,
            (w as f32 * s) as u32,
            (h as f32 * s) as u32,
            image::imageops::FilterType::Triangle,
        )
    } else if long < 1000 {
        image::imageops::resize(&gray, w * 2, h * 2, image::imageops::FilterType::CatmullRom)
    } else {
        gray
    };
    let mut png = Vec::new();
    image::codecs::png::PngEncoder::new_with_quality(
        &mut png,
        image::codecs::png::CompressionType::Fast,
        image::codecs::png::FilterType::NoFilter,
    )
    .write_image(gray.as_raw(), gray.width(), gray.height(), image::ExtendedColorType::L8)?;
    Ok(png)
}

/// Recognised text, one line per text line and a blank line between paragraphs.
/// `None` when the image holds no (believable) text.
pub fn recognize(png: Vec<u8>, langs: &str) -> Result<Option<String>> {
    let out = tool::run(
        Command::new("tesseract")
            .args(["stdin", "stdout", "-l", langs, "--psm", "3", "tsv"])
            // One thread per page: the indexer runs in the background and must stay out of the way.
            .env("OMP_THREAD_LIMIT", "1"),
        Some(png),
        TIMEOUT,
    )?;
    Ok(parse_tsv(&String::from_utf8_lossy(&out)))
}

/// Builds text from Tesseract's TSV output, keeping confident words only.
fn parse_tsv(tsv: &str) -> Option<String> {
    let mut text = String::new();
    let mut words = 0;
    let mut last: Option<(u32, u32, u32)> = None;
    for line in tsv.lines().skip(1) {
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 12 || cols[0] != "5" {
            continue;
        }
        let conf: f32 = cols[10].parse().unwrap_or(-1.0);
        let word = cols[11].trim();
        if word.is_empty() || conf < MIN_WORD_CONF {
            continue;
        }
        let key = (cols[2].parse().unwrap_or(0), cols[3].parse().unwrap_or(0), cols[4].parse().unwrap_or(0));
        match last {
            Some(l) if (l.0, l.1) != (key.0, key.1) => text.push_str("\n\n"),
            Some(l) if l.2 != key.2 => text.push('\n'),
            Some(_) => text.push(' '),
            None => {}
        }
        last = Some(key);
        text.push_str(word);
        if word.chars().filter(|c| c.is_alphanumeric()).count() >= 2 {
            words += 1;
        }
    }
    (words >= MIN_WORDS).then_some(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_locale_language() {
        let installed: Vec<String> = ["eng", "deu", "osd"].iter().map(|s| s.to_string()).collect();
        assert_eq!(pick_languages(&installed, "de_DE.UTF-8"), "eng+deu");
        assert_eq!(pick_languages(&installed, "fr_FR.UTF-8"), "eng");
        assert_eq!(pick_languages(&["osd".into(), "fra".into()], "en_US.UTF-8"), "fra");
    }

    #[test]
    fn keeps_confident_words() {
        let tsv = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext
5\t1\t1\t1\t1\t1\t0\t0\t1\t1\t96\tInvoice
5\t1\t1\t1\t1\t2\t0\t0\t1\t1\t91\tnumber
5\t1\t1\t1\t1\t3\t0\t0\t1\t1\t12\t~%
5\t1\t1\t1\t2\t1\t0\t0\t1\t1\t88\tTotal:
5\t1\t2\t1\t1\t1\t0\t0\t1\t1\t90\t$42.00";
        assert_eq!(parse_tsv(tsv).as_deref(), Some("Invoice number\nTotal:\n\n$42.00"));
        let noise = "h\n5\t1\t1\t1\t1\t1\t0\t0\t1\t1\t30\tIl\n5\t1\t1\t1\t1\t2\t0\t0\t1\t1\t70\tok";
        assert_eq!(parse_tsv(noise), None);
    }
}
