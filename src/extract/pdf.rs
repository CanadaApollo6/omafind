//! PDF text through poppler's `pdftotext`, and page images through `pdftoppm` for OCR.

use std::path::Path;
use std::process::Command;
use std::time::Duration;

use anyhow::Result;

use super::tool;

const TIMEOUT: Duration = Duration::from_secs(60);
pub const MAX_PAGES: usize = 300;

/// The text layer of each page.
pub fn pages(path: &Path) -> Result<Vec<String>> {
    let out = tool::run(
        Command::new("pdftotext").args(["-q", "-enc", "UTF-8", "-l", &MAX_PAGES.to_string()]).arg(path).arg("-"),
        None,
        TIMEOUT,
    )
    .map_err(|e| if super::tool::which("pdftotext").is_none() { e.context("install poppler") } else { e })?;
    let text = String::from_utf8_lossy(&out);
    // Form feeds end pages; the last one is followed by nothing.
    let mut pages: Vec<String> = text.split('\u{c}').map(str::to_string).collect();
    if pages.len() > 1 && pages.last().is_some_and(|p| p.trim().is_empty()) {
        pages.pop();
    }
    Ok(pages)
}

/// Pages (1-based) with too little text to be anything but a scan or a picture.
pub fn pages_without_text(pages: &[String]) -> Vec<usize> {
    pages
        .iter()
        .enumerate()
        .filter(|(_, p)| p.chars().filter(|c| c.is_alphanumeric()).count() < 40)
        .map(|(i, _)| i + 1)
        .collect()
}

/// Renders one page (1-based) as a grayscale PNG at `dpi`.
pub fn render_page(path: &Path, page: usize, dpi: u32) -> Result<Vec<u8>> {
    let p = page.to_string();
    tool::run(
        Command::new("pdftoppm")
            .args(["-q", "-gray", "-png", "-r", &dpi.to_string(), "-f", &p, "-l", &p, "-singlefile"])
            // No output root: the page goes to stdout. ("-" would be taken as a file name.)
            .arg(path),
        None,
        TIMEOUT,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_pages_without_text() {
        let pages = vec!["A page with plenty of words on it, more than forty letters.".to_string(), " \n 3 ".into()];
        assert_eq!(pages_without_text(&pages), vec![2]);
    }
}
