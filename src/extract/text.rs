//! Plain text, code, markdown and light markup.

use std::io::Read;
use std::path::Path;

use anyhow::{Result, bail};

/// Reads a text file as UTF-8 (lossy), refusing binary content.
pub fn read_text(path: &Path, max_bytes: u64) -> Result<String> {
    let mut buf = Vec::new();
    std::fs::File::open(path)?.take(max_bytes).read_to_end(&mut buf)?;
    if looks_binary(&buf) {
        bail!("binary content");
    }
    let mut s = String::from_utf8(buf).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned());
    if s.starts_with('\u{feff}') {
        s.drain(..3);
    }
    Ok(s)
}

/// True when a prefix of the content contains NUL bytes or is mostly non-text.
pub fn looks_binary(buf: &[u8]) -> bool {
    let head = &buf[..buf.len().min(8192)];
    if head.contains(&0) {
        return true;
    }
    let controls = head.iter().filter(|&&b| b < 0x09 || (b > 0x0d && b < 0x20)).count();
    controls * 20 > head.len().max(1)
}

/// Sniffs a file with an unknown extension to decide whether it's text worth indexing.
pub fn is_probably_text(path: &Path) -> bool {
    let mut buf = [0u8; 4096];
    let Ok(mut f) = std::fs::File::open(path) else { return false };
    let Ok(n) = f.read(&mut buf) else { return false };
    n > 0 && !looks_binary(&buf[..n]) && std::str::from_utf8(&buf[..n.min(4000)]).is_ok()
}

/// Removes tags, scripts and styles from HTML and decodes common entities.
pub fn strip_html(html: &str) -> String {
    let mut out = String::with_capacity(html.len() / 2);
    let lower = html.to_ascii_lowercase();
    let bytes = html.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'<' {
            for skip in ["script", "style", "noscript"] {
                if lower[i + 1..].starts_with(skip) {
                    let close = format!("</{skip}");
                    if let Some(end) = lower[i..].find(&close) {
                        i += end;
                    }
                }
            }
            let block = ["<p", "<br", "<div", "<li", "<h1", "<h2", "<h3", "<h4", "<tr", "</p", "</div", "</h"]
                .iter()
                .any(|t| lower[i..].starts_with(t));
            match html[i..].find('>') {
                Some(end) => i += end + 1,
                None => break,
            }
            if block && !out.ends_with('\n') {
                out.push('\n');
            }
            continue;
        }
        let next = html[i..].find('<').map(|n| i + n).unwrap_or(html.len());
        out.push_str(&decode_entities(&html[i..next]));
        i = next;
    }
    collapse_blank_lines(&out)
}

fn decode_entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    s.replace("&nbsp;", " ")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

fn collapse_blank_lines(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut blank = 0;
    for line in s.lines() {
        let t = line.trim_end();
        if t.trim().is_empty() {
            blank += 1;
            if blank > 1 {
                continue;
            }
        } else {
            blank = 0;
        }
        out.push_str(t);
        out.push('\n');
    }
    out
}

/// Markdown and code cells of a Jupyter notebook.
pub fn extract_notebook(json: &str) -> String {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(json) else { return String::new() };
    let mut out = String::new();
    for cell in v["cells"].as_array().into_iter().flatten() {
        match &cell["source"] {
            serde_json::Value::Array(lines) => {
                for l in lines {
                    out.push_str(l.as_str().unwrap_or_default());
                }
            }
            serde_json::Value::String(s) => out.push_str(s),
            _ => {}
        }
        out.push_str("\n\n");
    }
    out
}

/// Crude RTF to text: drops control words and groups like fonttbl.
pub fn strip_rtf(rtf: &str) -> String {
    let mut out = String::new();
    let mut depth_skip: Option<usize> = None;
    let mut depth = 0usize;
    let mut chars = rtf.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '{' => depth += 1,
            '}' => {
                if depth_skip == Some(depth) {
                    depth_skip = None;
                }
                depth = depth.saturating_sub(1);
            }
            '\\' => {
                let mut word = String::new();
                while let Some(&n) = chars.peek() {
                    if n.is_ascii_alphabetic() {
                        word.push(n);
                        chars.next();
                    } else {
                        break;
                    }
                }
                while let Some(&n) = chars.peek() {
                    if n.is_ascii_digit() || n == '-' {
                        chars.next();
                    } else {
                        break;
                    }
                }
                if chars.peek() == Some(&' ') {
                    chars.next();
                }
                if word.is_empty() {
                    if let Some(n) = chars.next() {
                        if n == '\'' {
                            chars.next();
                            chars.next();
                        } else if depth_skip.is_none() {
                            out.push(n);
                        }
                    }
                } else if matches!(word.as_str(), "fonttbl" | "colortbl" | "stylesheet" | "info" | "pict" | "header") {
                    depth_skip.get_or_insert(depth);
                } else if matches!(word.as_str(), "par" | "line") && depth_skip.is_none() {
                    out.push('\n');
                }
            }
            '\r' | '\n' => {}
            _ if depth_skip.is_none() => out.push(c),
            _ => {}
        }
    }
    out
}

/// First markdown heading or first short line, used as a document title.
pub fn guess_title(text: &str, ext: &str) -> Option<String> {
    if matches!(ext, "md" | "markdown" | "mdx" | "org" | "rst" | "txt" | "") {
        for line in text.lines().take(20) {
            let t = line.trim();
            if let Some(h) = t.strip_prefix("# ").or_else(|| t.strip_prefix("#+title: ")) {
                return Some(h.trim().chars().take(120).collect());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_is_stripped() {
        let s = strip_html(
            "<html><head><style>p{}</style></head><body><h1>Title</h1><p>Hello &amp; bye</p><script>x()</script></body></html>",
        );
        assert!(s.contains("Title"));
        assert!(s.contains("Hello & bye"));
        assert!(!s.contains("x()"));
        assert!(!s.contains("p{}"));
    }

    #[test]
    fn rtf_is_stripped() {
        let s = strip_rtf(r"{\rtf1\ansi{\fonttbl\f0\fswiss Helvetica;}\f0\pard Hello {\b world}\par Next}");
        assert!(s.contains("Hello world"), "{s}");
        assert!(!s.contains("Helvetica"));
    }

    #[test]
    fn binary_detection() {
        assert!(looks_binary(b"abc\0def"));
        assert!(!looks_binary("plain text\nwith lines".as_bytes()));
    }
}
