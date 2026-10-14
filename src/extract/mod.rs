//! File kinds and content readers.

pub mod image;
pub mod office;
pub mod pdf;
pub mod text;

use std::path::Path;

use anyhow::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Kind {
    Document,
    Code,
    Image,
    Audio,
    Video,
    Other,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Document => "document",
            Kind::Code => "code",
            Kind::Image => "image",
            Kind::Audio => "audio",
            Kind::Video => "video",
            Kind::Other => "other",
        }
    }

    pub fn parse(s: &str) -> Option<Kind> {
        Some(match s.to_ascii_lowercase().trim_end_matches('s') {
            "document" | "doc" | "text" | "pdf" | "note" => Kind::Document,
            "code" | "source" | "src" => Kind::Code,
            "image" | "photo" | "picture" | "pic" | "img" => Kind::Image,
            "audio" | "music" | "song" | "sound" => Kind::Audio,
            "video" | "movie" | "clip" => Kind::Video,
            "other" | "file" => Kind::Other,
            _ => return None,
        })
    }

    /// Group heading shown in result lists.
    pub fn group_title(self) -> &'static str {
        match self {
            Kind::Document => "Documents",
            Kind::Code => "Code",
            Kind::Image => "Images",
            Kind::Audio => "Audio",
            Kind::Video => "Videos",
            Kind::Other => "Other",
        }
    }

    /// Whether the text reader runs for this kind.
    pub fn has_text(self) -> bool {
        matches!(self, Kind::Document | Kind::Code)
    }
}

const CODE: &[&str] = &[
    "rs",
    "py",
    "pyi",
    "js",
    "mjs",
    "cjs",
    "ts",
    "tsx",
    "jsx",
    "go",
    "c",
    "h",
    "cpp",
    "hpp",
    "cc",
    "hh",
    "cxx",
    "java",
    "kt",
    "kts",
    "swift",
    "rb",
    "php",
    "sh",
    "bash",
    "zsh",
    "fish",
    "lua",
    "pl",
    "pm",
    "r",
    "scala",
    "cs",
    "fs",
    "m",
    "mm",
    "sql",
    "css",
    "scss",
    "sass",
    "less",
    "vue",
    "svelte",
    "zig",
    "nim",
    "ex",
    "exs",
    "erl",
    "hrl",
    "hs",
    "ml",
    "mli",
    "clj",
    "cljs",
    "dart",
    "toml",
    "yaml",
    "yml",
    "json",
    "jsonc",
    "ini",
    "cfg",
    "conf",
    "xml",
    "gradle",
    "cmake",
    "tf",
    "nix",
    "proto",
    "graphql",
    "qml",
    "el",
    "vim",
    "ps1",
    "bat",
    "asm",
    "s",
    "v",
    "sv",
    "vhd",
    "jl",
    "groovy",
    "hcl",
    "dockerfile",
    "mk",
    "kdl",
    "ron",
];

const DOCUMENT: &[&str] = &[
    "md", "markdown", "mdx", "txt", "text", "rst", "org", "tex", "bib", "csv", "tsv", "log", "adoc", "asciidoc",
    "html", "htm", "xhtml", "eml", "srt", "vtt", "pdf", "docx", "odt", "pptx", "odp", "xlsx", "xlsm", "xls", "ods",
    "epub", "rtf", "ipynb",
];

pub const RAW: &[&str] = &[
    "cr2", "cr3", "crw", "nef", "nrw", "arw", "srf", "sr2", "dng", "orf", "rw2", "raf", "pef", "srw", "x3f", "3fr",
    "iiq", "erf", "mrw", "kdc", "dcr", "mos", "rwl",
];

const IMAGE: &[&str] =
    &["jpg", "jpeg", "jpe", "png", "webp", "gif", "bmp", "tif", "tiff", "heic", "heif", "hif", "avif", "ico", "jxl"];

const AUDIO: &[&str] = &["mp3", "flac", "wav", "ogg", "oga", "opus", "m4a", "aac", "wma", "aiff", "aif", "alac", "mka"];

const VIDEO: &[&str] = &["mp4", "m4v", "mkv", "webm", "mov", "avi", "wmv", "flv", "mpg", "mpeg", "3gp", "ts", "mts"];

const CODE_NAMES: &[&str] =
    &["makefile", "dockerfile", "containerfile", "justfile", "cmakelists.txt", "pkgbuild", "rakefile", "gemfile"];

pub fn ext_of(path: &Path) -> String {
    path.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default()
}

/// Classifies a path by extension. Files without a known extension are `Other`
/// (the indexer may promote them to `Document` after sniffing for text).
pub fn kind_of(path: &Path) -> Kind {
    let ext = ext_of(path);
    let name = path.file_name().map(|n| n.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
    if CODE_NAMES.contains(&name.as_str()) {
        return Kind::Code;
    }
    let e = ext.as_str();
    // `.ts` is TypeScript far more often than MPEG transport streams on a dev machine.
    if CODE.contains(&e) {
        Kind::Code
    } else if DOCUMENT.contains(&e) {
        Kind::Document
    } else if IMAGE.contains(&e) || RAW.contains(&e) {
        Kind::Image
    } else if AUDIO.contains(&e) {
        Kind::Audio
    } else if VIDEO.contains(&e) {
        Kind::Video
    } else {
        Kind::Other
    }
}

/// Extracted text plus an optional title (used as the Gemma document title).
pub struct Extracted {
    pub text: String,
    pub title: Option<String>,
}

/// Reads the text content of a document or code file.
pub fn extract_text(path: &Path, max_bytes: u64) -> Result<Extracted> {
    let ext = ext_of(path);
    let text = match ext.as_str() {
        "pdf" => pdf::extract(path)?,
        "docx" | "pptx" | "odt" | "odp" | "epub" => office::extract_zip_doc(path, &ext)?,
        "xlsx" | "xlsm" | "xls" | "ods" => office::extract_sheet(path)?,
        "html" | "htm" | "xhtml" => text::strip_html(&text::read_text(path, max_bytes)?),
        "ipynb" => text::extract_notebook(&text::read_text(path, max_bytes)?),
        "rtf" => text::strip_rtf(&text::read_text(path, max_bytes)?),
        _ => text::read_text(path, max_bytes)?,
    };
    let mut text = text;
    if text.len() as u64 > max_bytes {
        let mut cut = max_bytes as usize;
        while !text.is_char_boundary(cut) {
            cut -= 1;
        }
        text.truncate(cut);
    }
    let title = text::guess_title(&text, &ext);
    Ok(Extracted { text, title })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds() {
        assert_eq!(kind_of(Path::new("/a/b.rs")), Kind::Code);
        assert_eq!(kind_of(Path::new("/a/Makefile")), Kind::Code);
        assert_eq!(kind_of(Path::new("/a/b.PDF")), Kind::Document);
        assert_eq!(kind_of(Path::new("/a/IMG_1.HEIC")), Kind::Image);
        assert_eq!(kind_of(Path::new("/a/DSC_1.NEF")), Kind::Image);
        assert_eq!(kind_of(Path::new("/a/song.flac")), Kind::Audio);
        assert_eq!(kind_of(Path::new("/a/x.bin")), Kind::Other);
        assert_eq!(Kind::parse("images"), Some(Kind::Image));
        assert_eq!(Kind::parse("docs"), Some(Kind::Document));
    }
}
