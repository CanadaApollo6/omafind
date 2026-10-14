//! Office Open XML, OpenDocument, EPUB and spreadsheets.

use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result};
use quick_xml::Reader;
use quick_xml::events::Event;

const MAX_XML_BYTES: u64 = 64 * 1024 * 1024;

/// Text of a zip-packaged document: docx, pptx, odt, odp or epub.
pub fn extract_zip_doc(path: &Path, ext: &str) -> Result<String> {
    let file = std::fs::File::open(path)?;
    let mut zip = zip::ZipArchive::new(file).context("not a zip package")?;
    let mut names: Vec<String> = (0..zip.len())
        .filter_map(|i| zip.by_index(i).ok().and_then(|f| f.name().ok().map(|n| n.to_string())))
        .filter(|n| match ext {
            "docx" => n == "word/document.xml" || n.starts_with("word/footnotes") || n.starts_with("word/endnotes"),
            "pptx" => n.starts_with("ppt/slides/slide") && n.ends_with(".xml"),
            "odt" | "odp" => n == "content.xml",
            "epub" => n.ends_with(".xhtml") || n.ends_with(".html") || n.ends_with(".htm"),
            _ => false,
        })
        .collect();
    // slide2 before slide10.
    names.sort_by_key(|n| {
        let digits: String = n.chars().filter(|c| c.is_ascii_digit()).collect();
        (digits.len(), digits, n.clone())
    });
    let mut out = String::new();
    for name in names {
        let mut xml = String::new();
        zip.by_name(&name)?.take(MAX_XML_BYTES).read_to_string(&mut xml)?;
        out.push_str(&xml_text(&xml));
        out.push_str("\n\n");
    }
    Ok(out)
}

/// Collects character data, breaking lines at paragraph-like elements.
pub fn xml_text(xml: &str) -> String {
    let mut reader = Reader::from_str(xml);
    let mut out = String::new();
    let mut skip_depth = 0usize;
    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) => {
                let name = e.local_name();
                let n = name.as_ref();
                if skip_depth > 0 || matches!(n, "script" | "style" | "head" | "instrText") {
                    skip_depth += 1;
                }
            }
            Ok(Event::End(e)) => {
                if skip_depth > 0 {
                    skip_depth -= 1;
                    continue;
                }
                let name = e.local_name();
                if matches!(name.as_ref(), "p" | "h" | "h1" | "h2" | "h3" | "h4" | "li" | "tr" | "div" | "title")
                    && !out.ends_with('\n')
                {
                    out.push('\n');
                }
            }
            Ok(Event::Empty(e)) => {
                if skip_depth == 0 {
                    match e.local_name().as_ref() {
                        "tab" => out.push('\t'),
                        "br" | "line-break" | "cr" => out.push('\n'),
                        "s" => out.push(' '),
                        _ => {}
                    }
                }
            }
            Ok(Event::Text(t)) if skip_depth == 0 => out.push_str(&t.into_inner()),
            Ok(Event::CData(t)) if skip_depth == 0 => out.push_str(&t.into_inner()),
            Ok(Event::GeneralRef(r)) if skip_depth == 0 => {
                if let Ok(Some(c)) = r.resolve_char_ref() {
                    out.push(c);
                } else {
                    out.push_str(match r.as_ref() {
                        "amp" => "&",
                        "lt" => "<",
                        "gt" => ">",
                        "quot" => "\"",
                        "apos" => "'",
                        "nbsp" => " ",
                        _ => "",
                    });
                }
            }
            Ok(Event::Eof) | Err(_) => break,
            _ => {}
        }
    }
    out
}

/// Spreadsheets: each row as tab-separated cells, sheets separated by their names.
pub fn extract_sheet(path: &Path) -> Result<String> {
    use calamine::Reader as _;
    let mut wb = calamine::open_workbook_auto(path).context("opening spreadsheet")?;
    let mut out = String::new();
    for name in wb.sheet_names().to_owned() {
        let Ok(range) = wb.worksheet_range(&name) else { continue };
        out.push_str(&format!("# {name}\n"));
        for row in range.rows().take(20_000) {
            let cells: Vec<String> = row.iter().map(|c| c.to_string()).collect();
            if cells.iter().any(|c| !c.is_empty()) {
                out.push_str(cells.join("\t").trim_end());
                out.push('\n');
            }
        }
        out.push('\n');
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn docx_xml() {
        let xml = r#"<w:document xmlns:w="x"><w:body><w:p><w:r><w:t>Hello</w:t></w:r><w:r><w:tab/><w:t>world &amp; co</w:t></w:r></w:p><w:p><w:r><w:t>Second</w:t></w:r></w:p></w:body></w:document>"#;
        assert_eq!(xml_text(xml), "Hello\tworld & co\nSecond\n");
    }

    #[test]
    fn odt_xml() {
        let xml = r#"<office:document-content xmlns:office="o" xmlns:text="t"><office:body><text:h>Title</text:h><text:p>One<text:s/>two</text:p></office:body></office:document-content>"#;
        assert_eq!(xml_text(xml), "Title\nOne two\n");
    }

    #[test]
    fn docx_roundtrip() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("t.docx");
        let mut zw = zip::ZipWriter::new(std::fs::File::create(&p).unwrap());
        zw.start_file("word/document.xml", zip::write::SimpleFileOptions::default()).unwrap();
        zw.write_all(
            br#"<w:document xmlns:w="x"><w:body><w:p><w:r><w:t>Quarterly plan</w:t></w:r></w:p></w:body></w:document>"#,
        )
        .unwrap();
        zw.finish().unwrap();
        assert!(extract_zip_doc(&p, "docx").unwrap().contains("Quarterly plan"));
    }
}
