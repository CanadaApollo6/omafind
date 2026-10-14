//! Splits text into retrieval chunks along line and paragraph boundaries.

pub struct Chunk {
    /// 1-based line where the chunk starts.
    pub line: usize,
    pub text: String,
}

/// Chunks prefer to end at a blank line once they reach `SOFT` bytes and always end by `HARD`.
const SOFT: usize = 700;
const HARD: usize = 1400;
pub const MAX_CHUNKS: usize = 4000;

pub fn chunk_text(text: &str) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut cur = String::new();
    let mut cur_line = 1;
    let flush = |cur: &mut String, line: usize, chunks: &mut Vec<Chunk>| {
        if !cur.trim().is_empty() {
            chunks.push(Chunk { line, text: std::mem::take(cur).trim_end().to_string() });
        }
        cur.clear();
    };
    for (i, line) in text.lines().enumerate() {
        let lineno = i + 1;
        if chunks.len() >= MAX_CHUNKS {
            break;
        }
        if line.trim().is_empty() {
            if cur.len() >= SOFT {
                flush(&mut cur, cur_line, &mut chunks);
            } else if !cur.is_empty() {
                cur.push('\n');
            }
            continue;
        }
        if cur.is_empty() {
            cur_line = lineno;
        }
        if line.len() > HARD {
            // Minified or wrapped-paragraph text: hard-split the line itself.
            flush(&mut cur, cur_line, &mut chunks);
            let mut rest = line;
            while !rest.is_empty() && chunks.len() < MAX_CHUNKS {
                let mut cut = rest.len().min(HARD);
                while !rest.is_char_boundary(cut) {
                    cut -= 1;
                }
                if cut < rest.len()
                    && let Some(sp) = rest[..cut].rfind(char::is_whitespace)
                    && sp > HARD / 2
                {
                    cut = sp;
                }
                chunks.push(Chunk { line: lineno, text: rest[..cut].trim().to_string() });
                rest = &rest[cut..];
            }
            continue;
        }
        if cur.len() + line.len() > HARD {
            flush(&mut cur, cur_line, &mut chunks);
            cur_line = lineno;
        }
        cur.push_str(line);
        cur.push('\n');
    }
    if chunks.len() < MAX_CHUNKS {
        flush(&mut cur, cur_line, &mut chunks);
    }
    chunks.retain(|c| !c.text.is_empty());
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_on_paragraphs_with_line_numbers() {
        let para = "word ".repeat(160);
        let text = format!("{para}\n\n{para}\n\nshort tail\n");
        let chunks = chunk_text(&text);
        assert_eq!(chunks.len(), 3, "{:?}", chunks.iter().map(|c| c.text.len()).collect::<Vec<_>>());
        assert_eq!(chunks[0].line, 1);
        assert_eq!(chunks[1].line, 3);
        assert_eq!(chunks[2].line, 5);
        assert_eq!(chunks[2].text, "short tail");
    }

    #[test]
    fn hard_splits_long_lines() {
        let text = "x".repeat(5000);
        let chunks = chunk_text(&text);
        assert!(chunks.len() >= 4);
        assert!(chunks.iter().all(|c| c.text.len() <= HARD));
    }

    #[test]
    fn empty_text_has_no_chunks() {
        assert!(chunk_text("\n\n   \n").is_empty());
    }
}
