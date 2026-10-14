//! The two-stage search path.
//!
//! **Fast** (every keystroke): file names (FTS5 prefix match), exact words (FTS5 BM25 over
//! chunks) and static-embedding nearest neighbours, fused with reciprocal rank fusion.
//!
//! **Deep** (after a pause): EmbeddingGemma embeds the query, searches the image vectors
//! and re-scores the fast stage's text shortlist. Text-to-text and text-to-image scores
//! are on different scales, so results stay grouped by kind instead of being merged.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use rusqlite::{Connection, OptionalExtension, params_from_iter};
use serde::Serialize;

use crate::config::{self, Config};
use crate::db;
use crate::embed::GemmaSlot;
use crate::embed::gemma::{CODE_QUERY_PREFIX, QUERY_PREFIX, document_input, dot};
use crate::embed::static_model::{Space, StaticEmbedder};
use crate::extract::Kind;
use crate::vecindex::StaticIndex;

/// (chunk id, file id, chunk text, file name, file kind, cached Gemma vector).
type ChunkRow = (i64, i64, String, String, String, Option<Vec<u8>>);

const RRF_K: f64 = 60.0;
const KEYWORD_K: usize = 100;
const STATIC_K: usize = 100;
const NAME_K: usize = 60;
/// Chunks the keyword stage may rank per query (~0.5 µs each).
const KEYWORD_BUDGET: i64 = 20_000;
const TERM_COUNT_MAX_AGE_SECS: u64 = 600;
const IMAGE_K: usize = 80;

#[derive(Debug, Clone, Default, Serialize)]
pub struct Query {
    pub text: String,
    pub kinds: Vec<Kind>,
    pub exts: Vec<String>,
    pub after: Option<i64>,
    pub before: Option<i64>,
    pub paths: Vec<String>,
}

impl Serialize for Kind {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

/// Parses `kind:image`, `ext:pdf`, `after:2024-03`, `before:2w`, `year:2023` and `in:path`
/// out of the query; everything else is the search text.
pub fn parse_query(input: &str) -> Query {
    let mut q = Query::default();
    let mut words = Vec::new();
    for word in input.split_whitespace() {
        let Some((key, val)) = word.split_once(':') else {
            words.push(word);
            continue;
        };
        let val = val.trim();
        match key.to_ascii_lowercase().as_str() {
            "kind" | "type" | "is" if !val.is_empty() => {
                for v in val.split(',') {
                    if let Some(k) = Kind::parse(v) {
                        q.kinds.push(k);
                    }
                }
            }
            "ext" if !val.is_empty() => {
                q.exts.extend(val.split(',').map(|e| e.trim_start_matches('.').to_ascii_lowercase()))
            }
            "after" | "since" => q.after = parse_date(val),
            "before" | "until" => q.before = parse_date(val),
            "year" => {
                if let Ok(y) = val.parse::<i32>() {
                    q.after = ymd(y, 1, 1);
                    q.before = ymd(y + 1, 1, 1);
                }
            }
            "in" | "path" if !val.is_empty() => q.paths.push(val.to_string()),
            _ => words.push(word),
        }
    }
    q.text = words.join(" ");
    q
}

fn ymd(y: i32, m: u32, d: u32) -> Option<i64> {
    chrono::NaiveDate::from_ymd_opt(y, m, d).and_then(|d| d.and_hms_opt(0, 0, 0)).map(|t| t.and_utc().timestamp())
}

/// `2024`, `2024-03`, `2024-03-11`, `today`, `yesterday`, or a relative age like `7d`, `2w`, `3m`, `1y`.
pub fn parse_date(s: &str) -> Option<i64> {
    let now = chrono::Utc::now();
    let today = now.date_naive().and_hms_opt(0, 0, 0)?.and_utc().timestamp();
    match s {
        "today" => return Some(today),
        "yesterday" => return Some(today - 86_400),
        _ => {}
    }
    if let Some(n) = s.strip_suffix(['d', 'w', 'm', 'y']).and_then(|n| n.parse::<i64>().ok()) {
        let days = match s.chars().last()? {
            'd' => n,
            'w' => n * 7,
            'm' => n * 30,
            _ => n * 365,
        };
        return Some(today - days * 86_400);
    }
    let parts: Vec<&str> = s.split('-').collect();
    let y = parts.first()?.parse().ok()?;
    let m = parts.get(1).map(|v| v.parse().ok()).unwrap_or(Some(1))?;
    let d = parts.get(2).map(|v| v.parse().ok()).unwrap_or(Some(1))?;
    ymd(y, m, d)
}

#[derive(Debug, Clone, Serialize)]
pub struct Hit {
    #[serde(skip)]
    pub file_id: i64,
    pub path: String,
    pub name: String,
    /// Parent folder, with the home directory shortened to `~`.
    pub dir: String,
    pub kind: Kind,
    pub score: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thumb: Option<String>,
    /// Capture time for photos, otherwise modification time (Unix seconds).
    pub date: i64,
    /// Which signals matched: name, keyword, semantic, gemma, vision.
    pub why: Vec<&'static str>,
    /// Deep stage: Gemma similarity to the query above its unrelated-content level.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sim: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Stage {
    Fast,
    Deep,
}

#[derive(Debug, Clone, Serialize)]
pub struct Group {
    pub kind: Kind,
    pub title: &'static str,
    pub hits: Vec<Hit>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Results {
    pub stage: Stage,
    pub query: String,
    pub elapsed_ms: f64,
    pub groups: Vec<Group>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// Ranked hits of one stage, plus what the deep stage needs from the fast one.
pub struct Outcome {
    pub stage: Stage,
    /// Sorted by score within each kind.
    pub hits: Vec<Hit>,
    pub elapsed_ms: f64,
    pub note: Option<String>,
    /// Chunk shortlist for Gemma, best first: (chunk id, file id).
    candidates: Vec<(i64, i64)>,
    /// File-name matches, best first.
    name_ranked: Vec<i64>,
    /// Fast-stage file ranking.
    fast_ranked: Vec<i64>,
}

impl Outcome {
    pub fn results(&self, query: &str, limit: usize) -> Results {
        Results {
            stage: self.stage,
            query: query.to_string(),
            elapsed_ms: self.elapsed_ms,
            groups: group(&self.hits, limit),
            note: self.note.clone(),
        }
    }

    /// Hits of one kind, best first.
    pub fn ranked(&self, kind: Kind) -> impl Iterator<Item = &Hit> {
        self.hits.iter().filter(move |h| h.kind == kind)
    }
}

/// Groups hits by kind in a fixed, Spotlight-like order.
pub fn group(hits: &[Hit], limit: usize) -> Vec<Group> {
    let order = [Kind::Document, Kind::Code, Kind::Image, Kind::Audio, Kind::Video, Kind::Other];
    order
        .iter()
        .filter_map(|&k| {
            let hits: Vec<Hit> = hits.iter().filter(|h| h.kind == k).take(limit).cloned().collect();
            (!hits.is_empty()).then(|| Group { kind: k, title: k.group_title(), hits })
        })
        .collect()
}

pub struct Engine {
    pub cfg: Config,
    pub statics: Arc<StaticEmbedder>,
    pub gemma: Arc<GemmaSlot>,
    pub db_path: PathBuf,
    baseline: std::sync::Mutex<Option<Baseline>>,
    pub static_index: StaticIndex,
    /// Chunks matching each keyword term, for the keyword query plan.
    term_counts: std::sync::Mutex<(Instant, HashMap<String, i64>)>,
}

/// A random sample of indexed Gemma vectors. A query's mean similarity to it is the
/// "unrelated" level, and only results clearly above it are shown: raw cosine
/// similarities sit in a narrow band whose position shifts from query to query.
struct Baseline {
    taken: Instant,
    images: Vec<Vec<f32>>,
    text: Vec<Vec<f32>>,
    code: Vec<Vec<f32>>,
}

const BASELINE_SAMPLE: usize = 256;
const BASELINE_MIN: usize = 16;
const BASELINE_MAX_AGE_SECS: u64 = 600;
/// A sample taken while Gemma vectors were still scarce (fresh index) is retaken sooner.
const BASELINE_PARTIAL_AGE_SECS: u64 = 2;

impl Baseline {
    fn load(conn: &Connection) -> Result<Self> {
        let mut rng = Rng::seeded();
        let image_ids: Vec<i64> = conn
            .prepare("SELECT id FROM files WHERE vision_state=?1")?
            .query_map([db::VISION_DONE], |r| r.get(0))?
            .collect::<Result<_, _>>()?;
        let mut images = Vec::new();
        if let Ok(mut stmt) = conn.prepare("SELECT embedding FROM vec_image WHERE rowid=?1") {
            for id in rng.sample(&image_ids, BASELINE_SAMPLE) {
                if let Some(b) = stmt.query_row([id], |r| r.get::<_, Vec<u8>>(0)).optional()? {
                    images.push(db::bytes_f32(&b));
                }
            }
        }
        let (lo, hi): (Option<i64>, Option<i64>) =
            conn.query_row("SELECT min(chunk_id), max(chunk_id) FROM gemma_chunks", [], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })?;
        let (mut text, mut code) = (Vec::new(), Vec::new());
        if let (Some(lo), Some(hi)) = (lo, hi) {
            let mut stmt = conn.prepare(
                "SELECT g.vec, f.kind FROM gemma_chunks g JOIN chunks c ON c.id=g.chunk_id JOIN files f ON f.id=c.file_id
                 WHERE g.chunk_id=?1",
            )?;
            for _ in 0..BASELINE_SAMPLE * 8 {
                if text.len() >= BASELINE_SAMPLE && code.len() >= BASELINE_SAMPLE {
                    break;
                }
                let id = lo + (rng.next() % (hi - lo + 1) as u64) as i64;
                if let Some((b, kind)) =
                    stmt.query_row([id], |r| Ok((r.get::<_, Vec<u8>>(0)?, r.get::<_, String>(1)?))).optional()?
                {
                    let list = if kind == "code" { &mut code } else { &mut text };
                    if list.len() < BASELINE_SAMPLE {
                        list.push(db::bytes_f32(&b));
                    }
                }
            }
        }
        Ok(Self { taken: Instant::now(), images, text, code })
    }

    /// Mean similarity of `q` to a sample, if the sample is big enough to mean anything.
    fn level(sample: &[Vec<f32>], q: &[f32]) -> Option<f64> {
        let usable: Vec<&Vec<f32>> = sample.iter().filter(|v| v.len() == q.len()).collect();
        (usable.len() >= BASELINE_MIN)
            .then(|| usable.iter().map(|v| dot(q, v) as f64).sum::<f64>() / usable.len() as f64)
    }
}

/// xorshift64*, enough for sampling.
struct Rng(u64);

impl Rng {
    fn seeded() -> Self {
        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
        Self((t.as_nanos() as u64) | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn sample<T: Copy>(&mut self, items: &[T], n: usize) -> Vec<T> {
        if items.len() <= n {
            return items.to_vec();
        }
        let mut v = items.to_vec();
        for i in 0..n {
            let j = i + (self.next() % (v.len() - i) as u64) as usize;
            v.swap(i, j);
        }
        v.truncate(n);
        v
    }
}

/// Which signals the fast stage may use (the evaluation compares them).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastMode {
    NamesOnly,
    Keyword,
    Full,
}

#[derive(Clone, Copy, Default)]
struct Signals {
    name: bool,
    keyword: bool,
    semantic: bool,
    /// A static-embedding neighbour that clearly stands out (see `STATIC_MIN_SIM`).
    strong: bool,
}

/// Neighbour lists always return something, relevant or not, so files that only the
/// static embeddings found are shown only when the similarity is high in absolute terms
/// and well above the rest of the list. Calibrated on eval/ (relevant: 0.35-0.6,
/// unrelated queries' best: 0.12-0.30).
const STATIC_MIN_SIM: f64 = 0.32;
const STATIC_MIN_LIFT: f64 = 0.2;

const STOPWORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "my", "me", "i", "is", "it", "that", "this",
    "at", "by", "from", "about", "where", "what", "which", "when", "how", "was", "were", "be", "we", "our", "your",
    "file", "files", "doc", "document", "note", "notes", "thing", "stuff", "some", "did", "do", "has", "have",
];

pub fn tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric()).filter(|t| !t.is_empty()).map(|t| t.to_lowercase()).collect()
}

fn fts_quote(t: &str) -> String {
    format!("\"{}\"", t.replace('"', "\"\""))
}

impl Engine {
    pub fn new(cfg: &Config, statics: Arc<StaticEmbedder>, gemma: Arc<GemmaSlot>) -> Self {
        Self {
            cfg: cfg.clone(),
            statics,
            gemma,
            db_path: config::db_path(),
            baseline: Default::default(),
            static_index: Default::default(),
            term_counts: std::sync::Mutex::new((Instant::now(), HashMap::new())),
        }
    }

    pub fn open(&self) -> Result<Connection> {
        db::open_readonly(&self.db_path)
    }

    /// Unrelated-similarity levels for a query: (text, code, images).
    fn baseline_levels(
        &self,
        conn: &Connection,
        qv: &[f32],
        code_qv: &[f32],
    ) -> (Option<f64>, Option<f64>, Option<f64>) {
        let mut guard = self.baseline.lock().unwrap();
        let fresh = guard.as_ref().is_some_and(|b| {
            let partial = b.text.len() < BASELINE_SAMPLE || b.code.len() < BASELINE_SAMPLE;
            let max_age = if partial { BASELINE_PARTIAL_AGE_SECS } else { BASELINE_MAX_AGE_SECS };
            b.taken.elapsed().as_secs() < max_age
        });
        if !fresh {
            match Baseline::load(conn) {
                Ok(b) => *guard = Some(b),
                Err(e) => tracing::warn!("sampling baseline vectors: {e:#}"),
            }
        }
        let Some(b) = guard.as_ref() else { return (None, None, None) };
        let text = Baseline::level(&b.text, qv);
        let code = Baseline::level(&b.code, code_qv).or(text);
        (text.or(code), code, Baseline::level(&b.images, qv))
    }

    pub fn fast(&self, conn: &Connection, q: &Query, mode: FastMode) -> Result<Outcome> {
        let started = Instant::now();
        if q.text.trim().is_empty() {
            return self.browse(conn, q, started);
        }
        let toks = tokens(&q.text);
        let terms: Vec<&String> = toks.iter().filter(|t| !STOPWORDS.contains(&t.as_str())).collect();

        // File names: all tokens as prefixes, falling back to any token.
        let mut name_ranked: Vec<i64> = Vec::new();
        if !toks.is_empty() {
            let all = toks.iter().map(|t| format!("{}*", fts_quote(t))).collect::<Vec<_>>().join(" AND ");
            name_ranked = names_match(conn, &format!("{{name}} : ({all})"))?;
            if name_ranked.len() < 10 && terms.len() > 1 {
                let any = terms.iter().map(|t| format!("{}*", fts_quote(t))).collect::<Vec<_>>().join(" OR ");
                for id in names_match(conn, &format!("{{name}} : ({any})"))? {
                    if !name_ranked.contains(&id) {
                        name_ranked.push(id);
                    }
                }
            }
        }

        // Chunk-level evidence, fused by rank.
        let mut chunk_score: HashMap<i64, (i64, f64, Signals)> = HashMap::new();
        let mut add = |list: &[(i64, i64, f64)], weight: f64, sig: fn(&mut Signals, bool)| {
            let mean = list.iter().map(|l| l.2).sum::<f64>() / list.len().max(1) as f64;
            for (rank, &(chunk, file, sim)) in list.iter().enumerate() {
                let e = chunk_score.entry(chunk).or_insert((file, 0.0, Signals::default()));
                e.1 += weight / (RRF_K + rank as f64 + 1.0);
                sig(&mut e.2, sim >= STATIC_MIN_SIM && sim - mean >= STATIC_MIN_LIFT);
            }
        };
        if mode != FastMode::NamesOnly
            && let Some(m) = self.keyword_query(conn, &terms)?
        {
            let kw: Vec<(i64, i64, f64)> = keyword_match(conn, &m)?.into_iter().map(|(c, f)| (c, f, 0.0)).collect();
            add(&kw, 1.0, |s, _| s.keyword = true);
        }
        if mode == FastMode::Full {
            let tv = self.statics.embed_one(Space::Text, &q.text);
            let sem = |s: &mut Signals, strong: bool| {
                s.semantic = true;
                s.strong |= strong;
            };
            add(&self.knn_chunks(conn, Space::Text, &tv)?, 1.0, sem);
            let cv = self.statics.embed_one(Space::Code, &q.text);
            add(&self.knn_chunks(conn, Space::Code, &cv)?, 1.0, sem);
        }

        let mut chunks: Vec<(i64, i64, f64, Signals)> =
            chunk_score.into_iter().map(|(c, (f, s, sig))| (c, f, s, sig)).collect();
        chunks.sort_by(|a, b| b.2.total_cmp(&a.2));
        let candidates: Vec<(i64, i64)> =
            chunks.iter().take(self.cfg.search.rescore_top).map(|&(c, f, _, _)| (c, f)).collect();

        // File-level: best chunk plus the file-name signal.
        let mut files: HashMap<i64, (f64, Option<i64>, Signals)> = HashMap::new();
        for &(c, f, s, sig) in &chunks {
            let e = files.entry(f).or_insert((0.0, None, Signals::default()));
            if e.1.is_none() {
                e.0 = s;
                e.1 = Some(c);
            }
            e.2.keyword |= sig.keyword;
            e.2.semantic |= sig.semantic;
            e.2.strong |= sig.strong;
        }
        for (rank, &f) in name_ranked.iter().enumerate() {
            let e = files.entry(f).or_insert((0.0, None, Signals::default()));
            e.0 += 1.5 / (RRF_K + rank as f64 + 1.0);
            e.2.name = true;
        }
        let mut ranked: Vec<(i64, f64, Option<i64>, Signals)> =
            files.into_iter().map(|(f, (s, c, sig))| (f, s, c, sig)).collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        ranked.truncate(400);

        let all_hits = self.load_hits(conn, q, &toks, &ranked)?;
        // Gemma sees the whole fast ranking; the user only sees files with real evidence.
        let fast_ranked = all_hits.iter().map(|h| h.file_id).collect();
        let shown: std::collections::HashSet<i64> =
            ranked.iter().filter(|r| r.3.name || r.3.keyword || r.3.strong).map(|r| r.0).collect();
        let hits = all_hits.into_iter().filter(|h| shown.contains(&h.file_id)).collect();
        Ok(Outcome {
            stage: Stage::Fast,
            hits,
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
            note: None,
            candidates,
            name_ranked,
            fast_ranked,
        })
    }

    /// Filter-only queries (`kind:image after:2024`) list the newest matching files.
    /// FTS5 query for the terms. BM25 scores every matching chunk (~0.5 µs each), so OR-ing
    /// common terms on a large index is slow while adding little: their weight is low. Past
    /// a budget, only the rarest terms are OR-ed, plus all terms together.
    fn keyword_query(&self, conn: &Connection, terms: &[&String]) -> Result<Option<String>> {
        if terms.is_empty() {
            return Ok(None);
        }
        let all_or = || terms.iter().map(|t| fts_quote(t)).collect::<Vec<_>>().join(" OR ");
        let mut counted: Vec<(i64, &String)> = Vec::with_capacity(terms.len());
        {
            let mut cache = self.term_counts.lock().unwrap();
            if cache.0.elapsed().as_secs() > TERM_COUNT_MAX_AGE_SECS {
                *cache = (Instant::now(), HashMap::new());
            }
            let mut stmt = conn.prepare_cached("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH ?1")?;
            for t in terms {
                let n = match cache.1.get(t.as_str()) {
                    Some(&n) => n,
                    None => {
                        let n: i64 = stmt.query_row([fts_quote(t)], |r| r.get(0)).unwrap_or(0);
                        cache.1.insert(t.to_string(), n);
                        n
                    }
                };
                counted.push((n, t));
            }
        }
        if counted.iter().map(|c| c.0).sum::<i64>() <= KEYWORD_BUDGET {
            return Ok(Some(all_or()));
        }
        counted.sort();
        let mut clauses = Vec::new();
        let mut spent = 0;
        for &(n, t) in &counted {
            if spent + n > KEYWORD_BUDGET {
                break;
            }
            spent += n;
            clauses.push(fts_quote(t));
        }
        if terms.len() > 1 {
            clauses.push(format!("({})", terms.iter().map(|t| fts_quote(t)).collect::<Vec<_>>().join(" AND ")));
        }
        Ok((!clauses.is_empty()).then(|| clauses.join(" OR ")))
    }

    /// Nearest chunks in a static space: (chunk id, file id, cosine similarity).
    fn knn_chunks(&self, conn: &Connection, space: Space, v: &[f32]) -> Result<Vec<(i64, i64, f64)>> {
        let rows = self.static_index.search(conn, space, v, STATIC_K)?;
        if tracing::enabled!(tracing::Level::TRACE) {
            let top: Vec<String> = rows.iter().take(5).map(|r| format!("{:.3}", r.2)).collect();
            let mean = rows.iter().map(|r| r.2).sum::<f64>() / rows.len().max(1) as f64;
            tracing::trace!("{space:?}: top {} mean {mean:.3}", top.join(" "));
        }
        Ok(rows)
    }

    fn browse(&self, conn: &Connection, q: &Query, started: Instant) -> Result<Outcome> {
        let (where_sql, args) = filter_sql(q);
        let sql = format!(
            "SELECT id FROM files WHERE 1=1 {where_sql} ORDER BY COALESCE(taken, mtime) DESC LIMIT {}",
            self.cfg.search.limit * 6
        );
        let ids: Vec<i64> =
            conn.prepare(&sql)?.query_map(params_from_iter(args), |r| r.get(0))?.collect::<Result<_, _>>()?;
        let n = ids.len() as f64;
        let ranked: Vec<_> =
            ids.iter().enumerate().map(|(i, &f)| (f, n - i as f64, None, Signals::default())).collect();
        let hits = self.load_hits(conn, q, &[], &ranked)?;
        Ok(Outcome {
            stage: Stage::Fast,
            hits,
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
            note: None,
            candidates: Vec::new(),
            name_ranked: Vec::new(),
            fast_ranked: Vec::new(),
        })
    }

    /// Loads file rows for ranked ids, applies the query filters and builds hits.
    fn load_hits(
        &self,
        conn: &Connection,
        q: &Query,
        toks: &[String],
        ranked: &[(i64, f64, Option<i64>, Signals)],
    ) -> Result<Vec<Hit>> {
        if ranked.is_empty() {
            return Ok(Vec::new());
        }
        let (where_sql, args) = filter_sql(q);
        let ids = ranked.iter().map(|r| r.0.to_string()).collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT id, path, name, kind, COALESCE(taken, mtime), thumb FROM files WHERE id IN ({ids}) {where_sql}"
        );
        let mut rows: HashMap<i64, (String, String, String, i64, Option<String>)> = HashMap::new();
        let mut stmt = conn.prepare(&sql)?;
        let mapped = stmt.query_map(params_from_iter(args), |r| {
            Ok((r.get::<_, i64>(0)?, (r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?)))
        })?;
        for row in mapped {
            let (id, v) = row?;
            rows.insert(id, v);
        }
        let mut chunk_stmt = conn.prepare_cached("SELECT text, line FROM chunks WHERE id=?1")?;
        let mut hits = Vec::new();
        for &(id, score, chunk, sig) in ranked {
            let Some((path, name, kind, date, thumb)) = rows.remove(&id) else { continue };
            let (snippet, line) = match chunk {
                Some(c) => chunk_stmt
                    .query_row([c], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))
                    .optional()?
                    .map(|(t, l)| (Some(snippet(&t, toks)), Some(l)))
                    .unwrap_or((None, None)),
                None => (None, None),
            };
            let mut why = Vec::new();
            if sig.name {
                why.push("name");
            }
            if sig.keyword {
                why.push("keyword");
            }
            if sig.semantic {
                why.push("semantic");
            }
            hits.push(Hit {
                file_id: id,
                dir: display_dir(&path),
                path,
                name,
                kind: Kind::parse(&kind).unwrap_or(Kind::Other),
                score,
                snippet,
                line,
                thumb,
                date,
                why,
                sim: None,
            });
        }
        Ok(hits)
    }

    /// Gemma stage. `cancelled` is polled between model calls.
    pub fn deep(&self, conn: &Connection, q: &Query, fast: &Outcome, cancelled: &dyn Fn() -> bool) -> Result<Outcome> {
        let started = Instant::now();
        if q.text.trim().is_empty() {
            return Ok(Outcome { stage: Stage::Deep, hits: fast.hits.clone(), elapsed_ms: 0.0, note: None, ..empty() });
        }
        let gemma = self.gemma.get()?;
        let qv = gemma.embed_query(&q.text, QUERY_PREFIX)?;
        let toks = tokens(&q.text);
        let wants = |k: Kind| q.kinds.is_empty() || q.kinds.contains(&k);
        let mut hits: Vec<Hit> = Vec::new();

        // Text: re-score the shortlist, reusing cached chunk vectors.
        let mut gemma_rank: Vec<(i64, f64, i64)> = Vec::new(); // (file, score, chunk)
        if !fast.candidates.is_empty() && !cancelled() {
            let ids = fast.candidates.iter().map(|c| c.0.to_string()).collect::<Vec<_>>().join(",");
            let mut stmt = conn.prepare(&format!(
                "SELECT c.id, c.file_id, c.text, f.name, f.kind, g.vec FROM chunks c JOIN files f ON f.id=c.file_id
                 LEFT JOIN gemma_chunks g ON g.chunk_id=c.id WHERE c.id IN ({ids})"
            ))?;
            let rows: Vec<ChunkRow> = stmt
                .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?)))?
                .collect::<Result<_, _>>()?;
            let mut vecs: HashMap<i64, Vec<f32>> = HashMap::new();
            let mut missing: Vec<(i64, String)> = Vec::new();
            for (cid, _, text, name, _, cached) in &rows {
                match cached.as_ref().map(|b| db::bytes_f32(b)) {
                    Some(v) if v.len() == gemma.dim => {
                        vecs.insert(*cid, v);
                    }
                    _ => missing.push((*cid, document_input(Some(name), text))),
                }
            }
            // Embed what isn't cached, best fast-stage candidates first, within the time budget.
            let order: HashMap<i64, usize> = fast.candidates.iter().enumerate().map(|(i, c)| (c.0, i)).collect();
            missing.sort_by_key(|m| order.get(&m.0).copied().unwrap_or(usize::MAX));
            let budget_ms = match self.cfg.search.rescore_budget_ms {
                0 if gemma.on_gpu() => 3000,
                0 => 600,
                ms => ms,
            };
            let embed_started = Instant::now();
            let mut fresh = Vec::new();
            let mut batch_size = 2;
            let mut rest = &missing[..];
            while !rest.is_empty() && !cancelled() {
                let spent = embed_started.elapsed().as_secs_f64() * 1000.0;
                let per_item = if fresh.is_empty() { 0.0 } else { spent / fresh.len() as f64 };
                if spent + per_item * batch_size as f64 > budget_ms as f64 {
                    break;
                }
                let (batch, tail) = rest.split_at(batch_size.min(rest.len()));
                rest = tail;
                let inputs: Vec<String> = batch.iter().map(|b| b.1.clone()).collect();
                let out = gemma.embed_texts(&inputs, self.cfg.search.rescore_max_tokens)?;
                for ((cid, _), v) in batch.iter().zip(out) {
                    fresh.push((*cid, v.clone()));
                    vecs.insert(*cid, v);
                }
                batch_size = 8;
            }
            if !rest.is_empty() {
                tracing::debug!("deep: {} of {} uncached chunks left to the fast ranking", rest.len(), missing.len());
            }
            self.cache_chunk_vectors(&fresh);
            let code_qv = gemma.embed_query(&q.text, CODE_QUERY_PREFIX)?;
            let (text_level, code_level, _) = self.baseline_levels(conn, &qv, &code_qv);
            // Margin over the unrelated level; without a baseline yet, over the shortlist's own mean
            // (which is higher, so the cut is stricter).
            let mut best: HashMap<i64, (f64, i64)> = HashMap::new();
            let mut raw = Vec::new();
            for (cid, fid, _, _, kind, _) in &rows {
                let Some(v) = vecs.get(cid) else { continue };
                let is_code = kind == "code";
                let s = dot(if is_code { &code_qv } else { &qv }, v) as f64;
                raw.push((*cid, *fid, s, is_code));
            }
            let fallback = if raw.is_empty() { 0.0 } else { raw.iter().map(|r| r.2).sum::<f64>() / raw.len() as f64 };
            for (cid, fid, s, is_code) in raw {
                let level = if is_code { code_level } else { text_level }.unwrap_or(fallback);
                let margin = s - level;
                let e = best.entry(fid).or_insert((f64::MIN, cid));
                if margin > e.0 {
                    *e = (margin, cid);
                }
            }
            gemma_rank = best.into_iter().map(|(f, (s, c))| (f, s, c)).collect();
            gemma_rank.sort_by(|a, b| b.1.total_cmp(&a.1));
        }

        // Fuse Gemma's ranking with the file-name and fast rankings.
        let mut fused: HashMap<i64, (f64, Option<i64>, Signals)> = HashMap::new();
        for (rank, &(f, _, c)) in gemma_rank.iter().enumerate() {
            let e = fused.entry(f).or_insert((0.0, Some(c), Signals::default()));
            e.0 += 2.0 / (RRF_K + rank as f64 + 1.0);
            e.2.semantic = true;
        }
        for (rank, &f) in fast.name_ranked.iter().enumerate() {
            let e = fused.entry(f).or_insert((0.0, None, Signals::default()));
            e.0 += 1.0 / (RRF_K + rank as f64 + 1.0);
            e.2.name = true;
        }
        for (rank, &f) in fast.fast_ranked.iter().enumerate() {
            let e = fused.entry(f).or_insert((0.0, None, Signals::default()));
            e.0 += 0.5 / (RRF_K + rank as f64 + 1.0);
        }
        let mut ranked: Vec<(i64, f64, Option<i64>, Signals)> =
            fused.into_iter().map(|(f, (s, c, sig))| (f, s, c, sig)).collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        let gemma_margins: HashMap<i64, f64> = gemma_rank.iter().map(|g| (g.0, g.1)).collect();
        let keyword_files: std::collections::HashSet<i64> =
            fast.hits.iter().filter(|h| h.why.contains(&"keyword")).map(|h| h.file_id).collect();
        let name_files: std::collections::HashSet<i64> = fast.name_ranked.iter().copied().collect();
        let text_margin = self.cfg.search.text_margin;
        ranked.retain(|r| {
            let named = name_files.contains(&r.0);
            let keyword = keyword_files.contains(&r.0);
            match gemma_margins.get(&r.0) {
                Some(&m) => named || m >= text_margin || (keyword && m >= text_margin / 2.0),
                None => named || keyword,
            }
        });
        for mut h in self.load_hits(conn, q, &toks, &ranked)? {
            if h.kind == Kind::Image {
                continue; // images come from the vision index below
            }
            if let Some(&sim) = gemma_margins.get(&h.file_id) {
                h.sim = Some(sim as f32);
                h.why.retain(|w| *w != "semantic");
                h.why.push("gemma");
            }
            // Keep the fast snippet when Gemma didn't see this file.
            if h.snippet.is_none()
                && let Some(f) = fast.hits.iter().find(|f| f.file_id == h.file_id)
            {
                h.snippet = f.snippet.clone();
                h.line = f.line;
            }
            hits.push(h);
        }

        // Images: text-to-image similarity against the vision index.
        if wants(Kind::Image) && !cancelled() {
            let knn = knn_rows(conn, "vec_image", &qv, IMAGE_K)?;
            let sims: Vec<f64> = knn.iter().map(|&(_, d)| 1.0 - d).collect();
            let (_, _, level) = self.baseline_levels(conn, &qv, &qv);
            let keep = match level {
                Some(l) => l + self.cfg.search.image_margin,
                None => image_threshold(&sims),
            };
            let name_hits: std::collections::HashSet<i64> = fast.name_ranked.iter().copied().collect();
            let mut ranked: Vec<(i64, f64, Option<i64>, Signals)> = knn
                .iter()
                .zip(&sims)
                .filter(|(_, s)| **s >= keep)
                .map(|(&(f, _), &s)| {
                    let sig = Signals { name: name_hits.contains(&f), ..Default::default() };
                    (f, s + if sig.name { 0.05 } else { 0.0 }, None, sig)
                })
                .collect();
            let image_sims: HashMap<i64, f64> =
                knn.iter().zip(&sims).map(|(&(f, _), &s)| (f, s - level.unwrap_or(0.0))).collect();
            // Images that matched by name but not by content still belong in the group.
            for h in fast.hits.iter().filter(|h| h.kind == Kind::Image) {
                if !ranked.iter().any(|r| r.0 == h.file_id) {
                    ranked.push((h.file_id, 0.0, None, Signals { name: true, ..Default::default() }));
                }
            }
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
            for mut h in self.load_hits(conn, q, &toks, &ranked)? {
                if h.score > 0.0 {
                    h.why.push("vision");
                }
                h.sim = image_sims.get(&h.file_id).map(|&s| s as f32);
                hits.push(h);
            }
        }

        let images_pending: i64 = conn
            .query_row("SELECT count(*) FROM files WHERE vision_state=?1", [db::VISION_PENDING], |r| r.get(0))
            .unwrap_or(0);
        let note = (images_pending > 0).then(|| format!("{images_pending} images are still being indexed"));
        Ok(Outcome { stage: Stage::Deep, hits, elapsed_ms: started.elapsed().as_secs_f64() * 1000.0, note, ..empty() })
    }

    /// Best effort: a busy writer just means the vectors get computed again next time.
    fn cache_chunk_vectors(&self, fresh: &[(i64, Vec<f32>)]) {
        if fresh.is_empty() {
            return;
        }
        let Ok(conn) = Connection::open(&self.db_path) else { return };
        let _ = conn.busy_timeout(std::time::Duration::from_millis(200));
        let Ok(tx) = conn.unchecked_transaction() else { return };
        for (cid, v) in fresh {
            let _ = tx.execute(
                "INSERT OR REPLACE INTO gemma_chunks(chunk_id, vec) VALUES (?1, ?2)",
                rusqlite::params![cid, db::f32_bytes(v)],
            );
        }
        let _ = tx.commit();
    }
}

/// What a streamed search should produce.
#[derive(Debug, Clone, Copy)]
pub struct StreamOpts {
    pub limit: usize,
    pub fast: bool,
    pub deep: bool,
    /// After the fast results, wait this long for the user to keep typing before running Gemma.
    pub pause_ms: u64,
}

impl Engine {
    /// Runs the fast stage, then (unless cancelled during the pause) the deep stage,
    /// handing each stage's results to `emit`. `wait` sleeps and returns false if the
    /// request was cancelled meanwhile.
    pub fn stream(
        &self,
        conn: &Connection,
        input: &str,
        opts: StreamOpts,
        wait: &dyn Fn(std::time::Duration) -> bool,
        cancelled: &dyn Fn() -> bool,
        emit: &mut dyn FnMut(&Results) -> Result<()>,
    ) -> Result<()> {
        let q = parse_query(input);
        let fast = self.fast(conn, &q, FastMode::Full)?;
        if opts.fast {
            emit(&fast.results(input, opts.limit))?;
        }
        if !opts.deep || q.text.trim().is_empty() {
            return Ok(());
        }
        if opts.fast && opts.pause_ms > 0 && !wait(std::time::Duration::from_millis(opts.pause_ms)) {
            return Ok(());
        }
        match self.deep(conn, &q, &fast, cancelled) {
            Ok(deep) if !cancelled() => emit(&deep.results(input, opts.limit)),
            Ok(_) => Ok(()),
            Err(e) => {
                tracing::warn!("deep search failed: {e:#}");
                let mut r = fast.results(input, opts.limit);
                r.stage = Stage::Deep;
                r.note = Some(format!("Gemma unavailable: {e:#}"));
                emit(&r)
            }
        }
    }
}

fn empty() -> Outcome {
    Outcome {
        stage: Stage::Fast,
        hits: Vec::new(),
        elapsed_ms: 0.0,
        note: None,
        candidates: Vec::new(),
        name_ranked: Vec::new(),
        fast_ranked: Vec::new(),
    }
}

/// Text-to-image cosine similarities sit in a narrow band and their absolute level varies
/// by query, so keep the images that stand out from the neighbourhood rather than using a
/// fixed cut-off.
fn image_threshold(sims: &[f64]) -> f64 {
    if sims.is_empty() {
        return f64::MAX;
    }
    let top = sims[0];
    let mut sorted = sims.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let median = sorted[sorted.len() / 2];
    (top - 0.06).max(median + 0.01).min(top)
}

fn names_match(conn: &Connection, m: &str) -> Result<Vec<i64>> {
    let mut stmt = conn.prepare_cached(
        "SELECT f.id FROM names_fts JOIN files f ON f.id = names_fts.rowid
         WHERE names_fts MATCH ?1 ORDER BY rank, length(f.name) LIMIT ?2",
    )?;
    let ids = stmt.query_map(rusqlite::params![m, NAME_K as i64], |r| r.get(0))?.collect::<Result<_, _>>();
    Ok(ids.unwrap_or_default())
}

fn keyword_match(conn: &Connection, m: &str) -> Result<Vec<(i64, i64)>> {
    let mut stmt = conn.prepare_cached(
        "SELECT c.id, c.file_id FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid
         WHERE chunks_fts MATCH ?1 ORDER BY rank LIMIT ?2",
    )?;
    let rows: rusqlite::Result<Vec<(i64, i64)>> =
        stmt.query_map(rusqlite::params![m, KEYWORD_K as i64], |r| Ok((r.get(0)?, r.get(1)?)))?.collect();
    Ok(rows.unwrap_or_default())
}

/// (rowid, cosine distance), nearest first. Missing tables (fresh index) give no rows.
fn knn_rows(conn: &Connection, table: &str, v: &[f32], k: usize) -> Result<Vec<(i64, f64)>> {
    let sql = format!("SELECT rowid, distance FROM {table} WHERE embedding MATCH ?1 AND k = ?2 ORDER BY distance");
    let Ok(mut stmt) = conn.prepare_cached(&sql) else { return Ok(Vec::new()) };
    let rows = stmt
        .query_map(rusqlite::params![db::f32_bytes(v), k as i64], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<Result<_, _>>()?;
    Ok(rows)
}

fn filter_sql(q: &Query) -> (String, Vec<rusqlite::types::Value>) {
    use rusqlite::types::Value;
    let mut sql = String::new();
    let mut args = Vec::new();
    if !q.kinds.is_empty() {
        sql.push_str(&format!(" AND kind IN ({})", vec!["?"; q.kinds.len()].join(",")));
        args.extend(q.kinds.iter().map(|k| Value::Text(k.as_str().into())));
    }
    if !q.exts.is_empty() {
        sql.push_str(&format!(" AND ext IN ({})", vec!["?"; q.exts.len()].join(",")));
        args.extend(q.exts.iter().map(|e| Value::Text(e.clone())));
    }
    if let Some(a) = q.after {
        sql.push_str(" AND COALESCE(taken, mtime) >= ?");
        args.push(Value::Integer(a));
    }
    if let Some(b) = q.before {
        sql.push_str(" AND COALESCE(taken, mtime) < ?");
        args.push(Value::Integer(b));
    }
    for p in &q.paths {
        sql.push_str(" AND path LIKE ? ESCAPE '\\'");
        let p = config::expand_tilde(p).to_string_lossy().replace('\\', "\\\\").replace('%', "\\%").replace('_', "\\_");
        args.push(Value::Text(format!("%{p}%")));
    }
    (sql, args)
}

/// A short excerpt around the first query word in the chunk.
pub fn snippet(text: &str, toks: &[String]) -> String {
    let lower = text.to_lowercase();
    let pos = toks
        .iter()
        .filter(|t| t.len() > 2 && !STOPWORDS.contains(&t.as_str()))
        .filter_map(|t| lower.find(t.as_str()))
        .min()
        .unwrap_or(0);
    // `lower` can differ in byte length from `text` for some scripts; clamp to a boundary.
    let mut start = pos.saturating_sub(60).min(text.len());
    while !text.is_char_boundary(start) {
        start -= 1;
    }
    let excerpt: String = text[start..].chars().take(360).collect();
    let collapsed = excerpt.split_whitespace().collect::<Vec<_>>().join(" ");
    if start > 0 { format!("…{collapsed}") } else { collapsed }
}

pub fn display_dir(path: &str) -> String {
    let parent = Path::new(path).parent().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
    let home = config::home_dir().to_string_lossy().into_owned();
    match parent.strip_prefix(&home) {
        Some(rest) if home.len() > 1 => format!("~{rest}"),
        _ => parent,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_filters() {
        let q = parse_query("dogs at the beach kind:image after:2023-06 ext:jpg,heic in:~/Pictures");
        assert_eq!(q.text, "dogs at the beach");
        assert_eq!(q.kinds, vec![Kind::Image]);
        assert_eq!(q.exts, vec!["jpg", "heic"]);
        assert_eq!(q.after, ymd(2023, 6, 1));
        assert_eq!(q.paths, vec!["~/Pictures"]);
        let y = parse_query("taxes year:2023");
        assert_eq!((y.after, y.before), (ymd(2023, 1, 1), ymd(2024, 1, 1)));
        assert!(parse_date("2w").unwrap() < chrono::Utc::now().timestamp());
        // Unknown keys stay part of the text.
        assert_eq!(parse_query("http://x").text, "http://x");
    }

    #[test]
    fn snippets_center_on_terms() {
        let text = format!("{} the sourdough starter needs feeding {}", "x ".repeat(100), "y ".repeat(100));
        let s = snippet(&text, &["sourdough".into()]);
        assert!(s.contains("sourdough starter"));
        assert!(s.starts_with('…'));
    }

    #[test]
    fn image_threshold_keeps_standouts() {
        let sims = [0.70, 0.66, 0.60, 0.59, 0.58, 0.58, 0.57];
        let t = image_threshold(&sims);
        assert!(t > 0.62 && t <= 0.66);
    }
}
