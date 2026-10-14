//! Measures whether content search beats file-name search: indexes a corpus into a
//! throwaway database and runs a query set through each search mode.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::db;
use crate::embed::GemmaSlot;
use crate::embed::static_model::StaticEmbedder;
use crate::index::{Indexer, Stats};
use crate::power::Budget;
use crate::search::{Engine, FastMode, Outcome, parse_query};

#[derive(Debug, Deserialize)]
struct QueryFile {
    query: Vec<EvalQuery>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EvalQuery {
    pub q: String,
    pub expect: Vec<String>,
    #[serde(rename = "type", default)]
    pub kind: String,
}

pub const MODES: &[&str] = &["filename", "keyword", "fast", "deep"];

#[derive(Debug, Default, Clone, Serialize)]
pub struct Score {
    pub queries: usize,
    pub hit1: f64,
    pub hit5: f64,
    pub hit10: f64,
    pub mrr10: f64,
    pub ms_mean: f64,
    pub ms_p95: f64,
    /// Mean results shown per query (all groups, capped at the per-group limit).
    pub shown: f64,
}

#[derive(Debug, Serialize)]
pub struct QueryResult {
    pub q: String,
    pub kind: String,
    /// Rank (1-based, within the expected file's kind group) per mode; None = not in the top 10.
    pub ranks: HashMap<String, Option<usize>>,
    pub ms: HashMap<String, f64>,
    pub shown: HashMap<String, usize>,
    /// Queries with no expected file only count towards `shown`.
    pub negative: bool,
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub files: i64,
    pub chunks: i64,
    pub index_secs: f64,
    /// Vision encoder plus (unless cold) Gemma vectors for every chunk.
    pub gemma_index_secs: f64,
    pub gemma_load_secs: f64,
    pub cold: bool,
    pub overall: Vec<(String, Score)>,
    pub by_type: Vec<(String, Vec<(String, Score)>)>,
    pub queries: Vec<QueryResult>,
}

/// With `cold`, Gemma chunk vectors are not precomputed, so deep search has to embed
/// within its time budget (the state right after indexing new files).
pub fn run(cfg: &Config, corpus: &Path, queries: &Path, modes: &[String], cold: bool) -> Result<Report> {
    let corpus = corpus.canonicalize().with_context(|| format!("corpus {}", corpus.display()))?;
    let qf: QueryFile = toml::from_str(&std::fs::read_to_string(queries)?).context("parsing queries")?;
    for q in &qf.query {
        for e in &q.expect {
            anyhow::ensure!(corpus.join(e).is_file(), "query {:?} expects missing file {e}", q.q);
        }
    }

    let mut cfg = cfg.clone();
    cfg.roots = vec![corpus.to_string_lossy().into_owned()];
    cfg.exclude.clear();
    cfg.indexing.thumbnails = false;
    let tmp = tempdir()?;
    let db_path = tmp.join("eval.db");

    let statics = Arc::new(StaticEmbedder::load(cfg.models.static_code_model)?);
    let gemma = Arc::new(GemmaSlot::new(&cfg));
    let started = Instant::now();
    let mut indexer =
        Indexer::with_conn(&cfg, db::open(&db_path)?, statics.clone(), gemma.clone(), Arc::new(Stats::default()))?;
    indexer.scan(&|| Budget::ALL)?;
    let needs_gemma = modes.iter().any(|m| m == "deep");
    let index_secs = started.elapsed().as_secs_f64();
    let started = Instant::now();
    if needs_gemma {
        while indexer.vision_pass(16, &|| false)? > 0 {}
        if !cold {
            while indexer.gemma_text_pass(64)? > 0 {}
        }
    }
    let gemma_index_secs = started.elapsed().as_secs_f64();
    let (files, chunks): (i64, i64) =
        indexer.conn().query_row("SELECT (SELECT count(*) FROM files), (SELECT count(*) FROM chunks)", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })?;
    drop(indexer);

    let mut engine = Engine::new(&cfg, statics, gemma.clone());
    engine.db_path = db_path.clone();
    let conn = engine.open()?;
    let mut gemma_load_secs = 0.0;
    if needs_gemma && cold {
        let t = Instant::now();
        gemma.get()?.embed_query("warm up", crate::embed::gemma::QUERY_PREFIX)?;
        gemma_load_secs = t.elapsed().as_secs_f64();
    }

    let mut results = Vec::new();
    for eq in &qf.query {
        let q = parse_query(&eq.q);
        let expected: Vec<String> = eq.expect.iter().map(|e| corpus.join(e).to_string_lossy().into_owned()).collect();
        let mut ranks = HashMap::new();
        let mut ms = HashMap::new();
        let mut shown = HashMap::new();
        let mut full: Option<Outcome> = None;
        for mode in modes {
            let (outcome, elapsed) = match mode.as_str() {
                "filename" => timed(|| engine.names(&conn, &q))?,
                "keyword" => timed(|| engine.fast(&conn, &q, FastMode::Keyword))?,
                "fast" => timed(|| engine.fast(&conn, &q, FastMode::Full))?,
                "deep" => {
                    let fast_ms;
                    let fast = match full.take() {
                        Some(f) => {
                            fast_ms = f.elapsed_ms;
                            f
                        }
                        None => {
                            let (f, t) = timed(|| engine.fast(&conn, &q, FastMode::Full))?;
                            fast_ms = t;
                            f
                        }
                    };
                    let (d, t) = timed(|| engine.deep(&conn, &q, &fast, &|| false))?;
                    (d, t + fast_ms)
                }
                other => anyhow::bail!("unknown mode {other}"),
            };
            ranks.insert(mode.clone(), rank_of(&outcome, &expected));
            shown.insert(
                mode.clone(),
                crate::search::group(&outcome.hits, cfg.search.limit).iter().map(|g| g.hits.len()).sum(),
            );
            ms.insert(mode.clone(), elapsed);
            if mode == "fast" {
                full = Some(outcome);
            }
        }
        results.push(QueryResult {
            q: eq.q.clone(),
            kind: eq.kind.clone(),
            ranks,
            ms,
            shown,
            negative: eq.expect.is_empty(),
        });
    }

    let overall = modes.iter().map(|m| (m.clone(), score(results.iter().filter(|r| !r.negative), m))).collect();
    let mut types: Vec<String> = results.iter().map(|r| r.kind.clone()).collect();
    types.sort();
    types.dedup();
    let by_type = types
        .into_iter()
        .map(|t| {
            let per = modes.iter().map(|m| (m.clone(), score(results.iter().filter(|r| r.kind == t), m))).collect();
            (t, per)
        })
        .collect();
    let _ = std::fs::remove_dir_all(&tmp);
    Ok(Report {
        files,
        chunks,
        index_secs,
        gemma_index_secs,
        gemma_load_secs,
        cold,
        overall,
        by_type,
        queries: results,
    })
}

fn timed(f: impl FnOnce() -> Result<Outcome>) -> Result<(Outcome, f64)> {
    let t = Instant::now();
    let o = f()?;
    Ok((o, t.elapsed().as_secs_f64() * 1000.0))
}

/// Best rank of any expected file within its own kind group, as the UI shows groups separately.
fn rank_of(o: &Outcome, expected: &[String]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for e in expected {
        let Some(hit) = o.hits.iter().find(|h| &h.path == e) else { continue };
        let r = o.ranked(hit.kind).position(|h| &h.path == e).map(|p| p + 1);
        best = match (best, r) {
            (Some(b), Some(r)) => Some(b.min(r)),
            (b, r) => b.or(r),
        };
    }
    best.filter(|&r| r <= 10)
}

fn score<'a>(rs: impl Iterator<Item = &'a QueryResult>, mode: &str) -> Score {
    let mut s = Score::default();
    let mut times = Vec::new();
    let mut positives = 0usize;
    for r in rs {
        let Some(rank) = r.ranks.get(mode) else { continue };
        s.queries += 1;
        s.shown += r.shown[mode] as f64;
        times.push(r.ms[mode]);
        if r.negative {
            continue;
        }
        positives += 1;
        if let Some(k) = *rank {
            s.hit1 += (k <= 1) as u8 as f64;
            s.hit5 += (k <= 5) as u8 as f64;
            s.hit10 += 1.0;
            s.mrr10 += 1.0 / k as f64;
        }
    }
    if s.queries > 0 {
        let n = s.queries as f64;
        let p = positives.max(1) as f64;
        s.hit1 /= p;
        s.hit5 /= p;
        s.hit10 /= p;
        s.mrr10 /= p;
        s.shown /= n;
        times.sort_by(|a, b| a.total_cmp(b));
        s.ms_mean = times.iter().sum::<f64>() / n;
        s.ms_p95 = times[((times.len() as f64 * 0.95).ceil() as usize).saturating_sub(1)];
    }
    s
}

fn tempdir() -> Result<PathBuf> {
    let dir = std::env::temp_dir().join(format!("omafind-eval-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub fn print(report: &Report, verbose: bool) {
    println!("Indexed {} files ({} chunks) in {:.2}s", report.files, report.chunks, report.index_secs);
    if report.cold {
        println!("Cold: no precomputed Gemma chunk vectors; Gemma loaded in {:.2}s\n", report.gemma_load_secs);
    } else {
        println!(
            "Gemma vectors for all chunks and images: {:.2}s (background work in the daemon)\n",
            report.gemma_index_secs
        );
    }
    let table = |rows: &[(String, Score)]| {
        println!(
            "  {:<9} {:>5} {:>6} {:>6} {:>6} {:>6} {:>6} {:>9} {:>8}",
            "mode", "n", "Hit@1", "Hit@5", "Hit@10", "MRR", "shown", "mean ms", "p95 ms"
        );
        for (m, s) in rows {
            println!(
                "  {:<9} {:>5} {:>6.2} {:>6.2} {:>6.2} {:>6.3} {:>6.1} {:>9.1} {:>8.1}",
                m, s.queries, s.hit1, s.hit5, s.hit10, s.mrr10, s.shown, s.ms_mean, s.ms_p95
            );
        }
    };
    println!("All queries with an expected file");
    table(&report.overall);
    for (t, rows) in &report.by_type {
        println!("\n{} queries", if t.is_empty() { "untyped" } else { t });
        table(rows);
    }
    if verbose {
        println!("\nPer query (rank within kind group, - = not in top 10):");
        for r in &report.queries {
            let ranks: Vec<String> = MODES
                .iter()
                .filter_map(|m| {
                    r.ranks.get(*m).map(|k| format!("{m}={}", k.map(|v| v.to_string()).unwrap_or("-".into())))
                })
                .collect();
            let shown: Vec<String> = MODES.iter().filter_map(|m| r.shown.get(*m).map(|n| n.to_string())).collect();
            println!("  [{:<8}] {:<52} {}  shown {}", r.kind, r.q, ranks.join(" "), shown.join("/"));
        }
    }
}
