//! Crawling, change detection and the indexing pipeline.
//!
//! Every file gets a row (name, kind, size, dates) so file-name search covers the whole
//! disk. Documents and code are read, chunked and embedded with the static models in the
//! same pass. Images are queued for the vision encoder, which runs in a separate, slower
//! lane that respects the power budget and processes the newest photos first.

use std::collections::{HashMap, HashSet};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use anyhow::{Context, Result};
use globset::{Glob, GlobSet, GlobSetBuilder};
use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;

use crate::chunk::{Chunk, chunk_text};
use crate::config::{self, Config};
use crate::db::{self, TEXT_DONE, TEXT_FAILED, TEXT_NONE, VISION_DONE, VISION_FAILED, VISION_NA, VISION_PENDING};
use crate::embed::GemmaSlot;
use crate::embed::gemma::document_input;
use crate::embed::static_model::{Space, StaticEmbedder};
use crate::extract::{self, Kind};
use crate::power::{self, Budget};

const BATCH: usize = 64;

/// Live counters, shared with the status endpoint.
#[derive(Default)]
pub struct Stats {
    pub scanned: AtomicU64,
    pub indexed: AtomicU64,
    pub removed: AtomicU64,
    pub moved: AtomicU64,
    pub errors: AtomicU64,
    pub images_done: AtomicU64,
    pub gemma_chunks_done: AtomicU64,
    pub scanning: AtomicBool,
}

#[derive(Serialize)]
pub struct StatsSnapshot {
    pub scanned: u64,
    pub indexed: u64,
    pub removed: u64,
    pub moved: u64,
    pub errors: u64,
    pub images_done: u64,
    pub gemma_chunks_done: u64,
    pub scanning: bool,
}

impl Stats {
    pub fn snapshot(&self) -> StatsSnapshot {
        StatsSnapshot {
            scanned: self.scanned.load(Ordering::Relaxed),
            indexed: self.indexed.load(Ordering::Relaxed),
            removed: self.removed.load(Ordering::Relaxed),
            moved: self.moved.load(Ordering::Relaxed),
            errors: self.errors.load(Ordering::Relaxed),
            images_done: self.images_done.load(Ordering::Relaxed),
            gemma_chunks_done: self.gemma_chunks_done.load(Ordering::Relaxed),
            scanning: self.scanning.load(Ordering::Relaxed),
        }
    }
}

/// Decides which paths are indexed at all.
#[derive(Clone)]
pub struct Filter {
    globs: GlobSet,
    prefixes: Vec<PathBuf>,
    roots: Vec<PathBuf>,
    include_hidden: bool,
}

impl Filter {
    pub fn new(cfg: &Config) -> Result<Self> {
        let mut b = GlobSetBuilder::new();
        let mut prefixes = vec![config::data_dir(), config::cache_dir()];
        if !cfg.index_encrypted {
            prefixes.extend(encrypted_mounts());
        }
        for pat in &cfg.exclude {
            let expanded = config::expand_tilde(pat);
            let s = expanded.to_string_lossy();
            if s.contains(['*', '?', '[', '{']) {
                b.add(Glob::new(&s).with_context(|| format!("bad exclude pattern {pat}"))?);
            } else {
                prefixes.push(expanded);
            }
        }
        Ok(Self { globs: b.build()?, prefixes, roots: cfg.root_paths(), include_hidden: cfg.include_hidden })
    }

    /// Exclusion by configured paths and patterns only (the walker handles hidden files).
    fn excluded_by_rules(&self, path: &Path) -> bool {
        self.prefixes.iter().any(|p| path.starts_with(p)) || self.globs.is_match(path)
    }

    /// Full check for watcher events: rules, being outside every root, and hidden components.
    pub fn excluded(&self, path: &Path) -> bool {
        if self.excluded_by_rules(path) {
            return true;
        }
        let Some(root) = self.roots.iter().filter(|r| path.starts_with(r)).max_by_key(|r| r.as_os_str().len()) else {
            return true;
        };
        if !self.include_hidden {
            let rel = path.strip_prefix(root).unwrap_or(path);
            return rel.components().any(|c| c.as_os_str().to_string_lossy().starts_with('.'));
        }
        false
    }
}

/// Mount points of decrypted views of encrypted folders. Full-disk encryption (LUKS on
/// the root or home partition) is not listed: there the index is encrypted at rest too.
pub fn encrypted_mounts() -> Vec<PathBuf> {
    let Ok(info) = std::fs::read_to_string("/proc/self/mountinfo") else { return Vec::new() };
    parse_encrypted_mounts(&info)
}

fn parse_encrypted_mounts(mountinfo: &str) -> Vec<PathBuf> {
    const FS_TYPES: &[&str] = &[
        "ecryptfs",
        "fuse.gocryptfs",
        "fuse.cryfs",
        "fuse.encfs",
        "fuse.securefs",
        "fuse.cryptomator",
        "fuse.sirikali",
    ];
    let mut out = Vec::new();
    for line in mountinfo.lines() {
        // id parent major:minor root mount-point options [optional...] - fstype source super-options
        let fields: Vec<&str> = line.split(' ').collect();
        let Some(dash) = fields.iter().position(|f| *f == "-") else { continue };
        let (Some(mount), Some(fstype), Some(source)) = (fields.get(4), fields.get(dash + 1), fields.get(dash + 2))
        else {
            continue;
        };
        let veracrypt = source.starts_with("/dev/mapper/veracrypt");
        if FS_TYPES.contains(fstype) || fstype.starts_with("fuse.cryptomator") || veracrypt {
            out.push(PathBuf::from(unescape_mount(mount)));
        }
    }
    out
}

/// mountinfo escapes space, tab, newline and backslash as octal (`\040`).
fn unescape_mount(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'\\' && i + 3 < b.len() && b[i + 1..i + 4].iter().all(|c| (b'0'..=b'7').contains(c)) {
            out.push((b[i + 1] - b'0') * 64 + (b[i + 2] - b'0') * 8 + (b[i + 3] - b'0'));
            i += 4;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Work item: a file that is new or changed.
struct Job {
    path: PathBuf,
    size: u64,
    mtime: i64,
    existing: Option<i64>,
}

enum Content {
    /// Name and dates only.
    Bare,
    Text {
        chunks: Vec<Chunk>,
        vecs: Vec<Vec<f32>>,
        space: Space,
    },
    Image {
        taken: Option<i64>,
    },
    Failed(String),
}

struct Prepared {
    job: Job,
    kind: Kind,
    fingerprint: Option<String>,
    content: Content,
}

pub struct Indexer {
    pub cfg: Config,
    conn: Connection,
    statics: Arc<StaticEmbedder>,
    gemma: Arc<GemmaSlot>,
    pub filter: Filter,
    pub stats: Arc<Stats>,
    pool: rayon::ThreadPool,
}

impl Indexer {
    pub fn new(cfg: &Config, statics: Arc<StaticEmbedder>, gemma: Arc<GemmaSlot>, stats: Arc<Stats>) -> Result<Self> {
        let conn = db::open(&config::db_path())?;
        Self::with_conn(cfg, conn, statics, gemma, stats)
    }

    pub fn with_conn(
        cfg: &Config,
        conn: Connection,
        statics: Arc<StaticEmbedder>,
        gemma: Arc<GemmaSlot>,
        stats: Arc<Stats>,
    ) -> Result<Self> {
        db::ensure_vec_tables(
            &conn,
            &db::VecDims { text: statics.text_dim, code: statics.code_dim, gemma: cfg.models.dim },
        )?;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(cfg.index_threads())
            .thread_name(|i| format!("omafind-index-{i}"))
            .start_handler(|_| power::lower_current_thread_priority())
            .build()?;
        Ok(Self { cfg: cfg.clone(), conn, statics, gemma, filter: Filter::new(cfg)?, stats, pool })
    }

    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    /// Walks the roots, indexes new and changed files and removes vanished ones.
    /// Returns the directories it walked (for the watcher).
    pub fn scan(&mut self, budget: &dyn Fn() -> Budget) -> Result<Vec<PathBuf>> {
        let started = Instant::now();
        // Encrypted folders may have been mounted or unmounted since the last scan.
        self.filter = Filter::new(&self.cfg)?;
        self.stats.scanning.store(true, Ordering::Relaxed);
        let mut dirs = Vec::new();
        for root in self.cfg.root_paths() {
            if !root.is_dir() {
                // Unmounted drive or missing folder: keep its entries for when it returns.
                tracing::warn!("root {} is not available; skipping", root.display());
                continue;
            }
            dirs.extend(self.scan_dir(&root, true, budget)?);
        }
        self.stats.scanning.store(false, Ordering::Relaxed);
        tracing::info!(
            "scan finished in {:.1}s: {} files seen, {} indexed, {} removed",
            started.elapsed().as_secs_f32(),
            self.stats.scanned.load(Ordering::Relaxed),
            self.stats.indexed.load(Ordering::Relaxed),
            self.stats.removed.load(Ordering::Relaxed)
        );
        Ok(dirs)
    }

    /// Indexes everything under `dir`. With `prune`, rows under `dir` whose files are gone are removed.
    pub fn scan_dir(&mut self, dir: &Path, prune: bool, budget: &dyn Fn() -> Budget) -> Result<Vec<PathBuf>> {
        let mut known: HashMap<String, (i64, i64, i64)> = HashMap::new();
        {
            let prefix = format!("{}/", dir.to_string_lossy().trim_end_matches('/'));
            let mut stmt =
                self.conn.prepare("SELECT id, path, size, mtime FROM files WHERE path >= ?1 AND path < ?2")?;
            let upper = format!("{}0", prefix.trim_end_matches('/'));
            let rows = stmt.query_map(params![prefix, upper], |r| {
                Ok((r.get::<_, String>(1)?, (r.get::<_, i64>(0)?, r.get::<_, i64>(2)?, r.get::<_, i64>(3)?)))
            })?;
            for row in rows {
                let (p, v) = row?;
                known.insert(p, v);
            }
        }

        let filter = self.filter.clone();
        let walker = ignore::WalkBuilder::new(dir)
            .hidden(!self.cfg.include_hidden)
            .git_ignore(self.cfg.respect_gitignore)
            .git_global(self.cfg.respect_gitignore)
            .git_exclude(self.cfg.respect_gitignore)
            .ignore(self.cfg.respect_gitignore)
            .parents(self.cfg.respect_gitignore)
            .follow_links(false)
            .filter_entry(move |e| !filter.excluded_by_rules(e.path()))
            .build();

        let mut dirs = Vec::new();
        let mut jobs = Vec::new();
        for entry in walker {
            let Ok(entry) = entry else { continue };
            let Some(ft) = entry.file_type() else { continue };
            if ft.is_dir() {
                dirs.push(entry.into_path());
                continue;
            }
            if !ft.is_file() {
                continue;
            }
            let Ok(meta) = entry.metadata() else { continue };
            self.stats.scanned.fetch_add(1, Ordering::Relaxed);
            let path_str = entry.path().to_string_lossy().into_owned();
            let size = meta.len();
            let mtime = mtime_secs(&meta);
            let existing = known.remove(&path_str);
            if let Some((_, s, m)) = existing
                && s == size as i64
                && m == mtime
            {
                continue;
            }
            jobs.push(Job { path: entry.into_path(), size, mtime, existing: existing.map(|e| e.0) });
            if jobs.len() >= BATCH {
                self.wait_for_budget(budget, |b| b.text);
                self.process(std::mem::take(&mut jobs))?;
            }
        }
        if !jobs.is_empty() {
            self.process(jobs)?;
        }
        if prune && !known.is_empty() {
            let tx = self.conn.unchecked_transaction()?;
            for (path, (id, _, _)) in known {
                // A row may have been taken over by a moved file during this scan.
                let current: Option<String> =
                    tx.query_row("SELECT path FROM files WHERE id=?1", [id], |r| r.get(0)).optional()?;
                if current.as_deref() != Some(path.as_str()) {
                    continue;
                }
                if let Some(thumb) = db::delete_file(&tx, id)? {
                    let _ = std::fs::remove_file(thumb);
                }
                self.stats.removed.fetch_add(1, Ordering::Relaxed);
            }
            tx.commit()?;
        }
        Ok(dirs)
    }

    fn wait_for_budget(&self, budget: &dyn Fn() -> Budget, need: impl Fn(Budget) -> bool) {
        let mut waited = 0;
        while !need(budget()) && waited < 600 {
            std::thread::sleep(std::time::Duration::from_secs(1));
            waited += 1;
        }
    }

    /// Re-indexes specific paths (watcher events). Directories are walked.
    pub fn update_paths(&mut self, paths: &[PathBuf], budget: &dyn Fn() -> Budget) -> Result<Vec<PathBuf>> {
        let mut jobs = Vec::new();
        let mut new_dirs = Vec::new();
        let mut gone = Vec::new();
        for p in paths {
            if self.filter.excluded(p) {
                continue;
            }
            match std::fs::symlink_metadata(p) {
                Ok(m) if m.is_dir() => new_dirs.extend(self.scan_dir(p, true, budget)?),
                Ok(m) if m.is_file() => {
                    let path_str = p.to_string_lossy();
                    let existing: Option<(i64, i64, i64)> = self
                        .conn
                        .query_row("SELECT id, size, mtime FROM files WHERE path=?1", [path_str.as_ref()], |r| {
                            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
                        })
                        .optional()?;
                    let mtime = mtime_secs(&m);
                    if existing.is_some_and(|(_, s, t)| s == m.len() as i64 && t == mtime) {
                        continue;
                    }
                    jobs.push(Job { path: p.clone(), size: m.len(), mtime, existing: existing.map(|e| e.0) });
                }
                Ok(_) => {}
                Err(_) => gone.push(p.clone()),
            }
        }
        // New paths first so moves are recognised before the old rows are dropped.
        while !jobs.is_empty() {
            let rest = jobs.split_off(jobs.len().min(BATCH));
            self.process(jobs)?;
            jobs = rest;
        }
        if !gone.is_empty() {
            self.remove_paths(&gone)?;
        }
        Ok(new_dirs)
    }

    /// Removes rows for vanished files or directories.
    pub fn remove_paths(&mut self, paths: &[PathBuf]) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        for p in paths {
            let s = p.to_string_lossy();
            let prefix = format!("{}/", s.trim_end_matches('/'));
            let upper = format!("{}0", s.trim_end_matches('/'));
            let ids: Vec<i64> = {
                let mut stmt = tx.prepare("SELECT id FROM files WHERE path=?1 OR (path >= ?2 AND path < ?3)")?;
                stmt.query_map(params![s.as_ref(), prefix, upper], |r| r.get(0))?.collect::<Result<_, _>>()?
            };
            for id in ids {
                if let Some(thumb) = db::delete_file(&tx, id)? {
                    let _ = std::fs::remove_file(thumb);
                }
                self.stats.removed.fetch_add(1, Ordering::Relaxed);
            }
        }
        tx.commit()?;
        Ok(())
    }

    fn process(&mut self, jobs: Vec<Job>) -> Result<()> {
        // Moves and renames: an unseen path whose content fingerprint matches a row whose
        // file no longer exists just takes over that row (no re-embedding).
        let mut rest = Vec::with_capacity(jobs.len());
        for job in jobs {
            let kind = extract::kind_of(&job.path);
            if job.existing.is_none()
                && kind != Kind::Other
                && let Some(fp) = fingerprint(&job.path, job.size)
                && self.try_move(&job, &fp)?
            {
                continue;
            }
            rest.push(job);
        }
        let statics = self.statics.clone();
        let cfg = self.cfg.clone();
        let prepared: Vec<Prepared> = self.pool.install(|| {
            use rayon::prelude::*;
            rest.into_par_iter().map(|job| prepare(job, &cfg, &statics)).collect()
        });
        let tx = self.conn.unchecked_transaction()?;
        for p in prepared {
            write_prepared(&tx, &p)?;
            self.stats.indexed.fetch_add(1, Ordering::Relaxed);
            if matches!(p.content, Content::Failed(_)) {
                self.stats.errors.fetch_add(1, Ordering::Relaxed);
            }
        }
        tx.commit()?;
        Ok(())
    }

    fn try_move(&self, job: &Job, fp: &str) -> Result<bool> {
        let mut stmt = self.conn.prepare_cached("SELECT id, path FROM files WHERE fingerprint=?1 AND size=?2")?;
        let candidates: Vec<(i64, String)> =
            stmt.query_map(params![fp, job.size as i64], |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<Result<_, _>>()?;
        for (id, old) in candidates {
            if Path::new(&old).exists() {
                continue;
            }
            let name = file_name(&job.path);
            self.conn.execute(
                "UPDATE files SET path=?2, name=?3, ext=?4, mtime=?5, indexed_at=?6 WHERE id=?1",
                params![id, job.path.to_string_lossy(), name, extract::ext_of(&job.path), job.mtime, db::now_secs()],
            )?;
            self.stats.moved.fetch_add(1, Ordering::Relaxed);
            return Ok(true);
        }
        Ok(false)
    }

    /// Embeds up to `limit` pending images with the vision encoder. Returns how many it handled.
    pub fn vision_pass(&mut self, limit: usize, stop: &dyn Fn() -> bool) -> Result<usize> {
        if !self.cfg.indexing.images {
            return Ok(0);
        }
        let pending: Vec<(i64, String)> = {
            let mut stmt = self.conn.prepare(
                "SELECT id, path FROM files WHERE vision_state=?1 ORDER BY COALESCE(taken, mtime) DESC LIMIT ?2",
            )?;
            stmt.query_map(params![VISION_PENDING, limit as i64], |r| Ok((r.get(0)?, r.get(1)?)))?
                .collect::<Result<_, _>>()?
        };
        if pending.is_empty() {
            return Ok(0);
        }
        let gemma = self.gemma.get()?;
        let mut n = 0;
        for (id, path) in pending {
            if stop() {
                break;
            }
            let p = PathBuf::from(&path);
            let result = extract::image::load(&p, self.cfg.max_image_bytes).and_then(|img| {
                let vec = gemma.embed_image(&img)?;
                let thumb = if self.cfg.indexing.thumbnails {
                    let dest = thumb_path(&path);
                    extract::image::write_thumbnail(&img, &dest).ok().map(|_| dest.to_string_lossy().into_owned())
                } else {
                    None
                };
                Ok((vec, thumb))
            });
            let tx = self.conn.unchecked_transaction()?;
            match result {
                Ok((vec, thumb)) => {
                    tx.execute("DELETE FROM vec_image WHERE rowid=?1", [id])?;
                    tx.execute(
                        "INSERT INTO vec_image(rowid, embedding) VALUES (?1, ?2)",
                        params![id, db::f32_bytes(&vec)],
                    )?;
                    tx.execute(
                        "UPDATE files SET vision_state=?2, thumb=?3, error=NULL WHERE id=?1",
                        params![id, VISION_DONE, thumb],
                    )?;
                    self.stats.images_done.fetch_add(1, Ordering::Relaxed);
                }
                Err(e) => {
                    tracing::debug!("vision failed for {path}: {e:#}");
                    tx.execute(
                        "UPDATE files SET vision_state=?2, error=?3 WHERE id=?1",
                        params![id, VISION_FAILED, format!("{e:#}")],
                    )?;
                    self.stats.errors.fetch_add(1, Ordering::Relaxed);
                }
            }
            tx.commit()?;
            n += 1;
        }
        Ok(n)
    }

    /// Pre-computes Gemma vectors for text chunks (only with `indexing.gemma_text`).
    pub fn gemma_text_pass(&mut self, limit: usize) -> Result<usize> {
        if !self.cfg.indexing.gemma_text {
            return Ok(0);
        }
        let rows: Vec<(i64, String, String, Option<String>)> = {
            let mut stmt = self.conn.prepare(
                "SELECT c.id, c.text, f.name, f.kind FROM chunks c JOIN files f ON f.id=c.file_id
                 LEFT JOIN gemma_chunks g ON g.chunk_id=c.id WHERE g.chunk_id IS NULL LIMIT ?1",
            )?;
            stmt.query_map([limit as i64], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?
                .collect::<Result<_, _>>()?
        };
        if rows.is_empty() {
            return Ok(0);
        }
        let gemma = self.gemma.get()?;
        let inputs: Vec<String> = rows.iter().map(|(_, text, name, _)| document_input(Some(name), text)).collect();
        let vecs = gemma.embed_texts(&inputs, self.cfg.search.rescore_max_tokens)?;
        let tx = self.conn.unchecked_transaction()?;
        for ((id, ..), v) in rows.iter().zip(vecs) {
            tx.execute(
                "INSERT OR REPLACE INTO gemma_chunks(chunk_id, vec) VALUES (?1, ?2)",
                params![id, db::f32_bytes(&v)],
            )?;
        }
        tx.commit()?;
        self.stats.gemma_chunks_done.fetch_add(rows.len() as u64, Ordering::Relaxed);
        Ok(rows.len())
    }

    pub fn pending_images(&self) -> Result<i64> {
        Ok(self.conn.query_row("SELECT count(*) FROM files WHERE vision_state=?1", [VISION_PENDING], |r| r.get(0))?)
    }
}

fn prepare(job: Job, cfg: &Config, statics: &StaticEmbedder) -> Prepared {
    let mut kind = extract::kind_of(&job.path);
    if kind == Kind::Other
        && job.path.extension().is_none()
        && job.size > 0
        && job.size <= 1 << 20
        && extract::text::is_probably_text(&job.path)
    {
        kind = Kind::Document;
    }
    let fingerprint = if kind != Kind::Other { fingerprint(&job.path, job.size) } else { None };
    let content = match kind {
        Kind::Document | Kind::Code => match text_content(&job.path, kind, cfg, statics) {
            Ok(c) => c,
            Err(e) => Content::Failed(format!("{e:#}")),
        },
        Kind::Image => Content::Image { taken: extract::image::taken(&job.path) },
        _ => Content::Bare,
    };
    Prepared { job, kind, fingerprint, content }
}

fn text_content(path: &Path, kind: Kind, cfg: &Config, statics: &StaticEmbedder) -> Result<Content> {
    let ext = extract::ext_of(path);
    // Binary office formats can be much larger than their text; cap the plain-text ones only.
    let plain =
        !matches!(ext.as_str(), "pdf" | "docx" | "pptx" | "odt" | "odp" | "epub" | "xlsx" | "xlsm" | "xls" | "ods");
    if plain && std::fs::metadata(path)?.len() > cfg.max_text_bytes {
        anyhow::bail!("larger than max_text_bytes");
    }
    let extracted = extract::extract_text(path, cfg.max_text_bytes)?;
    let chunks = chunk_text(&extracted.text);
    let space = if kind == Kind::Code { Space::Code } else { Space::Text };
    let name = file_name(path);
    // Static models see the file name too: it often carries the topic.
    let inputs: Vec<String> = chunks.iter().map(|c| format!("{name}\n{}", c.text)).collect();
    let vecs = statics.embed(space, &inputs);
    Ok(Content::Text { chunks, vecs, space })
}

fn write_prepared(tx: &Connection, p: &Prepared) -> Result<()> {
    let path = p.job.path.to_string_lossy();
    let (text_state, vision_state, taken, error) = match &p.content {
        Content::Bare => (TEXT_NONE, VISION_NA, None, None),
        Content::Text { .. } => (TEXT_DONE, VISION_NA, None, None),
        Content::Image { taken } => (TEXT_NONE, VISION_PENDING, *taken, None),
        Content::Failed(e) => (TEXT_FAILED, VISION_NA, None, Some(e.as_str())),
    };
    let id: i64 = tx.query_row(
        "INSERT INTO files(path, name, ext, kind, size, mtime, taken, fingerprint, text_state, vision_state, error, indexed_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
         ON CONFLICT(path) DO UPDATE SET kind=excluded.kind, size=excluded.size, mtime=excluded.mtime,
             taken=excluded.taken, fingerprint=excluded.fingerprint, text_state=excluded.text_state,
             vision_state=excluded.vision_state, error=excluded.error, indexed_at=excluded.indexed_at
         RETURNING id",
        params![
            path,
            file_name(&p.job.path),
            extract::ext_of(&p.job.path),
            p.kind.as_str(),
            p.job.size as i64,
            p.job.mtime,
            taken,
            p.fingerprint,
            text_state,
            vision_state,
            error,
            db::now_secs()
        ],
        |r| r.get(0),
    )?;
    if p.job.existing.is_some() {
        db::clear_file_content(tx, id)?;
    }
    if let Content::Text { chunks, vecs, space } = &p.content {
        let mut ins_chunk =
            tx.prepare_cached("INSERT INTO chunks(file_id, ord, line, text) VALUES (?1, ?2, ?3, ?4)")?;
        for (i, (c, v)) in chunks.iter().zip(vecs).enumerate() {
            ins_chunk.execute(params![id, i as i64, c.line as i64, c.text])?;
            crate::vecindex::insert(tx, tx.last_insert_rowid(), id, *space, v)?;
        }
    }
    Ok(())
}

pub fn file_name(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

pub fn mtime_secs(m: &std::fs::Metadata) -> i64 {
    m.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Cheap content identity: size plus a hash of the first and last 64 KiB.
fn fingerprint(path: &Path, size: u64) -> Option<String> {
    let mut f = std::fs::File::open(path).ok()?;
    let mut h = blake3::Hasher::new();
    h.update(&size.to_le_bytes());
    let mut buf = vec![0u8; 64 * 1024];
    let n = f.read(&mut buf).ok()?;
    h.update(&buf[..n]);
    if size > 128 * 1024 {
        f.seek(SeekFrom::End(-(64 * 1024))).ok()?;
        let n = f.read(&mut buf).ok()?;
        h.update(&buf[..n]);
    }
    Some(h.finalize().to_hex()[..32].to_string())
}

pub fn thumb_path(path: &str) -> PathBuf {
    let h = blake3::hash(path.as_bytes()).to_hex();
    config::thumbs_dir().join(&h[..2]).join(format!("{}.jpg", &h[2..24]))
}

/// Counts for `omafind status`.
#[derive(Serialize, Default)]
pub struct IndexCounts {
    pub files: i64,
    pub documents: i64,
    pub code: i64,
    pub images: i64,
    pub images_embedded: i64,
    pub images_pending: i64,
    pub images_failed: i64,
    pub chunks: i64,
    pub gemma_chunks: i64,
    pub text_failed: i64,
    pub db_bytes: u64,
}

pub fn counts(conn: &Connection, db_path: &Path) -> Result<IndexCounts> {
    let q = |sql: &str| -> Result<i64> { Ok(conn.query_row(sql, [], |r| r.get(0))?) };
    Ok(IndexCounts {
        files: q("SELECT count(*) FROM files")?,
        documents: q("SELECT count(*) FROM files WHERE kind='document'")?,
        code: q("SELECT count(*) FROM files WHERE kind='code'")?,
        images: q("SELECT count(*) FROM files WHERE kind='image'")?,
        images_embedded: q(&format!("SELECT count(*) FROM files WHERE vision_state={VISION_DONE}"))?,
        images_pending: q(&format!("SELECT count(*) FROM files WHERE vision_state={VISION_PENDING}"))?,
        images_failed: q(&format!("SELECT count(*) FROM files WHERE vision_state={VISION_FAILED}"))?,
        chunks: q("SELECT count(*) FROM chunks")?,
        gemma_chunks: q("SELECT count(*) FROM gemma_chunks")?,
        text_failed: q(&format!("SELECT count(*) FROM files WHERE text_state={TEXT_FAILED}"))?,
        db_bytes: ["", "-wal"]
            .iter()
            .filter_map(|s| std::fs::metadata(format!("{}{s}", db_path.display())).ok())
            .map(|m| m.len())
            .sum(),
    })
}

/// Paths that changed while nobody was watching are picked up by the periodic rescan; this
/// set dedups watcher events between flushes.
pub type PathSet = HashSet<PathBuf>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_encrypted_mounts() {
        let info = "\
36 35 98:0 / / rw,noatime master:1 - btrfs /dev/mapper/luks-root rw
41 36 0:44 / /home/me/Private rw,nosuid shared:30 - fuse.gocryptfs /home/me/.private rw
42 36 0:45 / /home/me/My\\040Vault rw - fuse.cryfs cryfs@/home/me/.vault rw
43 36 254:3 / /media/veracrypt1 rw - ext4 /dev/mapper/veracrypt1 rw
44 36 0:46 / /home/me/notes rw - fuse.sshfs host:/notes rw";
        assert_eq!(
            parse_encrypted_mounts(info),
            vec![
                PathBuf::from("/home/me/Private"),
                PathBuf::from("/home/me/My Vault"),
                PathBuf::from("/media/veracrypt1")
            ]
        );
    }
}
