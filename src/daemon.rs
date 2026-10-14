//! The background service: serves searches over the socket, watches the file system
//! with inotify, and spends idle time on the vision encoder.

use std::collections::HashSet;
use std::io::{BufRead, BufReader, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use crossbeam_channel::{Receiver, select};
use notify::{RecursiveMode, Watcher};
use serde_json::json;

use crate::config::{self, Config};
use crate::embed::GemmaSlot;
use crate::embed::static_model::StaticEmbedder;
use crate::index::{self, Indexer, PathSet, Stats};
use crate::ipc::{self, Request};
use crate::power::{self, Budget, OwnLoad};
use crate::search::{Engine, StreamOpts};

const DEBOUNCE: Duration = Duration::from_millis(1500);
const IDLE_POLL: Duration = Duration::from_secs(30);
const VISION_STEP: usize = 4;
const GEMMA_TEXT_STEP: usize = 8;

enum Control {
    Rescan,
}

struct Shared {
    cfg: Config,
    engine: Engine,
    stats: Arc<Stats>,
    gemma: Arc<GemmaSlot>,
    started: Instant,
    budget: Mutex<Budget>,
    phase: Mutex<&'static str>,
    watches: std::sync::atomic::AtomicUsize,
    control: crossbeam_channel::Sender<Control>,
}

pub fn run(cfg: Config) -> Result<()> {
    // Fewer malloc arenas: many short-lived connection threads otherwise each keep one.
    #[cfg(target_env = "gnu")]
    unsafe {
        libc::mallopt(libc::M_ARENA_MAX, 2);
    }
    let socket = config::socket_path();
    if ipc::connect_at(&socket).is_some() {
        anyhow::bail!("omafind daemon is already running ({})", socket.display());
    }
    let _ = std::fs::remove_file(&socket);
    if let Some(dir) = socket.parent() {
        std::fs::create_dir_all(dir)?;
    }

    let statics = Arc::new(StaticEmbedder::load(cfg.models.static_code_model)?);
    let gemma = Arc::new(GemmaSlot::new(&cfg));
    let stats = Arc::new(Stats::default());
    let indexer = Indexer::new(&cfg, statics.clone(), gemma.clone(), stats.clone())?;
    let (control, control_rx) = crossbeam_channel::unbounded();
    let shared = Arc::new(Shared {
        engine: Engine::new(&cfg, statics, gemma.clone()),
        cfg,
        stats,
        gemma,
        started: Instant::now(),
        budget: Mutex::new(Budget::ALL),
        phase: Mutex::new("starting"),
        watches: Default::default(),
        control,
    });

    let s = shared.clone();
    std::thread::Builder::new().name("omafind-indexer".into()).spawn(move || {
        if let Err(e) = index_loop(indexer, control_rx, &s) {
            tracing::error!("indexer stopped: {e:#}");
            *s.phase.lock().unwrap() = "failed";
        }
    })?;

    let s = shared.clone();
    std::thread::Builder::new().name("omafind-janitor".into()).spawn(move || {
        // Load the static index before the first search needs it.
        if let Err(e) = s.engine.open().and_then(|c| s.engine.static_index.warm(&c)) {
            tracing::debug!("static index: {e:#}");
        }
        loop {
            std::thread::sleep(Duration::from_secs(30));
            s.gemma.unload_if_idle();
            if let Err(e) = s.engine.open().and_then(|c| s.engine.static_index.compact(&c)) {
                tracing::debug!("static index: {e:#}");
            }
        }
    })?;

    let listener = UnixListener::bind(&socket).with_context(|| format!("binding {}", socket.display()))?;
    std::fs::set_permissions(&socket, std::fs::Permissions::from_mode(0o600))?;
    tracing::info!("listening on {}", socket.display());
    for conn in listener.incoming() {
        let Ok(conn) = conn else { continue };
        let s = shared.clone();
        std::thread::spawn(move || {
            if let Err(e) = handle(conn, &s) {
                tracing::debug!("connection: {e:#}");
            }
        });
    }
    Ok(())
}

fn handle(conn: UnixStream, s: &Shared) -> Result<()> {
    let mut reader = BufReader::new(conn.try_clone()?);
    let mut line = String::new();
    conn.set_read_timeout(Some(Duration::from_secs(5)))?;
    reader.read_line(&mut line)?;
    let mut out = conn.try_clone()?;
    let mut send = |v: &dyn erased::Json| -> Result<()> {
        let mut buf = v.to_json()?;
        buf.push('\n');
        out.write_all(buf.as_bytes())?;
        out.flush()?;
        Ok(())
    };
    let req: Request = match serde_json::from_str(line.trim()) {
        Ok(r) => r,
        Err(e) => return send(&json!({ "error": format!("bad request: {e}") })),
    };
    match req {
        Request::Search { q, limit, stages, pause_ms } => {
            let db = s.engine.open()?;
            let opts = StreamOpts {
                limit: limit.clamp(1, 200),
                fast: stages != "deep",
                deep: stages != "fast",
                pause_ms: pause_ms.min(5000),
            };
            let fd = conn.as_raw_fd();
            let wait = |d: Duration| !peer_activity(fd, d.as_millis() as i32);
            let cancelled = || peer_activity(fd, 0);
            let mut emit = |r: &crate::search::Results| send(r);
            s.engine.stream(&db, &q, opts, &wait, &cancelled, &mut emit)?;
            send(&json!({ "stage": "done" }))
        }
        Request::Status => send(&status(s)),
        Request::Rescan => {
            let _ = s.control.send(Control::Rescan);
            send(&json!({ "ok": true }))
        }
    }
}

/// True if the client closed the connection or sent more data within `timeout_ms`.
/// Either way the current request is stale.
fn peer_activity(fd: i32, timeout_ms: i32) -> bool {
    let mut p = libc::pollfd { fd, events: libc::POLLIN | libc::POLLRDHUP, revents: 0 };
    let n = unsafe { libc::poll(&mut p, 1, timeout_ms) };
    n > 0 && p.revents != 0
}

fn status(s: &Shared) -> serde_json::Value {
    let counts = s.engine.open().and_then(|c| index::counts(&c, &s.engine.db_path)).ok();
    let budget = *s.budget.lock().unwrap();
    json!({
        "running": true,
        "pid": std::process::id(),
        "uptime_s": s.started.elapsed().as_secs(),
        "phase": *s.phase.lock().unwrap(),
        "on_ac": power::on_ac(),
        "budget": { "text": budget.text, "gemma": budget.gemma },
        "gemma_loaded": s.gemma.is_loaded(),
        "watches": s.watches.load(Ordering::Relaxed),
        "progress": s.stats.snapshot(),
        "index": counts,
        "db": s.engine.db_path,
        "roots": s.cfg.root_paths(),
    })
}

struct Watches {
    watcher: notify::RecommendedWatcher,
    dirs: HashSet<PathBuf>,
    warned: bool,
}

impl Watches {
    fn add(&mut self, dirs: Vec<PathBuf>, shared: &Shared) {
        for d in dirs {
            if self.dirs.contains(&d) {
                continue;
            }
            match self.watcher.watch(&d, RecursiveMode::NonRecursive) {
                Ok(()) => {
                    self.dirs.insert(d);
                }
                Err(e) => {
                    if !self.warned {
                        tracing::warn!(
                            "cannot watch {} ({e}); changes there are picked up by the periodic rescan. \
                             Raise fs.inotify.max_user_watches if you have very many folders.",
                            d.display()
                        );
                        self.warned = true;
                    }
                }
            }
        }
        // Dirs that disappeared lose their inotify watch automatically.
        self.dirs.retain(|d| d.is_dir());
        shared.watches.store(self.dirs.len(), Ordering::Relaxed);
    }
}

fn index_loop(mut indexer: Indexer, control: Receiver<Control>, s: &Shared) -> Result<()> {
    power::lower_current_thread_priority();
    let own = Mutex::new(OwnLoad::new());
    let budget = || {
        let b = power::budget(&s.cfg, own.lock().unwrap().sample());
        *s.budget.lock().unwrap() = b;
        b
    };

    let (ev_tx, events) = crossbeam_channel::unbounded();
    let watcher = notify::recommended_watcher(move |res| {
        let _ = ev_tx.send(res);
    })?;
    let mut watches = Watches { watcher, dirs: HashSet::new(), warned: false };

    let set_phase = |p: &'static str| *s.phase.lock().unwrap() = p;
    set_phase("scanning");
    // Start watching the roots straight away so nothing is missed during the first scan.
    watches.add(s.cfg.root_paths().into_iter().filter(|p| p.is_dir()).collect(), s);
    let dirs = indexer.scan(&budget)?;
    watches.add(dirs, s);
    let mut last_scan = Instant::now();
    let rescan_every = Duration::from_secs(s.cfg.indexing.rescan_hours.max(1) * 3600);

    let mut pending = PathSet::new();
    let mut last_event = Instant::now();
    let mut full_rescan = false;
    let mut idle_work = true;
    let mut text_turn = false;

    loop {
        let timeout = if !pending.is_empty() {
            DEBOUNCE.saturating_sub(last_event.elapsed())
        } else if idle_work {
            Duration::ZERO
        } else {
            IDLE_POLL
        };
        if pending.is_empty() && !idle_work {
            set_phase("watching");
        }
        select! {
            recv(events) -> ev => match ev {
                Ok(Ok(ev)) => {
                    if ev.need_rescan() {
                        full_rescan = true;
                    }
                    if !matches!(ev.kind, notify::EventKind::Access(_)) {
                        pending.extend(ev.paths);
                        last_event = Instant::now();
                    }
                    continue;
                }
                Ok(Err(e)) => tracing::warn!("watch error: {e}"),
                Err(_) => anyhow::bail!("watcher channel closed"),
            },
            recv(control) -> c => if let Ok(Control::Rescan) = c { full_rescan = true },
            default(timeout) => {}
        }

        if !pending.is_empty() && last_event.elapsed() >= DEBOUNCE {
            set_phase("updating");
            let paths: Vec<PathBuf> = pending.drain().collect();
            let before = indexer.stats.indexed.load(Ordering::Relaxed);
            match indexer.update_paths(&paths, &budget) {
                Ok(new_dirs) => watches.add(new_dirs, s),
                Err(e) => tracing::warn!("update failed: {e:#}"),
            }
            let n = indexer.stats.indexed.load(Ordering::Relaxed) - before;
            if n > 0 {
                tracing::info!("re-indexed {n} changed files");
            }
            idle_work = true;
        }
        if full_rescan || last_scan.elapsed() > rescan_every {
            set_phase("scanning");
            match indexer.scan(&budget) {
                Ok(dirs) => watches.add(dirs, s),
                Err(e) => tracing::warn!("rescan failed: {e:#}"),
            }
            full_rescan = false;
            last_scan = Instant::now();
            idle_work = true;
        }
        if !pending.is_empty() {
            continue;
        }

        // Idle: give the vision encoder (and optional Gemma text vectors) a few items at a time
        // so new file events are never kept waiting for long.
        if idle_work {
            let b = budget();
            if !b.gemma {
                // Retried after the next poll interval.
                idle_work = false;
                set_phase("paused (power budget)");
                continue;
            }
            // Alternate between images and text vectors so neither starves the other.
            let stop = || !events.is_empty() || !control.is_empty();
            let mut did = 0;
            for step in 0..2 {
                if (step == 0) == text_turn {
                    set_phase("embedding text");
                    did += indexer.gemma_text_pass(GEMMA_TEXT_STEP).unwrap_or_else(|e| {
                        tracing::warn!("gemma text pass: {e:#}");
                        0
                    });
                } else {
                    set_phase("embedding images");
                    did += indexer.vision_pass(VISION_STEP, &stop).unwrap_or_else(|e| {
                        tracing::warn!("vision pass: {e:#}");
                        0
                    });
                }
                if did > 0 {
                    break;
                }
            }
            text_turn = !text_turn;
            if did == 0 {
                idle_work = false;
                if let Some(g) = s.gemma.loaded() {
                    g.unload_vision();
                }
                tracing::info!("index is up to date");
            }
        } else if indexer.pending_images().unwrap_or(0) > 0 {
            // Woken by the poll interval while images wait (e.g. after a power-budget pause).
            idle_work = true;
        }
    }
}

/// Lets `send` take both `Results` and ad-hoc `json!` values.
mod erased {
    pub trait Json {
        fn to_json(&self) -> anyhow::Result<String>;
    }
    impl<T: serde::Serialize> Json for T {
        fn to_json(&self) -> anyhow::Result<String> {
            Ok(serde_json::to_string(self)?)
        }
    }
}
