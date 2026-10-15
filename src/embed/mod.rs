//! Embedding models: static (fast tier) and EmbeddingGemma 2 (deep tier).

pub mod audio;
pub mod gemma;
pub mod static_model;

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex, mpsc};

use anyhow::Result;

use crate::config::Config;
use crate::db::now_secs;

/// Loads Gemma on first use and drops it again after a quiet period, so the daemon
/// only holds the model in memory while it is being used.
///
/// The model is always loaded on a helper thread started by `new`: ONNX Runtime's worker
/// threads inherit the priority of the thread that creates the session, and a query must
/// not end up on the indexer's nice-19 threads just because the indexer loaded it first.
pub struct GemmaSlot {
    cfg: Config,
    inner: Mutex<Option<Arc<gemma::Gemma>>>,
    last_used: AtomicI64,
    loader: Mutex<mpsc::Sender<mpsc::Sender<Result<gemma::Gemma>>>>,
}

impl GemmaSlot {
    /// Call from a normal-priority thread.
    pub fn new(cfg: &Config) -> Self {
        let (tx, rx) = mpsc::channel::<mpsc::Sender<Result<gemma::Gemma>>>();
        let loader_cfg = cfg.clone();
        std::thread::Builder::new()
            .name("omafind-gemma-loader".into())
            .spawn(move || {
                for reply in rx {
                    let _ = reply.send(gemma::Gemma::load(&loader_cfg));
                }
            })
            .expect("spawning the model loader thread");
        Self {
            cfg: cfg.clone(),
            inner: Mutex::new(None),
            last_used: AtomicI64::new(now_secs()),
            loader: Mutex::new(tx),
        }
    }

    pub fn get(&self) -> Result<Arc<gemma::Gemma>> {
        self.last_used.store(now_secs(), Ordering::Relaxed);
        let mut guard = self.inner.lock().unwrap();
        if let Some(g) = guard.as_ref() {
            return Ok(g.clone());
        }
        let started = std::time::Instant::now();
        let (tx, rx) = mpsc::channel();
        self.loader.lock().unwrap().send(tx)?;
        let g = Arc::new(rx.recv()??);
        tracing::info!("loaded {} in {:.1}s", g.describe(), started.elapsed().as_secs_f32());
        *guard = Some(g.clone());
        Ok(g)
    }

    /// The model if it is loaded, without counting as a use.
    pub fn loaded(&self) -> Option<Arc<gemma::Gemma>> {
        self.inner.lock().unwrap().clone()
    }

    pub fn is_loaded(&self) -> bool {
        self.inner.lock().unwrap().is_some()
    }

    /// Drops the model if nobody used it for `unload_after_secs`. In-flight users keep
    /// their `Arc` alive until they finish.
    pub fn unload_if_idle(&self) -> bool {
        let idle = now_secs() - self.last_used.load(Ordering::Relaxed);
        if idle < self.cfg.models.unload_after_secs as i64 {
            return false;
        }
        let mut guard = self.inner.lock().unwrap();
        if guard.take().is_some() {
            drop(guard);
            release_memory();
            tracing::info!("unloaded EmbeddingGemma after {idle}s idle");
            return true;
        }
        false
    }
}

/// Hands freed heap pages back to the system. glibc keeps them after a model is dropped, so
/// without this an idle daemon would stay at its peak size.
pub fn release_memory() {
    #[cfg(target_env = "gnu")]
    unsafe {
        libc::malloc_trim(0);
    }
}
