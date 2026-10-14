//! The daemon's Unix-socket protocol: one JSON request line per connection, answered
//! with JSON lines. A search answers with the fast results, then the deep results, then
//! `{"stage":"done"}`. Closing the connection cancels the request, which is how the
//! launcher drops a query the user has typed past.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::config;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "lowercase")]
pub enum Request {
    Search {
        q: String,
        #[serde(default = "default_limit")]
        limit: usize,
        /// "both" (default), "fast", "deep" or "names" (file names only).
        #[serde(default = "default_stages")]
        stages: String,
        #[serde(default = "default_pause")]
        pause_ms: u64,
    },
    Status,
    /// Walk all roots again (picks up anything the watcher missed).
    Rescan,
}

fn default_limit() -> usize {
    8
}
fn default_stages() -> String {
    "both".into()
}
fn default_pause() -> u64 {
    350
}

/// Connects to a running daemon, or returns None if there is none.
pub fn connect() -> Option<UnixStream> {
    connect_at(&config::socket_path())
}

pub fn connect_at(path: &Path) -> Option<UnixStream> {
    UnixStream::connect(path).ok()
}

/// Sends a request and calls `on_line` for every JSON line of the answer until the
/// daemon closes the connection or sends `{"stage":"done"}`.
pub fn request(
    stream: UnixStream,
    req: &Request,
    mut on_line: impl FnMut(serde_json::Value) -> Result<()>,
) -> Result<()> {
    let mut writer = stream.try_clone()?;
    let mut line = serde_json::to_string(req)?;
    line.push('\n');
    writer.write_all(line.as_bytes())?;
    writer.flush()?;
    stream.set_read_timeout(Some(Duration::from_secs(120)))?;
    let reader = BufReader::new(stream);
    for line in reader.lines() {
        let line = line.context("reading from the omafind daemon")?;
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(&line).context("bad reply from daemon")?;
        if let Some(err) = v.get("error").and_then(|e| e.as_str()) {
            anyhow::bail!("daemon: {err}");
        }
        let done = v.get("stage").and_then(|s| s.as_str()) == Some("done");
        on_line(v)?;
        if done {
            break;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_defaults() {
        let r: Request = serde_json::from_str(r#"{"op":"search","q":"tax forms"}"#).unwrap();
        match r {
            Request::Search { q, limit, stages, pause_ms } => {
                assert_eq!((q.as_str(), limit, stages.as_str(), pause_ms), ("tax forms", 8, "both", 350));
            }
            _ => panic!(),
        }
        assert!(matches!(serde_json::from_str(r#"{"op":"status"}"#).unwrap(), Request::Status));
    }
}
