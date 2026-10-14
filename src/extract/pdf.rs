//! PDF text through poppler's `pdftotext`, run as a child process so a hostile or
//! broken PDF can't take the daemon down, and with a time limit.

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

const TIMEOUT: Duration = Duration::from_secs(60);
const MAX_PAGES: &str = "300";

pub fn extract(path: &Path) -> Result<String> {
    let mut child = Command::new("pdftotext")
        .args(["-q", "-enc", "UTF-8", "-l", MAX_PAGES])
        .arg(path)
        .arg("-")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("running pdftotext (install poppler)")?;
    let mut stdout = child.stdout.take().expect("piped stdout");
    let reader = std::thread::spawn(move || {
        let mut s = Vec::new();
        let _ = stdout.read_to_end(&mut s);
        s
    });
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            let out = reader.join().unwrap_or_default();
            if !status.success() && out.is_empty() {
                bail!("pdftotext failed ({status})");
            }
            // Form feeds separate pages.
            return Ok(String::from_utf8_lossy(&out).replace('\u{c}', "\n\n"));
        }
        if start.elapsed() > TIMEOUT {
            let _ = child.kill();
            let _ = child.wait();
            bail!("pdftotext timed out");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}
