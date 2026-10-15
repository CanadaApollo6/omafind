//! Running helper programs (pdftotext, tesseract, ffmpeg, whisper-cli) as child processes,
//! so a hostile or broken file can't take the daemon down, and with a time limit.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

/// Runs `cmd`, feeding it `input` on stdin, and returns its stdout. Fails if it runs longer
/// than `timeout` or exits unsuccessfully without output.
pub fn run(cmd: &mut Command, input: Option<Vec<u8>>, timeout: Duration) -> Result<Vec<u8>> {
    let name = cmd.get_program().to_string_lossy().into_owned();
    let mut child = cmd
        .stdin(if input.is_some() { Stdio::piped() } else { Stdio::null() })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("running {name}"))?;
    if let Some(data) = input {
        let mut stdin = child.stdin.take().expect("piped stdin");
        std::thread::spawn(move || {
            let _ = stdin.write_all(&data);
        });
    }
    let mut stdout = child.stdout.take().expect("piped stdout");
    let out = std::thread::spawn(move || {
        let mut s = Vec::new();
        let _ = stdout.read_to_end(&mut s);
        s
    });
    let mut stderr = child.stderr.take().expect("piped stderr");
    let err = std::thread::spawn(move || {
        let mut s = Vec::new();
        let _ = stderr.by_ref().take(64 * 1024).read_to_end(&mut s);
        let _ = std::io::copy(&mut stderr, &mut std::io::sink());
        s
    });
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            let out = out.join().unwrap_or_default();
            let err = err.join().unwrap_or_default();
            if !status.success() && out.is_empty() {
                let msg = String::from_utf8_lossy(&err);
                let last = msg.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("").trim();
                bail!("{name} failed ({status}){}", if last.is_empty() { String::new() } else { format!(": {last}") });
            }
            return Ok(out);
        }
        if start.elapsed() > timeout {
            let _ = child.kill();
            let _ = child.wait();
            bail!("{name} timed out after {}s", timeout.as_secs());
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Finds a program on `PATH`.
pub fn which(program: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|d| d.join(program)).find(|p| {
        use std::os::unix::fs::PermissionsExt;
        p.metadata().is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
    })
}
