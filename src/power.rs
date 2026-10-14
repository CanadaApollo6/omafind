//! Being a good citizen: AC/battery state, load, and low-priority worker threads.

use std::path::Path;

use crate::config::{Config, num_cpus};

/// What background indexing may do right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    /// Extract text and compute static embeddings.
    pub text: bool,
    /// Run Gemma (vision encoder, background text vectors).
    pub gemma: bool,
}

impl Budget {
    pub const ALL: Budget = Budget { text: true, gemma: true };
}

/// True on AC power or when the machine has no battery.
pub fn on_ac() -> bool {
    let Ok(entries) = std::fs::read_dir("/sys/class/power_supply") else { return true };
    let mut has_battery = false;
    for e in entries.flatten() {
        let dir = e.path();
        let kind = read_trim(&dir.join("type"));
        match kind.as_deref() {
            Some("Mains") | Some("USB") => {
                if read_trim(&dir.join("online")).as_deref() == Some("1") {
                    return true;
                }
            }
            Some("Battery")
                // Peripheral batteries (mice, headsets) report scope=Device.
                if read_trim(&dir.join("scope")).as_deref() != Some("Device") => {
                    has_battery = true;
                }
            _ => {}
        }
    }
    !has_battery
}

/// 1-minute load average divided by the number of cores.
pub fn load_per_core() -> f32 {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| s.split_whitespace().next().and_then(|v| v.parse::<f32>().ok()))
        .map(|l| l / num_cpus() as f32)
        .unwrap_or(0.0)
}

fn read_trim(p: &Path) -> Option<String> {
    std::fs::read_to_string(p).ok().map(|s| s.trim().to_string())
}

/// Decides what may run, given the power state and how busy the machine is.
/// `own_load` is the load this process contributes (its busy worker threads), which
/// shouldn't count against itself.
pub fn budget(cfg: &Config, own_load: f32) -> Budget {
    let busy = (load_per_core() - own_load / num_cpus() as f32) > cfg.indexing.max_load;
    if busy {
        return Budget { text: false, gemma: false };
    }
    if on_ac() {
        return Budget::ALL;
    }
    match cfg.indexing.on_battery.as_str() {
        "full" => Budget::ALL,
        "pause" => Budget { text: false, gemma: false },
        _ => Budget { text: true, gemma: false },
    }
}

/// Lowers the calling thread to nice 19 and the idle I/O class, so indexing yields to
/// everything the user is doing. Query threads keep normal priority.
pub fn lower_current_thread_priority() {
    unsafe {
        let tid = libc::syscall(libc::SYS_gettid) as libc::id_t;
        libc::setpriority(libc::PRIO_PROCESS, tid, 19);
        const IOPRIO_WHO_PROCESS: libc::c_long = 1;
        const IOPRIO_CLASS_IDLE: libc::c_long = 3;
        libc::syscall(libc::SYS_ioprio_set, IOPRIO_WHO_PROCESS, tid as libc::c_long, IOPRIO_CLASS_IDLE << 13);
    }
}

/// This process's recent CPU use, in cores, smoothed like the kernel's 1-minute load
/// average so the two can be compared.
pub struct OwnLoad {
    last: Option<(std::time::Instant, f64)>,
    ema: f32,
}

impl Default for OwnLoad {
    fn default() -> Self {
        Self::new()
    }
}

impl OwnLoad {
    pub fn new() -> Self {
        Self { last: None, ema: 0.0 }
    }

    pub fn sample(&mut self) -> f32 {
        let now = std::time::Instant::now();
        let Some(cpu) = process_cpu_secs() else { return self.ema };
        if let Some((t, c)) = self.last {
            let dt = now.duration_since(t).as_secs_f64();
            if dt > 0.05 {
                let cores = ((cpu - c) / dt) as f32;
                let alpha = (1.0 - (-dt / 60.0).exp()) as f32;
                self.ema += alpha * (cores - self.ema);
            } else {
                return self.ema;
            }
        }
        self.last = Some((now, cpu));
        self.ema
    }
}

fn process_cpu_secs() -> Option<f64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    // Fields after the parenthesised command name; utime and stime are fields 14 and 15.
    let rest = &stat[stat.rfind(')')? + 2..];
    let f: Vec<&str> = rest.split_whitespace().collect();
    let ticks: f64 = f.get(11)?.parse::<f64>().ok()? + f.get(12)?.parse::<f64>().ok()?;
    let hz = unsafe { libc::sysconf(libc::_SC_CLK_TCK) } as f64;
    Some(ticks / hz.max(1.0))
}
