use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use clap::{Parser, Subcommand};

use omafind::config::{self, Config};
use omafind::embed::GemmaSlot;
use omafind::embed::static_model::StaticEmbedder;
use omafind::index::{self, Indexer, Stats};
use omafind::ipc::{self, Request};
use omafind::power::{self, Budget};
use omafind::search::{Engine, Results, StreamOpts};
use omafind::{daemon, db, eval, models};

#[derive(Parser)]
#[command(name = "omafind", version, about = "Local semantic file search: find files by what is in them.")]
struct Cli {
    /// Config file (default: ~/.config/omafind/config.toml).
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Search the index. Uses the daemon when it is running.
    #[command(alias = "s")]
    Search {
        /// Words to search for, plus optional filters: kind:image ext:pdf after:2024-03 before:2w year:2023 in:~/Documents
        #[arg(required = true, num_args = 1..)]
        query: Vec<String>,
        /// Print JSON lines, one per stage (fast, deep), then {"stage":"done"}.
        #[arg(long)]
        json: bool,
        /// Only the instant tier (names, keywords, static embeddings).
        #[arg(long, conflicts_with = "deep")]
        fast: bool,
        /// Only the Gemma tier.
        #[arg(long)]
        deep: bool,
        /// Match file names only (word starts, then anywhere in the name). Filters still apply.
        #[arg(long, conflicts_with_all = ["fast", "deep"])]
        names: bool,
        /// Results per group.
        #[arg(short = 'n', long)]
        limit: Option<usize>,
        /// Wait this long after the fast results before running Gemma (cancelled if the
        /// caller goes away first). Used by the launcher.
        #[arg(long, default_value_t = 0)]
        pause: u64,
        /// Search in this process even if the daemon is running.
        #[arg(long)]
        no_daemon: bool,
    },
    /// Index (or update) everything now, in the foreground.
    Index {
        /// Skip the vision encoder; images are still searchable by name.
        #[arg(long)]
        no_images: bool,
    },
    /// Run the background service: watches for changes and answers searches.
    Daemon,
    /// Render a large preview picture (a PDF page, a video frame, a HEIC or RAW photo) and
    /// print its path. Used by the launcher; pictures are cached.
    Preview {
        path: PathBuf,
        /// PDF page (1-based).
        #[arg(long)]
        page: Option<u32>,
        /// Video position, in milliseconds.
        #[arg(long)]
        at_ms: Option<i64>,
    },
    /// Show index and daemon status.
    Status {
        #[arg(long)]
        json: bool,
    },
    /// Download or list the embedding models.
    Models {
        #[command(subcommand)]
        cmd: ModelsCmd,
    },
    /// Measure search quality on a labelled corpus (see eval/).
    Eval {
        #[arg(long, default_value = "eval/corpus")]
        corpus: PathBuf,
        #[arg(long, default_value = "eval/queries.toml")]
        queries: PathBuf,
        /// Comma-separated: filename,keyword,fast,deep
        #[arg(long, default_value = "filename,keyword,fast,deep", value_delimiter = ',')]
        modes: Vec<String>,
        /// Don't precompute Gemma chunk vectors: measures deep search right after indexing.
        #[arg(long)]
        cold: bool,
        #[arg(long)]
        json: bool,
        #[arg(short, long)]
        verbose: bool,
    },
    /// Delete the index (or only its vectors, e.g. after changing models.dim).
    Reset {
        #[arg(long)]
        vectors: bool,
    },
    /// Print an example config file.
    Config {
        /// Print the config file location instead.
        #[arg(long)]
        path: bool,
    },
}

#[derive(Subcommand)]
enum ModelsCmd {
    /// Download the static models, EmbeddingGemma 2 and the speech model into
    /// ~/.local/share/omafind/models.
    Fetch {
        /// Gemma precisions to fetch: q4, int8, fp32 (default: what models.precision resolves to).
        #[arg(long, value_delimiter = ',')]
        precision: Vec<String>,
        /// Skip the code-specific static model.
        #[arg(long)]
        no_code: bool,
        /// Skip Gemma's audio encoder.
        #[arg(long)]
        no_audio: bool,
        /// Skip the speech-to-text model.
        #[arg(long)]
        no_speech: bool,
    },
    /// Show where each model was found.
    List,
}

fn main() {
    let cli = Cli::parse();
    let quiet = matches!(cli.cmd, Cmd::Search { .. } | Cmd::Status { .. } | Cmd::Config { .. } | Cmd::Preview { .. });
    let default = if quiet { "omafind=warn" } else { "omafind=info" };
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_env("OMAFIND_LOG").unwrap_or_else(|_| default.into()))
        .with_writer(std::io::stderr)
        .with_target(false)
        .init();
    if let Err(e) = run(cli) {
        eprintln!("omafind: {e:#}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<()> {
    if let Cmd::Config { path } = cli.cmd {
        if path {
            println!("{}", cli.config.unwrap_or_else(config::config_path).display());
        } else {
            print!("{}", config::EXAMPLE_CONFIG);
        }
        return Ok(());
    }
    let cfg = Config::load(cli.config.as_deref())?;
    match cli.cmd {
        Cmd::Search { query, json, fast, deep, names, limit, pause, no_daemon } => {
            let opts = StreamOpts {
                limit: limit.unwrap_or(cfg.search.limit),
                names_only: names,
                fast: !deep,
                deep: !fast,
                pause_ms: pause,
            };
            search(&cfg, &query.join(" "), opts, json, no_daemon)
        }
        Cmd::Index { no_images } => index(&cfg, no_images),
        Cmd::Daemon => daemon::run(cfg),
        Cmd::Preview { path, page, at_ms } => {
            println!("{}", omafind::preview::render(&cfg, &path, page, at_ms)?.display());
            Ok(())
        }
        Cmd::Status { json } => status(&cfg, json),
        Cmd::Models { cmd: ModelsCmd::Fetch { precision, no_code, no_audio, no_speech } } => {
            let precisions = if precision.is_empty() {
                vec![omafind::embed::gemma::resolve_precision(&cfg.models)]
            } else {
                precision
            };
            let speech = !no_speech && cfg.indexing.speech;
            models::fetch(
                &precisions,
                &models::FetchOpts {
                    code: !no_code && cfg.models.static_code_model,
                    audio: !no_audio && cfg.indexing.audio,
                    speech: speech.then(|| cfg.indexing.speech_model.clone()),
                },
            )?;
            if speech && !omafind::extract::speech::available() {
                eprintln!("Transcripts need whisper-cli: install whisper.cpp (Arch: pacman -S whisper-cpp).");
            }
            Ok(())
        }
        Cmd::Models { cmd: ModelsCmd::List } => {
            for (name, path) in models::describe() {
                match path {
                    Some(p) => println!("{name:<36} {}", p.display()),
                    None => println!("{name:<36} missing (run `omafind models fetch`)"),
                }
            }
            Ok(())
        }
        Cmd::Eval { corpus, queries, modes, cold, json, verbose } => {
            let report = eval::run(&cfg, &corpus, &queries, &modes, cold)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                eval::print(&report, verbose);
            }
            Ok(())
        }
        Cmd::Reset { vectors } => {
            anyhow::ensure!(ipc::connect().is_none(), "stop the daemon first (systemctl --user stop omafind)");
            let path = config::db_path();
            if vectors {
                db::reset_vectors(&db::open(&path)?)?;
                println!("Cleared vectors in {}; run `omafind index` to rebuild them.", path.display());
            } else {
                for suffix in ["", "-wal", "-shm"] {
                    let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
                }
                let _ = std::fs::remove_dir_all(config::thumbs_dir());
                let _ = std::fs::remove_dir_all(omafind::preview::previews_dir());
                println!("Deleted {} and the picture caches.", path.display());
            }
            Ok(())
        }
        Cmd::Config { .. } => unreachable!(),
    }
}

fn search(cfg: &Config, input: &str, opts: StreamOpts, json: bool, no_daemon: bool) -> Result<()> {
    let mut last: Option<serde_json::Value> = None;
    let stdout = std::io::stdout();
    let mut on_stage = |v: serde_json::Value| -> Result<()> {
        if json {
            let mut out = stdout.lock();
            serde_json::to_writer(&mut out, &v)?;
            out.write_all(b"\n")?;
            out.flush()?;
        } else if v.get("groups").is_some() {
            last = Some(v);
        }
        Ok(())
    };

    if !no_daemon && let Some(conn) = ipc::connect() {
        let stages = match (opts.names_only, opts.fast, opts.deep) {
            (true, _, _) => "names",
            (_, true, false) => "fast",
            (_, false, true) => "deep",
            _ => "both",
        };
        let req =
            Request::Search { q: input.to_string(), limit: opts.limit, stages: stages.into(), pause_ms: opts.pause_ms };
        ipc::request(conn, &req, &mut on_stage)?;
        if !json {
            print_results(last.as_ref());
        }
        return Ok(());
    }

    // No daemon: search in this process.
    anyhow::ensure!(
        config::db_path().exists(),
        "no index yet: run `omafind index`, or start the daemon with `systemctl --user enable --now omafind`"
    );
    let statics = Arc::new(StaticEmbedder::load(cfg.models.static_code_model)?);
    let engine = Engine::new(cfg, statics, Arc::new(GemmaSlot::new(cfg)));
    let conn = engine.open()?;
    let mut emit = |r: &Results| on_stage(serde_json::to_value(r)?);
    engine.stream(
        &conn,
        input,
        opts,
        &|d| {
            std::thread::sleep(d);
            true
        },
        &|| false,
        &mut emit,
    )?;
    if json {
        println!("{{\"stage\":\"done\"}}");
    } else {
        print_results(last.as_ref());
    }
    Ok(())
}

fn print_results(v: Option<&serde_json::Value>) {
    let Some(v) = v else { return };
    let groups = v["groups"].as_array().cloned().unwrap_or_default();
    if groups.is_empty() {
        println!("No matches.");
    }
    for g in groups {
        println!("\x1b[1m{}\x1b[0m", g["title"].as_str().unwrap_or(""));
        for h in g["hits"].as_array().into_iter().flatten() {
            // Where the match is: a line, a PDF page, a moment in a recording.
            let line = match (h["line"].as_i64(), h["page"].as_i64(), h["at_ms"].as_i64()) {
                (Some(l), _, _) => format!(":{l}"),
                (_, Some(p), _) => format!(" (page {p})"),
                (_, _, Some(ms)) if ms >= 1000 => format!(" (at {}:{:02})", ms / 60_000, ms / 1000 % 60),
                _ => String::new(),
            };
            println!(
                "  {}{}  \x1b[2m{}\x1b[0m",
                h["name"].as_str().unwrap_or(""),
                line,
                h["dir"].as_str().unwrap_or("")
            );
            if let Some(s) = h["snippet"].as_str() {
                let s: String = s.chars().take(110).collect();
                println!("      \x1b[2m{s}\x1b[0m");
            }
        }
        println!();
    }
    if let Some(note) = v["note"].as_str() {
        println!("({note})");
    }
}

fn index(cfg: &Config, no_images: bool) -> Result<()> {
    if let Some(conn) = ipc::connect() {
        ipc::request(conn, &Request::Rescan, |_| Ok(()))?;
        println!("The daemon is running; asked it to rescan. Follow progress with `omafind status`.");
        return Ok(());
    }
    let statics = Arc::new(StaticEmbedder::load(cfg.models.static_code_model)?);
    let gemma = Arc::new(GemmaSlot::new(cfg));
    let stats = Arc::new(Stats::default());
    let mut indexer = Indexer::new(cfg, statics, gemma, stats.clone())?;
    let t = std::time::Instant::now();
    indexer.scan(&|| Budget::ALL)?;
    let s = stats.snapshot();
    println!(
        "Scanned {} files in {:.1}s: {} indexed, {} moved, {} removed, {} errors.",
        s.scanned,
        t.elapsed().as_secs_f32(),
        s.indexed,
        s.moved,
        s.removed,
        s.errors
    );
    if no_images {
        return Ok(());
    }
    power::lower_current_thread_priority();
    let pending = indexer.pending()?;
    if pending.any() {
        println!("Slower work runs newest first; Ctrl-C is safe, the daemon resumes later.");
    }
    let stop = || false;
    run_lane(&mut indexer, pending.images, "images embedded", |ix| ix.vision_pass(8, &stop))?;
    run_lane(&mut indexer, pending.media, "recordings and videos embedded", |ix| ix.media_pass(1, &stop))?;
    // Camera photos wait for their image vector before OCR, so this comes after the images.
    let ocr = indexer.pending()?.ocr;
    run_lane(&mut indexer, ocr, "images and scans read (OCR)", |ix| ix.ocr_pass(4, &stop))?;
    run_lane(&mut indexer, pending.speech, "recordings transcribed", |ix| ix.speech_pass(&stop))?;
    let c = index::counts(indexer.conn(), &config::db_path())?;
    if c.images_failed > 0 {
        println!("{} images could not be embedded (see `omafind status`).", c.images_failed);
    }
    Ok(())
}

/// Runs one indexing lane to the end with a progress line.
fn run_lane(
    indexer: &mut Indexer,
    total: i64,
    what: &str,
    mut pass: impl FnMut(&mut Indexer) -> Result<usize>,
) -> Result<()> {
    if total == 0 {
        return Ok(());
    }
    let t = std::time::Instant::now();
    let mut done = 0;
    loop {
        let n = pass(indexer)?;
        if n == 0 {
            break;
        }
        done += n;
        let rate = done as f64 / t.elapsed().as_secs_f64();
        let left = (total as f64 - done as f64).max(0.0) / rate / 60.0;
        eprint!("\r  {what}: {done}/{total}  ~{left:.0} min left   ");
    }
    eprintln!("\r  {what}: {done} in {}          ", humanize_secs(t.elapsed().as_secs()));
    Ok(())
}

fn status(cfg: &Config, json: bool) -> Result<()> {
    let v = match ipc::connect() {
        Some(conn) => {
            let mut v = serde_json::Value::Null;
            ipc::request(conn, &Request::Status, |line| {
                v = line;
                Ok(())
            })?;
            v
        }
        None => {
            let path = config::db_path();
            let counts = if path.exists() { Some(index::counts(&db::open_readonly(&path)?, &path)?) } else { None };
            serde_json::json!({ "running": false, "index": counts, "db": path, "roots": cfg.root_paths() })
        }
    };
    if json {
        println!("{}", serde_json::to_string_pretty(&v)?);
        return Ok(());
    }
    let running = v["running"].as_bool().unwrap_or(false);
    if running {
        println!(
            "Daemon: running (pid {}, up {}), {}; {} folders watched; Gemma {}",
            v["pid"],
            humanize_secs(v["uptime_s"].as_u64().unwrap_or(0)),
            v["phase"].as_str().unwrap_or("?"),
            v["watches"],
            if v["gemma_loaded"].as_bool() == Some(true) { "loaded" } else { "not loaded" }
        );
        if v["budget"]["gemma"].as_bool() == Some(false) {
            println!(
                "        background embedding paused ({})",
                if v["on_ac"].as_bool() == Some(true) { "machine busy" } else { "on battery" }
            );
        }
    } else {
        println!("Daemon: not running (start it with `systemctl --user enable --now omafind`)");
    }
    let roots: Vec<String> =
        v["roots"].as_array().into_iter().flatten().filter_map(|r| r.as_str().map(String::from)).collect();
    println!("Roots:  {}", roots.join(", "));
    println!("Index:  {}", v["db"].as_str().unwrap_or(""));
    let i = &v["index"];
    if i.is_null() {
        println!("        (empty; run `omafind index`)");
        return Ok(());
    }
    let n = |k: &str| i[k].as_i64().unwrap_or(0);
    println!(
        "        {} files: {} documents, {} code, {} images; {} text chunks; {:.1} MB",
        n("files"),
        n("documents"),
        n("code"),
        n("images"),
        n("chunks"),
        i["db_bytes"].as_u64().unwrap_or(0) as f64 / 1e6
    );
    println!(
        "        images: {} embedded, {} pending, {} failed",
        n("images_embedded"),
        n("images_pending"),
        n("images_failed")
    );
    if n("ocr_read") + n("ocr_pending") > 0 {
        println!("        text in images and scans: {} read, {} pending", n("ocr_read"), n("ocr_pending"));
    }
    if n("audio") + n("video") > 0 {
        println!(
            "        {} recordings, {} videos: {} embedded, {} pending; {} transcribed, {} pending",
            n("audio"),
            n("video"),
            n("media_embedded"),
            n("media_pending"),
            n("transcribed"),
            n("speech_pending")
        );
    }
    Ok(())
}

fn humanize_secs(s: u64) -> String {
    match s {
        0..60 => format!("{s}s"),
        60..3600 => format!("{}m", s / 60),
        3600..86400 => format!("{}h{}m", s / 3600, s % 3600 / 60),
        _ => format!("{}d{}h", s / 86400, s % 86400 / 3600),
    }
}
