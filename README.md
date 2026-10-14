<img src="assets/icon.svg" width="64" height="64" alt="">

# omafind

Find files on Linux by what is in them. Type "the email about the lease renewal",
"sourdough starter" or "mountains at sunset" and omafind finds the document, the note or
the photo, even if its file name is `IMG_4021.HEIC` or `scan0003.pdf`.

Everything runs on your machine. omafind indexes your files into one local SQLite file
and searches it in two tiers:

- **Instant (while you type):** file names, keywords (SQLite FTS5) and small static
  embeddings ([Model2Vec](https://github.com/MinishLab/model2vec)) for every chunk of
  every document. Typically 15–45 ms on a million-chunk index.
- **Deep (when you pause):** [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)
  re-scores the top 100 text candidates and searches photos with its vision encoder.
  There is no captioning step: images and queries share one embedding space. Deep
  results stream in after the instant ones.

The first release targets Linux and [Omarchy](https://omarchy.org): a CLI, a background
daemon and a launcher plugin for the Omarchy shell.

## Install

### From source (any distribution)

You need Rust 1.90+, libheif and poppler (`pdftotext`):

```sh
sudo pacman -S --needed rust libheif poppler    # Arch / Omarchy
git clone https://github.com/CanadaApollo6/omafind && cd omafind
dist/install.sh
```

The script does the following:
- Builds omafind and installs `~/.local/bin/omafind`.
- Downloads the models (~370 MB, pinned and checked by SHA-256).
- Writes `~/.config/omafind/config.toml`.
- Enables the `omafind` systemd user service, which builds the index in the background.
- On Omarchy, installs and enables the **Find** launcher plugin.

Bind a key to the plugin in `~/.config/hypr/bindings.lua`:

```lua
o.bind("SUPER + CTRL + SLASH", "Find files", "omarchy-shell shell toggle omafind.search")
```

`CUDA=1 dist/install.sh` builds with ONNX Runtime's CUDA provider. It needs cuDNN 9
(`pacman -S cudnn`). Without cuDNN omafind falls back to the CPU.

### Arch package

`dist/arch/PKGBUILD` builds two packages:
- `omafind-git`: the binary, the user service, and the plugin under `/usr/share/omafind/omarchy`.
- `omafind-models`: the models, under `/usr/share/omafind/models`.

```sh
cd dist/arch && makepkg -si
systemctl --user enable --now omafind
mkdir -p ~/.config/omarchy/plugins && cp -r /usr/share/omafind/omarchy/omafind.search ~/.config/omarchy/plugins/
omarchy plugin enable omafind.search
```

## Use

```sh
omafind status                               # index and daemon state
omafind search lease renewal email           # grouped results: documents, code, images
omafind search kind:image dog on the beach
omafind search "tax return" after:2024-01 ext:pdf
omafind search --names invoice 2023          # file names only
omafind search --json budget spreadsheet     # one JSON line per stage, for scripts
omafind index                                # index now, in the foreground
```

Filters can be mixed with words:

| Filter | Meaning |
|---|---|
| `kind:image` (`document`, `code`, `image`), also `type:` / `is:` | one kind of file |
| `ext:pdf` | file extension |
| `after:2024-03`, `since:2w`, `before:2023`, `until:3d`, `year:2022` | modified (photos: taken) date |
| `in:~/Documents`, `path:work` | folder |

**File names only.** When you know roughly what a file is called, `--names` (or **Tab** in
the launcher) skips contents and Gemma:
- Every word must start a word in the name: "tax 23" finds `Tax-Return-2023.pdf`.
- Typed text also matches anywhere in the name: "4021" finds `IMG_4021.HEIC`.
- Only if nothing matches that way, any single word will do.
- Filters still apply.

To open the launcher straight into name mode, bind
`omarchy-shell shell toggle omafind.search '{"names":true}'`.

In the launcher:
- **Enter** opens the result.
- **Alt+Enter** or **Ctrl+Enter** shows it in the file manager.
- **Shift+Enter** copies its path.
- **Tab** switches between content search and file names only.

## What gets indexed

- **Documents:**
  - Text, Markdown, Org, reStructuredText, LaTeX, HTML, e-mail and subtitles.
  - CSV, Jupyter notebooks and RTF.
  - PDF, via `pdftotext`.
  - Word/Writer, PowerPoint/Impress, Excel/Calc and EPUB.
- **Code:** about 60 languages plus Makefiles, Dockerfiles, PKGBUILDs and similar. Code uses a code-specific static model.
- **Images:** JPEG, PNG, WebP, GIF, TIFF, BMP, AVIF and JPEG XL. HEIC/HEIF through libheif. Camera RAW (CR2/CR3, NEF, ARW, DNG, ORF, RW2, RAF and more) through the embedded preview.
- **Not yet:** audio and video are recognised and searchable by name only.

By default omafind indexes your home folder. It respects `.gitignore` and skips hidden
files and caches (Trash, browser profiles, package caches, `node_modules`, build output).
It also skips anything that looks like a key or keystore: `*.pem`, `*.key`, `id_ed25519*`,
`*.kdbx`, `*.gpg` and so on. `omafind config` prints every option with its default.

The daemon keeps the index current:
- It watches folders with inotify, so new, edited, moved and deleted files show up within seconds.
- Re-indexing is skipped for unchanged or merely renamed files: their content fingerprint already matches.
- It rescans every 12 hours, to catch changes inotify cannot see (other machines writing to a share, for example).

## Privacy

- **Stays local:** the index never leaves your machine. omafind makes no network
  connections except `omafind models fetch`, which downloads the pinned model files.
- **Private files:** the index and the search socket are created readable by your user
  only (`0600`). The index holds text from your files, so treat it like the files.
- **Encrypted folders:** folders inside a mounted gocryptfs, CryFS, EncFS, eCryptfs,
  securefs, Cryptomator or VeraCrypt volume are skipped. Otherwise their contents would
  end up in plain text in the index. Set `index_encrypted = true` if your index lives on
  an encrypted disk anyway. Full-disk encryption (LUKS) protects the index like
  everything else.
- **Forgetting files:** deleting a file removes it from the index. `omafind reset`
  deletes the whole index.

## Resource use

omafind is built to run all the time without being noticed.

- **Indexing pace:**
  - The indexer runs at low CPU and I/O priority and pauses while the machine is busy (load above 75%).
  - On battery it only does cheap text work by default. Gemma work waits for the charger (`indexing.on_battery`).
- **Memory:**
  - The daemon idles at about 185 MB with the static models loaded.
  - Gemma (q4 on CPU) adds 500–700 MB while it embeds or searches, and is unloaded after 5 idle minutes.
  - The in-memory search index costs about 40 bytes per chunk: 40 MB for a million chunks.
- **Disk:** the index is about the size of the text it holds. Thumbnails go to `~/.cache/omafind`.

## Measurements

Measured on an i5-13600K (20 threads) with 32 GB RAM, CPU only. These are the numbers
behind the "hard parts" of local semantic search.

**First-time indexing**:
- **Text and code** were measured on 99,021 files (2 GB) from `/usr/share/doc` and `~/.cargo/registry/src`:

  | Metric | Result |
  |---|---|
  | Indexing time | 120 s (~825 files/s) |
  | Chunks | 1,078,242 |
  | Index size | 2.1 GB |
  | No-change rescan | 0.2 s |
  | Fast search, through the daemon | 13–44 ms |
- **Images** go through the Gemma vision encoder at ~0.85 s each on the CPU (140 tokens), so 1,000 photos take about 15 minutes of background time. Text search works during that time; images join the results as they are embedded. With CUDA the encoder is much faster and uses 280 tokens.
- **Gemma text vectors** are precomputed in the background as well (~85 ms per chunk on the CPU). Until a chunk has one, deep search embeds it on demand within a time budget (`search.rescore_budget_ms`).

**Search quality.** `eval/` holds a synthetic corpus of 45 personal files:
- notes, journal entries, recipes, finance and health records;
- school and work documents;
- a little code.

It comes with 50 queries that describe a file without using its name (for example "the
doctor said to cut back on salt"), plus 5 negative queries about topics that are not in
the corpus. Reproduce with `omafind eval --corpus eval/corpus --queries eval/queries.toml`.

| mode | Hit@1 | Hit@5 | Hit@10 | MRR | results shown | mean ms |
|---|---|---|---|---|---|---|
| file name only | 0.26 | 0.26 | 0.26 | 0.260 | 0.6 | 0.1 |
| keywords (BM25) | 0.78 | 0.92 | 0.98 | 0.848 | 8.8 | 0.3 |
| instant tier | 0.86 | 0.98 | 0.98 | 0.903 | 8.9 | 0.7 |
| deep tier (Gemma) | **0.92** | **1.00** | **1.00** | **0.953** | **2.7** | 30.0 |

- **Names vs. content:** file-name search finds the right file for a quarter of the queries. The deep tier puts it first for 92% of them.
- **Shown results:** the deep tier also shows far fewer results, because it only shows files clearly above the query's "unrelated" level. The five negative queries return nothing.
- **Precomputed vectors:** the table assumes the background job has already computed Gemma vectors.
- **Right after indexing** (`--cold`), deep search embeds candidates on demand within its budget. It then scores Hit@1 0.88, Hit@10 0.98 and MRR 0.923, at 225 ms mean and 273 ms p95, and the negative queries still show nothing.
- **Coverage:** image search was checked by hand on a folder of wallpapers. The labelled eval covers documents and code only.

**How the instant tier stays fast:**
- Every chunk's static vector is stored twice: as int8 for exact scoring, and as 256 sign bits.
- The daemon keeps the sign bits in memory and scans them by Hamming distance (3 ms for 876k vectors).
- The 8,000 nearest are narrowed down with the float query, and 800 are re-scored exactly.
- Against an exhaustive search this finds 97–99% of the true top 10. Brute-force float search in SQLite took 280 ms for the same table.

## How it works

```
                      ┌────────────── omafind daemon (systemd user service) ───────────────┐
 files ── inotify ──▶ │ extract ─▶ chunk ─▶ static embed ─▶ SQLite: files, chunks,         │
                      │   (text, PDF, Office, images)        FTS5, static vectors,         │
                      │                                      Gemma text and image vectors  │
                      │ idle or plugged in: Gemma vision for images, Gemma text for chunks │
                      └──────────▲─────────────────────────────────────────────────────────┘
 launcher / CLI ── Unix socket ──┘   query ─▶ instant tier ─▶ (pause) ─▶ deep tier
```

| Piece | Where |
|---|---|
| Crawl, inotify watches, change detection, power and load throttling | `src/index.rs`, `src/daemon.rs`, `src/power.rs` |
| Extraction and chunking | `src/extract/`, `src/chunk.rs` |
| Static models and EmbeddingGemma 2 (ONNX Runtime) | `src/embed/` |
| Index schema | `src/db.rs` |
| In-memory static vector index | `src/vecindex.rs` |
| Query parsing, fusion, Gemma re-scoring, relevance cut-offs | `src/search.rs` |
| Launcher plugin (Quickshell / QML) | `dist/omarchy/omafind.search/` |

**Model details:**
- EmbeddingGemma 2 runs from the [ONNX export](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX), text and vision encoders only.
- It uses 4-bit weights on the CPU (`models.precision`) and fp32 on CUDA.
- Vectors are shortened to 256 dimensions (Matryoshka, `models.dim`).

**Relevance:** raw similarity scores drift from query to query. For each query, omafind
measures the similarity to a random sample of the index and shows only results clearly
above that level. Tune with `search.text_margin` and `search.image_margin`.

## Roadmap

- **Phase 2:**
  - OCR for scanned PDFs and screenshots.
  - Audio and video: speech-to-text, plus Gemma's audio encoder.
  - Richer previews.
  - Labels for people and pets.
- External and removable drives with their own index.
- A standalone Tauri app.

## License

Apache License 2.0; see [LICENSE](LICENSE). The models are not part of this repository
and have their own licenses: EmbeddingGemma 2 is Apache 2.0, the Model2Vec models are MIT.
See [NOTICE](NOTICE).

Created in [T3 Code](https://t3.codes).
