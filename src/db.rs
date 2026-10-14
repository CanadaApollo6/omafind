//! The single SQLite index: file details, FTS5 keyword indexes, static chunk vectors and the
//! sqlite-vec image table.

use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::sync::Once;

use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension, params};

pub const SCHEMA_VERSION: i64 = 2;

/// Vision state of a file.
pub const VISION_NA: i64 = 0;
pub const VISION_PENDING: i64 = 1;
pub const VISION_DONE: i64 = 2;
pub const VISION_FAILED: i64 = 3;

/// Text state of a file.
pub const TEXT_PENDING: i64 = 0;
pub const TEXT_DONE: i64 = 1;
pub const TEXT_NONE: i64 = 2;
pub const TEXT_FAILED: i64 = 3;

static VEC_INIT: Once = Once::new();

fn register_sqlite_vec() {
    VEC_INIT.call_once(|| unsafe {
        #[allow(clippy::missing_transmute_annotations)]
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(sqlite_vec::sqlite3_vec_init as *const ())));
    });
}

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS files(
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    ext TEXT NOT NULL,
    kind TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime INTEGER NOT NULL,
    taken INTEGER,
    fingerprint TEXT,
    text_state INTEGER NOT NULL DEFAULT 0,
    vision_state INTEGER NOT NULL DEFAULT 0,
    thumb TEXT,
    error TEXT,
    indexed_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS files_kind ON files(kind);
CREATE INDEX IF NOT EXISTS files_text_state ON files(text_state);
CREATE INDEX IF NOT EXISTS files_vision_state ON files(vision_state);
CREATE INDEX IF NOT EXISTS files_fingerprint ON files(fingerprint);

-- AUTOINCREMENT: ids are never reused, so caches keyed by chunk id can't go stale and the
-- in-memory static index can pick up new chunks as "id above the last one seen".
CREATE TABLE IF NOT EXISTS chunks(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    ord INTEGER NOT NULL,
    line INTEGER NOT NULL,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_file ON chunks(file_id);

-- Static (Model2Vec) vectors: sign bits for the in-memory scan, int8 for the re-rank.
CREATE TABLE IF NOT EXISTS static_bits(
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    space INTEGER NOT NULL,
    bits BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS static_vecs(
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    file_id INTEGER NOT NULL,
    vec BLOB NOT NULL
);

-- Gemma vectors for text chunks, filled on demand by deep search (or ahead of time).
CREATE TABLE IF NOT EXISTS gemma_chunks(
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    vec BLOB NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS names_fts USING fts5(
    name, path, content='files', content_rowid='id',
    tokenize='unicode61 remove_diacritics 2', prefix='2 3'
);
CREATE TRIGGER IF NOT EXISTS files_ai AFTER INSERT ON files BEGIN
    INSERT INTO names_fts(rowid, name, path) VALUES (new.id, new.name, new.path);
END;
CREATE TRIGGER IF NOT EXISTS files_ad AFTER DELETE ON files BEGIN
    INSERT INTO names_fts(names_fts, rowid, name, path) VALUES ('delete', old.id, old.name, old.path);
END;
CREATE TRIGGER IF NOT EXISTS files_au AFTER UPDATE OF name, path ON files BEGIN
    INSERT INTO names_fts(names_fts, rowid, name, path) VALUES ('delete', old.id, old.name, old.path);
    INSERT INTO names_fts(rowid, name, path) VALUES (new.id, new.name, new.path);
END;

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='id',
    tokenize='porter unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
"#;

/// Opens (and creates or migrates) the index.
pub fn open(path: &Path) -> Result<Connection> {
    register_sqlite_vec();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let fresh = !path.exists();
    let conn = Connection::open(path).with_context(|| format!("opening {}", path.display()))?;
    if fresh {
        // The index holds file contents: keep it private to the user.
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
    conn.busy_timeout(std::time::Duration::from_secs(10))?;
    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         PRAGMA synchronous=NORMAL;
         PRAGMA foreign_keys=ON;
         PRAGMA cache_size=-8000;
         PRAGMA temp_store=MEMORY;",
    )?;
    conn.execute_batch(SCHEMA)?;
    let version: Option<i64> = get_meta(&conn, "schema_version")?.and_then(|v| v.parse().ok());
    match version {
        None => set_meta(&conn, "schema_version", &SCHEMA_VERSION.to_string())?,
        Some(v) if v == SCHEMA_VERSION => {}
        Some(v) => anyhow::bail!(
            "index schema {v} is not supported by this build (expects {SCHEMA_VERSION}); run `omafind reset`"
        ),
    }
    Ok(conn)
}

/// Opens the index read-only (search path). Fails if it doesn't exist yet.
pub fn open_readonly(path: &Path) -> Result<Connection> {
    register_sqlite_vec();
    anyhow::ensure!(path.exists(), "no index at {} yet; run `omafind index` or start the daemon", path.display());
    let conn = Connection::open_with_flags(
        path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    conn.busy_timeout(std::time::Duration::from_secs(5))?;
    conn.execute_batch("PRAGMA cache_size=-8000; PRAGMA temp_store=MEMORY;")?;
    Ok(conn)
}

pub fn get_meta(conn: &Connection, key: &str) -> Result<Option<String>> {
    Ok(conn.query_row("SELECT value FROM meta WHERE key=?1", [key], |r| r.get(0)).optional()?)
}

pub fn set_meta(conn: &Connection, key: &str, value: &str) -> Result<()> {
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?1, ?2) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        params![key, value],
    )?;
    Ok(())
}

/// Vector dimensions. Static vectors follow the static models; the image table follows the
/// configured Matryoshka dimension.
pub struct VecDims {
    pub text: usize,
    pub code: usize,
    pub gemma: usize,
}

/// Creates the image vec0 table, or checks that stored vectors match the models in use.
pub fn ensure_vec_tables(conn: &Connection, dims: &VecDims) -> Result<()> {
    for (table, key, dim) in [
        ("static_text", "dim_text", dims.text),
        ("static_code", "dim_code", dims.code),
        ("vec_image", "dim_gemma", dims.gemma),
    ] {
        match get_meta(conn, key)? {
            Some(existing) if existing == dim.to_string() => {}
            Some(existing) => anyhow::bail!(
                "{table} holds {existing}-dimensional vectors but the current models produce {dim}; \
                 run `omafind reset --vectors` and re-index"
            ),
            None => {
                if table == "vec_image" {
                    conn.execute_batch(&format!(
                        "CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0(embedding float[{dim}] distance_metric=cosine);"
                    ))?;
                }
                set_meta(conn, key, &dim.to_string())?;
            }
        }
    }
    Ok(())
}

/// Removes everything derived from a file's content, keeping the file row.
pub fn clear_file_content(conn: &Connection, file_id: i64) -> Result<()> {
    conn.execute("DELETE FROM chunks WHERE file_id=?1", [file_id])?;
    conn.execute("DELETE FROM vec_image WHERE rowid=?1", [file_id])?;
    Ok(())
}

/// Deletes a file and everything derived from it. Returns its thumbnail path, if any.
pub fn delete_file(conn: &Connection, file_id: i64) -> Result<Option<String>> {
    let thumb: Option<String> =
        conn.query_row("SELECT thumb FROM files WHERE id=?1", [file_id], |r| r.get(0)).optional()?.flatten();
    clear_file_content(conn, file_id)?;
    conn.execute("DELETE FROM files WHERE id=?1", [file_id])?;
    Ok(thumb)
}

/// Drops all vector tables and Gemma caches, so the next index run rebuilds them.
pub fn reset_vectors(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "DROP TABLE IF EXISTS vec_image;
         DELETE FROM gemma_chunks;
         DELETE FROM meta WHERE key IN ('dim_text','dim_code','dim_gemma','static_text','static_code','gemma');
         DELETE FROM chunks;
         UPDATE files SET text_state=0 WHERE kind IN ('document','code');
         UPDATE files SET vision_state=1 WHERE vision_state<>0;",
    )?;
    Ok(())
}

pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

pub fn bytes_f32(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

pub fn now_secs() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::static_model::Space;

    #[test]
    fn schema_and_vectors_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let conn = open(&dir.path().join("t.sqlite")).unwrap();
        ensure_vec_tables(&conn, &VecDims { text: 4, code: 4, gemma: 4 }).unwrap();
        conn.execute(
            "INSERT INTO files(path,name,ext,kind,size,mtime,indexed_at) VALUES ('/a/b.txt','b.txt','txt','document',1,1,1)",
            [],
        )
        .unwrap();
        conn.execute("INSERT INTO chunks(file_id,ord,line,text) VALUES (1,0,1,'hello sourdough world')", []).unwrap();
        let tx = conn.unchecked_transaction().unwrap();
        crate::vecindex::insert(&tx, 1, 1, Space::Text, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        tx.commit().unwrap();
        let index = crate::vecindex::StaticIndex::default();
        let hits = index.search(&conn, Space::Text, &[0.9, 0.1, 0.0, 0.0], 5).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!((hits[0].0, hits[0].1), (1, 1));
        assert!(hits[0].2 > 0.99);
        assert!(index.search(&conn, Space::Code, &[0.9, 0.1, 0.0, 0.0], 5).unwrap().is_empty());
        conn.execute("INSERT INTO vec_image(rowid, embedding) VALUES (1, ?1)", [f32_bytes(&[1.0, 0.0, 0.0, 0.0])])
            .unwrap();
        let hit: i64 = conn
            .query_row(
                "SELECT rowid FROM vec_image WHERE embedding MATCH ?1 AND k=1",
                [f32_bytes(&[0.9, 0.1, 0.0, 0.0])],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(hit, 1);
        let fts: i64 = conn
            .query_row("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'sourdough'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(fts, 1);
        let name: i64 =
            conn.query_row("SELECT rowid FROM names_fts WHERE names_fts MATCH 'b*'", [], |r| r.get(0)).unwrap();
        assert_eq!(name, 1);
        delete_file(&conn, 1).unwrap();
        let left: i64 = conn
            .query_row("SELECT (SELECT count(*) FROM static_bits) + (SELECT count(*) FROM static_vecs)", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(left, 0);
        // Deleted chunks are skipped even before the in-memory index is compacted.
        assert!(index.search(&conn, Space::Text, &[0.9, 0.1, 0.0, 0.0], 5).unwrap().is_empty());
        assert!(ensure_vec_tables(&conn, &VecDims { text: 8, code: 4, gemma: 4 }).is_err());
    }
}
