//! Static-vector search that stays fast on a whole disk. Every chunk's sign bits (32 bytes
//! for 256 dimensions) are held in memory and scanned by Hamming distance; the shortlist is
//! narrowed with the float query against those bits, then re-ranked exactly with the stored
//! int8 vectors. Brute-force float search in sqlite-vec costs ~0.3 µs per vector, about
//! 300 ms for a million chunks; this takes a few milliseconds.

use std::sync::Mutex;

use anyhow::Result;
use rusqlite::{Connection, OptionalExtension, params};

use crate::embed::static_model::Space;

/// Candidates kept by the Hamming scan, and how many of them get an exact re-rank.
const SHORTLIST: usize = 8000;
const REFINE: usize = 800;

fn space_id(space: Space) -> usize {
    match space {
        Space::Text => 0,
        Space::Code => 1,
    }
}

/// Sign bits, least significant bit first.
pub fn sign_bits(v: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; v.len().div_ceil(8)];
    for (i, x) in v.iter().enumerate() {
        if *x > 0.0 {
            out[i / 8] |= 1 << (i % 8);
        }
    }
    out
}

/// Symmetric int8 quantization. Cosine similarity ignores scale, so none is stored.
pub fn quantize(v: &[f32]) -> Vec<u8> {
    let max = v.iter().fold(0f32, |m, x| m.max(x.abs())).max(f32::MIN_POSITIVE);
    v.iter().map(|x| (x / max * 127.0).round() as i8 as u8).collect()
}

/// Stores a chunk's static vector. Rows go away with their chunk (ON DELETE CASCADE).
pub fn insert(tx: &Connection, chunk_id: i64, file_id: i64, space: Space, v: &[f32]) -> Result<()> {
    tx.prepare_cached("INSERT INTO static_bits(chunk_id, space, bits) VALUES (?1, ?2, ?3)")?.execute(params![
        chunk_id,
        space_id(space) as i64,
        sign_bits(v)
    ])?;
    tx.prepare_cached("INSERT INTO static_vecs(chunk_id, file_id, vec) VALUES (?1, ?2, ?3)")?.execute(params![
        chunk_id,
        file_id,
        quantize(v)
    ])?;
    Ok(())
}

#[derive(Default)]
struct Part {
    words: usize,
    ids: Vec<i64>,
    codes: Vec<u64>,
}

impl Part {
    fn push(&mut self, id: i64, bits: &[u8]) {
        let words = bits.len().div_ceil(8);
        if self.words == 0 {
            self.words = words;
        }
        if words != self.words {
            return;
        }
        for w in bits.chunks(8) {
            let mut b = [0u8; 8];
            b[..w.len()].copy_from_slice(w);
            self.codes.push(u64::from_le_bytes(b));
        }
        self.ids.push(id);
    }

    /// Positions of the `k` codes nearest to `q` by Hamming distance (ties broken by order).
    fn shortlist(&self, q: &[u64], k: usize) -> Vec<usize> {
        let n = self.ids.len();
        if n <= k {
            return (0..n).collect();
        }
        let mut dist = Vec::with_capacity(n);
        let mut hist = vec![0usize; self.words * 64 + 1];
        for c in self.codes.chunks_exact(self.words) {
            let d: u32 = c.iter().zip(q).map(|(a, b)| (a ^ b).count_ones()).sum();
            hist[d as usize] += 1;
            dist.push(d as u16);
        }
        // Everything closer than `cut` fits; fill the rest with codes at exactly `cut`.
        let (mut taken, mut cut) = (0, 0);
        while taken + hist[cut] <= k {
            taken += hist[cut];
            cut += 1;
        }
        let mut ties = k - taken;
        let mut out = Vec::with_capacity(k);
        for (i, &d) in dist.iter().enumerate() {
            let d = d as usize;
            if d < cut {
                out.push(i);
            } else if d == cut && ties > 0 {
                ties -= 1;
                out.push(i);
            }
        }
        out
    }

    /// The float query's dot product with a code's ±1 signs, up to a constant.
    fn asymmetric(&self, pos: usize, q: &[f32]) -> f32 {
        let mut s = 0.0;
        for (wi, &w) in self.codes[pos * self.words..(pos + 1) * self.words].iter().enumerate() {
            let mut w = w;
            while w != 0 {
                let bit = wi * 64 + w.trailing_zeros() as usize;
                s += q.get(bit).copied().unwrap_or(0.0);
                w &= w - 1;
            }
        }
        s
    }
}

#[derive(Default)]
struct Loaded {
    parts: [Part; 2],
    /// Highest chunk id loaded. Chunk ids only grow (AUTOINCREMENT), so newer rows are
    /// exactly those above it; deleted rows linger until `compact` and are skipped.
    max_id: i64,
}

impl Loaded {
    fn len(&self) -> usize {
        self.parts.iter().map(|p| p.ids.len()).sum()
    }

    fn sync(&mut self, conn: &Connection) -> Result<()> {
        let mut stmt =
            conn.prepare_cached("SELECT chunk_id, space, bits FROM static_bits WHERE chunk_id > ?1 ORDER BY chunk_id")?;
        let mut rows = stmt.query([self.max_id])?;
        while let Some(r) = rows.next()? {
            let id: i64 = r.get(0)?;
            let space: i64 = r.get(1)?;
            let bits: Vec<u8> = r.get(2)?;
            if let Some(p) = self.parts.get_mut(space as usize) {
                p.push(id, &bits);
            }
            self.max_id = id;
        }
        Ok(())
    }
}

/// In-memory sign-bit index over `static_bits`, loaded on first use.
#[derive(Default)]
pub struct StaticIndex {
    inner: Mutex<Option<Loaded>>,
}

impl StaticIndex {
    /// Loads the index (if needed) and picks up rows added since the last call.
    pub fn warm(&self, conn: &Connection) -> Result<()> {
        let mut guard = self.inner.lock().unwrap();
        let loaded = guard.get_or_insert_with(Loaded::default);
        loaded.sync(conn)
    }

    /// Nearest chunks in a static space: (chunk id, file id, cosine similarity), best first.
    pub fn search(&self, conn: &Connection, space: Space, q: &[f32], k: usize) -> Result<Vec<(i64, i64, f64)>> {
        let candidates: Vec<i64> = {
            let mut guard = self.inner.lock().unwrap();
            let loaded = guard.get_or_insert_with(Loaded::default);
            loaded.sync(conn)?;
            let part = &loaded.parts[space_id(space)];
            if part.ids.is_empty() {
                return Ok(Vec::new());
            }
            let qb: Vec<u64> = sign_bits(q)
                .chunks(8)
                .map(|w| {
                    let mut b = [0u8; 8];
                    b[..w.len()].copy_from_slice(w);
                    u64::from_le_bytes(b)
                })
                .collect();
            let mut short: Vec<(usize, f32)> =
                part.shortlist(&qb, SHORTLIST).into_iter().map(|i| (i, part.asymmetric(i, q))).collect();
            short.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            short.truncate(REFINE.max(k));
            short.into_iter().map(|(i, _)| part.ids[i]).collect()
        };

        let qn = q.iter().map(|x| x * x).sum::<f32>().sqrt().max(f32::MIN_POSITIVE);
        let mut stmt = conn.prepare_cached("SELECT file_id, vec FROM static_vecs WHERE chunk_id=?1")?;
        let mut out = Vec::with_capacity(candidates.len());
        for id in candidates {
            let Some((file, v)) =
                stmt.query_row([id], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?))).optional()?
            else {
                continue;
            };
            let (mut dot, mut norm) = (0f32, 0f32);
            for (a, &b) in q.iter().zip(&v) {
                let b = b as i8 as f32;
                dot += a * b;
                norm += b * b;
            }
            out.push((id, file, (dot / (qn * norm.sqrt().max(f32::MIN_POSITIVE))) as f64));
        }
        out.sort_by(|a, b| b.2.total_cmp(&a.2));
        out.truncate(k);
        Ok(out)
    }

    /// Rebuilds the index once deleted chunks make up a noticeable share of it. Loads the
    /// fresh copy without holding the lock, so searches carry on meanwhile.
    pub fn compact(&self, conn: &Connection) -> Result<()> {
        let live: i64 = conn.query_row("SELECT count(*) FROM static_bits", [], |r| r.get(0))?;
        let held = self.inner.lock().unwrap().as_ref().map(Loaded::len);
        let Some(held) = held else { return Ok(()) };
        let live = live as usize;
        if held <= live + (live / 8).max(1000) {
            return Ok(());
        }
        let mut fresh = Loaded::default();
        fresh.sync(conn)?;
        let mut guard = self.inner.lock().unwrap();
        fresh.sync(conn)?;
        tracing::debug!("static index compacted: {held} -> {} entries", fresh.len());
        *guard = Some(fresh);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shortlist_keeps_nearest_codes() {
        let mut p = Part::default();
        let base = [0u8; 32];
        for i in 0..100u8 {
            let mut b = base;
            // Code i differs from the all-zero query in i % 50 bits.
            for bit in 0..(i % 50) as usize {
                b[bit / 8] |= 1 << (bit % 8);
            }
            p.push(i as i64, &b);
        }
        let got = p.shortlist(&[0u64; 4], 4);
        let mut ids: Vec<i64> = got.iter().map(|&i| p.ids[i]).collect();
        ids.sort();
        assert_eq!(ids, vec![0, 1, 50, 51]);
    }

    #[test]
    fn quantized_vectors_keep_their_direction() {
        let v: Vec<f32> = (0..256).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
        let q: Vec<f32> = quantize(&v).iter().map(|&b| b as i8 as f32).collect();
        let dot: f32 = v.iter().zip(&q).map(|(a, b)| a * b).sum();
        let cos = dot / (v.iter().map(|x| x * x).sum::<f32>().sqrt() * q.iter().map(|x| x * x).sum::<f32>().sqrt());
        assert!(cos > 0.999, "{cos}");
        assert_eq!(sign_bits(&[1.0, -1.0, 0.5, 0.0, 2.0, -3.0, 1.0, 1.0, 1.0]), vec![0b1101_0101, 1]);
    }
}
