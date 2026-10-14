//! Prints Gemma similarities of queries against every image and chunk in an index, for
//! calibrating cut-offs: `cargo run --release --example probe -- "query" ...` (uses OMAFIND_DB).
use omafind::{config, db, embed::gemma};
fn main() -> anyhow::Result<()> {
    let cfg = config::Config::load(None)?;
    let g = gemma::Gemma::load(&cfg)?;
    let conn = db::open_readonly(&config::db_path())?;
    let mut imgs = Vec::new();
    let mut st = conn.prepare("SELECT f.name, v.embedding FROM vec_image v JOIN files f ON f.id=v.rowid")?;
    for r in st.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?)))? {
        let (n, b) = r?;
        imgs.push((n, db::bytes_f32(&b)));
    }
    let mut docs = Vec::new();
    let mut st = conn.prepare(
        "SELECT f.name, g.vec FROM gemma_chunks g JOIN chunks c ON c.id=g.chunk_id JOIN files f ON f.id=c.file_id",
    )?;
    for r in st.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?)))? {
        let (n, b) = r?;
        docs.push((n, db::bytes_f32(&b)));
    }
    for q in std::env::args().skip(1) {
        let qv = g.embed_query(&q, gemma::QUERY_PREFIX)?;
        let mut si: Vec<(f32, &str)> = imgs.iter().map(|(n, v)| (gemma::dot(&qv, v), n.as_str())).collect();
        si.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut sd: Vec<(f32, &str)> = docs.iter().map(|(n, v)| (gemma::dot(&qv, v), n.as_str())).collect();
        sd.sort_by(|a, b| b.0.total_cmp(&a.0));
        sd.dedup_by(|a, b| a.1 == b.1);
        let stats = |v: &Vec<(f32, &str)>| {
            let n = v.len() as f32;
            let m = v.iter().map(|x| x.0).sum::<f32>() / n;
            let sd = (v.iter().map(|x| (x.0 - m).powi(2)).sum::<f32>() / n).sqrt();
            (m, sd)
        };
        let (mi, si_sd) = stats(&si);
        let (md, sd_sd) = stats(&sd);
        println!("\n== {q}   images mean {mi:.3} sd {si_sd:.3} | docs mean {md:.3} sd {sd_sd:.3}");
        for (s, n) in si.iter().take(4) {
            println!("  img {s:.3} z={:.1} +{:.3} {n}", (s - mi) / si_sd, s - mi);
        }
        for (s, n) in sd.iter().take(4) {
            println!("  doc {s:.3} z={:.1} +{:.3} {n}", (s - md) / sd_sd, s - md);
        }
    }
    Ok(())
}
