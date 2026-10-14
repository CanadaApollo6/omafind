//! Gemma text throughput: `cargo run --release --example bench_text -- q4 256`
fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut cfg = omafind::config::Config::default();
    cfg.models.precision = args.get(1).cloned().unwrap_or("q4".into());
    cfg.models.device = args.get(3).cloned().unwrap_or("cpu".into());
    let max_tokens: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(256);
    let g = omafind::embed::gemma::Gemma::load(&cfg)?;
    let chunk = "The landlord said the plumber would come Tuesday but nobody showed up again, so we \
                 have had cold showers for the whole week and I am keeping a log of every call. "
        .repeat(5);
    let texts: Vec<String> = (0..48).map(|i| format!("title: note{i} | text: {chunk}")).collect();
    g.embed_texts(&texts[..4], max_tokens)?;
    for n in [1, 16, 48] {
        let t = std::time::Instant::now();
        g.embed_texts(&texts[..n], max_tokens)?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        println!(
            "{} {} tok: {n:>2} chunks in {ms:>7.1} ms ({:.1} ms/chunk)",
            cfg.models.precision,
            max_tokens,
            ms / n as f64
        );
    }
    let t = std::time::Instant::now();
    g.embed_query("cold showers landlord", omafind::embed::gemma::QUERY_PREFIX)?;
    println!("query: {:.1} ms", t.elapsed().as_secs_f64() * 1000.0);
    Ok(())
}
