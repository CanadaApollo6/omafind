// csvsum: sum one numeric column of a csv file.
// usage: csvsum <file> <column-name-or-index>
// Handles a header row and skips blank or non-numeric cells with a warning.
use std::env;
use std::fs;
use std::process;

fn find_column(header: &str, wanted: &str) -> Option<usize> {
    // accept either a 0-based index or the header name
    if let Ok(idx) = wanted.parse::<usize>() {
        return Some(idx);
    }
    header
        .split(',')
        .position(|h| h.trim().eq_ignore_ascii_case(wanted))
}

fn sum_column(text: &str, wanted: &str) -> Result<(f64, usize), String> {
    let mut lines = text.lines();
    let header = lines.next().ok_or("file is empty")?;
    let col = find_column(header, wanted)
        .ok_or_else(|| format!("no column called '{}'", wanted))?;

    let mut total = 0.0;
    let mut rows = 0;
    for (lineno, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let cell = match line.split(',').nth(col) {
            Some(c) => c.trim(),
            None => {
                eprintln!("line {}: too few fields", lineno + 2);
                continue;
            }
        };
        match cell.parse::<f64>() {
            Ok(v) => {
                total += v;
                rows += 1;
            }
            Err(_) => eprintln!("line {}: '{}' is not a number", lineno + 2, cell),
        }
    }
    Ok((total, rows))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: csvsum <file> <column>");
        process::exit(2);
    }
    let text = match fs::read_to_string(&args[1]) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("could not read {}: {}", args[1], e);
            process::exit(1);
        }
    };
    match sum_column(&text, &args[2]) {
        Ok((total, rows)) => println!("sum={:.2} rows={}", total, rows),
        Err(msg) => {
            eprintln!("error: {}", msg);
            process::exit(1);
        }
    }
}
