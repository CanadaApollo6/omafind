//! Log-mel features for the EmbeddingGemma 2 audio encoder, a port of transformers'
//! `Gemma4AudioFeatureExtractor` with the model's settings: 16 kHz, 20 ms Hann frames every
//! 10 ms, a 512-point FFT magnitude, 128 HTK mel bands from 0 to 8 kHz and `ln(x + 0.001)`.

use std::f64::consts::PI;
use std::sync::OnceLock;

pub const MELS: usize = 128;
const FRAME: usize = 320;
const HOP: usize = 160;
const FFT: usize = 512;
const BINS: usize = FFT / 2 + 1;
const MEL_FLOOR: f32 = 0.001;
/// The extractor pads the waveform to a multiple of this many samples.
const PAD_MULTIPLE: usize = 128;

pub struct Features {
    /// [frames, MELS], zero where the mask is false.
    pub mel: Vec<f32>,
    pub mask: Vec<bool>,
}

impl Features {
    pub fn frames(&self) -> usize {
        self.mask.len()
    }

    /// Soft tokens the encoder produces: two stride-2 convolutions over the frame mask.
    pub fn tokens(&self) -> usize {
        let mut mask = self.mask.clone();
        for _ in 0..2 {
            let t_out = (mask.len() + 2 - 3) / 2 + 1;
            mask = mask.into_iter().step_by(2).take(t_out).collect();
        }
        mask.iter().filter(|m| **m).count()
    }
}

pub fn features(samples: &[f32]) -> Features {
    let padded_len = samples.len().div_ceil(PAD_MULTIPLE) * PAD_MULTIPLE;
    // Semicausal padding: FRAME/2 zeros in front, so the first frame is centred on t=0.
    let pad_left = FRAME / 2;
    let mut wave = vec![0f32; pad_left + padded_len];
    wave[pad_left..pad_left + samples.len()].copy_from_slice(samples);
    let real_end = pad_left + samples.len();
    let unfold = FRAME + 1;
    let frames = if wave.len() >= unfold { (wave.len() - unfold) / HOP + 1 } else { 0 };

    let window = hann();
    let filters = mel_filters();
    let mut mel = vec![0f32; frames * MELS];
    let mut mask = vec![false; frames];
    let mut re = vec![0f64; FFT];
    let mut im = vec![0f64; FFT];
    let mut mag = vec![0f64; BINS];
    for f in 0..frames {
        let start = f * HOP;
        // A frame counts only if every sample of its window is real audio.
        let end = start + unfold - 1;
        mask[f] = end >= pad_left && end < real_end;
        if !mask[f] {
            continue;
        }
        re.iter_mut().for_each(|x| *x = 0.0);
        im.iter_mut().for_each(|x| *x = 0.0);
        for i in 0..FRAME {
            re[i] = wave[start + i] as f64 * window[i];
        }
        fft(&mut re, &mut im);
        for (k, m) in mag.iter_mut().enumerate() {
            *m = (re[k] * re[k] + im[k] * im[k]).sqrt();
        }
        let row = &mut mel[f * MELS..(f + 1) * MELS];
        for (m, out) in row.iter_mut().enumerate() {
            let mut s = 0f64;
            for (k, &v) in mag.iter().enumerate() {
                s += v * filters[k * MELS + m];
            }
            *out = (s as f32 + MEL_FLOOR).ln();
        }
    }
    Features { mel, mask }
}

/// Periodic Hann window.
fn hann() -> &'static [f64] {
    static W: OnceLock<Vec<f64>> = OnceLock::new();
    W.get_or_init(|| (0..FRAME).map(|n| 0.5 - 0.5 * (2.0 * PI * n as f64 / FRAME as f64).cos()).collect())
}

/// [BINS, MELS] triangular HTK-scale filters without normalisation (transformers'
/// `mel_filter_bank(norm=None, mel_scale="htk")`).
fn mel_filters() -> &'static [f64] {
    static F: OnceLock<Vec<f64>> = OnceLock::new();
    F.get_or_init(|| {
        let hz_to_mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
        let mel_to_hz = |m: f64| 700.0 * (10f64.powf(m / 2595.0) - 1.0);
        let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(8000.0));
        let edges: Vec<f64> = (0..MELS + 2).map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (MELS + 1) as f64)).collect();
        let mut out = vec![0f64; BINS * MELS];
        for k in 0..BINS {
            let freq = 8000.0 * k as f64 / (BINS - 1) as f64;
            for m in 0..MELS {
                let down = (freq - edges[m]) / (edges[m + 1] - edges[m]);
                let up = (edges[m + 2] - freq) / (edges[m + 2] - edges[m + 1]);
                out[k * MELS + m] = down.min(up).max(0.0);
            }
        }
        out
    })
}

/// In-place iterative radix-2 FFT; `re.len()` must be a power of two.
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * PI / len as f64;
        let (wr, wi) = (ang.cos(), ang.sin());
        for start in (0..n).step_by(len) {
            let (mut cr, mut ci) = (1.0, 0.0);
            for k in 0..len / 2 {
                let (a, b) = (start + k, start + k + len / 2);
                let tr = re[b] * cr - im[b] * ci;
                let ti = re[b] * ci + im[b] * cr;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                let next = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = next;
            }
        }
        len <<= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_matches_dft() {
        let x: Vec<f64> = (0..16).map(|i| ((i * 7 % 5) as f64) - 2.0).collect();
        let (mut re, mut im) = (x.clone(), vec![0.0; 16]);
        fft(&mut re, &mut im);
        for k in 0..16 {
            let (mut r, mut i) = (0.0, 0.0);
            for (n, v) in x.iter().enumerate() {
                let a = -2.0 * PI * (k * n) as f64 / 16.0;
                r += v * a.cos();
                i += v * a.sin();
            }
            assert!((re[k] - r).abs() < 1e-9 && (im[k] - i).abs() < 1e-9);
        }
    }

    #[test]
    fn frame_and_token_counts_match_the_processor() {
        // One second: transformers gives 100 valid frames of 101 and 25 tokens.
        let f = features(&vec![0.1; 16_000]);
        assert_eq!(f.frames(), (16_000 + 160 - 321) / 160 + 1);
        assert_eq!(f.mask.iter().filter(|m| **m).count(), 99);
        assert_eq!(f.tokens(), 25);
        // A 1 kHz tone peaks in the mel band around 1 kHz.
        let tone: Vec<f32> = (0..16_000).map(|i| (2.0 * PI * 1000.0 * i as f64 / 16_000.0).sin() as f32).collect();
        let f = features(&tone);
        let row = &f.mel[50 * MELS..51 * MELS];
        let peak = row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        assert!((40..=50).contains(&peak), "peak band {peak}");
    }
}
