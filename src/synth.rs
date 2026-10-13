//! CW signal synthesizer used for testing and benchmarking the decoder under
//! realistic impairments: noise, timing jitter of a hand key ("fist"), odd
//! dash/dot weighting, Farnsworth spacing, speed drift, QSB fading, impulse
//! noise (QRN) and nearby interfering signals.

use crate::morse::pattern_for;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

#[derive(Clone, Debug)]
pub struct SignalSpec {
    pub text: String,
    pub freq: f32,
    pub wpm: f32,
    /// Signal-to-noise ratio in a 2500 Hz bandwidth (dB).
    pub snr_db: f32,
    /// Log-normal sigma applied to every element/gap length.
    pub jitter: f32,
    /// Dash length in dots (3.0 nominal).
    pub dash_ratio: f32,
    /// Mark weighting: >1 lengthens marks and shortens spaces.
    pub weight: f32,
    /// Farnsworth overall speed (stretches character/word gaps).
    pub farnsworth_wpm: Option<f32>,
    /// Relative speed change from start to end of the message.
    pub drift: f32,
    /// Fading depth 0..1 and rate in Hz.
    pub qsb_depth: f32,
    pub qsb_hz: f32,
    /// Start time in seconds.
    pub start: f32,
    /// Rise/fall time (ms).
    pub rise_ms: f32,
}

impl Default for SignalSpec {
    fn default() -> Self {
        Self {
            text: "CQ CQ DE W1AW K".into(),
            freq: 700.0,
            wpm: 20.0,
            snr_db: 10.0,
            jitter: 0.0,
            dash_ratio: 3.0,
            weight: 1.0,
            farnsworth_wpm: None,
            drift: 0.0,
            qsb_depth: 0.0,
            qsb_hz: 0.2,
            start: 0.5,
            rise_ms: 5.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Scene {
    pub rate: u32,
    pub signals: Vec<SignalSpec>,
    /// Impulse noise events per second.
    pub qrn_rate: f32,
    /// Impulse amplitude relative to noise sigma.
    pub qrn_amp: f32,
    pub tail: f32,
    pub seed: u64,
    /// Noise sigma per sample. If `None`, chosen from the first signal's SNR
    /// with signal amplitude 0.1.
    pub noise_sigma: Option<f32>,
}

impl Default for Scene {
    fn default() -> Self {
        Self { rate: 8000, signals: vec![], qrn_rate: 0.0, qrn_amp: 20.0, tail: 1.0, seed: 1, noise_sigma: None }
    }
}

/// Builds the keying timeline (on, seconds) for a signal.
pub fn timeline(spec: &SignalSpec, rng: &mut StdRng) -> Vec<(bool, f32)> {
    let text = crate::morse::from_display(&spec.text);
    let words: Vec<&str> = text.split_whitespace().collect();
    let total_chars: usize = spec.text.len().max(1);
    let jit = Normal::new(0.0f32, spec.jitter.max(1e-6)).unwrap();
    let mut out = Vec::new();
    let mut nchar = 0usize;
    let j = |rng: &mut StdRng| if spec.jitter > 0.0 { jit.sample(rng).exp() } else { 1.0 };
    for (wi, w) in words.iter().enumerate() {
        for (ci, c) in w.bytes().enumerate() {
            let progress = nchar as f32 / total_chars as f32;
            let wpm = spec.wpm * (1.0 + spec.drift * progress);
            let dot = 1.2 / wpm;
            // Farnsworth: gaps computed for the slower overall speed.
            let gap_unit = match spec.farnsworth_wpm {
                Some(f) if f < wpm => {
                    let ta = (60.0 * wpm - 37.2 * f) / (f * wpm);
                    ta / 19.0
                }
                _ => dot,
            };
            let wshift = (spec.weight - 1.0) * dot * 0.5;
            if ci > 0 {
                out.push((false, (3.0 * gap_unit - wshift) * j(rng)));
            } else if wi > 0 {
                out.push((false, (7.0 * gap_unit - wshift) * j(rng)));
            }
            let Some(pat) = pattern_for(c) else { continue };
            for (ei, e) in pat.bytes().enumerate() {
                if ei > 0 {
                    out.push((false, (dot - wshift).max(dot * 0.2) * j(rng)));
                }
                let len = if e == b'-' { spec.dash_ratio * dot } else { dot };
                out.push((true, (len + wshift) * j(rng)));
            }
            nchar += 1;
        }
        nchar += 1;
    }
    out
}

/// Renders a scene to samples. Returns (samples, duration seconds).
pub fn render(scene: &Scene) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(scene.seed);
    let fs = scene.rate as f32;
    let amp = 0.1f32;
    let sigma = scene.noise_sigma.unwrap_or_else(|| {
        let snr = scene.signals.first().map(|s| s.snr_db).unwrap_or(20.0);
        noise_sigma_for(amp, snr, fs)
    });
    let tls: Vec<Vec<(bool, f32)>> = scene.signals.iter().map(|s| timeline(s, &mut rng)).collect();
    let mut dur = 0.0f32;
    for (s, tl) in scene.signals.iter().zip(&tls) {
        dur = dur.max(s.start + tl.iter().map(|x| x.1).sum::<f32>());
    }
    dur += scene.tail;
    let n = (dur * fs) as usize;
    let mut out = vec![0.0f32; n];
    for (s, tl) in scene.signals.iter().zip(&tls) {
        // Each signal's amplitude follows its own SNR relative to the noise.
        let a = amp_for(sigma, s.snr_db, fs);
        // Build a smooth keying envelope.
        let mut env = vec![0.0f32; n];
        let mut t = s.start;
        for &(on, d) in tl {
            if on {
                let i0 = (t * fs) as usize;
                let i1 = ((t + d) * fs) as usize;
                for e in env.iter_mut().take(i1.min(n)).skip(i0) {
                    *e = 1.0;
                }
            }
            t += d;
        }
        // Raised-cosine edges by smoothing with a Hann kernel.
        let k = ((s.rise_ms / 1000.0) * fs).max(1.0) as usize;
        let kern: Vec<f32> = (0..k).map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * (i as f32 + 0.5) / k as f32).cos()).collect();
        let ks: f32 = kern.iter().sum();
        let mut acc = vec![0.0f32; n];
        for i in 0..n {
            if env[i] == 0.0 {
                continue;
            }
            for (j, kv) in kern.iter().enumerate() {
                if i + j < n {
                    acc[i + j] += kv / ks;
                }
            }
        }
        let phase0: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
        let qsb_phase: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
        for i in 0..n {
            if acc[i] == 0.0 {
                continue;
            }
            let tt = i as f32 / fs;
            let fade = if s.qsb_depth > 0.0 {
                1.0 - s.qsb_depth * (0.5 + 0.5 * (std::f32::consts::TAU * s.qsb_hz * tt + qsb_phase).sin())
            } else {
                1.0
            };
            out[i] += a * fade * acc[i] * (std::f32::consts::TAU * s.freq * tt + phase0).sin();
        }
    }
    let normal = Normal::new(0.0f32, sigma).unwrap();
    for v in out.iter_mut() {
        *v += normal.sample(&mut rng);
    }
    // Impulse noise: short decaying bursts.
    if scene.qrn_rate > 0.0 {
        let count = (scene.qrn_rate * dur) as usize;
        for _ in 0..count {
            let i0 = rng.gen_range(0..n);
            let len = rng.gen_range((0.002 * fs) as usize..(0.03 * fs) as usize);
            let a = scene.qrn_amp * sigma * rng.gen_range(0.3..1.0);
            for j in 0..len {
                if i0 + j < n {
                    out[i0 + j] += a * (-(j as f32) / (len as f32 / 3.0)).exp() * normal.sample(&mut rng) / sigma;
                }
            }
        }
    }
    out
}

/// Noise sigma per sample giving `snr_db` in 2500 Hz for a sine of `amp`.
pub fn noise_sigma_for(amp: f32, snr_db: f32, fs: f32) -> f32 {
    let ps = amp * amp / 2.0;
    let pn_2500 = ps / 10f32.powf(snr_db / 10.0);
    (pn_2500 * (fs / 2.0) / 2500.0).sqrt()
}

fn amp_for(sigma: f32, snr_db: f32, fs: f32) -> f32 {
    let pn_2500 = sigma * sigma * 2500.0 / (fs / 2.0);
    (2.0 * pn_2500 * 10f32.powf(snr_db / 10.0)).sqrt()
}

/// Character error rate via Levenshtein distance, ignoring repeated spaces.
pub fn cer(reference: &str, hyp: &str) -> f32 {
    let norm = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ");
    let r: Vec<u8> = norm(reference).into_bytes();
    let h: Vec<u8> = norm(hyp).into_bytes();
    if r.is_empty() {
        return if h.is_empty() { 0.0 } else { 1.0 };
    }
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    let mut cur = vec![0; h.len() + 1];
    for i in 1..=r.len() {
        cur[0] = i;
        for j in 1..=h.len() {
            let sub = prev[j - 1] + usize::from(r[i - 1] != h[j - 1]);
            cur[j] = sub.min(prev[j] + 1).min(cur[j - 1] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[h.len()] as f32 / r.len() as f32
}
