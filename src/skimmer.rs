//! Multi-channel skimmer: finds every keyed CW signal in the audio passband
//! and runs an independent decoder on each one.

use crate::callsign::parse_callsign;
use crate::decoder::{CwDecoder, DecoderParams};
use crate::dsp::channel::ChannelDemod;
use crate::dsp::spectrum::Spectrum;
use crate::dsp::Lowpass4;
use crate::lm::LanguageModel;
use num_complex::Complex32;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

/// Internal processing sample rate.
pub const RATE: u32 = 8000;
/// Input samples per envelope frame (2 ms).
pub const FRAME: usize = 16;
pub const FFT_SIZE: usize = 512;
pub const FFT_HOP: usize = 128;

#[derive(Clone, Debug)]
pub struct SkimmerConfig {
    pub min_freq: f32,
    pub max_freq: f32,
    pub max_channels: usize,
    /// Detection threshold above the noise floor (dB, per FFT bin).
    pub detect_db: f32,
    /// Minimum spacing between channels (Hz).
    pub min_spacing: f32,
    /// Seconds without a mark before an automatic channel is dropped.
    pub channel_timeout: f32,
    pub params: DecoderParams,
    /// Automatically create channels for detected signals.
    pub auto_detect: bool,
}

impl Default for SkimmerConfig {
    fn default() -> Self {
        Self {
            min_freq: 250.0,
            max_freq: 2800.0,
            max_channels: 40,
            detect_db: 12.0,
            min_spacing: 30.0,
            channel_timeout: 25.0,
            params: DecoderParams::default(),
            auto_detect: true,
        }
    }
}

pub struct Channel {
    pub id: u64,
    pub demod: ChannelDemod,
    pub decoder: CwDecoder,
    pub text: String,
    pub created: f64,
    pub last_mark: f64,
    pub pinned: bool,
    pub level_db: f32,
    /// Spectral activity gate: something is transmitting at this frequency.
    pub active: bool,
    /// Recent mark/space decisions (one per frame) for artefact detection.
    marks: VecDeque<bool>,
    /// Text not yet published (held until the channel qualifies).
    pending_out: String,
    /// The decoded text looks like real Morse (see [`text_quality`]).
    qualified: bool,
    /// Last time the keying fitted Morse timing (noise keeps `last_mark`
    /// fresh, so lifetime is based on this instead).
    last_good: f64,
    words: String,
    /// Language-model plausibility of the recent text (see
    /// [`text_plausibility`]), once there is enough of it.
    plaus: Option<f32>,
    call_counts: HashMap<String, u32>,
    prev_word: String,
    frames: Vec<Complex32>,
    /// Speed-matched filter (I, Q, bandwidth) applied after the fixed
    /// 100 Hz demodulator, so the unfiltered frames stay available.
    post: Option<(Lowpass4, Lowpass4, f32)>,
    /// Baseband frames (before the speed-matched filter) since the channel
    /// was created, kept until it is shown: whenever the measured speed
    /// changes, the start is decoded again with it (see `redecode_start`).
    start_frames: Option<Vec<Complex32>>,
    /// Dot length the start was last decoded with, and how many times.
    start_dot: Option<f32>,
    start_redecodes: u8,
    /// Length of `text` when the channel was created (revived text).
    start_text_len: usize,
}

#[derive(Clone, Debug)]
pub struct ChannelInfo {
    pub id: u64,
    pub freq: f32,
    pub wpm: Option<f32>,
    pub snr_db: f32,
    pub text: String,
    pub tentative: String,
    pub is_mark: bool,
    pub pinned: bool,
    pub last_mark_age: f32,
    pub callsign: Option<String>,
    /// Timing-fit score (0..1): how well the keying matches Morse timing.
    pub fit: f32,
    /// The text reads poorly (likely weak or garbled): shown dimmed.
    pub doubtful: bool,
}

#[derive(Clone, Debug)]
pub struct Spot {
    pub call: String,
    pub freq: f32,
    pub wpm: f32,
    pub snr_db: f32,
    pub time: f64,
    pub channel: u64,
    pub count: u32,
    /// Heard at least twice on its channel (a single sighting, even after
    /// DE, may be a decoding error).
    pub confirmed: bool,
}

#[derive(Clone, Debug)]
pub struct TextEvent {
    pub channel: u64,
    pub freq: f32,
    pub text: String,
}

pub struct Skimmer {
    pub cfg: SkimmerConfig,
    lm: Arc<LanguageModel>,
    spectrum: Spectrum,
    fast: Vec<f32>,
    peak: Vec<f32>,
    trough: Vec<f32>,
    /// Short peak-hold (decays ~8 dB/s) for the per-channel activity gate.
    recent: Vec<f32>,
    persist: Vec<u16>,
    pub noise_floor: f32,
    pub channels: Vec<Channel>,
    next_id: u64,
    /// Seconds of audio processed.
    pub time: f64,
    wf_acc: Vec<f32>,
    wf_n: u32,
    pub waterfall: VecDeque<Vec<f32>>,
    pub events: Vec<TextEvent>,
    pub spots: Vec<Spot>,
    /// Recent audio, replayed into newly created channels so the start of a
    /// transmission (before detection triggered) is not lost.
    preroll: VecDeque<f32>,
    /// Frequencies where artefacts of strong signals were found: (Hz, until).
    suppressed: Vec<(f32, f64)>,
    last_artifact_check: f64,
    graveyard: Vec<Retired>,
    hops: u64,
    /// The latest spectrum frame was a broadband burst (static crash).
    burst: bool,
    burst_hops: u32,
}

const PREROLL_SECS: f32 = 1.5;
/// Frames of mark history kept per channel (5 s).
const MARK_HIST: usize = 2500;

pub fn bin_hz() -> f32 {
    RATE as f32 / FFT_SIZE as f32
}

/// Minimum timing-fit score for a channel to be shown (real CW measured
/// 0.8-0.97 on live recordings; noise 0.35-0.68).
const MIN_FIT: f32 = 0.78;

/// Minimum mean Morse elements per character for a channel to be shown.
/// Noise and artefacts decode as short characters (E T I A N M S, ~1.6
/// elements); real text averages ~3.
const MIN_QUALITY: f32 = 1.8;

/// Minimum fraction of dashes among decoded elements. Noise bursts are
/// short, so noise decodes almost entirely as dots (E I S H 5); real text
/// is ~35-45 % dashes.
const MIN_DASH_FRAC: f32 = 0.22;

/// Below this plausibility a channel's text is junk and the channel is
/// hidden; below `DOUBTFUL_PLAUS` it is shown dimmed. Measured on live
/// logs: clean QSO text -1 to -2.2, garbled but real -2.4 to -3.1, noise
/// decoded as text -2.6 to -3.6 (the bands overlap, so only clear junk is
/// hidden). Code groups (random characters, as in practice traffic) are
/// implausible too but made of long characters, while noise decodes as
/// short ones; only text below `JUNK_MAX_ELEMS` elements per character is
/// hidden for implausibility.
const MIN_PLAUS: f32 = -3.15;
const JUNK_MAX_ELEMS: f32 = 3.3;
const DOUBTFUL_PLAUS: f32 = -2.45;
/// Text made of very short characters (E I S T…, under this many elements
/// per character; real text averages 2.5-3.3) that also reads poorly is
/// what noise and other stations' clicks decode as: hidden too. Measured
/// on 20 m recordings: junk -2.6 to -3.1 with 1.9-2.3 elements per character.
const SHORT_JUNK_ELEMS: f32 = 2.3;
const SHORT_JUNK_PLAUS: f32 = -2.55;

/// Mean language-model log-probability per character of the last ~100
/// characters, ignoring callsigns (random letter strings by nature). None
/// until there are 30 characters to judge.
pub fn text_plausibility(lm: &LanguageModel, text: &str) -> Option<f32> {
    let start = text.len().saturating_sub(100);
    let tail = text.get(start..).unwrap_or(text);
    // The first word may be cut.
    let mut words = tail.split_whitespace();
    if start > 0 {
        words.next();
    }
    let kept: Vec<&str> = words.filter(|w| parse_callsign(w).is_none()).collect();
    let k = kept.join(" ");
    (k.len() >= 30).then(|| lm.score(&format!(" {k} ")))
}

/// Splits a callsign sent twice with the word gap squeezed out
/// ("K5ZDK5ZD" -> "K5ZD K5ZD"); calls are routinely sent twice.
fn split_doubled_calls(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 4);
    for (i, w) in text.split(' ').enumerate() {
        if i > 0 {
            out.push(' ');
        }
        let h = w.len() / 2;
        if w.len() >= 6 && w.len() % 2 == 0 && w.is_char_boundary(h) && w[..h] == w[h..] && parse_callsign(&w[..h]).is_some() {
            out.push_str(&w[..h]);
            out.push(' ');
            out.push_str(&w[h..]);
        } else {
            out.push_str(w);
        }
    }
    out
}

/// Replaces near-miss decodes of a callsign confirmed on this channel
/// (a dit lost or gained in a fade) with the call itself.
fn snap_calls(text: &str, calls: &HashMap<String, u32>) -> String {
    let confirmed: Vec<&String> = calls.iter().filter(|(_, &n)| n >= 2).map(|(c, _)| c).collect();
    if confirmed.is_empty() {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    for (i, w) in text.split(' ').enumerate() {
        if i > 0 {
            out.push(' ');
        }
        let near: Vec<&&String> = if w.len() >= 3 && !confirmed.iter().any(|c| c.as_str() == w) {
            confirmed.iter().filter(|c| crate::callsign::morse_distance(c, w) == 1).collect()
        } else {
            Vec::new()
        };
        match near.as_slice() {
            [c] => out.push_str(c),
            _ => out.push_str(w),
        }
    }
    out
}

/// Text statistics over the last 40 characters (ignoring spaces): mean Morse
/// elements per character, fraction of dashes, and characters considered.
pub fn text_quality(text: &str) -> (f32, f32, usize) {
    let mut n = 0usize;
    let mut tot = 0usize;
    let mut dashes = 0usize;
    for b in text.bytes().rev() {
        if b == b' ' {
            continue;
        }
        if let Some(p) = crate::morse::pattern_for(b) {
            tot += p.len();
            dashes += p.bytes().filter(|&c| c == b'-').count();
            n += 1;
        }
        if n >= 40 {
            break;
        }
    }
    if n == 0 {
        return (0.0, 0.0, 0);
    }
    (tot as f32 / n as f32, dashes as f32 / tot as f32, n)
}

/// A recently timed-out channel, kept so a station that pauses and comes
/// back on the same frequency continues in the same row.
struct Retired {
    id: u64,
    freq: f32,
    at: f64,
    text: String,
    call_counts: HashMap<String, u32>,
}

/// Maximum |Pearson correlation| between two mark histories over lags of
/// ±100 ms, computed only where both channels are keying (so two stations
/// taking turns are not mistaken for an echo).
fn lagged_corr(a: &VecDeque<bool>, b: &VecDeque<bool>) -> f32 {
    let n = a.len().min(b.len());
    let (oa, ob) = (a.len() - n, b.len() - n);
    let x: Vec<bool> = a.iter().skip(oa).copied().collect();
    let y: Vec<bool> = b.iter().skip(ob).copied().collect();
    // Activity: a mark within ±0.6 s.
    let act = |v: &[bool]| -> Vec<bool> {
        let mut pre = vec![0u32; v.len() + 1];
        for (i, &m) in v.iter().enumerate() {
            pre[i + 1] = pre[i] + u32::from(m);
        }
        (0..v.len()).map(|i| pre[(i + 301).min(v.len())] > pre[i.saturating_sub(300)]).collect()
    };
    let (ax, ay) = (act(&x), act(&y));
    let mut best = 0.0f32;
    for lag in (-50i32..=50).step_by(5) {
        let (mut sx, mut sy, mut sxy, mut cnt) = (0f64, 0f64, 0f64, 0f64);
        for i in 0..n {
            let j = i as i32 + lag;
            if j < 0 || j as usize >= n {
                continue;
            }
            let j = j as usize;
            if !(ax[i] && ay[j]) {
                continue;
            }
            let (xv, yv) = (f64::from(u8::from(x[i])), f64::from(u8::from(y[j])));
            sx += xv;
            sy += yv;
            sxy += xv * yv;
            cnt += 1.0;
        }
        if cnt < 400.0 {
            continue;
        }
        let (mx, my) = (sx / cnt, sy / cnt);
        let cov = sxy / cnt - mx * my;
        let var = (mx * (1.0 - mx)) * (my * (1.0 - my));
        if var > 1e-9 {
            best = best.max((cov / var.sqrt()).abs() as f32);
        }
    }
    best
}

impl Skimmer {
    pub fn new(cfg: SkimmerConfig, lm: Arc<LanguageModel>) -> Self {
        let nb = FFT_SIZE / 2;
        Self {
            cfg,
            lm,
            spectrum: Spectrum::new(FFT_SIZE, FFT_HOP),
            fast: vec![0.0; nb],
            peak: vec![0.0; nb],
            trough: vec![0.0; nb],
            recent: vec![0.0; nb],
            persist: vec![0; nb],
            noise_floor: 1e-9,
            channels: Vec::new(),
            next_id: 1,
            time: 0.0,
            wf_acc: vec![0.0; nb],
            wf_n: 0,
            waterfall: VecDeque::new(),
            events: Vec::new(),
            spots: Vec::new(),
            preroll: VecDeque::new(),
            suppressed: Vec::new(),
            last_artifact_check: 0.0,
            graveyard: Vec::new(),
            hops: 0,
            burst: false,
            burst_hops: 0,
        }
    }

    /// Adds a decoder at a fixed frequency (e.g. user click). Returns its id.
    pub fn add_channel(&mut self, freq: f32, pinned: bool) -> u64 {
        if let Some(c) = self.channels.iter_mut().find(|c| (c.demod.freq as f32 - freq).abs() < self.cfg.min_spacing * 0.5) {
            c.pinned |= pinned;
            return c.id;
        }
        let revived = self
            .graveyard
            .iter()
            .position(|r| (r.freq - freq).abs() < 40.0)
            .map(|i| self.graveyard.remove(i));
        let was_revived = revived.is_some();
        let id = match &revived {
            Some(r) => r.id,
            None => {
                self.next_id += 1;
                self.next_id - 1
            }
        };
        let (rtext, rcalls) = match revived {
            Some(r) => (r.text + " ", r.call_counts),
            None => (String::new(), HashMap::new()),
        };
        let mut demod = ChannelDemod::new(RATE as f32, freq as f64, 100.0, FRAME);
        let mut decoder = CwDecoder::new(demod.frame_ms(), self.cfg.params.clone(), self.lm.clone());
        // Warm up on the pre-roll audio.
        let (a, b) = self.preroll.as_slices();
        let mut frames = Vec::new();
        demod.process(a, &mut frames);
        demod.process(b, &mut frames);
        let amps: Vec<f32> = frames.iter().map(|z| z.norm()).collect();
        decoder.warmup(&amps);
        let start_frames = Some(frames);
        self.channels.push(Channel {
            id,
            demod,
            decoder,
            text: rtext.clone(),
            created: self.time,
            last_mark: self.time,
            pinned,
            level_db: -200.0,
            active: true,
            marks: VecDeque::with_capacity(MARK_HIST),
            pending_out: String::new(),
            qualified: pinned || was_revived,
            last_good: self.time,
            words: String::new(),
            post: None,
            start_dot: None,
            start_redecodes: 0,
            start_text_len: rtext.len(),
            plaus: None,
            call_counts: rcalls,
            prev_word: String::new(),
            frames: Vec::new(),
            start_frames,
        });
        id
    }

    pub fn remove_channel(&mut self, id: u64) {
        self.channels.retain(|c| c.id != id);
    }

    /// Precisely measures the frequency of the strongest tone within
    /// `near ± span` Hz over the last second of audio (1 Hz Goertzel scan
    /// plus parabolic interpolation). Returns `None` if there is not yet
    /// enough audio or no clear tone. Result: (Hz, peak-to-median dB).
    pub fn measure_freq(&self, near: f32, span: f32) -> Option<(f32, f32)> {
        let need = RATE as usize; // 1 s
        if self.preroll.len() < need + RATE as usize / 5 {
            return None;
        }
        let x: Vec<f32> = self.preroll.iter().skip(self.preroll.len() - need).copied().collect();
        let n = x.len();
        let win: Vec<f32> = (0..n).map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / n as f32).cos()).collect();
        let power = |f: f32| -> f32 {
            let w = std::f32::consts::TAU * f / RATE as f32;
            let c = 2.0 * w.cos();
            let (mut s1, mut s2) = (0.0f32, 0.0f32);
            for i in 0..n {
                let s0 = x[i] * win[i] + c * s1 - s2;
                s2 = s1;
                s1 = s0;
            }
            s1 * s1 + s2 * s2 - c * s1 * s2
        };
        let lo = (near - span).max(50.0);
        let hi = (near + span).min(RATE as f32 / 2.0 - 50.0);
        let freqs: Vec<f32> = (0..=((hi - lo) as usize)).map(|i| lo + i as f32).collect();
        let p: Vec<f32> = freqs.iter().map(|&f| power(f)).collect();
        let (imax, &pmax) = p.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1))?;
        let mut sorted = p.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let med = sorted[sorted.len() / 2].max(1e-20);
        let snr_db = 10.0 * (pmax / med).log10();
        if snr_db < 6.0 || imax == 0 || imax + 1 >= p.len() {
            return None;
        }
        let (a, b, c) = (p[imax - 1].max(1e-20).ln(), pmax.ln(), p[imax + 1].max(1e-20).ln());
        let den = a - 2.0 * b + c;
        let off = if den.abs() > 1e-9 { (0.5 * (a - c) / den).clamp(-0.5, 0.5) } else { 0.0 };
        Some((freqs[imax] + off, snr_db))
    }

    /// The radio was retuned: every signal moved by `df` Hz in the audio
    /// passband. Moves the decoders with them (keeping their text) and drops
    /// those that left the passband.
    pub fn shift(&mut self, df: f32) {
        let (lo, hi) = (self.cfg.min_freq, self.cfg.max_freq);
        for c in &mut self.channels {
            let f = c.demod.freq + df as f64;
            c.demod.set_freq(f);
        }
        let (now, events, spots) = (self.time, &mut self.events, &mut self.spots);
        self.channels.retain_mut(|c| {
            let keep = (c.demod.freq as f32) > lo - 20.0 && (c.demod.freq as f32) < hi + 20.0;
            if !keep {
                Self::dropping(c, now, events, spots);
            }
            keep
        });
        // Spectral history no longer lines up.
        let nb = self.peak.len();
        self.peak = vec![0.0; nb];
        self.trough = vec![0.0; nb];
        self.persist = vec![0; nb];
        self.preroll.clear();
        self.suppressed.clear();
        for r in &mut self.graveyard {
            r.freq += df;
        }
        for c in &mut self.channels {
            c.marks.clear();
        }
    }

    pub fn clear(&mut self) {
        self.channels.clear();
        self.graveyard.clear();
        self.spots.clear();
    }

    /// Processes audio at [`RATE`].
    pub fn process(&mut self, samples: &[f32]) {
        // Work in FFT-hop sized pieces so burst erasures line up in time.
        for piece in samples.chunks(FFT_HOP) {
            self.process_piece(piece);
        }
    }

    fn process_piece(&mut self, samples: &[f32]) {
        let mut spectra: Vec<Vec<f32>> = Vec::new();
        self.spectrum.process(samples, |p| spectra.push(p.to_vec()));
        for p in &spectra {
            self.on_spectrum(p);
        }
        self.preroll.extend(samples.iter().copied());
        let max = (PREROLL_SECS * RATE as f32) as usize;
        if self.preroll.len() > max {
            let n = self.preroll.len() - max;
            self.preroll.drain(..n);
        }
        let now = self.time;
        let (params, lm) = (&self.cfg.params, &self.lm);
        for ch in &mut self.channels {
            ch.frames.clear();
            ch.demod.process(samples, &mut ch.frames);
            ch.decoder.set_erasure(self.burst && (5..60).contains(&self.burst_hops));
            for i in 0..ch.frames.len() {
                let raw = ch.frames[i];
                let z = match &mut ch.post {
                    Some((li, lq, _)) => Complex32::new(li.process(raw.re), lq.process(raw.im)),
                    None => raw,
                };
                ch.decoder.push_frame(z.norm());
                if let Some(f) = &mut ch.start_frames {
                    f.push(raw);
                    // Keep the start only until the channel is shown, and
                    // at most 20 s.
                    if ch.qualified || f.len() as f32 * ch.demod.frame_ms() > 20_000.0 {
                        ch.start_frames = None;
                    }
                }
                if let (Some(t), Some(_)) = (ch.decoder.timing(), &ch.start_frames) {
                    // A few times at most: noise channels' speed wanders.
                    if ch.start_redecodes < 3 && ch.start_dot.is_none_or(|d| !(0.8..=1.25).contains(&(t.dot / d))) {
                        Self::redecode_start(ch, params, lm);
                    }
                }
                // AFC only on clean marks so it cannot wander off onto
                // a neighbour.
                let strong = ch.decoder.snr_db() > 12.0;
                ch.demod.afc(z, ch.decoder.is_mark() && strong);
                if ch.marks.len() >= MARK_HIST {
                    ch.marks.pop_front();
                }
                ch.marks.push_back(ch.decoder.is_mark());
            }
            if ch.decoder.is_mark() {
                ch.last_mark = now;
            }
            // While the channel decodes well, let the AFC window follow the
            // signal so a drifting transmitter is tracked.
            // Slowly (over ~3 s), so AFC measurement noise cannot random-walk
            // the channel away.
            if ch.decoder.fit().0 >= MIN_FIT {
                let (lo, hi) = ch.demod.afc_limit;
                let c = 0.5 * (lo + hi);
                let c = c + 0.005 * (ch.demod.freq - c);
                ch.demod.afc_limit = (c - 12.0, c + 12.0);
            }
            if let Some(t) = ch.decoder.timing() {
                // Match the filter to the keying speed.
                let bw = (ch.decoder.params().bw_factor * 1000.0 / t.dot).clamp(30.0, 200.0);
                let fs = 1000.0 / ch.demod.frame_ms();
                match &mut ch.post {
                    Some((li, lq, b)) if (bw - *b).abs() > 1.0 => {
                        li.retune(fs, bw / 2.0);
                        lq.retune(fs, bw / 2.0);
                        *b = bw;
                    }
                    Some(_) => {}
                    None => ch.post = Some((Lowpass4::new(fs, bw / 2.0), Lowpass4::new(fs, bw / 2.0), bw)),
                }
            }
            let txt = ch.decoder.take_text();
            if !txt.is_empty() {
                Self::deliver(ch, &txt, now, false, &self.lm, &mut self.events, &mut self.spots);
            }
        }
        self.time += samples.len() as f64 / RATE as f64;
        self.housekeeping();
    }

    /// Flushes all decoders (end of input).
    pub fn finish(&mut self) {
        for ch in &mut self.channels {
            ch.decoder.finish();
            let txt = ch.decoder.take_text();
            Self::deliver(ch, &txt, self.time, true, &self.lm, &mut self.events, &mut self.spots);
        }
    }

    /// The decoder has measured the speed (for the first time, or found it
    /// different): decode the channel's start again through the filter
    /// matched to it. Until the speed was known the start was filtered wide
    /// (100 Hz), so the first characters of every transmission were copied
    /// through several dB more noise, and an early speed estimate from that
    /// noisy start can be well off. Nothing of the channel has been shown
    /// yet, so its text so far is simply replaced.
    fn redecode_start(ch: &mut Channel, params: &DecoderParams, lm: &Arc<LanguageModel>) {
        let (Some(t), Some(frames)) = (ch.decoder.timing().cloned(), &ch.start_frames) else { return };
        let bw = (params.bw_factor * 1000.0 / t.dot).clamp(30.0, 200.0);
        let fs = 1000.0 / ch.demod.frame_ms();
        let (mut li, mut lq) = (Lowpass4::new(fs, bw / 2.0), Lowpass4::new(fs, bw / 2.0));
        let amps: Vec<f32> = frames.iter().map(|z| Complex32::new(li.process(z.re), lq.process(z.im)).norm()).collect();
        // Start from the timing already measured: re-measuring on the
        // narrow-filtered start can go wrong with a sloppy fist whose short
        // gaps the filter partly fills in.
        let mut d = CwDecoder::new(ch.demod.frame_ms(), params.clone(), lm.clone());
        d.seed_timing(t.clone());
        d.warmup(&amps);
        ch.decoder = d;
        // The filter carries on from the end of the replayed start.
        ch.post = Some((li, lq, bw));
        ch.start_dot = Some(t.dot);
        ch.start_redecodes += 1;
        ch.text.truncate(ch.start_text_len);
        ch.pending_out.clear();
    }

    /// Appends decoded text to a channel and publishes it once the channel
    /// looks like real Morse rather than noise or an artefact.
    fn deliver(ch: &mut Channel, txt: &str, now: f64, final_: bool, lm: &LanguageModel, events: &mut Vec<TextEvent>, spots: &mut Vec<Spot>) {
        let snapped = snap_calls(&split_doubled_calls(txt), &ch.call_counts);
        let txt = snapped.as_str();
        ch.text.push_str(txt);
        if ch.text.len() > 4000 {
            let cut = ch.text.len() - 3000;
            ch.text.drain(..cut);
        }
        ch.pending_out.push_str(txt);
        // Show a channel once its timing looks like real keying: most marks
        // and gaps land near Morse durations. Noise (including AGC-pumped
        // band noise next to a strong station) gives random durations.
        // Hysteresis: hide again if the fit collapses (station gone).
        // Once there is enough text, also require it to look like Morse
        // rather than noise, which decodes as dot-only characters (E I S H 5).
        let (fit, runs) = ch.decoder.fit();
        let (el, dash, n) = text_quality(&ch.text);
        ch.plaus = text_plausibility(lm, &ch.text);
        let text_ok = if n >= 8 {
            let junk = ch.plaus.is_some_and(|p| (p < MIN_PLAUS && el < JUNK_MAX_ELEMS) || (p < SHORT_JUNK_PLAUS && el < SHORT_JUNK_ELEMS));
            el >= MIN_QUALITY && dash >= MIN_DASH_FRAC && !junk
        } else {
            n >= 3 && dash >= 0.2
        };
        if ch.pinned {
            ch.qualified = true;
        } else if runs >= 20 || (final_ && runs >= 10) {
            if fit >= MIN_FIT && text_ok {
                ch.qualified = true;
            } else if fit < MIN_FIT - 0.13 || (n >= 8 && !text_ok) {
                if ch.qualified {
                    Self::hide(ch, now, events, spots);
                }
                ch.qualified = false;
            }
        }
        if ch.qualified {
            Self::publish(ch, now, events, spots);
        } else if ch.pending_out.len() > 200 {
            let cut = ch.pending_out.len() - 200;
            ch.pending_out.drain(..cut);
        }
    }

    fn publish(ch: &mut Channel, now: f64, events: &mut Vec<TextEvent>, spots: &mut Vec<Spot>) {
        let out = std::mem::take(&mut ch.pending_out);
        if !out.is_empty() {
            events.push(TextEvent { channel: ch.id, freq: ch.demod.freq as f32, text: out.clone() });
            Self::scan_calls(ch, &out, now, spots);
        }
    }

    /// Publishes the word the decoder was holding back (in case of an error
    /// sign) before the channel is hidden, so the station's last word is kept.
    fn hide(ch: &mut Channel, now: f64, events: &mut Vec<TextEvent>, spots: &mut Vec<Spot>) {
        let txt = ch.decoder.release_held();
        ch.text.push_str(&txt);
        ch.pending_out.push_str(&txt);
        Self::publish(ch, now, events, spots);
    }

    /// Called before a channel is removed: keeps its held-back last word.
    fn dropping(ch: &mut Channel, now: f64, events: &mut Vec<TextEvent>, spots: &mut Vec<Spot>) {
        if ch.qualified {
            Self::hide(ch, now, events, spots);
        }
    }

    fn scan_calls(ch: &mut Channel, txt: &str, now: f64, spots: &mut Vec<Spot>) {
        ch.words.push_str(txt);
        while let Some(pos) = ch.words.find(' ') {
            let w: String = ch.words[..pos].to_string();
            ch.words.drain(..=pos);
            if w.is_empty() {
                continue;
            }
            if let Some(call) = parse_callsign(&w).filter(|c| crate::callsign::valid_prefix(c)) {
                // A near miss of a call already confirmed here, or spotted
                // elsewhere, counts toward that call.
                let near = |c: &str| c != call && crate::callsign::morse_distance(c, &call) == 1;
                let call = ch
                    .call_counts
                    .iter()
                    .filter(|(c, &n)| n >= 2 && near(c))
                    .max_by_key(|(_, &n)| n)
                    .map(|(c, _)| c.clone())
                    .or_else(|| spots.iter().filter(|s| s.confirmed && near(&s.call)).max_by_key(|s| s.count).map(|s| s.call.clone()))
                    .unwrap_or(call);
                let n = ch.call_counts.entry(call.clone()).or_insert(0);
                *n += 1;
                // Twice on this channel, or already spotted earlier (a net
                // control is heard every few minutes, longer than a
                // channel remembers its calls).
                let confirmed = *n >= 2 || spots.iter().any(|s| s.call == call);
                let strong_ctx = matches!(ch.prev_word.as_str(), "DE" | "CQ" | "TEST");
                if *n >= 2 || strong_ctx {
                    let wpm = ch.decoder.wpm().unwrap_or(0.0);
                    let snr = ch.decoder.snr_db();
                    let f = ch.demod.freq as f32;
                    if let Some(s) = spots.iter_mut().find(|s| s.call == call) {
                        s.freq = f;
                        s.wpm = wpm;
                        s.snr_db = snr;
                        s.time = now;
                        s.channel = ch.id;
                        s.count += 1;
                        s.confirmed |= confirmed;
                    } else {
                        spots.push(Spot { call, freq: f, wpm, snr_db: snr, time: now, channel: ch.id, count: 1, confirmed });
                    }
                }
            }
            ch.prev_word = w;
        }
    }

    fn bin_range(&self) -> (usize, usize) {
        let hz = bin_hz();
        let lo = ((self.cfg.min_freq / hz).floor() as usize).max(3);
        let hi = ((self.cfg.max_freq / hz).ceil() as usize).min(FFT_SIZE / 2 - 3);
        (lo, hi)
    }

    fn on_spectrum(&mut self, p: &[f32]) {
        // Waterfall rows: average 3 hops (~48 ms per row).
        for (a, &v) in self.wf_acc.iter_mut().zip(p) {
            *a += v;
        }
        self.wf_n += 1;
        if self.wf_n >= 3 {
            let row: Vec<f32> = self.wf_acc.iter().map(|&v| 10.0 * (v / self.wf_n as f32 + 1e-14).log10()).collect();
            self.waterfall.push_back(row);
            while self.waterfall.len() > 400 {
                self.waterfall.pop_front();
            }
            self.wf_acc.iter_mut().for_each(|v| *v = 0.0);
            self.wf_n = 0;
        }

        let (lo, hi) = self.bin_range();
        // Broadband static crashes raise the whole spectrum at once. Treat
        // such frames as erasures instead of letting every bin look keyed.
        let mut inst: Vec<f32> = p[lo..hi].to_vec();
        inst.sort_unstable_by(|a, b| a.total_cmp(b));
        let med = inst[inst.len() / 2];
        self.hops += 1;
        let top = inst[inst.len() - 1];
        // A crash is flat: the floor comes up close to the strongest signal.
        // (Codec noise riding on a strong keyed tone stays far below it.)
        self.burst = self.hops > 30 && med > self.noise_floor * 5.0 && med * 3000.0 > top;
        if self.burst {
            self.burst_hops += 1;
            // A sustained rise is a real change of band noise, not a crash.
            if self.burst_hops < 60 {
                return;
            }
        } else {
            self.burst_hops = 0;
        }
        for i in 0..p.len() {
            self.fast[i] = 0.65 * self.fast[i] + 0.35 * p[i];
            self.peak[i] = (self.peak[i] * 0.993).max(self.fast[i]);
            self.trough[i] = (self.trough[i] * 1.02).min(self.fast[i]).max(1e-15);
            self.recent[i] = (self.recent[i] * 0.97).max(self.fast[i]);
        }
        let mut v: Vec<f32> = self.fast[lo..hi].to_vec();
        v.sort_unstable_by(|a, b| a.total_cmp(b));
        let nf = v[v.len() / 2].max(1e-15);
        self.noise_floor = 0.9 * self.noise_floor + 0.1 * nf;
        let nf = self.noise_floor;

        for ch in &mut self.channels {
            let b = ((ch.demod.freq as f32 / bin_hz()).round() as usize).min(p.len() - 1);
            ch.level_db = 10.0 * (self.peak[b] / nf).log10();
            let r = self.recent[b.saturating_sub(1)..(b + 2).min(p.len())].iter().fold(0.0f32, |m, &v| m.max(v));
            let gate = 10f32.powf((self.cfg.detect_db - 2.0) / 10.0);
            ch.active = r > nf * gate;
            ch.decoder.set_gate(ch.active);
        }

        if !self.cfg.auto_detect {
            return;
        }
        let det = 10f32.powf(self.cfg.detect_db / 10.0);
        let hz = bin_hz();
        for b in lo..hi {
            let pk = self.peak[b];
            let is_max = (b - 2..=b + 2).all(|j| j == b || self.peak[j] < pk);
            let keyed = pk / self.trough[b].max(nf * 0.3) > 5.0;
            if is_max && pk > nf * det && keyed {
                self.persist[b] = self.persist[b].saturating_add(1);
            } else {
                self.persist[b] = 0;
            }
            if self.persist[b] < 6 {
                continue;
            }
            // Parabolic interpolation of the peak position.
            let (a, c0, c) = (self.peak[b - 1].ln(), pk.ln(), self.peak[b + 1].ln());
            let den = a - 2.0 * c0 + c;
            let off = if den.abs() > 1e-9 { 0.5 * (a - c) / den } else { 0.0 };
            let f = (b as f32 + off.clamp(-0.5, 0.5)) * hz;
            if self.suppressed.iter().any(|&(sf, _)| (sf - f).abs() < self.cfg.min_spacing) {
                continue;
            }
            if self.channels.iter().any(|ch| (ch.demod.freq as f32 - f).abs() < self.cfg.min_spacing) {
                continue;
            }
            // Ignore key clicks / skirts of much stronger nearby signals.
            let lvl = 10.0 * (pk / nf).log10();
            if self.channels.iter().any(|ch| {
                let df = (ch.demod.freq as f32 - f).abs();
                df < 200.0 && ch.level_db - lvl > 20.0 + df / 10.0
            }) {
                continue;
            }
            if self.channels.len() >= self.cfg.max_channels {
                // Replace the weakest unpinned channel if this one is stronger.
                let weakest = self
                    .channels
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| !c.pinned)
                    .min_by(|a, b| a.1.level_db.total_cmp(&b.1.level_db))
                    .map(|(i, c)| (i, c.level_db));
                match weakest {
                    Some((i, l)) if l + 6.0 < lvl => {
                        let mut c = self.channels.remove(i);
                        Self::dropping(&mut c, self.time, &mut self.events, &mut self.spots);
                    }
                    _ => continue,
                }
            }
            self.add_channel(f, false);
        }
    }

    /// Removes channels whose keying is just an artefact of a much stronger
    /// signal: harmonics, intermodulation, key clicks. Their marks almost
    /// all fall inside (or at the edges of) the strong signal's marks.
    fn reject_artifacts(&mut self) {
        let n = self.channels.len();
        let mut dead = vec![false; n];
        for (si, strong) in self.channels.iter().enumerate() {
            let len = strong.marks.len();
            if len < 500 || dead[si] {
                continue;
            }
            // Dilate the strong channel's marks by ±20 ms (key-click edges).
            let mut pre = vec![0u32; len + 1];
            for (i, &m) in strong.marks.iter().enumerate() {
                pre[i + 1] = pre[i] + u32::from(m);
            }
            let r = 10usize;
            let dil = |i: usize| pre[(i + r + 1).min(len)] > pre[i.saturating_sub(r)];
            for (wi, weak) in self.channels.iter().enumerate() {
                // A nearby channel of similar strength may be the same signal
                // picked up twice (both inside its skirt); it must then key
                // in step with it.
                let df = (strong.demod.freq - weak.demod.freq).abs() as f32;
                let twin = df < 80.0 && strong.level_db >= weak.level_db;
                if wi == si || dead[wi] || weak.pinned || (!twin && strong.level_db < weak.level_db + 10.0) {
                    continue;
                }
                let wl = weak.marks.len().min(len);
                let (wo, so) = (weak.marks.len() - wl, len - wl);
                let mut tot = 0u32;
                let mut inside = 0u32;
                for i in 0..wl {
                    if weak.marks[wo + i] {
                        tot += 1;
                        if dil(so + i) {
                            inside += 1;
                        }
                    }
                }
                if std::env::var_os("CWSK_DEBUG_ART").is_some() && tot >= 100 {
                    eprintln!("t={:.0} weak {:.0}Hz ({:.0}dB) vs strong {:.0}Hz ({:.0}dB): {}/{} = {:.2}", self.time, weak.demod.freq, weak.level_db, strong.demod.freq, strong.level_db, inside, tot, inside as f32 / tot as f32);
                }
                let corr = if tot >= 100 { lagged_corr(&strong.marks, &weak.marks) } else { 0.0 };
                if std::env::var_os("CWSK_DEBUG_ART").is_some() && tot >= 100 {
                    eprintln!("    corr {corr:.2}");
                }
                let much_stronger = strong.level_db >= weak.level_db + 10.0;
                if tot >= 100 && ((much_stronger && (inside as f32 > 0.92 * tot as f32 || corr > 0.5)) || (twin && corr > 0.7)) {
                    dead[wi] = true;
                }
            }
        }
        let until = self.time + 60.0;
        let mut i = 0;
        let (now, events, spots) = (self.time, &mut self.events, &mut self.spots);
        self.channels.retain_mut(|c| {
            let keep = !dead[i];
            i += 1;
            if !keep {
                Self::dropping(c, now, events, spots);
                self.suppressed.push((c.demod.freq as f32, until));
            }
            keep
        });
    }

    fn housekeeping(&mut self) {
        let now = self.time;
        if now - self.last_artifact_check >= 1.0 {
            self.last_artifact_check = now;
            self.reject_artifacts();
            self.suppressed.retain(|&(_, t)| t > now);
        }
        // Re-check every channel continuously, not only when it decodes new
        // text: a station that stopped leaves a decoder chewing on noise.
        for c in &mut self.channels {
            let (fit, runs) = c.decoder.fit();
            if fit >= MIN_FIT {
                c.last_good = now;
            }
            if !c.pinned && c.qualified && runs >= 20 && fit < MIN_FIT - 0.13 {
                Self::hide(c, now, &mut self.events, &mut self.spots);
                c.qualified = false;
            }
        }
        let timeout = self.cfg.channel_timeout as f64;
        let mut retired = Vec::new();
        let (events, spots) = (&mut self.events, &mut self.spots);
        self.channels.retain_mut(|c| {
            let keep = c.pinned || (now - c.last_mark < timeout && now - c.last_good < timeout);
            if !keep {
                Self::dropping(c, now, events, spots);
            }
            if !keep && c.qualified {
                retired.push(Retired { id: c.id, freq: c.demod.freq as f32, at: now, text: c.text.clone(), call_counts: c.call_counts.clone() });
            }
            keep
        });
        self.graveyard.extend(retired);
        self.graveyard.retain(|r| now - r.at < 180.0);
        // Merge channels that drifted together via AFC.
        let mut i = 0;
        while i < self.channels.len() {
            let mut j = i + 1;
            let mut removed = false;
            while j < self.channels.len() {
                let df = (self.channels[i].demod.freq - self.channels[j].demod.freq).abs() as f32;
                if df < self.cfg.min_spacing * 0.6 {
                    let (a, b) = (&self.channels[i], &self.channels[j]);
                    let drop_j = b.pinned <= a.pinned && (a.pinned || a.decoder.total_marks >= b.decoder.total_marks);
                    let k = if drop_j { j } else { i };
                    let mut c = self.channels.remove(k);
                    Self::dropping(&mut c, now, &mut self.events, &mut self.spots);
                    if drop_j {
                        continue;
                    } else {
                        removed = true;
                        break;
                    }
                }
                j += 1;
            }
            if !removed {
                i += 1;
            }
        }
        self.spots.retain(|s| now - s.time < 900.0);
    }

    /// Debug view of every channel, qualified or not:
    /// (freq, timing fit, fit runs, elements/char, dash fraction, qualified, text).
    pub fn debug_channels(&self) -> Vec<(f32, f32, u32, f32, f32, bool, String)> {
        self.channels
            .iter()
            .map(|c| {
                let (fit, n) = c.decoder.fit();
                let (q, d, _) = text_quality(&c.text);
                (c.demod.freq as f32, fit, n, q, d, c.qualified, c.text.clone())
            })
            .collect()
    }

    pub fn channel_info(&self) -> Vec<ChannelInfo> {
        let mut v: Vec<ChannelInfo> = self
            .channels
            .iter()
            .filter(|c| c.qualified || c.pinned)
            .map(|c| {
                let callsign = c
                    .call_counts
                    .iter()
                    .filter(|(_, &n)| n >= 2)
                    .max_by_key(|(_, &n)| n)
                    .map(|(k, _)| k.clone())
                    .or_else(|| self.spots.iter().filter(|s| s.channel == c.id).max_by(|a, b| a.time.total_cmp(&b.time)).map(|s| s.call.clone()));
                ChannelInfo {
                    id: c.id,
                    freq: c.demod.freq as f32,
                    wpm: c.decoder.wpm(),
                    snr_db: c.decoder.snr_db(),
                    text: c.text.clone(),
                    tentative: c.decoder.tentative(),
                    is_mark: c.decoder.is_mark(),
                    pinned: c.pinned,
                    last_mark_age: (self.time - c.last_mark) as f32,
                    callsign,
                    fit: c.decoder.fit().0,
                    doubtful: c.plaus.is_some_and(|p| p < DOUBTFUL_PLAUS),
                }
            })
            .collect();
        v.sort_by(|a, b| a.freq.total_cmp(&b.freq));
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_doubled_calls_only() {
        assert_eq!(split_doubled_calls("TEST K5ZDK5ZD TU"), "TEST K5ZD K5ZD TU");
        assert_eq!(split_doubled_calls("HIHI TUTU 599599"), "HIHI TUTU 599599");
    }
}
