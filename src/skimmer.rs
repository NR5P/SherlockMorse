//! Multi-channel skimmer: finds every keyed CW signal in the audio passband
//! and runs an independent decoder on each one.

use crate::callsign::parse_callsign;
use crate::decoder::{CwDecoder, DecoderParams};
use crate::dsp::channel::ChannelDemod;
use crate::dsp::spectrum::Spectrum;
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
    words: String,
    call_counts: HashMap<String, u32>,
    prev_word: String,
    frames: Vec<Complex32>,
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
}

const PREROLL_SECS: f32 = 1.5;

pub fn bin_hz() -> f32 {
    RATE as f32 / FFT_SIZE as f32
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
        }
    }

    /// Adds a decoder at a fixed frequency (e.g. user click). Returns its id.
    pub fn add_channel(&mut self, freq: f32, pinned: bool) -> u64 {
        if let Some(c) = self.channels.iter_mut().find(|c| (c.demod.freq as f32 - freq).abs() < self.cfg.min_spacing * 0.5) {
            c.pinned |= pinned;
            return c.id;
        }
        let id = self.next_id;
        self.next_id += 1;
        let mut demod = ChannelDemod::new(RATE as f32, freq as f64, 100.0, FRAME);
        let mut decoder = CwDecoder::new(demod.frame_ms(), self.cfg.params.clone(), self.lm.clone());
        // Warm up on the pre-roll audio.
        let (a, b) = self.preroll.as_slices();
        let mut frames = Vec::new();
        demod.process(a, &mut frames);
        demod.process(b, &mut frames);
        let amps: Vec<f32> = frames.iter().map(|z| z.norm()).collect();
        decoder.warmup(&amps);
        self.channels.push(Channel {
            id,
            demod,
            decoder,
            text: String::new(),
            created: self.time,
            last_mark: self.time,
            pinned,
            level_db: -200.0,
            active: true,
            words: String::new(),
            call_counts: HashMap::new(),
            prev_word: String::new(),
            frames: Vec::new(),
        });
        id
    }

    pub fn remove_channel(&mut self, id: u64) {
        self.channels.retain(|c| c.id != id);
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
        self.channels.retain(|c| (c.demod.freq as f32) > lo - 20.0 && (c.demod.freq as f32) < hi + 20.0);
        // Spectral history no longer lines up.
        let nb = self.peak.len();
        self.peak = vec![0.0; nb];
        self.trough = vec![0.0; nb];
        self.persist = vec![0; nb];
        self.preroll.clear();
    }

    pub fn clear(&mut self) {
        self.channels.clear();
        self.spots.clear();
    }

    /// Processes audio at [`RATE`].
    pub fn process(&mut self, samples: &[f32]) {
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
        for ch in &mut self.channels {
            ch.frames.clear();
            ch.demod.process(samples, &mut ch.frames);
            for i in 0..ch.frames.len() {
                let z = ch.frames[i];
                ch.decoder.push_frame(z.norm());
                // AFC only on clean marks so it cannot wander off onto
                // a neighbour.
                let strong = ch.decoder.snr_db() > 12.0;
                ch.demod.afc(z, ch.decoder.is_mark() && strong);
            }
            if ch.decoder.is_mark() {
                ch.last_mark = now;
            }
            if let Some(t) = ch.decoder.timing() {
                // Match the filter to the keying speed.
                let bw = (ch.decoder.params().bw_factor * 1000.0 / t.dot).clamp(30.0, 200.0);
                ch.demod.set_bandwidth(bw);
            }
            let txt = ch.decoder.take_text();
            if !txt.is_empty() {
                ch.text.push_str(&txt);
                if ch.text.len() > 4000 {
                    let cut = ch.text.len() - 3000;
                    ch.text.drain(..cut);
                }
                self.events.push(TextEvent { channel: ch.id, freq: ch.demod.freq as f32, text: txt.clone() });
                Self::scan_calls(ch, &txt, now, &mut self.spots);
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
            if !txt.is_empty() {
                ch.text.push_str(&txt);
                self.events.push(TextEvent { channel: ch.id, freq: ch.demod.freq as f32, text: txt.clone() });
                Self::scan_calls(ch, &txt, self.time, &mut self.spots);
            }
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
            if let Some(call) = parse_callsign(&w) {
                let n = ch.call_counts.entry(call.clone()).or_insert(0);
                *n += 1;
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
                    } else {
                        spots.push(Spot { call, freq: f, wpm, snr_db: snr, time: now, channel: ch.id, count: 1 });
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
                        self.channels.remove(i);
                    }
                    _ => continue,
                }
            }
            self.add_channel(f, false);
        }
    }

    fn housekeeping(&mut self) {
        let now = self.time;
        let timeout = self.cfg.channel_timeout as f64;
        self.channels.retain(|c| c.pinned || now - c.last_mark < timeout);
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
                    if drop_j {
                        self.channels.remove(j);
                        continue;
                    } else {
                        self.channels.remove(i);
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

    pub fn channel_info(&self) -> Vec<ChannelInfo> {
        let mut v: Vec<ChannelInfo> = self
            .channels
            .iter()
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
                }
            })
            .collect();
        v.sort_by(|a, b| a.freq.total_cmp(&b.freq));
        v
    }
}
