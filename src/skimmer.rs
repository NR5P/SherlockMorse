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
    /// A shown channel that has proven real (timing fit reached
    /// `STRONG_FIT`) is hidden for poor timing only after this many
    /// seconds without good keying: a weak station's fit dips in fades.
    pub dropout_s: f32,
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
            dropout_s: 30.0,
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
    /// Peak level over the noise right around the channel (dB): what the
    /// decoder actually has to work with. Not comparable between channels
    /// (`level_db` is), since the noise reference differs.
    pub local_db: f32,
    /// Spectral activity gate: something is transmitting at this frequency.
    pub active: bool,
    /// Recent mark/space decisions (one per frame) for artefact detection.
    marks: VecDeque<bool>,
    /// Recent envelope (one per frame), same span as `marks`.
    amps: VecDeque<f32>,
    /// The same, before the speed-matched filter (see [`speed_watchdog`]).
    wide: VecDeque<f32>,
    /// Strong enough that the post filter is kept at least `STRONG_BW` wide.
    strong_bw: bool,
    /// Created next to another strong channel, or had one created next to
    /// it: the two keep narrow demodulators while they are neighbours.
    crowd_pair: bool,
    /// That demodulator's bandwidth (Hz), matched to the pair's spacing.
    crowd_bw: f32,
    /// The last 512 ms of raw demodulator output, for finding a second
    /// carrier inside the passband (see [`Skimmer::sub_peaks`]).
    raw_ring: VecDeque<Complex32>,
    sub_checked: usize,
    /// (offset Hz, times seen in a row) of a second carrier.
    sub_cand: (f32, u8),
    /// Made from a second carrier found inside another channel's passband
    /// (a proven neighbour: pairing applies down to `CROWDED_MIN_HZ`).
    sub_spawned: bool,
    /// Slowly decaying maximum of the decoder's SNR: a real signal in this
    /// passband, even one torn up by a second station (whose instantaneous
    /// SNR swings between 4 and 30 dB), against a channel on a skirt.
    sub_snr: f32,
    wide_checked: usize,
    /// Text not yet published (held until the channel qualifies).
    pending_out: String,
    /// The decoded text looks like real Morse (see [`text_quality`]).
    qualified: bool,
    /// Best timing fit seen (a channel that reached `STRONG_FIT` is real).
    best_fit: f32,
    /// Last time the keying fitted Morse timing (noise keeps `last_mark`
    /// fresh, so lifetime is based on this instead).
    last_good: f64,
    words: String,
    /// Language-model plausibility of the recent text (see
    /// [`text_plausibility`]), once there is enough of it.
    plaus: Option<f32>,
    call_counts: HashMap<String, u32>,
    /// Recent call-length words with when they were heard, for piecing
    /// together a weak call from several differing copies.
    heard: Vec<(f64, String)>,
    prev_word: String,
    /// The word before `prev_word` ("CQ" in "CQ POTA K1ABC").
    prev2_word: String,
    /// A call heard in no telling context yet: spotted if the next word
    /// shows it was addressed or signing (DE, K, KN, ...).
    pending_call: Option<String>,
    /// A call cut short after DE or CQ (W3P): joined with the next word
    /// if that makes a whole call (W3PYF; a letter gap read as a word gap).
    /// (The call, and whether its own sighting is to be retracted on a join.)
    pending_join: Option<(String, bool)>,
    /// Contest exchanges heard here (NAME NUMBER), by name: how often.
    exch: HashMap<String, u32>,
    /// Own sightings of calls in a glued-letter pair (WN4DW / N4DW), the
    /// pair's sightings being pooled under one of them.
    glue_own: HashMap<String, u32>,
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
    /// Instead of text: why a channel was shown, hidden or removed, with
    /// its measurements at the time (for the channel log).
    pub note: Option<String>,
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
    /// Per-bin noise reference: the median of nearby bins (about ±125 Hz).
    /// The receiver's filter shapes the band noise (on the FTDX3000 it is
    /// ~6 dB stronger at 500-1000 Hz than on average, and falls off above
    /// 2 kHz): a weak signal on the skirt sits well above its own noise but
    /// below the overall median, and one in the middle looks stronger
    /// against the overall median than it is.
    local_floor: Vec<f32>,
    pub channels: Vec<Channel>,
    next_id: u64,
    /// Seconds of audio processed.
    pub time: f64,
    wf_acc: Vec<f32>,
    wf_n: u32,
    pub waterfall: VecDeque<Vec<f32>>,
    pub events: Vec<TextEvent>,
    pub spots: Vec<Spot>,
    /// The channel being added comes from `sub_peaks`.
    spawning: bool,
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
    /// The latest spectrum frame was muted audio (far below the noise),
    /// and for how many frames in a row.
    muted: bool,
    muted_hops: u32,
}

const PREROLL_SECS: f32 = 1.5;
/// Frames of mark history kept per channel (5 s).
const MARK_HIST: usize = 2500;
/// A channel this far over the noise floor is a station, not the noise
/// keyed by the receiver's AGC (AGC junk was 5-14 dB; stations 20-40).
const AGC_JUNK_DB: f32 = 15.0;

pub fn bin_hz() -> f32 {
    RATE as f32 / FFT_SIZE as f32
}

/// Minimum timing-fit score for a channel to be shown (real CW measured
/// 0.8-0.97 on live recordings; noise 0.35-0.68).
const MIN_FIT: f32 = 0.78;

/// Above this speed a channel must fit Morse timing at least `FAST_MIN_FIT`
/// to be shown.
const FAST_WPM: f32 = 30.0;
const FAST_MIN_FIT: f32 = 0.86;

/// Timing fit above which keying is certainly real CW (live recordings:
/// real 0.8-0.97, noise 0.35-0.68): such a channel is never hidden for its
/// text.
const STRONG_FIT: f32 = 0.85;
/// Another strong channel `CROWDED_MIN_HZ`..`CROWDED_HZ` away makes a new
/// channel "crowded": both get a `CROWDED_BW` demodulator passband instead
/// of 100 Hz. (Closer than the minimum is a strong signal's own spectral
/// spread, not another station: pairing a 39 dB station with its own
/// 33 Hz sideband pinned and narrowed it and garbled its copy.)
const CROWDED_MIN_HZ: f32 = 18.0;
/// ...and for a neighbour the band detector found (see `add_channel`).
const CROWDED_DETECT_MIN_HZ: f32 = 40.0;
const CROWDED_HZ: f32 = 60.0;
const CROWDED_BW: f32 = 50.0;
/// A pair's demodulator passband as a fraction of its spacing, at least
/// `CROWDED_BW_MIN` (a 20 Hz passband smears 30 WPM dits, but a partner
/// 25 Hz away smears everything).
const CROWDED_BW_FRAC: f32 = 0.8;
const CROWDED_BW_MIN: f32 = 16.0;
/// ...when both are within this much of each other in level.
const CROWDED_DIFF_DB: f32 = 10.0;
/// At creation (before either has read anything) both must be this strong.
const CROWDED_START_DB: f32 = 25.0;
/// How far the AFC may move a crowded channel from where it was created.
const CROWDED_AFC_HZ: f64 = 6.0;
/// Raw frames kept per channel (512 ms at 2 ms) and pieces (~14 ms) between looks
/// for a second carrier (see `sub_peaks`).
const SUB_RING: usize = 256;
/// A second carrier must exceed its mirror image about the channel's own
/// carrier by this power ratio (6 dB): keying sidebands are symmetric.
const SUB_ASYMMETRY: f32 = 4.0;
/// The parent must be a real signal in its own passband (decoder SNR, dB),
/// not a channel on another station's skirt.
const SUB_MIN_SNR_DB: f32 = 15.0;
const SUB_FRAMES: usize = 70;
/// Frames between looks at the unfiltered keying (see `speed_watchdog`).
const WATCH_FRAMES: usize = 600;
/// How long a channel keeps its recent words for piecing calls together.
const HEARD_SECS: f64 = 600.0;
/// Keying fit needed to trust a single sighting of a call, and the fastest
/// keying trusted (noise decodes as very fast text; contest CW tops out
/// around 45 WPM).
const SINGLE_SPOT_FIT: f32 = 0.85;
const SINGLE_SPOT_MAX_WPM: f32 = 48.0;
/// Envelope SNR (dB) needed for a single sighting; less on a channel with
/// no text to judge (see `spot`).
const SINGLE_SPOT_SNR: f32 = 20.0;
const SINGLE_SPOT_SNR_CLEAN: f32 = 12.0;
/// Below this SNR (dB) a channel must fit Morse timing at least
/// `LOW_SNR_MIN_FIT` to be shown.
const LOW_SNR_DB: f32 = 6.0;
const LOW_SNR_MIN_FIT: f32 = 0.82;

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

/// Splits callsigns run together with a neighbour when the word gap was
/// squeezed out: a call sent twice ("K5ZDK5ZD" -> "K5ZD K5ZD"; calls are
/// routinely sent twice) and "DE" glued to a call ("DEN4XYZ" -> "DE N4XYZ";
/// DE prefixes are not issued to amateurs).
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
            out.push_str(&split_glued(w).join(" "));
        }
    }
    out
}

/// Splits CQ, DE and DX run into what follows them ("CQCQDE" -> "CQ CQ
/// DE", "CQDXE71A" -> "CQ DX E71A", "DEN4XYZ" -> "DE N4XYZ"). Only where
/// the rest is CQ/DE/DX/K/TEST or a call with an issued prefix: no word
/// starts with CQ, DE prefixes are not issued, and a DX call (DX1ABC,
/// Philippines) stays whole when it parses as a call itself.
fn split_glued(w: &str) -> Vec<&str> {
    let call = |r: &str| parse_callsign(r).is_some_and(|c| crate::callsign::valid_prefix(&c));
    let ok_rest = |r: &str| matches!(r, "DE" | "DX" | "K" | "TEST") || r.starts_with("CQ") || call(r) || ((r.starts_with("DE") || r.starts_with("DX")) && call(&r[2..]));
    for head in ["CQ", "DE", "DX"] {
        if w.len() > 2 && w.starts_with(head) {
            let rest = &w[2..];
            let take = match head {
                "CQ" => ok_rest(rest),
                "DE" => w.len() >= 6 && call(rest),
                _ => w.len() >= 6 && call(rest) && !call(w),
            };
            if take {
                let mut v = vec![&w[..2]];
                v.extend(split_glued(rest));
                return v;
            }
        }
    }
    // The closing K run onto a call ("XE1FRFK"): only when that makes a
    // 4-letter suffix, which real calls almost never have.
    let suffix = w.bytes().rev().take_while(u8::is_ascii_uppercase).count();
    if w.len() >= 5 && w.ends_with('K') && suffix == 4 && call(&w[..w.len() - 1]) {
        return vec![&w[..w.len() - 1], "K"];
    }
    vec![w]
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
    heard: Vec<(f64, String)>,
}

/// Pearson correlation of two envelope histories (log amplitude) while the
/// first channel is keying (a mark within ±0.6 s per `marks_a`): the band
/// noise the receiver's AGC pumps moves against a strong signal, an
/// unrelated station does not. (Over the whole span, two stations simply
/// taking turns would look anti-correlated too.)
fn amp_corr(a: &VecDeque<f32>, b: &VecDeque<f32>, marks_a: &VecDeque<bool>) -> f32 {
    let n = a.len().min(b.len()).min(marks_a.len());
    if n < 500 {
        return 0.0;
    }
    let (oa, ob, om) = (a.len() - n, b.len() - n, marks_a.len() - n);
    let mut pre = vec![0u32; n + 1];
    for i in 0..n {
        pre[i + 1] = pre[i] + u32::from(marks_a[om + i]);
    }
    let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0f64, 0f64, 0f64, 0f64, 0f64);
    let mut cnt = 0usize;
    for i in 0..n {
        if pre[(i + 301).min(n)] == pre[i.saturating_sub(300)] {
            continue;
        }
        cnt += 1;
        let x = f64::from(a[oa + i].max(1e-9).ln());
        let y = f64::from(b[ob + i].max(1e-9).ln());
        sx += x;
        sy += y;
        sxx += x * x;
        syy += y * y;
        sxy += x * y;
    }
    if cnt < 500 {
        return 0.0;
    }
    let nf = cnt as f64;
    let cov = sxy / nf - (sx / nf) * (sy / nf);
    let var = (sxx / nf - (sx / nf).powi(2)) * (syy / nf - (sy / nf).powi(2));
    if var <= 1e-12 { 0.0 } else { (cov / var.sqrt()) as f32 }
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
            local_floor: vec![1e-9; nb],
            channels: Vec::new(),
            next_id: 1,
            time: 0.0,
            wf_acc: vec![0.0; nb],
            wf_n: 0,
            waterfall: VecDeque::new(),
            events: Vec::new(),
            spots: Vec::new(),
            spawning: false,
            preroll: VecDeque::new(),
            suppressed: Vec::new(),
            last_artifact_check: 0.0,
            graveyard: Vec::new(),
            hops: 0,
            burst: false,
            burst_hops: 0,
            muted: false,
            muted_hops: 0,
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
        let (rtext, rcalls, rheard) = match revived {
            Some(r) => (r.text + " ", r.call_counts, r.heard),
            None => (String::new(), HashMap::new(), Vec::new()),
        };
        // Next to another channel (a contest band packs stations 40-60 Hz
        // apart), start narrow so this one hears mostly its own station.
        let lvl = {
            // (As `level_db` is measured.)
            let b = ((freq / bin_hz()).round() as usize).min(self.peak.len() - 1);
            let nf = self.noise_floor;
            10.0 * (self.peak[b] / self.local_floor[b].clamp(nf * 0.1, nf).max(1e-15)).log10()
        };
        let mut crowded = false;
        let mut crowd_bw = CROWDED_BW;
        if lvl >= CROWDED_START_DB {
            for c in &mut self.channels {
                let d = (c.demod.freq as f32 - freq).abs();
                // A neighbour found by the band detector within 40 Hz is more
                // often a strong signal's own spread than another station;
                // one found as a second carrier in a passband is proven.
                let min_hz = if self.spawning || c.sub_spawned { CROWDED_MIN_HZ } else { CROWDED_DETECT_MIN_HZ };
                if (min_hz..CROWDED_HZ).contains(&d) && c.level_db >= CROWDED_START_DB && (c.level_db - lvl).abs() <= CROWDED_DIFF_DB {
                    // The passband keeps the partner out: a little under
                    // the spacing (and the AFC closer still).
                    let bw = (CROWDED_BW_FRAC * d).clamp(CROWDED_BW_MIN, CROWDED_BW);
                    let afc = (0.15 * d as f64).clamp(2.0, CROWDED_AFC_HZ);
                    crowded = true;
                    crowd_bw = crowd_bw.min(bw);
                    c.crowd_pair = true;
                    c.crowd_bw = c.crowd_bw.min(bw);
                    c.demod.set_bandwidth(c.crowd_bw);
                    c.demod.afc_limit = (c.demod.freq - afc, c.demod.freq + afc);
                }
            }
        }
        if std::env::var_os("CWSK_DEBUG").is_some() {
            let nb: Vec<String> = self.channels.iter().filter(|c| (c.demod.freq as f32 - freq).abs() < CROWDED_HZ).map(|c| format!("{:.0}Hz {:.0}dB", c.demod.freq, c.level_db)).collect();
            eprintln!("new channel t={:.1} {freq:.0} Hz lvl {lvl:.0} dB crowded={crowded} near {nb:?}", self.time);
        }
        let mut demod = ChannelDemod::new(RATE as f32, freq as f64, if crowded { crowd_bw } else { 100.0 }, FRAME);
        // (The AFC of each is pulled toward the other while both key, so
        // it is held close to the creation frequency, which next to a
        // known channel is the bin centre, not an interpolation.)
        if crowded {
            let afc = (0.15 * crowd_bw as f64 / CROWDED_BW_FRAC as f64).clamp(2.0, CROWDED_AFC_HZ);
            demod.afc_limit = (freq as f64 - afc, freq as f64 + afc);
        }
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
            local_db: -200.0,
            active: true,
            marks: VecDeque::with_capacity(MARK_HIST),
            amps: VecDeque::with_capacity(MARK_HIST),
            wide: VecDeque::with_capacity(MARK_HIST),
            wide_checked: 0,
            strong_bw: false,
            crowd_pair: crowded,
            crowd_bw,
            raw_ring: VecDeque::with_capacity(SUB_RING),
            sub_checked: 0,
            sub_cand: (0.0, 0),
            sub_spawned: self.spawning,
            sub_snr: 0.0,
            pending_out: String::new(),
            qualified: pinned || was_revived,
            last_good: self.time,
            words: String::new(),
            best_fit: 0.0,
            post: None,
            start_dot: None,
            start_redecodes: 0,
            start_text_len: rtext.len(),
            plaus: None,
            call_counts: rcalls,
            heard: rheard,
            prev_word: String::new(),
            prev2_word: String::new(),
            pending_call: None,
            pending_join: None,
            exch: HashMap::new(),
            glue_own: HashMap::new(),
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
                Self::dropping(c, now, events, spots, "out of the passband after tuning");
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
        // Muted only if every spectrum frame of this piece was: the decoder
        // takes one erasure flag per piece, and a piece ending in a short
        // gap of digital silence (an MP3 between elements) still holds marks.
        // (And only after about 5 s of it: an MP3 has stretches of digital
        // silence between words, seconds long at 5 WPM, which are true
        // spaces.)
        let mut all_muted = !spectra.is_empty();
        for p in &spectra {
            self.on_spectrum(p);
            all_muted &= self.muted_hops > 300;
        }
        self.preroll.extend(samples.iter().copied());
        let max = (PREROLL_SECS * RATE as f32) as usize;
        if self.preroll.len() > max {
            let n = self.preroll.len() - max;
            self.preroll.drain(..n);
        }
        let now = self.time;
        let (params, lm) = (&self.cfg.params, &self.lm);
        // A channel paired at creation with a neighbour (see `add_channel`)
        // keeps its narrow demodulator until no channel is within
        // `CROWDED_HZ` of it any more.
        let freqs: Vec<f32> = self.channels.iter().map(|c| c.demod.freq as f32).collect();
        for (i, ch) in self.channels.iter_mut().enumerate() {
            if ch.crowd_pair && !freqs.iter().enumerate().any(|(j, f)| j != i && (f - freqs[i]).abs() < CROWDED_HZ) {
                ch.crowd_pair = false;
                ch.crowd_bw = CROWDED_BW;
                ch.demod.set_bandwidth(100.0);
                let f = ch.demod.freq;
                ch.demod.afc_limit = (f - 12.0, f + 12.0);
            }
            ch.frames.clear();
            ch.demod.process(samples, &mut ch.frames);
            ch.decoder.set_erasure((self.burst && (5..60).contains(&self.burst_hops)) || all_muted);
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
                    ch.amps.pop_front();
                    ch.wide.pop_front();
                }
                ch.marks.push_back(ch.decoder.is_mark());
                ch.amps.push_back(z.norm());
                ch.wide.push_back(raw.norm());
                if ch.raw_ring.len() >= SUB_RING {
                    ch.raw_ring.pop_front();
                }
                ch.raw_ring.push_back(raw);
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
            Self::speed_watchdog(ch, now);
            let snr = ch.decoder.snr_db();
            if snr >= crate::decoder::STRONG_BW_ON {
                ch.strong_bw = true;
            } else if snr < crate::decoder::STRONG_BW_OFF {
                ch.strong_bw = false;
            }
            if let Some(t) = ch.decoder.timing() {
                // Match the filter to the keying speed.
                let bw = ch.decoder.params().post_bw(t.dot, ch.local_db, ch.strong_bw);
                if std::env::var_os("CWSK_BWDBG").is_some() && ch.post.as_ref().is_some_and(|p| (bw - p.2).abs() > 0.1 * p.2) {
                    eprintln!("BW t={:.0} {:.0} Hz level {:.1} dB snr {:.1} -> {bw:.1}", now, ch.demod.freq, ch.level_db, ch.decoder.snr_db());
                }
                let fs = 1000.0 / ch.demod.frame_ms();
                match &mut ch.post {
                    Some((li, lq, b)) if (bw - *b).abs() > 0.1 * *b => {
                        li.retune(fs, bw / 2.0);
                        lq.retune(fs, bw / 2.0);
                        *b = bw;
                        ch.decoder.set_bandwidth(bw);
                    }
                    Some(_) => {}
                    None => {
                        ch.post = Some((Lowpass4::new(fs, bw / 2.0), Lowpass4::new(fs, bw / 2.0), bw));
                        ch.decoder.set_bandwidth(bw);
                    }
                }
            }
            let txt = ch.decoder.take_text();
            if !txt.is_empty() {
                Self::deliver(ch, &txt, now, false, &self.lm, self.cfg.dropout_s, &mut self.events, &mut self.spots);
            }
        }
        self.sub_peaks();
        self.time += samples.len() as f64 / RATE as f64;
        self.housekeeping();
    }

    /// Flushes all decoders (end of input).
    pub fn finish(&mut self) {
        for ch in &mut self.channels {
            ch.decoder.finish();
            let txt = ch.decoder.take_text();
            Self::deliver(ch, &txt, self.time, true, &self.lm, self.cfg.dropout_s, &mut self.events, &mut self.spots);
        }
    }

    /// A second station inside a strong channel's passband, too close to
    /// show as a peak of its own in the band spectrum (under two bins,
    /// ~30 Hz): looked for in the channel's own raw demodulator output,
    /// 512 ms at a time (2 Hz resolution), as a second carrier 18-60 Hz
    /// from the channel's own (the strongest within 15 Hz of the centre)
    /// within 10 dB of it. Seen three seconds running
    /// on a channel that is not reading well, it gets a channel of its
    /// own, and the two are paired with narrow passbands (`add_channel`).
    fn sub_peaks(&mut self) {
        let mut found: Vec<f32> = Vec::new();
        let ms = self.channels.first().map_or(2.0, |c| c.demod.frame_ms());
        for ch in &mut self.channels {
            ch.sub_checked += 1;
            if ch.sub_checked < SUB_FRAMES || ch.raw_ring.len() < SUB_RING {
                continue;
            }
            ch.sub_checked = 0;
            // Only a real signal in its own passband that is failing to read.
            let reading_well = ch.qualified && ch.plaus.is_none_or(|p| p >= DOUBTFUL_PLAUS);
            ch.sub_snr = ch.decoder.snr_db().max(ch.sub_snr * 0.95);
            if ch.crowd_pair || ch.level_db < CROWDED_START_DB || ch.sub_snr < SUB_MIN_SNR_DB || reading_well {
                ch.sub_cand = (0.0, 0);
                continue;
            }
            let x: Vec<Complex32> = ch.raw_ring.iter().copied().collect();
            let power = |hz: f32| -> f32 {
                let w = -std::f32::consts::TAU * hz * ms / 1000.0;
                let mut acc = Complex32::new(0.0, 0.0);
                for (n, v) in x.iter().enumerate() {
                    let (s, c) = (w * n as f32).sin_cos();
                    acc += v * Complex32::new(c, s);
                }
                acc.norm_sqr()
            };
            // Carriers in the passband: local maxima within 10 dB of the
            // strongest. The channel's own is the one nearest its centre;
            // a second one at least `CROWDED_MIN_HZ` from it is a partner.
            let offsets: Vec<f32> = (-48..=48).map(|k| k as f32 * 1.25).collect();
            let ps: Vec<f32> = offsets.iter().map(|&d| power(d)).collect();
            let pmax = ps.iter().copied().fold(0.0f32, f32::max).max(1e-20);
            let peaks: Vec<(f32, f32)> = (2..offsets.len() - 2)
                .filter(|&i| ps[i] >= pmax * 0.1 && ps[i] >= ps[i - 1] && ps[i] > ps[i + 1] && ps[i] >= ps[i - 2] && ps[i] > ps[i + 2])
                .map(|i| (offsets[i], ps[i]))
                .collect();
            let own = peaks.iter().copied().filter(|(d, _)| d.abs() <= 15.0).max_by(|a, b| a.1.total_cmp(&b.1));
            // A station's own keying sidebands sit at both sides of its
            // carrier alike; a second station does not have a mirror image.
            let at = |hz: f32| -> f32 {
                let i = ((hz / 1.25).round() as i32 + 48).clamp(0, offsets.len() as i32 - 1) as usize;
                ps[i]
            };
            let best = own.and_then(|(d0, _)| {
                peaks
                    .iter()
                    .copied()
                    .filter(|(d, p)| (d - d0).abs() >= CROWDED_MIN_HZ && *p >= SUB_ASYMMETRY * at(2.0 * d0 - d))
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .map(|(d, _)| (d0, d))
            });
            match best {
                Some((_, d)) if (d - ch.sub_cand.0).abs() < 6.0 => ch.sub_cand = (d, ch.sub_cand.1.saturating_add(1)),
                Some((_, d)) => ch.sub_cand = (d, 1),
                None => ch.sub_cand = (0.0, 0),
            }
            if ch.sub_cand.1 >= 3 {
                if let Some((d0, d)) = best {
                    // Centre on the own carrier first (the AFC had settled
                    // between the two), then the partner at the true spacing.
                    let f0 = ch.demod.freq + d0 as f64;
                    ch.demod.set_freq(f0);
                    found.push((f0 + (d - d0) as f64) as f32);
                }
                ch.sub_cand = (0.0, 0);
            }
        }
        for f in found {
            if !self.channels.iter().any(|c| (c.demod.freq as f32 - f).abs() < self.cfg.min_spacing * 0.5) {
                if std::env::var_os("CWSK_DEBUG").is_some() {
                    eprintln!("sub-peak t={:.1}: second carrier at {f:.0} Hz", self.time);
                }
                self.spawning = true;
                self.add_channel(f, false);
                self.spawning = false;
            }
        }
    }

    /// A faster station on the frequency of a slow one is smeared by the
    /// filter matched to the slow one (30 Hz for 12 WPM; a 35 WPM dit is
    /// 34 ms) into blobs the decoder reads as the slow station's dashes,
    /// so the speed is never re-measured and the fast station is lost for
    /// as long as it keeps sending (contest QSOs after a slow CQ). Watch
    /// the keying before the filter: when it is clearly explained by a much
    /// faster speed, switch the decoder and the filter to it.
    fn speed_watchdog(ch: &mut Channel, now: f64) {
        let Some(dot) = ch.decoder.timing().map(|t| t.dot) else { return };
        // (Only a slow model's filter can hide a station; one for 27 WPM
        // or faster passes everything up to 60 WPM.)
        // (And only while the channel reads badly: a slow station copied
        // well, or keyed cleanly, is left alone whatever its clicks look
        // like unfiltered.)
        let reading_well = ch.plaus.is_some_and(|p| p >= DOUBTFUL_PLAUS) || ch.decoder.fit().0 >= 0.9;
        if dot < 45.0 || reading_well || ch.wide.len() < MARK_HIST || ch.wide_checked < WATCH_FRAMES || std::env::var_os("CWSK_NOWATCH").is_some() {
            ch.wide_checked += 1;
            return;
        }
        ch.wide_checked = 0;
        let mut sorted: Vec<f32> = ch.wide.iter().copied().collect();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let (noise, peak) = (sorted[sorted.len() / 4], sorted[sorted.len() * 95 / 100]);
        // (Strong, clean keying only: a weak signal's blips and clicks
        // would argue for a very fast station.)
        if peak < 6.0 * noise {
            return;
        }
        let (thr, hyst) = (noise + 0.5 * (peak - noise), 0.1 * (peak - noise));
        let ms = ch.demod.frame_ms();
        let mut runs: Vec<crate::decoder::Run> = Vec::new();
        let (mut state, mut len) = (false, 0usize);
        for &x in &ch.wide {
            let want = if state { x > thr - hyst } else { x > thr + hyst };
            if want != state && len > 0 {
                runs.push(if state { crate::decoder::Run::Mark(len as f32 * ms) } else { crate::decoder::Run::Space(len as f32 * ms) });
                len = 0;
            }
            state = want;
            len += 1;
        }
        let marks = runs.iter().filter(|r| matches!(r, crate::decoder::Run::Mark(_))).count();
        if marks < 12 {
            return;
        }
        let Some(tm) = crate::decoder::estimate_timing(&runs) else { return };
        // (Up to 42 WPM: faster "keying" is a strong signal's edge chatter.)
        if tm.dot > 0.6 * dot || tm.dot < 28.5 || !crate::decoder::clearly_better(&runs, dot, tm.dot) {
            return;
        }
        // Real keying at that speed has both dits and dashes, and element
        // gaps of a dit; static crashes and clicks on a slow station's
        // channel make random short blips that an estimate can still fit.
        let d = tm.dot;
        let (mut dits, mut dashes, mut egaps, mut gaps) = (0, 0, 0, 0);
        for r in &runs {
            match *r {
                crate::decoder::Run::Mark(m) if (0.6..=1.5).contains(&(m / d)) => dits += 1,
                crate::decoder::Run::Mark(m) if (2.2..=4.2).contains(&(m / d)) => dashes += 1,
                crate::decoder::Run::Space(g) if g < 2.0 * d => {
                    gaps += 1;
                    if (0.6..=1.6).contains(&(g / d)) {
                        egaps += 1;
                    }
                }
                _ => {}
            }
        }
        if dits < 6 || dashes < 4 || gaps < 8 || egaps * 3 < gaps * 2 {
            return;
        }
        if std::env::var_os("CWSK_DEBUG").is_some() {
            eprintln!("speed watchdog t={now:.0} {:.0} Hz: dot {dot:.0} -> {:.0} ms ({:.0} WPM) from the wide keying", ch.demod.freq, tm.dot, tm.wpm());
        }
        let bw = ch.decoder.params().post_bw(tm.dot, ch.local_db, ch.strong_bw);
        let fs = 1000.0 / ms;
        ch.decoder.reseed_timing(tm);
        ch.decoder.set_bandwidth(bw);
        match &mut ch.post {
            Some((li, lq, b)) => {
                li.retune(fs, bw / 2.0);
                lq.retune(fs, bw / 2.0);
                *b = bw;
            }
            None => ch.post = Some((Lowpass4::new(fs, bw / 2.0), Lowpass4::new(fs, bw / 2.0), bw)),
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
        let bw = params.post_bw(t.dot, ch.local_db, ch.strong_bw);
        let fs = 1000.0 / ch.demod.frame_ms();
        let (mut li, mut lq) = (Lowpass4::new(fs, bw / 2.0), Lowpass4::new(fs, bw / 2.0));
        let amps: Vec<f32> = frames.iter().map(|z| Complex32::new(li.process(z.re), lq.process(z.im)).norm()).collect();
        // Start from the timing already measured: re-measuring on the
        // narrow-filtered start can go wrong with a sloppy fist whose short
        // gaps the filter partly fills in.
        let mut d = CwDecoder::new(ch.demod.frame_ms(), params.clone(), lm.clone());
        d.seed_timing(t.clone());
        d.set_bandwidth(bw);
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
    fn deliver(ch: &mut Channel, txt: &str, now: f64, final_: bool, lm: &LanguageModel, dropout_s: f32, events: &mut Vec<TextEvent>, spots: &mut Vec<Spot>) {
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
        // Clearly real keying (noise never fits Morse timing this well) is
        // a real station even when its text is garbled (a handover, QRM, a
        // sloppy fist): show it, dimmed, rather than hide it.
        // (Echoes and other artefacts of strong signals are keyed too, but
        // decode as runs of E I S T; real text averages 2.5-3.3 elements per
        // character, and even garbled text keeps its dashes: a steady
        // pattern of dits like "UHVHVHV" is not a station talking.)
        let text_ok = text_ok || (fit >= STRONG_FIT && ch.decoder.snr_db() >= 12.0 && (n < 8 || (el >= SHORT_JUNK_ELEMS && dash >= MIN_DASH_FRAC)));
        if ch.pinned {
            ch.qualified = true;
        } else if runs >= 20 || (final_ && runs >= 10) {
            // Noise decodes as very fast "keying" that only just passes the
            // timing test (40 m scan: every channel shown above 30 WPM fit
            // 0.78-0.83 and was junk; real fast CW fits 0.94-0.98).
            // Likewise in the noise floor: a channel under 6 dB SNR whose
            // timing only just passes was junk that flashed up for a few
            // seconds (a third of the rows shown on a night of 40/80 m).
            let min_fit = if ch.decoder.wpm().is_some_and(|w| w > FAST_WPM) {
                FAST_MIN_FIT
            } else if ch.decoder.snr_db() < LOW_SNR_DB {
                LOW_SNR_MIN_FIT
            } else {
                MIN_FIT
            };
            if fit >= min_fit && text_ok {
                if !ch.qualified {
                    Self::note(ch, now, events, "shown");
                }
                ch.qualified = true;
            } else if (fit < MIN_FIT - 0.13 && !Self::holding(ch, now, dropout_s)) || (n >= 8 && !text_ok) {
                if ch.qualified {
                    let why = if fit < MIN_FIT - 0.13 { "hidden: keying no longer fits Morse timing" } else { "hidden: text looks like noise" };
                    Self::note(ch, now, events, why);
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
            events.push(TextEvent { channel: ch.id, freq: ch.demod.freq as f32, text: out.clone(), note: None });
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

    /// Called before a channel is removed: keeps its held-back last word,
    /// and notes why a shown channel went away.
    fn dropping(ch: &mut Channel, now: f64, events: &mut Vec<TextEvent>, spots: &mut Vec<Spot>, why: &str) {
        if std::env::var_os("CWSK_DEBUG").is_some() {
            eprintln!("drop t={now:.1} {:.0} Hz ({:.0} dB, pair={}): {why}", ch.demod.freq, ch.level_db, ch.crowd_pair);
        }
        if ch.qualified {
            Self::note(ch, now, events, &format!("removed: {why}"));
            Self::hide(ch, now, events, spots);
        }
    }

    /// A shown channel that has proven real keeps its row through a
    /// stretch of poor timing (a fade) up to the dropout time.
    fn holding(ch: &Channel, now: f64, dropout_s: f32) -> bool {
        ch.best_fit >= STRONG_FIT && now - ch.last_good <= f64::from(dropout_s)
    }

    /// Records a change in a channel's visibility with its measurements.
    fn note(ch: &Channel, now: f64, events: &mut Vec<TextEvent>, what: &str) {
        let (fit, runs) = ch.decoder.fit();
        let (el, dash, _) = text_quality(&ch.text);
        let tail: String = ch.text.chars().rev().take(40).collect::<Vec<_>>().into_iter().rev().collect();
        let note = format!(
            "{what} | fit {fit:.2} ({runs} runs), snr {:.0} dB, {} wpm, {el:.2} el/char, dash {dash:.2}, plaus {}, age {:.0} s, last mark {:.0} s ago, last good keying {:.0} s ago | ...{}",
            ch.decoder.snr_db(),
            ch.decoder.wpm().map(|w| format!("{w:.0}")).unwrap_or("-".into()),
            ch.plaus.map(|p| format!("{p:.2}")).unwrap_or("-".into()),
            now - ch.created,
            now - ch.last_mark,
            now - ch.last_good,
            crate::morse::to_display(tail.trim()),
        );
        events.push(TextEvent { channel: ch.id, freq: ch.demod.freq as f32, text: String::new(), note: Some(note) });
    }

    fn scan_calls(ch: &mut Channel, txt: &str, now: f64, spots: &mut Vec<Spot>) {
        ch.words.push_str(txt);
        while let Some(pos) = ch.words.find(' ') {
            let w: String = ch.words[..pos].to_string();
            ch.words.drain(..=pos);
            if w.is_empty() {
                continue;
            }
            let bare = w.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '/');
            if (3..=10).contains(&bare.len()) && bare.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'/') {
                ch.heard.retain(|(t, _)| now - t < HEARD_SECS);
                if ch.heard.len() >= 80 {
                    ch.heard.remove(0);
                }
                ch.heard.push((now, bare.to_string()));
            }
            // A call just before DE or a turn-over (K, KN, BK, ...) is
            // being addressed or is signing: as good a sighting as one
            // after DE ("W2GLN DE K9UIY", "... W9UZ K").
            if let Some(c) = ch.pending_call.take() {
                if matches!(w.as_str(), "DE" | "K" | "KN" | "BK" | "AR" | "SK" | "<KN>" | "<AR>" | "<SK>") {
                    Self::spot(ch, c, now, spots, true);
                }
            }
            if let Some((c, retract)) = ch.pending_join.take() {
                let joined = format!("{c}{bare}");
                let word_of_its_own = matches!(bare, "E" | "T" | "K" | "R" | "N" | "I" | "ES" | "DE" | "TU" | "UR" | "GM" | "GA" | "GE" | "GN" | "OK" | "HR" | "HW" | "OM" | "FB" | "CQ" | "BK" | "KN" | "AR" | "SK" | "TT" | "NN" | "AS" | "CL" | "VY" | "DX" | "QRL" | "QRZ" | "QSL" | "QTH" | "PSE" | "TNX" | "RST" | "AGN" | "NR" | "WX" | "HI");
                if (1..=3).contains(&bare.len())
                    && (!retract || (bare.len() <= 2 && !word_of_its_own))
                    && parse_callsign(&joined).as_deref() == Some(joined.as_str())
                    && crate::callsign::standard_shape(&joined)
                    && crate::callsign::valid_prefix(&joined)
                {
                    // (The cut call was counted, and maybe spotted, as a
                    // call of its own: that sighting is this one.)
                    if retract {
                        if let Some(n) = ch.call_counts.get_mut(&c) {
                            *n = n.saturating_sub(1);
                        }
                        spots.retain(|s| !(s.channel == ch.id && s.call == c && s.count <= 1));
                    }
                    *ch.call_counts.entry(joined.clone()).or_insert(0) += 1;
                    // (A cut call joined keeps its own context: none.)
                    Self::spot(ch, joined, now, spots, !retract);
                    ch.prev2_word = std::mem::replace(&mut ch.prev_word, w);
                    continue;
                }
            }
            let calls = Self::calls_in(&w, ch, spots);
            for (call, signing) in calls {
                // A near miss of a call already confirmed here, or spotted
                // elsewhere, counts toward that call.
                let near = |c: &str| c != call && crate::callsign::morse_distance(c, &call) == 1;
                let local_rival = ch.call_counts.iter().filter(|(c, &n)| n >= 2 && near(c)).max_by_key(|(_, &n)| n).map(|(c, _)| c.clone());
                let band_rival = spots.iter().filter(|s| s.confirmed && near(&s.call)).max_by_key(|s| s.count).map(|s| s.call.clone());
                // Unless several differing copies of the call here (a weak,
                // fading station) agree better on another: then that call,
                // confirmed by their number. (A bad copy confirmed twice
                // first would otherwise swallow the good ones: DL1DGR for
                // DL1BGR.) Not against a call established elsewhere on the
                // band, as a DX call is (EZ1A heard for E71A in a pileup).
                let anchor = local_rival.as_deref().unwrap_or(&call);
                let copies: Vec<&str> = ch
                    .heard
                    .iter()
                    .map(|(_, h)| h.as_str())
                    .filter(|h| crate::callsign::element_distance(h, &call) <= 2 || crate::callsign::element_distance(h, anchor) <= 2)
                    .collect();
                let voted = if band_rival.is_some() && local_rival.is_none() { None } else { crate::callsign::consensus(&copies) }
                    .filter(|v| *v == call || copies.len() >= 4)
                    .filter(|v| {
                        let support = copies.iter().filter(|c| crate::callsign::element_distance(c, v) <= 1).count() as u32;
                        !spots.iter().any(|s| s.confirmed && s.call != *v && crate::callsign::morse_distance(&s.call, v) <= 2 && s.count >= 3 * support)
                    });
                if let Some(v) = &voted {
                    spots.retain(|s| !(s.channel == ch.id && &s.call != v && s.count <= 2 && crate::callsign::element_distance(&s.call, v) <= 2));
                }
                let unsnapped = voted.is_none() && local_rival.is_none() && band_rival.is_none();
                let mut call = voted.clone().or(local_rival).or(band_rival).unwrap_or(call);
                // The call sent twice or three times in a row with a copy
                // garbled ("CX55K CX5FK"): signing, as after DE, when this
                // copy fits them all best. The garbled copies are forgotten.
                let mut repeated = false;
                if unsnapped {
                    let mut group: Vec<&str> = ch
                        .heard
                        .iter()
                        .rev()
                        .take(4)
                        .filter(|(t, h)| now - t < 20.0 && *h != call && crate::callsign::morse_distance(h, &call) <= 2)
                        .map(|(_, h)| h.as_str())
                        .collect();
                    if !group.is_empty() {
                        group.push(&call);
                        if let Some(r) = crate::callsign::repeat_vote(&group).filter(|r| *r == call) {
                            let drop: Vec<String> = group.iter().filter(|g| **g != r).map(|g| g.to_string()).collect();
                            ch.call_counts.retain(|c, _| !drop.contains(c));
                            spots.retain(|s| !(s.channel == ch.id && !s.confirmed && drop.contains(&s.call)));
                            call = r;
                            repeated = true;
                        }
                    }
                }
                // A letter glued on the front of a call (WN4DW for N4DW, EN2MA
                // for N2MA: a stray dit, the previous word's tail) or lost
                // from it (T7FT for YT7FT): one station. Its sightings are
                // pooled under the copy heard more often by itself, the
                // shorter at a tie (nine of ten such pairs on the CWT and
                // night sets were glue), and the pool moves when the other
                // copy overtakes it ("DE EA5GS" eighteen times after one
                // A5GS cut out of a garble).
                let front = |a: &str, b: &str| a.len() == b.len() + 1 && a.ends_with(b) && a.as_bytes()[0].is_ascii_alphabetic();
                // (Or a call confirmed elsewhere on the band: EK0II on W0ABE's
                // channel for K0II, a caller spotted on N8KH's.)
                if !ch.call_counts.contains_key(&call)
                    && !ch.glue_own.contains_key(&call)
                    && spots.iter().any(|s| s.confirmed && s.channel != ch.id && front(&call, &s.call) && crate::callsign::standard_shape(&s.call))
                {
                    call = call[1..].to_string();
                }
                if let Some(other) = ch.call_counts.keys().chain(ch.glue_own.keys()).find(|c| front(c, &call) || front(&call, c)).cloned() {
                    let other_own = ch.glue_own.get(&other).copied().unwrap_or_else(|| ch.call_counts.get(&other).copied().unwrap_or(0));
                    ch.glue_own.insert(other.clone(), other_own);
                    let own = ch.glue_own.entry(call.clone()).or_insert(0);
                    *own += 1;
                    let own = *own;
                    let keep_other = other_own > own || (other_own == own && other.len() < call.len());
                    if std::env::var_os("CWSK_NOTES").is_some() {
                        eprintln!("GLUE {:.1} Hz: {call} ({own}) vs {other} ({other_own}) -> {}", ch.demod.freq, if keep_other { &other } else { &call });
                    }
                    let loser = if keep_other { std::mem::replace(&mut call, other) } else { other };
                    let k = ch.call_counts.remove(&loser).unwrap_or(0);
                    *ch.call_counts.entry(call.clone()).or_insert(0) += k;
                    spots.retain(|s| !(s.channel == ch.id && s.call == loser));
                }
                // A call cut short by a letter gap ("VA3R A" for VA3RA, "M5T
                // XJ"): joined with the next word if that makes a call and
                // the next word is not one of its own (E, T, K, R, N, ES...).
                let cut = !crate::callsign::standard_shape(&call) || ch.call_counts.get(&call).is_none_or(|&n| n == 0);
                let n = ch.call_counts.entry(call.clone()).or_insert(0);
                *n += 1;
                if voted.is_some() {
                    *n = (*n).max(2);
                }
                // (Contest and activation calls: "TU K1GU" from the running
                // station, "CQ CWT K1AJ", "CQ POTA W3PYF".)
                let cq_word = ch.prev2_word == "CQ" && (2..=5).contains(&ch.prev_word.len()) && ch.prev_word.bytes().all(|c| c.is_ascii_uppercase());
                let strong_ctx = repeated || signing || cq_word || matches!(ch.prev_word.as_str(), "DE" | "CQ" | "TEST" | "TU");
                if strong_ctx && !repeated && call.len() <= 4 && !crate::callsign::standard_shape(&call) {
                    ch.pending_join = Some((call.clone(), false));
                } else if cut && !strong_ctx {
                    // (Not a call that came signing or called: "DE W3IQ W M"
                    // is W3IQ and a stray letter.)
                    ch.pending_join = Some((call.clone(), true));
                }
                if *n >= 2 || strong_ctx {
                    Self::spot(ch, call, now, spots, strong_ctx);
                } else {
                    ch.pending_call = Some(call);
                }
            }
            Self::exchange_context(ch, &w, now, spots);
            ch.prev2_word = std::mem::replace(&mut ch.prev_word, w);
        }
    }

    /// A call answered with a contest exchange is being addressed, as
    /// good a sighting as one before DE: the running station sends
    /// "K0II KEN 3910" to each caller (its exchange repeats, the caller's
    /// call does not), or "JO7INF 5NN 25". The caller's call is a word
    /// already recognised here, and the exchange one heard before (any
    /// NAME NUMBER pair once could be text; the first caller is not
    /// spotted on the strength of a later repeat, it was two busts for
    /// no real call).
    fn exchange_context(ch: &mut Channel, w: &str, now: f64, spots: &mut Vec<Spot>) {
        let w = w.trim_matches(|c: char| !c.is_ascii_alphanumeric());
        let prev = ch.prev_word.trim_matches(|c: char| !c.is_ascii_alphanumeric()).to_string();
        let prev2 = ch.prev2_word.trim_matches(|c: char| !c.is_ascii_alphanumeric()).to_string();
        let recognised = |c: &str| ch.call_counts.contains_key(c) && parse_callsign(c).is_some();
        let name_ok = |s: &str| {
            (2..=6).contains(&s.len())
                && s.bytes().all(|c| c.is_ascii_uppercase())
                && !matches!(
                    s,
                    "DE" | "CQ" | "TU" | "TEST" | "CWT" | "KN" | "BK" | "AR" | "SK" | "AGN" | "QRL" | "QRZ" | "RST" | "NR" | "ES" | "PSE" | "TNX" | "TKS" | "UR" | "QSL" | "QTH" | "GM" | "GA" | "GE" | "GN" | "HR" | "POTA" | "SOTA" | "DX" | "QRS" | "QRQ" | "AS" | "GL" | "OM" | "FB" | "HI" | "CUL" | "CL" | "BT" | "GUD" | "VY" | "NN" | "TT" | "AA" | "ANT" | "RIG" | "WX" | "PWR" | "NAME" | "SRI" | "HW" | "QSO" | "QRM" | "QRN" | "QSB"
                )
        };
        let nr_ok = |s: &str| (2..=5).contains(&s.len()) && s.bytes().all(|c| c.is_ascii_alphanumeric()) && s.bytes().any(|c| c.is_ascii_digit());
        // "<call> 5NN": the report answers the caller.
        let report = (w.starts_with("5NN") || w.starts_with("599")) && w.len() <= 6 && w[3..].bytes().all(|c| c.is_ascii_digit() || c == b'N' || c == b'T');
        if report && recognised(&prev) {
            Self::spot(ch, prev, now, spots, true);
            return;
        }
        let (name, cand) = if name_ok(&prev) && nr_ok(w) {
            (prev.clone(), prev2.clone())
        } else {
            // The name and number run together ("JOE2884").
            let k = w.bytes().position(|c| c.is_ascii_digit()).unwrap_or(0);
            if k > 0 && name_ok(&w[..k]) && nr_ok(&w[k..]) && parse_callsign(w).is_none() {
                (w[..k].to_string(), prev.clone())
            } else {
                return;
            }
        };
        let e = ch.exch.entry(name).or_insert(0);
        *e += 1;
        if *e >= 2 && recognised(&cand) {
            Self::spot(ch, cand, now, spots, true);
        }
    }

    /// The callsigns in a decoded word: the word itself, or two calls run
    /// together where one of them is already known ("KA4WLXE71A"), maybe
    /// with DE, BK or K between them ("XE1FRFBKH6KB": XE1FRF DE KH6KB, DE
    /// run together reads as B). The flag: signing after a DE in the word.
    fn calls_in(w: &str, ch: &Channel, spots: &[Spot]) -> Vec<(String, bool)> {
        let ok = |c: &str| parse_callsign(c).filter(|c| crate::callsign::valid_prefix(c));
        if let Some(c) = ok(w) {
            return vec![(c, false)];
        }
        // (After a prosign run on in front: "<SK>XE1FRF...".)
        let w = w.trim_start_matches(|c: char| !c.is_ascii_alphanumeric());
        // A signing word run on in front at speed ("TESTW5TM", "CQK1AJ"):
        // the call, counted as heard but not as a clean single sighting
        // (the gap lost, the letters may be too).
        for sep in ["CQ", "DE", "TEST", "TU"] {
            if let Some(y) = w.strip_prefix(sep).and_then(ok).filter(|y| y.len() >= 4 && crate::callsign::standard_shape(y)) {
                return vec![(y, false)];
            }
        }
        let known = |c: &str| ch.call_counts.get(c).is_some_and(|&n| n >= 2) || spots.iter().any(|s| s.call == c);
        let splits = || (3..w.len().saturating_sub(2)).filter(|&k| w.is_char_boundary(k)).map(|k| (&w[..k], &w[k..]));
        // A known or whole call, a turn-over word, a whole call (first:
        // BKH6KB is call shaped too).
        for (a, b) in splits() {
            if let Some(x) = ok(a).filter(|x| known(x) || x.len() >= 4 && crate::callsign::standard_shape(x)) {
                for (sep, de) in [("DE", true), ("B", true), ("BK", false), ("K", false)] {
                    if let Some(y) = b.strip_prefix(sep).and_then(ok).filter(|y| y.len() >= 4 && crate::callsign::standard_shape(y)) {
                        return vec![(x, false), (y, de)];
                    }
                }
            }
        }
        // (Two full-length calls rarely come out of noise by chance.) Of
        // several ways to cut, the one with the known call(s), then two
        // calls from one country (a station repeating its call, a QSO
        // partner nearby), then the longer first call: "W0ABEK0II" is
        // W0ABE K0II, not W0AB EK0II; "EA5BSEA5GS" EA5BS EA5GS, not
        // EA5BSE A5GS.
        let same_country = |x: &str, y: &str| crate::callsign::entity(x).is_some_and(|e| crate::callsign::entity(y).as_deref() == Some(e.as_str()));
        splits()
            .filter_map(|(a, b)| match (ok(a), ok(b)) {
                (Some(x), Some(y)) if known(&x) || known(&y) || (x.len() >= 4 && y.len() >= 4) => Some((x, y)),
                _ => None,
            })
            .max_by_key(|(x, y)| (u8::from(known(x)) + u8::from(known(y)), same_country(x, y), x.len()))
            .map(|(x, y)| vec![(x, false), (y, false)])
            .unwrap_or_default()
    }

    /// Records a sighting of `call` on this channel as a spot; `strong`:
    /// it was signing or being called (after DE/CQ, before DE/K).
    fn spot(ch: &mut Channel, call: String, now: f64, spots: &mut Vec<Spot>, strong: bool) {
        // Twice on this channel, or already spotted earlier (a net control
        // is heard every few minutes, longer than a channel remembers its
        // calls). Or once, signing or being called, on clean keying, with
        // the shape of a real call and not a near copy of another call
        // heard this session: CW Skimmer spots such single sightings too
        // (it checks them against a call database; we use only the shape).
        let n = ch.call_counts.get(&call).copied().unwrap_or(1);
        let lookalike = |c: &str| c != call && (crate::callsign::morse_distance(c, &call) <= 2 || (c.len() >= 3 && (c.contains(call.as_str()) || call.contains(c))));
        // Same first four characters give or take an element or two: a
        // partial or busted copy (DL1BM next to DL1DGR, E71TDX next to E71A).
        let head = |c: &str, max: usize| c != call && c.len() >= 4 && call.len() >= 4 && crate::callsign::morse_distance(&c[..4], &call[..4]) <= max;
        // Strong enough to be copied right: on a noisy night band, single
        // sightings under 20 dB were two garbles for every real call; a
        // short channel with nothing but the call and CQ/DE on it (no text
        // to judge, `plaus` None) is clean down to 12 dB (20 m by day).
        let snr = ch.decoder.snr_db();
        let strong_enough = snr >= SINGLE_SPOT_SNR || (ch.plaus.is_none() && snr >= SINGLE_SPOT_SNR_CLEAN);
        let clean_single = strong
            && strong_enough
            && crate::callsign::standard_shape(&call)
            && ch.decoder.fit().0 >= SINGLE_SPOT_FIT
            && ch.decoder.wpm().is_some_and(|w| w <= SINGLE_SPOT_MAX_WPM)
            && !spots.iter().any(|s| lookalike(&s.call) || head(&s.call, 1))
            && !ch.call_counts.keys().any(|c| lookalike(c) || head(c, 2));
        let confirmed = n >= 2 || spots.iter().any(|s| s.call == call) || clean_single;
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
        Self::drop_variants(spots);
    }

    /// Drops spots that are garbled copies of an established call: a
    /// fragment of it (XE1F for XE1FRF), it with a letter run on (K9UIYR),
    /// or within two Morse elements (E7AOA for E71A). Unconfirmed spots go
    /// if such a call is confirmed; a confirmed one heard once if the other
    /// call is confirmed and heard twice, others only if the other call has
    /// been heard four times as often. Real stations rarely have calls
    /// that similar; a sighting that is one is almost always a misread.
    fn drop_variants(spots: &mut Vec<Spot>) {
        let variant = |a: &str, b: &str| {
            a != b && ((a.len() >= 3 && (b.contains(a) || a.contains(b))) || crate::callsign::morse_distance(a, b) <= 2)
        };
        // Against established calls anywhere on the band, not only nearby:
        // in a pileup the callers send the DX call too, each with its own
        // copying errors (E71A heard as E71EE, E7AOA, XE71A ...).
        let all: Vec<(String, u32, bool)> = spots.iter().map(|s| (s.call.clone(), s.count, s.confirmed)).collect();
        spots.retain(|s| {
            !all.iter().any(|(c, n, conf)| {
                c != &s.call
                    && variant(&s.call, c)
                    && ((!s.confirmed && (*conf || *n >= 2)) || (s.confirmed && s.count == 1 && *conf && *n >= 2) || (s.confirmed && *n >= 4 && *n >= 4 * s.count))
            })
        });
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
        // Muted audio (the radio mutes the receiver while transmitting, a
        // recording has a gap of digital silence): everything 30 dB under
        // the noise floor. Keep the noise floors through it, or they
        // collapse and the first seconds after it look 70 dB above the
        // noise. (Only the floors: an MP3 has digital silence between
        // elements, and the rest must keep tracking.)
        self.muted = self.hops > 30 && med < self.noise_floor * 1e-3 && top < self.noise_floor;
        self.muted_hops = if self.muted { self.muted_hops + 1 } else { 0 };
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
        if !self.muted {
            self.noise_floor = 0.9 * self.noise_floor + 0.1 * nf;
        }
        let nf = self.noise_floor;
        // Local floor: medians of 16-bin blocks, interpolated between block
        // centres, smoothed like the overall floor.
        const BLK: usize = 16;
        let meds: Vec<(f32, f32)> = (lo..hi)
            .step_by(BLK)
            .map(|b0| {
                let b1 = (b0 + BLK).min(hi);
                let mut v: Vec<f32> = self.fast[b0..b1].to_vec();
                v.sort_unstable_by(|a, b| a.total_cmp(b));
                ((b0 + b1) as f32 / 2.0, v[v.len() / 2].max(1e-15))
            })
            .collect();
        for b in lo..hi {
            let x = b as f32 + 0.5;
            let k = meds.partition_point(|m| m.0 <= x);
            let m = if k == 0 {
                meds[0].1
            } else if k == meds.len() {
                meds[k - 1].1
            } else {
                let (a, c) = (meds[k - 1], meds[k]);
                a.1 + (c.1 - a.1) * (x - a.0) / (c.0 - a.0)
            };
            if !self.muted {
                self.local_floor[b] = 0.9 * self.local_floor[b] + 0.1 * m;
            }
        }
        // Detection and the activity gate: never above the overall floor
        // (finds weak signals on the skirts without making a noisy stretch
        // of band harder).
        let floors: Vec<f32> = self.local_floor.iter().map(|l| l.clamp(nf * 0.1, nf)).collect();
        let floor = |b: usize| floors[b];

        for ch in &mut self.channels {
            let b = ((ch.demod.freq as f32 / bin_hz()).round() as usize).min(p.len() - 1);
            let local = self.local_floor[b].max(nf * 0.1);
            ch.level_db = 10.0 * (self.peak[b] / floor(b)).log10();
            // Level over the noise right around the signal: in the middle
            // of the receiver's passband the noise is several dB above the
            // overall median, and this picks the envelope filter.
            ch.local_db = 10.0 * (self.peak[b] / local).log10();
            // Activity gate: energy at the channel 9 dB (with the default
            // detection threshold) over the noise. Against the overall
            // floor: in the middle of the passband the local one would gate
            // off a faded straight-key signal's marks.
            let r = self.recent[b.saturating_sub(1)..(b + 2).min(p.len())].iter().fold(0.0f32, |m, &v| m.max(v));
            let gate = 10f32.powf((self.cfg.detect_db - 3.0) / 10.0);
            ch.active = r > floor(b) * gate;
            ch.decoder.set_gate(ch.active);
        }

        if !self.cfg.auto_detect {
            return;
        }
        let det = 10f32.powf(self.cfg.detect_db / 10.0);
        let hz = bin_hz();
        // A peak next to a strong channel of comparable level (the pair a
        // crowded channel is made of, see `add_channel`) is not interpolated:
        // the neighbour's skirt pulls the estimate toward it (two stations
        // 45 Hz apart came out 20 Hz apart, both wrong).
        let strong: Vec<(usize, f32)> = self.channels.iter().filter(|c| c.level_db >= CROWDED_START_DB).map(|c| ((c.demod.freq as f32 / hz).round() as usize, c.level_db)).collect();
        for b in lo..hi {
            let pk = self.peak[b];
            let nf = floor(b);
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
            let lvl = 10.0 * (pk / nf).log10();
            let paired = lvl >= CROWDED_START_DB && strong.iter().any(|&(t, l)| (2..=4).contains(&t.abs_diff(b)) && (l - lvl).abs() <= CROWDED_DIFF_DB);
            let off = if den.abs() > 1e-9 && !paired { 0.5 * (a - c) / den } else { 0.0 };
            let f = (b as f32 + off.clamp(-0.5, 0.5)) * hz;
            if self.suppressed.iter().any(|&(sf, _)| (sf - f).abs() < self.cfg.min_spacing) {
                continue;
            }
            if self.channels.iter().any(|ch| (ch.demod.freq as f32 - f).abs() < self.cfg.min_spacing) {
                continue;
            }
            // Ignore key clicks / skirts of much stronger nearby signals.
            // Judge the strong signal by its level now (fast decay), not
            // its peak hold: clicks and skirts exist only while it keys,
            // and in a QSO the other station sends while it is silent
            // (W2GLN 165 Hz from a 47 dB K9UIY was never picked up).
            let recent = &self.recent;
            if self.channels.iter().any(|ch| {
                let df = (ch.demod.freq as f32 - f).abs();
                let cb = ((ch.demod.freq as f32 / hz).round() as usize).clamp(1, recent.len() - 2);
                let now_db = 10.0 * (recent[cb - 1..=cb + 1].iter().fold(0.0f32, |m, &v| m.max(v)) / nf).log10();
                df < 200.0 && ch.level_db.min(now_db) - lvl > 20.0 + df / 10.0
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
                        Self::dropping(&mut c, self.time, &mut self.events, &mut self.spots, "replaced by a stronger signal (channel limit)");
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
        let mut why = vec![String::new(); n];
        // Removed for keying against the strong signal (marks in its gaps).
        let mut anti = vec![false; n];
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
                // (Not two strong channels created as neighbours: those are
                // two stations by construction, each heard through a narrow
                // demodulator, and in a QSO they key in turn.)
                let twin = df < 80.0 && strong.level_db >= weak.level_db && !(strong.crowd_pair && weak.crowd_pair);
                if wi == si || dead[wi] || weak.pinned || (!twin && strong.level_db < weak.level_db + 10.0) {
                    continue;
                }
                let wl = weak.marks.len().min(len);
                let (wo, so) = (weak.marks.len() - wl, len - wl);
                let mut tot = 0u32;
                let mut inside = 0u32;
                let mut covered = 0u32;
                for i in 0..wl {
                    let c = dil(so + i);
                    covered += u32::from(c);
                    if weak.marks[wo + i] {
                        tot += 1;
                        if c {
                            inside += 1;
                        }
                    }
                }

                if std::env::var_os("CWSK_DEBUG_ART").is_some() && tot >= 100 {
                    eprintln!("ART t={:.0} weak {:.0}Hz ({:.0}dB) vs strong {:.0}Hz ({:.0}dB): inside {:.2} ampcorr {:.2}", self.time, weak.demod.freq, weak.level_db, strong.demod.freq, strong.level_db, inside as f32 / tot as f32, amp_corr(&strong.amps, &weak.amps, &strong.marks));
                }
                let corr = if tot >= 100 { lagged_corr(&strong.marks, &weak.marks) } else { 0.0 };
                if std::env::var_os("CWSK_DEBUG_ART").is_some() && tot >= 100 {
                    eprintln!("    corr {corr:.2} covered {:.2}", covered as f32 / wl.max(1) as f32);
                }
                let much_stronger = strong.level_db >= weak.level_db + 10.0;
                let inside_frac = inside as f32 / tot.max(1) as f32;
                // A busy strong signal's (dilated) marks cover most of the
                // time, so many of a real station's marks land inside them by
                // chance; an artefact's marks avoid the strong signal's gaps
                // almost completely.
                // Skirts, clicks and AGC junk sit near the strong signal or key
                // badly; a station far away with good Morse timing is real
                // even if it happens to send at the same time (weak1: a
                // 2170 Hz station 1.5 kHz from a QSO, 93% "inside").
                let far_and_clean = df > 400.0 && weak.decoder.fit().0 >= MIN_FIT;
                // Against a busy strong signal "inside" means little: with
                // its dilated marks covering 90% of the time, a station
                // keying on its own lands 90% inside. An artefact misses the
                // strong signal's gaps far more completely than chance, or
                // sits close by, or fades with it (1458 Hz under a 1206 Hz
                // ragchew, 95% inside at 90% cover: dropped for good).
                let cov_frac = covered as f32 / wl.max(1) as f32;
                let avoids_gaps = df <= 200.0
                    || (tot >= 100 && amp_corr(&strong.amps, &weak.amps, &strong.marks) > 0.15)
                    || 1.0 - inside_frac < 0.3 * (1.0 - cov_frac);
                // The receiver's AGC turns the gain down while the strong
                // station keys, so band noise elsewhere dips in step with
                // it: inverted "keying" that decodes as junk. Its envelope
                // moves against the strong one's; a real weak station's own
                // keying dominates its envelope instead. (On 40 m at night
                // next to W1AW: dozens of junk channels, all -0.24..-0.45.)
                // Only a channel down at the noise: the AGC makes "keying"
                // out of noise, not out of a station 10-20 dB under the
                // strong one, whose envelope also dips with its marks. (A
                // CWT: K4NMR at 40 dB beside KG9X at 52 was dropped for a
                // minute at a time, and QSO partners at home likewise.)
                let agc = much_stronger && weak.level_db < AGC_JUNK_DB && tot >= 100 && amp_corr(&strong.amps, &weak.amps, &strong.marks) < -0.25;
                if tot >= 100 && ((much_stronger && ((inside_frac > 0.92 && avoids_gaps && !far_and_clean) || corr > 0.5)) || (twin && corr > 0.7) || agc) {
                    dead[wi] = true;
                    anti[wi] = inside_frac < 0.3 && !agc && weak.qualified;
                    let rule = if agc { "agc" } else if twin && corr > 0.7 { "twin" } else if inside_frac > 0.92 { "inside" } else { "corr" };
                    why[wi] = format!(" ({rule}, {:.0} Hz at {:.0} dB)", strong.demod.freq, strong.level_db);
                }
            }
        }
        let mut i = 0;
        let (now, events, spots) = (self.time, &mut self.events, &mut self.spots);
        self.channels.retain_mut(|c| {
            let keep = !dead[i];
            i += 1;
            if !keep {
                // An artefact found young is kept away for a minute; a
                // channel that lived and was shown is a station's, caught
                // idling between overs (its noise keyed by a neighbour),
                // and the next station on that frequency must not be
                // locked out (K8PK, 33 s after KO4VW's over).
                // Likewise a channel keying in the strong signal's gaps: that
                // is the AGC pumping an idle frequency, not an artefact made
                // by the strong signal, and the station that was there may
                // come back out of a fade (KB1FGC, locked out mid-over).
                let hold = if (c.qualified && now - c.created > 20.0) || anti[i - 1] { 10.0 } else { 60.0 };
                Self::dropping(c, now, events, spots, &format!("artifact or duplicate of a stronger signal{}", why[i - 1]));
                self.suppressed.push((c.demod.freq as f32, now + hold));
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
            if runs >= 20 {
                c.best_fit = c.best_fit.max(fit);
            }
            if !c.pinned && c.qualified && runs >= 20 && fit < MIN_FIT - 0.13 && !Self::holding(c, now, self.cfg.dropout_s) {
                Self::note(c, now, &mut self.events, "hidden: keying no longer fits Morse timing (housekeeping)");
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
                Self::dropping(c, now, events, spots, "timed out");
            }
            if !keep && c.qualified {
                retired.push(Retired { id: c.id, freq: c.demod.freq as f32, at: now, text: c.text.clone(), call_counts: c.call_counts.clone(), heard: c.heard.clone() });
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
                    // Keep the established channel: pinned, then shown, then
                    // the older. (Not the one with more marks: a skirt channel
                    // walked onto a strong signal by its AFC can split every
                    // dash into dits, ARRL 15 WPM.)
                    let rank = |c: &Channel| (c.pinned, c.qualified, -c.created);
                    let drop_j = rank(a).partial_cmp(&rank(b)).is_some_and(|o| o.is_ge());
                    let k = if drop_j { j } else { i };
                    let mut c = self.channels.remove(k);
                    Self::dropping(&mut c, now, &mut self.events, &mut self.spots, "merged with a channel on the same signal");
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

    /// Everything about every channel, shown or not, for the recording's
    /// display log: what the decoder measured and why the channel is or
    /// is not shown.
    pub fn debug_json(&self) -> Vec<serde_json::Value> {
        self.channels
            .iter()
            .map(|c| {
                let (fit, runs) = c.decoder.fit();
                let (el, dash, _) = text_quality(&c.text);
                let t = c.decoder.timing();
                let tail: String = c.text.chars().rev().take(120).collect::<Vec<_>>().into_iter().rev().collect();
                serde_json::json!({
                    "id": c.id,
                    "audio_hz": c.demod.freq,
                    "level_db": c.level_db,
                    "snr_db": c.decoder.snr_db(),
                    "active": c.active,
                    "squelch_open": c.decoder.squelch_open(),
                    "shown": c.qualified,
                    "pinned": c.pinned,
                    "fit": fit,
                    "runs": runs,
                    "wpm": c.decoder.wpm(),
                    "dot_ms": t.map(|t| t.dot),
                    "dash_ms": t.map(|t| t.dash),
                    "egap_ms": t.map(|t| t.egap),
                    "cgap_ms": t.map(|t| t.cgap),
                    "wgap_ms": t.map(|t| t.wgap()),
                    "el_per_char": el,
                    "dash_frac": dash,
                    "plaus": c.plaus,
                    "doubtful": c.plaus.is_some_and(|p| p < DOUBTFUL_PLAUS),
                    "age_s": self.time - c.created,
                    "last_mark_s": self.time - c.last_mark,
                    "last_good_s": self.time - c.last_good,
                    "start_redecodes": c.start_redecodes,
                    "text": tail,
                    "tentative": c.decoder.tentative(),
                })
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
        assert_eq!(split_doubled_calls("CQ DEN4XYZ K"), "CQ DE N4XYZ K");
        assert_eq!(split_doubled_calls("DENVER DEPARTED"), "DENVER DEPARTED");
        assert_eq!(split_doubled_calls("CQCQCQCQDE XE1FRF"), "CQ CQ CQ CQ DE XE1FRF");
        assert_eq!(split_doubled_calls("CQDXE71A E71A"), "CQ DX E71A E71A");
        assert_eq!(split_doubled_calls("QRZ? DXE71A"), "QRZ? DX E71A");
        assert_eq!(split_doubled_calls("DX1ABC CQD"), "DX1ABC CQD");
        assert_eq!(split_doubled_calls("DE XE1FRF XE1FRFK"), "DE XE1FRF XE1FRF K");
        assert_eq!(split_doubled_calls("DE W1AK"), "DE W1AK");
    }
}
