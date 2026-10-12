//! Single-channel probabilistic CW decoder.
//!
//! Pipeline (per channel):
//!
//! 1. **Level tracking / slicing** – the envelope (one value per frame, ~2 ms)
//!    is compared with an adaptive threshold placed between a noise-floor
//!    quantile tracker and a signal-level tracker, with hysteresis, squelch
//!    and a speed-dependent debounce.
//! 2. **Run builder** – the binary signal becomes alternating mark/space
//!    durations.
//! 3. **Timing model** – dot/dash/gap lengths are bootstrapped by a maximum
//!    likelihood search over speeds, then tracked adaptively so hand-sent
//!    "fists" with odd weighting and Farnsworth spacing are followed.
//! 4. **Beam search** – every mark/space is scored under each possible
//!    interpretation (dot/dash; element/character/word gap) with log-normal
//!    likelihoods. Extra hypotheses cover a mark being a noise burst and a
//!    space being a QSB drop-out inside an element. Completed characters are
//!    scored by a trigram language model of CW traffic. Text is committed
//!    once all surviving hypotheses agree on it.

use crate::lm::LanguageModel;
use crate::morse::{MorseLookup, MAX_ELEMS};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct DecoderParams {
    /// Weight of the language model relative to timing likelihoods.
    pub lm_weight: f32,
    pub beam_width: usize,
    /// Log-domain standard deviation for mark lengths.
    pub sigma_mark: f32,
    /// Log-domain standard deviation for space lengths.
    pub sigma_space: f32,
    /// Log-probability penalty for treating a mark as a noise burst.
    pub blip_penalty: f32,
    /// Log-probability penalty for treating a space as a fade drop-out.
    pub dropout_penalty: f32,
    /// Squelch: minimum signal/noise amplitude ratio to open.
    pub squelch_open: f32,
    pub squelch_close: f32,
    /// Threshold position between noise (0) and signal (1).
    pub threshold: f32,
    /// Per-frame decay of the signal level toward the noise floor
    /// (lets the threshold follow QSB fades).
    pub sig_decay: f32,
    /// Demodulator bandwidth as a multiple of 1/dot.
    pub bw_factor: f32,
}

impl DecoderParams {
    /// Defaults with overrides from `CWSK_PARAMS="name=value,..."` (tuning aid).
    pub fn from_env() -> Self {
        let mut p = Self::default();
        if let Ok(s) = std::env::var("CWSK_PARAMS") {
            for kv in s.split(',') {
                let mut it = kv.splitn(2, '=');
                let (Some(k), Some(v)) = (it.next(), it.next()) else { continue };
                let Ok(v) = v.trim().parse::<f32>() else { continue };
                match k.trim() {
                    "lm_weight" => p.lm_weight = v,
                    "beam_width" => p.beam_width = v as usize,
                    "sigma_mark" => p.sigma_mark = v,
                    "sigma_space" => p.sigma_space = v,
                    "blip_penalty" => p.blip_penalty = v,
                    "dropout_penalty" => p.dropout_penalty = v,
                    "squelch_open" => p.squelch_open = v,
                    "squelch_close" => p.squelch_close = v,
                    "threshold" => p.threshold = v,
                    "sig_decay" => p.sig_decay = v,
                    "bw_factor" => p.bw_factor = v,
                    _ => eprintln!("unknown param {k}"),
                }
            }
        }
        p
    }
}

impl Default for DecoderParams {
    fn default() -> Self {
        Self {
            lm_weight: 0.7,
            beam_width: 48,
            sigma_mark: 0.30,
            sigma_space: 0.38,
            blip_penalty: -5.0,
            dropout_penalty: -5.0,
            squelch_open: 2.3,
            squelch_close: 1.9,
            threshold: 0.5,
            sig_decay: 0.0015,
            bw_factor: 2.8,
        }
    }
}

/// Element timing estimates in milliseconds.
#[derive(Clone, Debug)]
pub struct TimingModel {
    pub dot: f32,
    pub dash: f32,
    pub egap: f32,
    pub cgap: f32,
}

impl TimingModel {
    fn from_dot(dot: f32) -> Self {
        Self { dot, dash: 3.0 * dot, egap: dot, cgap: 3.0 * dot }
    }
    pub fn wgap(&self) -> f32 {
        // Word gaps scale with character gaps (keeps Farnsworth consistent).
        (self.cgap * 7.0 / 3.0).max(self.dot * 5.0)
    }
    pub fn wpm(&self) -> f32 {
        1200.0 / self.dot
    }
}

#[inline]
fn lognorm(x: f32, center: f32, sigma: f32) -> f32 {
    let z = (x.max(1e-3) / center).ln() / sigma;
    -0.5 * z * z - sigma.ln()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Run {
    Mark(f32),
    Space(f32),
    /// Long silence: finish the current word.
    Flush,
}

#[derive(Clone)]
struct Hyp {
    score: f32,
    bits: u8,
    len: u8,
    pend_mark: f32,
    pend_space: f32,
    ctx: [u8; 2],
    out: Vec<u8>,
}

impl Hyp {
    fn key(&self) -> (u8, u8, u16, u16, [u8; 2]) {
        (
            self.bits,
            self.len,
            (self.pend_mark * 0.5) as u16,
            (self.pend_space * 0.5) as u16,
            self.ctx,
        )
    }
}

/// Beam-search decoder over mark/space durations.
pub struct BeamDecoder {
    pub params: DecoderParams,
    pub timing: TimingModel,
    lm: Arc<LanguageModel>,
    morse: MorseLookup,
    hyps: Vec<Hyp>,
    committed: Vec<u8>,
    scratch: Vec<Hyp>,
}

impl BeamDecoder {
    pub fn new(params: DecoderParams, lm: Arc<LanguageModel>, timing: TimingModel) -> Self {
        Self {
            params,
            timing,
            lm,
            morse: MorseLookup::new(),
            hyps: vec![Self::root()],
            committed: Vec::new(),
            scratch: Vec::new(),
        }
    }

    fn root() -> Hyp {
        Hyp { score: 0.0, bits: 0, len: 0, pend_mark: 0.0, pend_space: 0.0, ctx: [b' ', b' '], out: Vec::new() }
    }

    /// Takes text committed so far.
    pub fn take_committed(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.committed)
    }

    /// Text of the best hypothesis not yet committed (for live display).
    pub fn tentative(&self) -> String {
        self.hyps.first().map(|h| String::from_utf8_lossy(&h.out).into_owned()).unwrap_or_default()
    }

    fn mark_options(&self, m: f32) -> [(bool, f32); 2] {
        let t = &self.timing;
        let s = self.params.sigma_mark;
        let dot = lognorm(m, t.dot, s);
        // Over-long dashes are common with hand keys: widen the upper tail.
        let dash = if m > t.dash { lognorm(m, t.dash, s * 1.8) + (1.8f32).ln() } else { lognorm(m, t.dash, s) };
        [(false, dot), (true, dash)]
    }

    /// Returns log-likelihoods for (element gap, character gap, word gap).
    fn space_options(&self, sp: f32) -> [f32; 3] {
        let t = &self.timing;
        let s = self.params.sigma_space;
        let wg = t.wgap();
        let e = lognorm(sp, t.egap, s) + (0.55f32).ln();
        let c = lognorm(sp, t.cgap, s) + (0.33f32).ln();
        let w = if sp >= wg { -s.ln() } else { lognorm(sp, wg, s) } + (0.12f32).ln();
        [e, c, w]
    }

    fn emit(&self, h: &mut Hyp, c: u8) {
        let lp = self.lm.logp(h.ctx[0], h.ctx[1], c);
        h.score += self.params.lm_weight * (lp + self.lm.char_bonus);
        h.ctx = [h.ctx[1], c];
        h.out.push(c);
    }

    /// Ends the current character; returns false if the elements are invalid.
    fn end_char(&self, h: &mut Hyp, word: bool) -> bool {
        if h.len > 0 {
            match self.morse.get(h.bits, h.len) {
                Some(c) => self.emit(h, c),
                None => return false,
            }
            h.bits = 0;
            h.len = 0;
        }
        if word && h.ctx[1] != b' ' {
            self.emit(h, b' ');
        }
        true
    }

    pub fn push(&mut self, run: Run) {
        let mut next = std::mem::take(&mut self.scratch);
        next.clear();
        let dot = self.timing.dot;
        match run {
            Run::Mark(m) => {
                for h in &self.hyps {
                    if h.pend_space > 0.0 {
                        let opts = self.space_options(h.pend_space);
                        for (k, &ll) in opts.iter().enumerate() {
                            let mut n = h.clone();
                            n.score += ll;
                            let ok = match k {
                                0 => n.len < MAX_ELEMS,
                                1 => self.end_char(&mut n, false),
                                _ => self.end_char(&mut n, true),
                            };
                            if !ok {
                                continue;
                            }
                            n.pend_space = 0.0;
                            n.pend_mark = m;
                            next.push(n);
                        }
                        if m < 0.6 * dot {
                            let mut n = h.clone();
                            n.pend_space += m;
                            n.score += self.params.blip_penalty + 2.0 * (m / dot);
                            next.push(n);
                        }
                    } else {
                        let mut n = h.clone();
                        n.pend_mark += m;
                        next.push(n);
                    }
                }
            }
            Run::Space(sp) => {
                for h in &self.hyps {
                    if h.pend_mark > 0.0 {
                        for (dash, ll) in self.mark_options(h.pend_mark) {
                            if h.len >= MAX_ELEMS {
                                continue;
                            }
                            let mut n = h.clone();
                            n.bits = (n.bits << 1) | u8::from(dash);
                            n.len += 1;
                            if !self.morse.is_prefix(n.bits, n.len) {
                                continue;
                            }
                            n.score += ll;
                            n.pend_mark = 0.0;
                            n.pend_space = sp;
                            next.push(n);
                        }
                        if sp < 0.6 * dot {
                            let mut n = h.clone();
                            n.pend_mark += sp;
                            n.score += self.params.dropout_penalty + 2.0 * (sp / dot);
                            next.push(n);
                        }
                    } else {
                        let mut n = h.clone();
                        n.pend_space += sp;
                        next.push(n);
                    }
                }
            }
            Run::Flush => {
                for h in &self.hyps {
                    let mut cands: Vec<Hyp> = Vec::new();
                    if h.pend_mark > 0.0 {
                        for (dash, ll) in self.mark_options(h.pend_mark) {
                            if h.len >= MAX_ELEMS {
                                continue;
                            }
                            let mut n = h.clone();
                            n.bits = (n.bits << 1) | u8::from(dash);
                            n.len += 1;
                            n.score += ll;
                            cands.push(n);
                        }
                    } else {
                        cands.push(h.clone());
                    }
                    for mut n in cands {
                        n.pend_mark = 0.0;
                        n.pend_space = 0.0;
                        if self.end_char(&mut n, true) {
                            next.push(n);
                        }
                    }
                }
                if next.is_empty() {
                    // Everything was an invalid character: drop the partial.
                    let mut n = self.hyps[0].clone();
                    n.bits = 0;
                    n.len = 0;
                    n.pend_mark = 0.0;
                    n.pend_space = 0.0;
                    next.push(n);
                }
            }
        }
        if next.is_empty() {
            // Dead end (e.g. over-long element sequence): restart at best.
            let mut n = self.hyps[0].clone();
            n.bits = 0;
            n.len = 0;
            n.pend_mark = if let Run::Mark(m) = run { m } else { 0.0 };
            n.pend_space = if let Run::Space(s) = run { s } else { 0.0 };
            next.push(n);
        }
        self.scratch = std::mem::replace(&mut self.hyps, next);
        self.prune();
        if matches!(run, Run::Flush) {
            self.hyps.truncate(1);
        }
        self.commit(matches!(run, Run::Flush));
    }

    fn prune(&mut self) {
        self.hyps.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
        let best = self.hyps[0].score;
        let mut seen = std::collections::HashSet::with_capacity(self.hyps.len());
        let width = self.params.beam_width;
        self.hyps.retain(|h| h.score > best - 30.0 && seen.insert(h.key()));
        self.hyps.truncate(width);
        // Renormalise to keep scores bounded.
        for h in &mut self.hyps {
            h.score -= best;
        }
    }

    fn commit(&mut self, all: bool) {
        if all {
            let out = std::mem::take(&mut self.hyps[0].out);
            self.committed.extend_from_slice(&out);
            return;
        }
        // Common prefix of all hypotheses.
        let first = &self.hyps[0].out;
        let mut n = first.len();
        for h in &self.hyps[1..] {
            n = n.min(h.out.len());
            n = first.iter().zip(&h.out).take(n).take_while(|(a, b)| a == b).count();
            if n == 0 {
                break;
            }
        }
        // Bound latency: force-commit the best hypothesis' oldest characters.
        const MAX_PENDING: usize = 10;
        let mut forced = 0;
        if first.len().saturating_sub(n) > MAX_PENDING {
            forced = first.len() - MAX_PENDING;
        }
        let take = n.max(forced);
        if take == 0 {
            return;
        }
        let prefix: Vec<u8> = first[..take].to_vec();
        self.committed.extend_from_slice(&prefix);
        self.hyps.retain(|h| h.out.len() >= take && h.out[..take] == prefix[..]);
        for h in &mut self.hyps {
            h.out.drain(..take);
        }
    }

    /// Hard-decision classification of a run under the current timing model,
    /// used to adapt the model.
    pub fn adapt(&mut self, run: Run) {
        let a = 0.08;
        match run {
            Run::Mark(m) => {
                let opts = self.mark_options(m);
                let t = &mut self.timing;
                if opts[1].1 > opts[0].1 {
                    if m < t.dot * 6.0 {
                        t.dash += a * (m - t.dash);
                    }
                } else if m > t.dot * 0.4 {
                    t.dot += a * (m - t.dot);
                }
                // Keep the model sane.
                t.dot = t.dot.clamp(20.0, 240.0);
                t.dash = t.dash.clamp(t.dot * 2.0, t.dot * 5.0);
            }
            Run::Space(s) => {
                let o = self.space_options(s);
                let t = &mut self.timing;
                if o[0] >= o[1] && o[0] >= o[2] {
                    if s > t.dot * 0.4 {
                        t.egap += a * (s - t.egap);
                    }
                } else if o[1] >= o[2] {
                    t.cgap += a * (s - t.cgap);
                }
                t.egap = t.egap.clamp(t.dot * 0.5, t.dot * 2.0);
                t.cgap = t.cgap.clamp(t.dot * 2.0, t.dot * 15.0);
            }
            Run::Flush => {}
        }
    }
}

/// Maximum-likelihood speed estimate from a set of runs.
pub fn estimate_timing(runs: &[Run]) -> Option<TimingModel> {
    let marks: Vec<f32> = runs.iter().filter_map(|r| if let Run::Mark(m) = r { Some(*m) } else { None }).collect();
    let spaces: Vec<f32> = runs.iter().filter_map(|r| if let Run::Space(s) = r { Some(*s) } else { None }).collect();
    if marks.len() < 4 {
        return None;
    }
    let sm = 0.30f32;
    let ss = 0.40f32;
    const FLOOR: f32 = -3.0;
    let mut best = (f32::NEG_INFINITY, 0.0f32);
    // 5-60 WPM.
    let mut d = 20.0f32;
    while d < 240.0 {
        let mut score = 0.0;
        // Outliers (noise bursts, drop-outs) get a capped penalty so they
        // cannot drag the estimate.
        for &m in &marks {
            let a = lognorm(m, d, sm);
            let b = lognorm(m, 3.0 * d, sm);
            score += a.max(b).max(FLOOR);
        }
        for &s in &spaces {
            if s > 15.0 * d {
                continue;
            }
            let a = lognorm(s, d, ss);
            let b = lognorm(s, 3.0 * d, ss);
            let c = if s > 7.0 * d { -ss.ln() } else { lognorm(s, 7.0 * d, ss) };
            score += 0.6 * a.max(b).max(c).max(FLOOR);
        }
        // Mild prior toward common speeds (12-35 WPM).
        let wpm = 1200.0 / d;
        score -= 0.5 * ((wpm / 22.0).ln() / 0.8).powi(2);
        if score > best.0 {
            best = (score, d);
        }
        d *= 1.02;
    }
    let dot = best.1;
    if dot <= 0.0 {
        return None;
    }
    let mut tm = TimingModel::from_dot(dot);
    // Refine with class means.
    let mean = |v: &mut Vec<f32>| if v.len() >= 2 { Some(v.iter().sum::<f32>() / v.len() as f32) } else { None };
    let mut dots: Vec<f32> = marks.iter().copied().filter(|&m| m < 2.0 * dot).collect();
    let mut dashes: Vec<f32> = marks.iter().copied().filter(|&m| m >= 2.0 * dot && m < 6.0 * dot).collect();
    if let Some(v) = mean(&mut dots) {
        tm.dot = v;
    }
    if let Some(v) = mean(&mut dashes) {
        tm.dash = v.clamp(tm.dot * 2.0, tm.dot * 5.0);
    }
    let mut eg: Vec<f32> = spaces.iter().copied().filter(|&s| s < 2.0 * dot).collect();
    if let Some(v) = mean(&mut eg) {
        tm.egap = v.clamp(tm.dot * 0.5, tm.dot * 2.0);
    }
    // Character gaps outnumber word gaps roughly 4:1 in text, so a low
    // percentile of the long spaces is the character gap. This also copes
    // with Farnsworth spacing where gaps are far longer than 3 dots.
    let mut long: Vec<f32> = spaces.iter().copied().filter(|&s| s >= 2.0 * tm.egap.max(dot)).collect();
    if long.len() >= 3 {
        long.sort_by(|a, b| a.total_cmp(b));
        let v = long[(long.len() as f32 * 0.35) as usize];
        tm.cgap = v.clamp(tm.dot * 2.0, tm.dot * 15.0);
    }
    Some(tm)
}

/// Words made only of E and T (single dots/dashes) are what noise and
/// interference decode as; real traffic almost never contains them.
pub fn is_noise_word(w: &[u8]) -> bool {
    !w.is_empty() && w.iter().all(|&c| c == b'E' || c == b'T')
}

/// Converts a sampled envelope into decoded text.
pub struct CwDecoder {
    pub frame_ms: f32,
    params: DecoderParams,
    lm: Arc<LanguageModel>,
    // level tracking (linear amplitude)
    noise_q: f32,
    sig: f32,
    level_frames: u32,
    init_buf: Vec<f32>,
    squelch_open: bool,
    gate: bool,
    erasure: bool,
    // slicer / run builder
    state: bool,
    run_frames: u32,
    cand: u32,
    flushed: bool,
    // timing
    boot: Vec<Run>,
    history: std::collections::VecDeque<Run>,
    marks_since_check: u32,
    beam: Option<BeamDecoder>,
    text: Vec<u8>,
    word: Vec<u8>,
    pub frames_since_mark: u64,
    pub total_marks: u64,
}

impl CwDecoder {
    pub fn new(frame_ms: f32, params: DecoderParams, lm: Arc<LanguageModel>) -> Self {
        Self {
            frame_ms,
            params,
            lm,
            noise_q: 0.0,
            sig: 0.0,
            level_frames: 0,
            init_buf: Vec::new(),
            squelch_open: false,
            gate: true,
            erasure: false,
            state: false,
            run_frames: 0,
            cand: 0,
            flushed: true,
            boot: Vec::new(),
            history: Default::default(),
            marks_since_check: 0,
            beam: None,
            text: Vec::new(),
            word: Vec::new(),
            frames_since_mark: 0,
            total_marks: 0,
        }
    }

    pub fn params(&self) -> &DecoderParams {
        &self.params
    }

    pub fn timing(&self) -> Option<&TimingModel> {
        self.beam.as_ref().map(|b| &b.timing)
    }

    pub fn wpm(&self) -> Option<f32> {
        self.timing().map(|t| t.wpm())
    }

    /// Signal-to-noise ratio of the tracked levels in dB.
    pub fn snr_db(&self) -> f32 {
        let nm = self.noise_q * 1.65;
        if nm <= 0.0 {
            return 0.0;
        }
        20.0 * (self.sig / nm).max(1e-6).log10()
    }

    pub fn is_mark(&self) -> bool {
        self.state
    }

    /// Marks the following frames as unreliable (broadband static crash):
    /// they are treated as space and do not disturb the level trackers.
    pub fn set_erasure(&mut self, e: bool) {
        self.erasure = e;
    }

    /// External activity gate (spectral energy present at this frequency).
    pub fn set_gate(&mut self, g: bool) {
        self.gate = g;
    }

    pub fn squelch_open(&self) -> bool {
        self.squelch_open
    }

    /// Takes newly decoded text.
    pub fn take_text(&mut self) -> String {
        if let Some(b) = &mut self.beam {
            let c = b.take_committed();
            self.word.extend_from_slice(&c);
        }
        // Release complete words, dropping noise artefacts.
        while let Some(p) = self.word.iter().position(|&c| c == b' ') {
            let w: Vec<u8> = self.word.drain(..=p).collect();
            let body = &w[..w.len() - 1];
            if !is_noise_word(body) {
                self.text.extend_from_slice(&w);
            }
        }
        String::from_utf8_lossy(&std::mem::take(&mut self.text)).into_owned()
    }

    pub fn tentative(&self) -> String {
        self.beam.as_ref().map(|b| b.tentative()).unwrap_or_default()
    }

    /// Primes the level trackers with recent frames, then decodes them. Used
    /// when a channel is created so the start of the transmission is sliced
    /// with converged levels instead of the noise-only start-up state.
    pub fn warmup(&mut self, frames: &[f32]) {
        for &x in frames {
            self.track_levels(x.max(1e-12));
        }
        // Levels now reflect the signal; replay with slicing.
        let (q, s) = (self.noise_q, self.sig);
        for &x in frames {
            self.push_frame(x);
        }
        // Keep the converged levels if the replay drifted them.
        self.noise_q = self.noise_q.max(q * 0.5);
        self.sig = self.sig.max(s * 0.5);
    }

    fn track_levels(&mut self, x: f32) {
        // Skip the filter's start-up transient, then initialise from a
        // short block of settled frames.
        self.level_frames += 1;
        if self.level_frames <= 40 {
            return;
        }
        if self.level_frames <= 70 {
            self.init_buf.push(x);
            if self.level_frames == 70 {
                let mut v = std::mem::take(&mut self.init_buf);
                v.sort_by(|a, b| a.total_cmp(b));
                // Guard against digital silence at the start of a stream.
                self.noise_q = v[v.len() / 4].max(v[v.len() - 1] * 1e-3).max(1e-9);
                self.sig = v[v.len() - 1].max(self.noise_q * 1.65);
            }
            return;
        }
        // Converge faster during the first second.
        let eta = if self.level_frames < 600 { 0.06 } else { 0.02 };
        if x < self.noise_q {
            self.noise_q *= 1.0 - eta * 0.75;
        } else {
            self.noise_q *= 1.0 + eta * 0.25;
        }
        // Never let the floor fall more than 60 dB below the signal level
        // (e.g. after a stretch of digital silence).
        self.noise_q = self.noise_q.max(self.sig * 1e-3);
        let nm = self.noise_q * 1.65;
        if x > self.sig {
            self.sig += 0.04 * (x - self.sig);
        } else if self.state {
            self.sig += 0.01 * (x - self.sig);
        }
        self.sig += self.params.sig_decay * (nm - self.sig);
        if self.sig < nm {
            self.sig = nm;
        }
    }

    /// Feeds one envelope amplitude frame.
    pub fn push_frame(&mut self, x: f32) {
        let x = x.max(1e-12);
        if self.erasure {
            self.slice(false);
            return;
        }
        // Noise floor: 25th-percentile tracker. Signal: fast attack on peaks,
        // slower tracking during marks, decay toward the noise floor.
        self.track_levels(x);
        if self.noise_q <= 0.0 {
            self.slice(false);
            return;
        }
        let nm = self.noise_q * 1.65;
        let ratio = self.sig / nm;
        if self.squelch_open {
            if ratio < self.params.squelch_close {
                self.squelch_open = false;
            }
        } else if ratio > self.params.squelch_open {
            self.squelch_open = true;
        }
        let span = self.sig - nm;
        let thr = nm + self.params.threshold * span;
        let hyst = 0.1 * span;
        let want = if !self.squelch_open || !self.gate {
            false
        } else if self.state {
            x > thr - hyst
        } else {
            x > thr + hyst
        };
        self.slice(want);
    }

    fn confirm_frames(&self) -> u32 {
        let dot = self.beam.as_ref().map(|b| b.timing.dot).unwrap_or(40.0);
        ((0.18 * dot / self.frame_ms).round() as u32).clamp(1, 12)
    }

    fn slice(&mut self, want: bool) {
        self.run_frames += 1;
        if self.state {
            self.frames_since_mark = 0;
        } else {
            self.frames_since_mark += 1;
        }
        if want != self.state {
            self.cand += 1;
        } else {
            self.cand = 0;
        }
        if self.cand >= self.confirm_frames() {
            let dur = (self.run_frames - self.cand) as f32 * self.frame_ms;
            let was_mark = self.state;
            self.state = want;
            self.run_frames = self.cand;
            self.cand = 0;
            if was_mark {
                self.total_marks += 1;
                self.on_run(Run::Mark(dur));
            } else if !self.flushed {
                self.on_run(Run::Space(dur));
            }
            self.flushed = false;
        } else if !self.state && !self.flushed {
            // Flush after a long silence so the last word is displayed.
            // While bootstrapping, wait longer: the gaps may be Farnsworth.
            let wg = self.beam.as_ref().map(|b| b.timing.wgap()).unwrap_or(1000.0);
            let dur = self.run_frames as f32 * self.frame_ms;
            if dur > (wg * 1.6).max(250.0) {
                self.on_run(Run::Space(dur));
                self.on_run(Run::Flush);
                self.flushed = true;
            }
        }
    }

    fn on_run(&mut self, run: Run) {
        if !matches!(run, Run::Flush) {
            self.history.push_back(run);
            if self.history.len() > 64 {
                self.history.pop_front();
            }
        }
        if self.beam.is_none() {
            if !matches!(run, Run::Flush) {
                self.boot.push(run);
            }
            let nmarks = self.boot.iter().filter(|r| matches!(r, Run::Mark(_))).count();
            if nmarks >= 8 || (matches!(run, Run::Flush) && nmarks >= 4) {
                if let Some(tm) = estimate_timing(&self.boot) {
                    if std::env::var_os("CWSK_DEBUG").is_some() {
                        eprintln!("boot: {:?} wpm={:.1} snr={:.1}dB runs={:?}", tm, tm.wpm(), self.snr_db(), &self.boot);
                    }
                    let mut b = BeamDecoder::new(self.params.clone(), self.lm.clone(), tm);
                    for r in std::mem::take(&mut self.boot) {
                        b.adapt(r);
                        b.push(r);
                    }
                    if matches!(run, Run::Flush) {
                        b.push(Run::Flush);
                    }
                    self.beam = Some(b);
                } else if matches!(run, Run::Flush) {
                    self.boot.clear();
                }
            } else if matches!(run, Run::Flush) && nmarks < 4 {
                // Too little to estimate speed; drop to avoid stale bootstrap.
                if self.boot.len() > 40 {
                    self.boot.clear();
                }
            }
            return;
        }
        if let Run::Mark(_) = run {
            self.marks_since_check += 1;
        }
        // Periodically re-estimate speed in case a different operator (or
        // a speed change) appears on this frequency.
        if self.marks_since_check >= 16 {
            self.marks_since_check = 0;
            let hist: Vec<Run> = self.history.iter().copied().collect();
            if let Some(tm) = estimate_timing(&hist) {
                let b = self.beam.as_mut().unwrap();
                let r = tm.dot / b.timing.dot;
                if !(0.7..=1.43).contains(&r) {
                    if std::env::var_os("CWSK_DEBUG").is_some() {
                        eprintln!("re-estimate: {:?} -> {:?}", b.timing, tm);
                    }
                    b.timing = tm;
                } else {
                    // Character gaps can be stuck classified as word gaps
                    // (Farnsworth); trust the spacing statistics.
                    let rc = tm.cgap / b.timing.cgap;
                    if !(0.75..=1.33).contains(&rc) {
                        b.timing.cgap = tm.cgap.clamp(b.timing.dot * 2.0, b.timing.dot * 15.0);
                    }
                }
            }
        }
        let b = self.beam.as_mut().unwrap();
        b.adapt(run);
        b.push(run);
    }

    /// Forces out any pending text (e.g. at end of a file).
    pub fn finish(&mut self) {
        if self.beam.is_none() && !self.boot.is_empty() {
            self.on_run(Run::Flush);
        }
        if !self.flushed {
            let dur = self.run_frames as f32 * self.frame_ms;
            if self.state {
                self.on_run(Run::Mark(dur));
            } else {
                self.on_run(Run::Space(dur));
            }
            self.on_run(Run::Flush);
            self.flushed = true;
        }
        if let Some(b) = &mut self.beam {
            let c = b.take_committed();
            self.word.extend_from_slice(&c);
        }
        if !self.word.ends_with(b" ") && !self.word.is_empty() {
            self.word.push(b' ');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morse::pattern_for;

    fn runs_for(text: &str, dot: f32) -> Vec<Run> {
        let mut v = Vec::new();
        for (wi, w) in text.split(' ').enumerate() {
            if wi > 0 {
                v.push(Run::Space(7.0 * dot));
            }
            for (ci, c) in w.bytes().enumerate() {
                if ci > 0 {
                    v.push(Run::Space(3.0 * dot));
                }
                for (ei, e) in pattern_for(c).unwrap().bytes().enumerate() {
                    if ei > 0 {
                        v.push(Run::Space(dot));
                    }
                    v.push(Run::Mark(if e == b'-' { 3.0 * dot } else { dot }));
                }
            }
        }
        v.push(Run::Space(7.0 * dot));
        v.push(Run::Flush);
        v
    }

    #[test]
    fn perfect_timing_decodes() {
        let lm = Arc::new(LanguageModel::new());
        let runs = runs_for("CQ CQ DE W1AW K", 60.0);
        let tm = estimate_timing(&runs).unwrap();
        assert!((tm.dot - 60.0).abs() < 3.0, "{tm:?}");
        let mut b = BeamDecoder::new(DecoderParams::default(), lm, tm);
        for r in runs {
            b.push(r);
        }
        let out = String::from_utf8(b.take_committed()).unwrap();
        assert_eq!(out.trim(), "CQ CQ DE W1AW K");
    }
}
