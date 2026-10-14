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
use crate::morse::{can_extend, MorseLookup, BK, ERROR};
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
    /// Narrowest speed-matched post filter (Hz) for a strong signal.
    pub min_bw: f32,
    /// Narrowest post filter for a weak signal (Hz): below `weak_db` peak
    /// level over the noise (per FFT bin) the floor moves from `min_bw`
    /// down to this, reaching it `weak_span_db` lower. Noise matters more
    /// than smearing the gaps then.
    pub narrow_bw: f32,
    pub weak_db: f32,
    pub weak_span_db: f32,
    /// Signal-level tracking rate during marks, and attack rate outside.
    pub sig_track: f32,
    pub sig_attack: f32,
    /// Flicker shorter than this fraction of a dot is ignored (once the
    /// speed is known).
    pub debounce: f32,
    /// Weight of the amplitude evidence of a run (soft decision): the cost
    /// of calling a mark noise, or a space a fade inside an element, is
    /// this times how far (in noise standard deviations, integrated over
    /// the run in dot units) the envelope sat on the run's side of the
    /// threshold.
    pub soft_weight: f32,
    /// Log-probability penalty for finding an element hidden in a space
    /// (a dit too faded to cross the threshold), before its evidence.
    pub insert_penalty: f32,
    /// Floor on the log-likelihood of a mark far shorter than a dot being a
    /// dot: senders clip dits (rushed keying, a dash running into the next
    /// element), and the language model should get a say instead of the
    /// timing ruling it out.
    pub clipped_dit: f32,
    /// Language-model bonus for finishing a word (3+ characters) that was
    /// copied in the last few dozen words: CW repeats calls, CQ, reports.
    pub repeat_bonus: f32,
    /// Correct finished words that are one Morse element away from exactly
    /// one known word (see `LanguageModel::correct`); 0 turns it off.
    pub word_fix: f32,
    /// Find mark/space runs by a Viterbi search over per-frame likelihoods
    /// (minimum run length, evidence summed over each run) instead of
    /// thresholding frame by frame (1 = on).
    pub viterbi: f32,
    /// Scale of the per-frame log-likelihood ratios in that search.
    pub vit_gain: f32,
    /// Cap on one independent sample's log-likelihood ratio (clicks, fades).
    pub vit_cap: f32,
    /// Decision delay of the search, in dots.
    pub vit_lag: f32,
    /// Apply the squelch to the Viterbi search once the speed is known.
    pub vit_squelch: f32,
}

impl DecoderParams {
    /// Bandwidth (Hz) of the speed-matched post filter for a signal of this
    /// dot length (ms) and peak level over the noise (dB).
    pub fn post_bw(&self, dot: f32, level_db: f32) -> f32 {
        let t = ((self.weak_db - level_db) / self.weak_span_db.max(0.1)).clamp(0.0, 1.0);
        let floor = self.min_bw + t * (self.narrow_bw - self.min_bw);
        (self.bw_factor * 1000.0 / dot).clamp(floor, 200.0)
    }

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
                    "min_bw" => p.min_bw = v,
                    "narrow_bw" => p.narrow_bw = v,
                    "weak_db" => p.weak_db = v,
                    "weak_span_db" => p.weak_span_db = v,
                    "sig_track" => p.sig_track = v,
                    "sig_attack" => p.sig_attack = v,
                    "debounce" => p.debounce = v,
                    "soft_weight" => p.soft_weight = v,
                    "insert_penalty" => p.insert_penalty = v,
                    "clipped_dit" => p.clipped_dit = v,
                    "repeat_bonus" => p.repeat_bonus = v,
                    "word_fix" => p.word_fix = v,
                    "viterbi" => p.viterbi = v,
                    "vit_gain" => p.vit_gain = v,
                    "vit_cap" => p.vit_cap = v,
                    "vit_lag" => p.vit_lag = v,
                    "vit_squelch" => p.vit_squelch = v,

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
            lm_weight: 0.5,
            beam_width: 48,
            sigma_mark: 0.30,
            sigma_space: 0.38,
            blip_penalty: -2.0,
            dropout_penalty: -2.0,
            squelch_open: 2.3,
            squelch_close: 1.9,
            threshold: 0.5,
            sig_decay: 0.0015,
            bw_factor: 1.6,
            min_bw: 30.0,
            narrow_bw: 15.0,
            weak_db: 20.0,
            weak_span_db: 6.0,
            sig_track: 0.02,
            sig_attack: 0.01,
            debounce: 0.24,
            soft_weight: 0.7,
            insert_penalty: -3.0,
            clipped_dit: -4.5,
            repeat_bonus: 2.0,
            word_fix: 1.0,
            viterbi: 0.0,
            vit_gain: 8.0,
            vit_cap: 10.0,
            vit_lag: 4.0,
            vit_squelch: 1.0,

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
    /// This sender's word gap relative to the nominal one (7/3 of the
    /// character gap), learned from the gaps read as word gaps. Hand
    /// senders often run words together or spread them apart.
    pub wscale: f32,
}

impl TimingModel {
    fn from_dot(dot: f32) -> Self {
        Self { dot, dash: 3.0 * dot, egap: dot, cgap: 3.0 * dot, wscale: 1.0 }
    }
    pub fn wgap(&self) -> f32 {
        // Word gaps scale with character gaps (keeps Farnsworth consistent).
        (self.cgap * 7.0 / 3.0).max(self.dot * 5.0) * self.wscale
    }
    /// Character speed. A dot plus the gap after it is two units whatever
    /// the keying weight or the slicer's bias (which lengthens marks and
    /// shortens gaps by the same amount: on ARRL practice files the dot
    /// alone read ~17% slow).
    pub fn wpm(&self) -> f32 {
        2400.0 / (self.dot + self.egap)
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
    ctx: [u8; 4],
    /// The word being decoded (truncated), for the repetition bonus.
    word: [u8; 12],
    wlen: u8,
    out: Vec<u8>,
}

impl Hyp {
    fn key(&self) -> (u8, u8, u16, u16, [u8; 4]) {
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
    /// The last few dozen committed words, and the word being committed.
    recent: std::collections::VecDeque<Vec<u8>>,
    cword: Vec<u8>,
    /// Smoothed fraction of runs that fit Morse timing (0..1).
    pub fit: f32,
    pub fit_n: u32,
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
            recent: Default::default(),
            cword: Vec::new(),
            fit: 0.6,
            fit_n: 0,
        }
    }

    fn root() -> Hyp {
        Hyp { score: 0.0, bits: 0, len: 0, pend_mark: 0.0, pend_space: 0.0, ctx: [b' '; 4], word: [0; 12], wlen: 0, out: Vec::new() }
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
        let mut dot = lognorm(m, t.dot, s);
        if m < t.dot {
            dot = dot.max(self.params.clipped_dit);
        }
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
        if c == ERROR {
            // Not in the language model: a fixed, fairly rare cost. It ends
            // the word, and the next word starts with fresh context.
            h.score += self.params.lm_weight * (-4.0 + self.lm.char_bonus);
            h.ctx = [b' '; 4];
            h.wlen = 0;
            h.out.extend_from_slice(&[ERROR, b' ']);
            return;
        }
        let lp = self.lm.logp4(&h.ctx, c);
        h.score += self.params.lm_weight * (lp + self.lm.char_bonus);
        h.ctx = [h.ctx[1], h.ctx[2], h.ctx[3], c];
        h.out.push(c);
        if c == b' ' {
            let n = h.wlen as usize;
            // (Longer than the buffer: never matched.)
            let w = &h.word[..n.min(h.word.len())];
            let before = &h.out[..h.out.len().saturating_sub(1 + n)];
            if (3..=h.word.len()).contains(&n) && self.recently_copied(w, before) {
                h.score += self.params.lm_weight * self.params.repeat_bonus;
            }
            h.wlen = 0;
        } else if (h.wlen as usize) < h.word.len() {
            h.word[h.wlen as usize] = c;
            h.wlen += 1;
        } else {
            h.wlen = h.word.len() as u8 + 1;
        }
    }

    /// True if `w` was one of the recently committed words, or appears as a
    /// word earlier in this hypothesis' own uncommitted text.
    fn recently_copied(&self, w: &[u8], before: &[u8]) -> bool {
        if w.len() > 12 {
            return false;
        }
        self.recent.iter().any(|r| r.as_slice() == w) || before.split(|&c| c == b' ').any(|x| x == w)
    }

    /// Ends the current character; returns false if the elements are invalid.
    fn end_char(&self, h: &mut Hyp, word: bool) -> bool {
        if h.len > 0 {
            match self.morse.get(h.bits, h.len) {
                Some(BK) => {
                    self.emit(h, b'B');
                    self.emit(h, b'K');
                }
                Some(c) => self.emit(h, c),
                None => return false,
            }
            h.bits = 0;
            h.len = 0;
        }
        if word && h.ctx[3] != b' ' {
            self.emit(h, b' ');
        }
        true
    }

    /// Pushes a run with no amplitude information (as if strong).
    pub fn push(&mut self, run: Run) {
        self.push_soft(run, None);
    }

    /// Cost (log-probability) of explaining away a run whose amplitude
    /// evidence is `ev` (noise-sigma x ms): a mark as a noise burst, or a
    /// space as a fade inside an element. Without evidence, falls back to
    /// a duration-based penalty for short runs only.
    fn explain_away(&self, base: f32, len: f32, ev: Option<f32>) -> Option<f32> {
        let dot = self.timing.dot;
        let ev = ev.filter(|_| self.params.soft_weight > 0.0);
        match ev {
            Some(e) if len < 2.0 * dot => Some(base - self.params.soft_weight * e.max(0.0) / dot),
            None if len < 0.6 * dot => Some(base + 2.0 * (len / dot)),
            _ => None,
        }
    }

    /// Ways to explain a space as gap + faded element + gap: for each kind
    /// of the gap before (element/character/word) and element (dot/dash),
    /// the best placement as (score, length of the gap after). The score
    /// adds the timing likelihoods to the amplitude evidence in the window
    /// (see `prefix_above_median`: partly relative to the rest of the gap,
    /// since in a fade the threshold lags high above a dit that is still
    /// clearly out of the noise).
    fn hidden_elements(&self, sp: f32, prof: &[f32]) -> Vec<(usize, bool, f32, f32)> {
        let mut out = Vec::new();
        let t = &self.timing;
        let w = self.params.soft_weight;
        if w <= 0.0 || prof.len() < 4 || sp < 2.2 * t.dot {
            return out;
        }
        let fm = sp / prof.len() as f32;
        let pre = prefix_above_median(prof);
        for dash in [false, true] {
            let len = if dash { t.dash } else { t.dot };
            let wf = ((len / fm).round() as usize).max(1);
            let min_gap = 0.5 * t.egap;
            if sp < len + 2.0 * min_gap {
                continue;
            }
            let mark_ll = self.mark_options(len)[usize::from(dash)].1;
            let step = (wf / 4).max(1);
            let mut best = [(f32::NEG_INFINITY, 0.0f32); 3];
            let mut i = (min_gap / fm).ceil() as usize;
            while i + wf <= prof.len() {
                let s1 = i as f32 * fm;
                let s2 = sp - s1 - len;
                if s2 < min_gap {
                    break;
                }
                let ev = w * (pre[i + wf] - pre[i]) / t.dot;
                let after = self.space_options(s2).into_iter().fold(f32::NEG_INFINITY, f32::max);
                let before = self.space_options(s1);
                for k in 0..3 {
                    let sc = before[k] + mark_ll + ev + after;
                    if sc > best[k].0 {
                        best[k] = (sc, s2);
                    }
                }
                i += step;
            }
            for (k, &(sc, s2)) in best.iter().enumerate() {
                if sc.is_finite() {
                    // `after` is scored again when the gap completes.
                    let after = self.space_options(s2).into_iter().fold(f32::NEG_INFINITY, f32::max);
                    out.push((k, dash, s2, sc - after + self.params.insert_penalty));
                }
            }
        }
        out
    }

    /// Ways to explain a long mark as element + short gap + element, the
    /// gap filled in by noise: for each kind of the first element, the best
    /// split as (score, length of the remaining mark). Mirror image of
    /// `hidden_elements`, with the evidence of a dip in the envelope.
    fn hidden_gaps(&self, m: f32, prof: &[f32]) -> Vec<(bool, f32, f32)> {
        let mut out = Vec::new();
        let t = &self.timing;
        let w = self.params.soft_weight;
        if w <= 0.0 || prof.len() < 4 || m < 2.2 * t.dot {
            return out;
        }
        let fm = m / prof.len() as f32;
        let pre = prefix_above_median(prof);
        let wf = ((t.egap / fm).round() as usize).max(1);
        let gap_ll = self.space_options(t.egap)[0];
        let min_mark = 0.5 * t.dot;
        let step = (wf / 4).max(1);
        let mut best = [(f32::NEG_INFINITY, 0.0f32); 2];
        let mut i = (min_mark / fm).ceil() as usize;
        while i + wf <= prof.len() {
            let m1 = i as f32 * fm;
            let m2 = m - m1 - t.egap;
            if m2 < min_mark {
                break;
            }
            let ev = -w * (pre[i + wf] - pre[i]) / t.dot;
            let after = self.mark_options(m2).into_iter().map(|o| o.1).fold(f32::NEG_INFINITY, f32::max);
            for (d, ll) in self.mark_options(m1) {
                let sc = ll + gap_ll + ev + after;
                let k = usize::from(d);
                if sc > best[k].0 {
                    best[k] = (sc, m2);
                }
            }
            i += step;
        }
        for (k, &(sc, m2)) in best.iter().enumerate() {
            if sc.is_finite() {
                let after = self.mark_options(m2).into_iter().map(|o| o.1).fold(f32::NEG_INFINITY, f32::max);
                out.push((k == 1, m2, sc - after + self.params.insert_penalty));
            }
        }
        out
    }

    /// Classifies a completed space for a hypothesis: 0 = element gap,
    /// 1 = character gap, 2 = word gap. False if that makes it invalid.
    fn apply_space(&self, n: &mut Hyp, k: usize) -> bool {
        match k {
            0 => can_extend(n.bits, n.len),
            1 => self.end_char(n, false),
            _ => self.end_char(n, true),
        }
    }

    /// Appends an element; false if no character starts that way.
    fn apply_mark(&self, n: &mut Hyp, dash: bool) -> bool {
        if !can_extend(n.bits, n.len) {
            return false;
        }
        n.bits = (n.bits << 1) | u8::from(dash);
        n.len += 1;
        self.morse.is_prefix(n.bits, n.len)
    }

    /// Pushes a run with its per-frame amplitude evidence `prof` (noise
    /// sigma x ms above the threshold), if known.
    pub fn push_soft(&mut self, run: Run, prof: Option<&[f32]>) {
        let total: Option<f32> = prof.map(|p| p.iter().sum());
        let ev = match run {
            Run::Mark(_) => total,
            _ => total.map(|t| -t),
        };
        let hidden = match (run, prof) {
            (Run::Space(sp), Some(p)) => self.hidden_elements(sp, p),
            _ => Vec::new(),
        };
        let gaps = match (run, prof) {
            (Run::Mark(m), Some(p)) => self.hidden_gaps(m, p),
            _ => Vec::new(),
        };
        let mut next = std::mem::take(&mut self.scratch);
        next.clear();
        match run {
            Run::Mark(m) => {
                for h in &self.hyps {
                    if h.pend_space > 0.0 {
                        let opts = self.space_options(h.pend_space);
                        for (k, &ll) in opts.iter().enumerate() {
                            let mut n = h.clone();
                            n.score += ll;
                            if !self.apply_space(&mut n, k) {
                                continue;
                            }
                            n.pend_space = 0.0;
                            n.pend_mark = m;
                            for &(d1, m2, sc) in &gaps {
                                let mut g = n.clone();
                                if self.apply_mark(&mut g, d1) && can_extend(g.bits, g.len) {
                                    g.score += sc;
                                    g.pend_mark = m2;
                                    next.push(g);
                                }
                            }
                            next.push(n);
                        }
                        if let Some(c) = self.explain_away(self.params.blip_penalty, m, ev) {
                            let mut n = h.clone();
                            n.pend_space += m;
                            n.score += c;
                            next.push(n);
                        }
                    } else {
                        // At the start of a word (after a pause) a short
                        // mark may be a noise blip too, not only inside
                        // the transmission.
                        if h.pend_mark == 0.0 && h.len == 0 {
                            if let Some(c) = self.explain_away(self.params.blip_penalty, m, ev) {
                                let mut n = h.clone();
                                n.score += c;
                                next.push(n);
                            }
                        }
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
                            let mut n = h.clone();
                            if !self.apply_mark(&mut n, dash) {
                                continue;
                            }
                            n.score += ll;
                            n.pend_mark = 0.0;
                            n.pend_space = sp;
                            for &(k, hdash, s2, sc) in &hidden {
                                let mut m = n.clone();
                                if self.apply_space(&mut m, k) && self.apply_mark(&mut m, hdash) {
                                    m.score += sc;
                                    m.pend_space = s2;
                                    next.push(m);
                                }
                            }
                            next.push(n);
                        }
                        if let Some(c) = self.explain_away(self.params.dropout_penalty, sp, ev) {
                            let mut n = h.clone();
                            n.pend_mark += sp;
                            n.score += c;
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
                            if !can_extend(h.bits, h.len) {
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

    /// Remembers committed words for the repetition bonus.
    fn note_committed(&mut self, text: &[u8]) {
        for &c in text {
            if c == b' ' || c == ERROR {
                if self.cword.len() >= 3 {
                    self.recent.push_back(std::mem::take(&mut self.cword));
                    if self.recent.len() > 40 {
                        self.recent.pop_front();
                    }
                }
                self.cword.clear();
            } else {
                self.cword.push(c);
            }
        }
    }

    fn commit(&mut self, all: bool) {
        if all {
            let out = std::mem::take(&mut self.hyps[0].out);
            self.note_committed(&out);
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
        self.note_committed(&prefix);
        self.committed.extend_from_slice(&prefix);
        self.hyps.retain(|h| h.out.len() >= take && h.out[..take] == prefix[..]);
        for h in &mut self.hyps {
            h.out.drain(..take);
        }
    }

    /// Hard-decision classification of a run under the current timing model,
    /// used to adapt the model.
    pub fn adapt(&mut self, run: Run) {
        // Timing fit: does this run land near a Morse duration (1 or 3 units
        // for marks; 1, 3 or 7+ for spaces)? Real keying, even sloppy, mostly
        // does; noise produces random durations.
        let ll = match run {
            Run::Mark(m) => Some(self.mark_options(m).iter().map(|o| o.1).fold(f32::MIN, f32::max) + self.params.sigma_mark.ln()),
            Run::Space(sp) if sp < self.timing.wgap() * 2.0 => {
                let o = self.space_options(sp);
                Some(o[0].max(o[1] - (0.33f32 / 0.55).ln()).max(o[2] - (0.12f32 / 0.55).ln()) - (0.55f32).ln() + self.params.sigma_space.ln())
            }
            _ => None,
        };
        if let Some(ll) = ll {
            // ll > -1.1 means within ~1.5 sigma of a Morse duration.
            let hit = if ll > -1.1 { 1.0 } else { 0.0 };
            self.fit += 0.05 * (hit - self.fit);
            self.fit_n += 1;
        }
        let a = 0.08;
        match run {
            Run::Mark(m) => {
                let opts = self.mark_options(m);
                let t = &mut self.timing;
                if opts[1].1 > opts[0].1 {
                    if m < t.dot * 6.0 {
                        t.dash += a * (m - t.dash);
                    }
                } else if m > t.dot * 0.5 {
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
                t.cgap = t.cgap.clamp(t.dot * 2.0, t.dot * 25.0);
            }
            Run::Flush => {}
        }
    }
}

/// Prefix sums of a run's evidence profile measured halfway between the
/// threshold and the run's median, so a window sum rewards both standing
/// out from the rest of the run and coming close to the threshold.
fn prefix_above_median(prof: &[f32]) -> Vec<f32> {
    let mut sorted = prof.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let med = sorted[sorted.len() / 2];
    let mut pre = Vec::with_capacity(prof.len() + 1);
    pre.push(0.0f32);
    for &z in prof {
        pre.push(pre.last().unwrap() + z - 0.5 * med);
    }
    pre
}

/// Log-likelihood of mark/space durations under a dot length `d` (ms),
/// with outliers capped, plus a mild prior toward common speeds.
fn model_score(d: f32, marks: &[f32], spaces: &[f32]) -> f32 {
    let sm = 0.30f32;
    let ss = 0.40f32;
    const FLOOR: f32 = -3.0;
    // Dash/dot ratio: 3 nominally, but hand senders range widely; fit it
    // too (with a prior toward 3) so a heavy dash does not inflate the dot.
    let mut score = f32::NEG_INFINITY;
    for r in [2.6f32, 3.0, 3.4, 3.8, 4.3] {
        let mut sc = -0.5 * ((r / 3.0).ln() / 0.15).powi(2);
        for &m in marks {
            let a = lognorm(m, d, sm);
            let b = lognorm(m, r * d, sm);
            sc += a.max(b).max(FLOOR);
        }
        score = score.max(sc);
    }
    for &s in spaces {
        if s > 15.0 * d {
            continue;
        }
        let a = lognorm(s, d, ss);
        let b = lognorm(s, 3.0 * d, ss);
        let c = if s > 7.0 * d { -ss.ln() } else { lognorm(s, 7.0 * d, ss) };
        score += 0.6 * a.max(b).max(c).max(FLOOR);
    }
    let wpm = 1200.0 / d;
    score - 0.5 * ((wpm / 22.0).ln() / 0.8).powi(2)
}

/// Filters out noise blips: keeps marks/spaces of at least 1/5 of the
/// 75th-percentile mark (a real dot is never shorter than that).
fn solid_runs(runs: &[Run]) -> (Vec<f32>, Vec<f32>) {
    let marks: Vec<f32> = runs.iter().filter_map(|r| if let Run::Mark(m) = r { Some(*m) } else { None }).collect();
    let spaces: Vec<f32> = runs.iter().filter_map(|r| if let Run::Space(s) = r { Some(*s) } else { None }).collect();
    if marks.is_empty() {
        return (marks, spaces);
    }
    let mut sorted = marks.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let l90 = sorted[(sorted.len() * 9 / 10).min(sorted.len() - 1)];
    let min_len = (l90 / 4.5).max(16.0);
    (marks.into_iter().filter(|&m| m >= min_len).collect(), spaces.into_iter().filter(|&s| s >= min_len).collect())
}

/// True if dot length `new` explains the runs clearly better than `cur`.
pub fn clearly_better(runs: &[Run], cur: f32, new: f32) -> bool {
    let (m, s) = solid_runs(runs);
    if m.len() < 6 {
        return false;
    }
    model_score(new, &m, &s) > model_score(cur, &m, &s) + 0.5 * m.len() as f32
}

/// True if dot length `new` explains the runs better than `cur` at all.
pub fn better(runs: &[Run], cur: f32, new: f32) -> bool {
    let (m, s) = solid_runs(runs);
    m.len() >= 6 && model_score(new, &m, &s) > model_score(cur, &m, &s)
}

/// Character gap from the first few long gaps: the lower group if they
/// split into two (character and word gaps), otherwise judged by length:
/// under 5 dots they are character gaps, over 9 Farnsworth character gaps,
/// and in between most likely word gaps (keep the 3-dot default).
fn early_cgap(long: &mut [f32], dot: f32) -> Option<f32> {
    // One or two gaps (possibly stretched by a fade) are too few to judge.
    if long.len() < 3 {
        return None;
    }
    long.sort_by(|a, b| a.total_cmp(b));
    let jump = (1..long.len()).max_by(|&i, &j| (long[i] / long[i - 1]).total_cmp(&(long[j] / long[j - 1])));
    if let Some(i) = jump.filter(|&i| long[i] / long[i - 1] >= 1.6) {
        let lower = &long[..i];
        return Some(lower[lower.len() / 2]);
    }
    let med = long[long.len() / 2];
    if med < 5.0 * dot || med > 9.0 * dot {
        Some(med)
    } else {
        None
    }
}

/// Spaces clearly longer than an element gap (over 2.2 x the 25th-percentile
/// mark, which is about a dot).
fn boot_long_gaps(runs: &[Run]) -> usize {
    let mut marks: Vec<f32> = runs.iter().filter_map(|r| if let Run::Mark(m) = r { Some(*m) } else { None }).collect();
    if marks.is_empty() {
        return 0;
    }
    marks.sort_by(|a, b| a.total_cmp(b));
    let dot = marks[marks.len() / 4];
    runs.iter().filter(|r| matches!(r, Run::Space(s) if *s > 2.2 * dot)).count()
}

/// Maximum-likelihood speed estimate from a set of runs.
pub fn estimate_timing(runs: &[Run]) -> Option<TimingModel> {
    let total_marks = runs.iter().filter(|r| matches!(r, Run::Mark(_))).count();
    if total_marks < 4 {
        return None;
    }
    // In noise, short blips can outnumber real elements; anchor on the
    // solid marks.
    let (marks, spaces) = solid_runs(runs);
    if marks.len() < 3 {
        return None;
    }
    let mut best = (f32::NEG_INFINITY, 0.0f32);
    // 5-60 WPM.
    let mut d = 20.0f32;
    while d < 240.0 {
        let score = model_score(d, &marks, &spaces);
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
    // Refine with robust class medians. Ignore marks far shorter than a dot
    // (noise blips) so they cannot drag the speed estimate up.
    let median = |v: &mut Vec<f32>| {
        if v.len() >= 2 {
            v.sort_by(|a, b| a.total_cmp(b));
            Some(v[v.len() / 2])
        } else {
            None
        }
    };
    let mean = |v: &mut Vec<f32>| if v.len() >= 2 { Some(v.iter().sum::<f32>() / v.len() as f32) } else { None };
    let mut dots: Vec<f32> = marks.iter().copied().filter(|&m| m >= 0.5 * dot && m < 2.0 * dot).collect();
    let mut dashes: Vec<f32> = marks.iter().copied().filter(|&m| m >= 2.0 * dot && m < 6.0 * dot).collect();
    if let Some(v) = median(&mut dots) {
        tm.dot = v.clamp(dot * 0.75, dot * 1.33).clamp(20.0, 240.0);
    }
    if let Some(v) = median(&mut dashes) {
        tm.dash = v.clamp(tm.dot * 2.0, tm.dot * 5.0);
    }
    let mut eg: Vec<f32> = spaces.iter().copied().filter(|&s| s >= 0.5 * dot && s < 2.0 * dot).collect();
    if let Some(v) = mean(&mut eg) {
        tm.egap = v.clamp(tm.dot * 0.5, tm.dot * 2.0);
    }
    // Character gaps outnumber word gaps roughly 4:1 in text, so a low
    // percentile of the long spaces is the character gap. This also copes
    // with Farnsworth spacing where gaps are far longer than 3 dots.
    let mut long: Vec<f32> = spaces.iter().copied().filter(|&s| s >= 2.0 * tm.egap.max(dot)).collect();
    if let Some(v) = early_cgap(&mut long, tm.dot) {
        tm.cgap = v.clamp(tm.dot * 2.0, tm.dot * 25.0);
    }
    Some(tm)
}

/// Splits the long gaps of a sender (character and word gaps) into two
/// clusters by 1-D k-means on log durations. Returns (character gap, word
/// gap) when they are clearly distinct and both common enough to trust.
pub fn gap_clusters(gaps: &[f32]) -> Option<(f32, f32)> {
    if gaps.len() < 10 {
        return None;
    }
    let mut v: Vec<f32> = gaps.iter().map(|g| g.max(1.0).ln()).collect();
    v.sort_by(|a, b| a.total_cmp(b));
    let (mut lo, mut hi) = (v[v.len() / 4], v[v.len() * 9 / 10]);
    let mut split = 0;
    for _ in 0..12 {
        let mid = 0.5 * (lo + hi);
        split = v.partition_point(|&x| x < mid);
        if split == 0 || split == v.len() {
            return None;
        }
        lo = v[..split].iter().sum::<f32>() / split as f32;
        hi = v[split..].iter().sum::<f32>() / (v.len() - split) as f32;
    }
    let frac_hi = (v.len() - split) as f32 / v.len() as f32;
    // Word gaps are roughly 1 in 4-5 of the long gaps in text.
    if hi - lo < (1.3f32).ln() || !(0.08..=0.55).contains(&frac_hi) {
        return None;
    }
    Some((lo.exp(), hi.exp()))
}

/// Words made only of E and T (single dots/dashes) are what noise and
/// interference decode as; real traffic almost never contains them.
/// How long (ms) a finished word is held back waiting for a possible error
/// sign; the next word releases it sooner.
const HOLD_MS: f32 = 4000.0;

/// Applies error signs to text: each erases the word in progress, or the
/// previous word if it follows a word gap.
pub fn apply_errors(text: &[u8]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(text.len());
    let mut i = 0;
    while i < text.len() {
        if text[i] == ERROR {
            if out.last().is_none_or(|&c| c == b' ') {
                while out.last() == Some(&b' ') {
                    out.pop();
                }
            }
            while out.last().is_some_and(|&c| c != b' ') {
                out.pop();
            }
            // Skip the space that follows the sign.
            if text.get(i + 1) == Some(&b' ') {
                i += 1;
            }
        } else {
            out.push(text[i]);
        }
        i += 1;
    }
    out
}

pub fn is_noise_word(w: &[u8]) -> bool {
    !w.is_empty() && w.iter().all(|&c| c == b'E' || c == b'T')
}

/// ln I0(u) for u >= 0 (modified Bessel function), accurate to ~1e-3.
fn ln_i0(u: f32) -> f32 {
    if u < 3.0 {
        let q = u * u / 4.0;
        (1.0 + q * (1.0 + q * (0.25 + q * (1.0 / 36.0 + q / 576.0)))).ln()
    } else {
        u - 0.5 * (2.0 * std::f32::consts::PI * u).ln() + (1.0 + 1.0 / (8.0 * u) + 9.0 / (128.0 * u * u)).ln()
    }
}

/// Fixed-lag Viterbi search for the key state. States are "mark for c
/// frames" and "space for c frames", c = 1..k, the last one absorbing (a run
/// is at least k frames). Each frame adds half its log-likelihood ratio to
/// the mark states and subtracts it from the space states, so a run's
/// evidence is summed over the whole run instead of being thresholded frame
/// by frame: a dit sitting in the noise is kept if, as a whole, it is more
/// likely signal than noise, and a dash is not split by a few noisy frames.
struct KeyViterbi {
    /// Minimum run length (frames); may change between frames.
    k: usize,
    /// Score of state (m, c) at `m * KMAX + c`: m = 0 mark, 1 space; c + 1
    /// frames into the run, the last count (k - 1) absorbing.
    score: Vec<f32>,
    next: Vec<f32>,
    /// Per frame: the best previous state of each state.
    back: std::collections::VecDeque<Vec<u8>>,
    /// Per frame: the evidence passed through to the run builder.
    ev: std::collections::VecDeque<f32>,
    /// Backpointer buffers for reuse.
    spare: Vec<Vec<u8>>,
}

const KMAX: usize = 100;

impl KeyViterbi {
    fn new(k: usize, mark: bool) -> Self {
        let mut score = vec![-1e9; 2 * KMAX];
        let k = k.clamp(1, KMAX);
        score[usize::from(!mark) * KMAX + k - 1] = 0.0;
        Self { k, score, next: vec![-1e9; 2 * KMAX], back: Default::default(), ev: Default::default(), spare: Vec::new() }
    }

    /// Changes the minimum run length. Runs already longer than the new
    /// minimum fold into its absorbing state; backpointers stay valid.
    fn set_k(&mut self, k: usize) {
        let k = k.clamp(1, KMAX);
        if k < self.k {
            for m in 0..2 {
                let b = m * KMAX;
                let top = self.score[b + k - 1..b + self.k].iter().copied().fold(f32::MIN, f32::max);
                self.score[b + k - 1] = top;
                self.score[b + k..b + self.k].iter_mut().for_each(|s| *s = -1e9);
            }
        }
        self.k = k;
    }

    /// One frame: `llr` is the log-likelihood ratio mark/space, `enter`
    /// the log-probabilities per frame of starting a mark / a space.
    fn step(&mut self, llr: f32, ev: f32, enter: [f32; 2]) {
        let k = self.k;
        let mut bp = if self.back.len() > 64 { self.spare.pop().unwrap_or_default() } else { Vec::new() };
        bp.resize(2 * KMAX, 0);
        for m in 0..2 {
            let base = m * KMAX;
            let other = (1 - m) * KMAX + k - 1;
            let half = if m == 0 { 0.5 * llr } else { -0.5 * llr };
            for c in 0..k {
                let (mut best, mut arg) = if c == 0 {
                    (self.score[other] + enter[m], other)
                } else {
                    (self.score[base + c - 1], base + c - 1)
                };
                if c == k - 1 && self.score[base + c] > best {
                    best = self.score[base + c];
                    arg = base + c;
                }
                self.next[base + c] = best + half;
                bp[base + c] = arg as u8;
            }
        }
        let top = (0..2).flat_map(|m| (0..k).map(move |c| m * KMAX + c)).map(|i| self.next[i]).fold(f32::MIN, f32::max);
        for m in 0..2 {
            for c in 0..KMAX {
                let i = m * KMAX + c;
                self.score[i] = if c < k { self.next[i] - top } else { -1e9 };
            }
        }
        self.back.push_back(bp);
        self.ev.push_back(ev);
    }

    /// Decisions for the oldest frames, keeping the newest `lag` undecided.
    fn decide(&mut self, lag: usize, out: &mut Vec<(bool, f32)>) {
        if self.back.len() <= lag {
            return;
        }
        let n = self.back.len() - lag;
        let mut st = (0..2 * KMAX).max_by(|&a, &b| self.score[a].total_cmp(&self.score[b])).unwrap_or(0);
        let mut path = vec![0u8; self.back.len()];
        for (i, bp) in self.back.iter().enumerate().rev() {
            path[i] = st as u8;
            st = bp[st] as usize;
        }
        for i in 0..n {
            out.push(((path[i] as usize) < KMAX, self.ev[i]));
        }
        self.spare.extend(self.back.drain(..n));
        self.spare.truncate(64);
        self.ev.drain(..n);
    }
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
    block: Vec<f32>,
    block_p25: std::collections::VecDeque<f32>,
    squelch_open: bool,
    gate: bool,
    erasure: bool,
    // slicer / run builder
    state: bool,
    run_frames: u32,
    cand: u32,
    /// Per-frame amplitude evidence (noise-sigma x ms above the threshold)
    /// of the current run, including its trailing candidate (debounce)
    /// frames.
    run_z: Vec<f32>,
    flushed: bool,
    // timing
    boot: Vec<Run>,
    boot_prof: Vec<Option<Vec<f32>>>,
    history: std::collections::VecDeque<Run>,
    /// Recent character/word gaps (not pauses), for `gap_clusters`.
    long_gaps: std::collections::VecDeque<f32>,
    marks_since_check: u32,
    beam: Option<BeamDecoder>,
    text: Vec<u8>,
    word: Vec<u8>,
    /// Last complete word, held back briefly in case the sender follows it
    /// with the error sign.
    held: Vec<u8>,
    held_frames: u32,
    pub frames_since_mark: u64,
    pub total_marks: u64,
    /// A long silence just ended: the next over (maybe another station)
    /// is held back for a few elements to check its speed first.
    pause_pending: bool,
    over_buf: Option<Vec<(Run, Option<Vec<f32>>)>>,
    /// Debug: per-frame trace (CWSK_DUMP=path), one decoder only.
    dump: Option<std::io::BufWriter<std::fs::File>>,
    /// Viterbi key-state search (`params.viterbi`); `state`/`cand` then
    /// keep the immediate threshold decision (level tracking, `is_mark`),
    /// and `vstate`/`vcand` build the runs from the search's decisions.
    vit: Option<KeyViterbi>,
    vstate: bool,
    vcand: u32,
    vfsm: u64,
    /// Bandwidth (Hz) of the envelope filter: how many independent
    /// samples per second the frames carry.
    bw: f32,
    vit_out: Vec<(bool, f32)>,
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
            block: Vec::with_capacity(100),
            block_p25: Default::default(),
            squelch_open: false,
            gate: true,
            erasure: false,
            state: false,
            run_frames: 0,
            cand: 0,
            run_z: Vec::new(),
            flushed: true,
            boot: Vec::new(),
            boot_prof: Vec::new(),
            history: Default::default(),
            long_gaps: Default::default(),
            marks_since_check: 0,
            beam: None,
            text: Vec::new(),
            word: Vec::new(),
            held: Vec::new(),
            held_frames: 0,
            frames_since_mark: 0,
            total_marks: 0,
            pause_pending: false,
            over_buf: None,
            dump: std::env::var_os("CWSK_DUMP").and_then(|p| std::fs::File::create(p).ok()).map(std::io::BufWriter::new),
            vit: None,
            vstate: false,
            vcand: 0,
            vfsm: 0,
            bw: 100.0,
            vit_out: Vec::new(),
        }
    }

    /// Tells the decoder the bandwidth (Hz) of the envelope filter.
    pub fn set_bandwidth(&mut self, bw: f32) {
        self.bw = bw;
    }

    /// Starts decoding with a known timing instead of measuring it first.
    pub fn seed_timing(&mut self, tm: TimingModel) {
        self.beam = Some(BeamDecoder::new(self.params.clone(), self.lm.clone(), tm));
    }

    pub fn params(&self) -> &DecoderParams {
        &self.params
    }

    pub fn timing(&self) -> Option<&TimingModel> {
        self.beam.as_ref().map(|b| &b.timing)
    }

    /// Timing-fit score (fraction of recent runs near Morse durations) and
    /// how many runs it is based on.
    pub fn fit(&self) -> (f32, u32) {
        self.beam.as_ref().map(|b| (b.fit, b.fit_n)).unwrap_or((0.0, 0))
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
        self.release_words();
        // Nobody sends the error sign long after the word: release it.
        if !self.held.is_empty() && self.held_frames as f32 * self.frame_ms > HOLD_MS {
            self.text.append(&mut self.held);
        }
        String::from_utf8_lossy(&std::mem::take(&mut self.text)).into_owned()
    }

    /// Releases the held word now (e.g. the station has stopped).
    pub fn release_held(&mut self) -> String {
        self.release_words();
        self.text.append(&mut self.held);
        String::from_utf8_lossy(&std::mem::take(&mut self.text)).into_owned()
    }

    /// Moves complete words from the beam into the output, dropping noise
    /// artefacts and applying the error sign.
    fn release_words(&mut self) {
        if let Some(b) = &mut self.beam {
            let c = b.take_committed();
            self.word.extend_from_slice(&c);
        }
        while let Some(p) = self.word.iter().position(|&c| c == b' ') {
            let w: Vec<u8> = self.word.drain(..=p).collect();
            let body = &w[..w.len() - 1];
            if let Some(e) = body.iter().position(|&c| c == ERROR) {
                // The error sign erases the word in progress, or if it came
                // after a word gap, the previous word.
                if body[..e].is_empty() {
                    self.held.clear();
                }
            } else if !is_noise_word(body) {
                self.text.append(&mut self.held);
                // A human copying fills in a garbled letter from the word;
                // do the same when the fix is unambiguous.
                // Only on clearly real keying: on noise it would turn junk
                // into plausible-looking words.
                let real = self.beam.as_ref().is_some_and(|b| b.fit >= 0.8);
                let fixed = if self.params.word_fix > 0.0 && real { self.lm.correct(body) } else { None };
                self.held = match fixed {
                    Some(f) => [f, b" "].concat(),
                    None => w,
                };
                self.held_frames = 0;
            }
        }
    }

    /// Text not yet released: the held word, then the live decode.
    pub fn tentative(&self) -> String {
        let mut b = self.held.clone();
        b.extend_from_slice(&self.word);
        if let Some(beam) = &self.beam {
            b.extend_from_slice(beam.tentative().as_bytes());
        }
        String::from_utf8_lossy(&apply_errors(&b)).into_owned()
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
        // Noise floor by minimum statistics: the 25th percentile of each
        // 200 ms block, minimum over the last ~1.6 s. Every transmission has
        // gaps, so marks (however long) cannot raise it, yet it follows
        // changing band noise within a couple of seconds.
        self.block.push(x);
        if self.block.len() >= 100 {
            let mut b = std::mem::take(&mut self.block);
            b.sort_by(|a, c| a.total_cmp(c));
            self.block_p25.push_back(b[b.len() / 4]);
            if self.block_p25.len() > 8 {
                self.block_p25.pop_front();
            }
            self.noise_q = self.block_p25.iter().copied().fold(f32::MAX, f32::min);
            self.block = b;
            self.block.clear();
        }
        // Never let the floor fall more than 60 dB below the signal level
        // (e.g. after a stretch of digital silence).
        self.noise_q = self.noise_q.max(self.sig * 1e-3);
        let nm = self.noise_q * 1.65;
        // Signal level: the mean mark amplitude. A slow attack outside marks
        // lets a new signal pull it up without noise spikes inflating it.
        if self.state {
            self.sig += self.params.sig_track * (x - self.sig);
        } else if x > self.sig {
            self.sig += self.params.sig_attack * (x - self.sig);
        }
        self.sig += self.params.sig_decay * (nm - self.sig);
        if self.sig < nm {
            self.sig = nm;
        }
    }

    /// Feeds one envelope amplitude frame.
    pub fn push_frame(&mut self, x: f32) {
        let x = x.max(1e-12);
        let vit = self.params.viterbi > 0.0;
        if self.erasure {
            if vit {
                // No information either way.
                self.debounce(false);
                self.vit_frame(0.0, 0.0);
            } else {
                self.slice(false, 0.0);
            }
            return;
        }
        // Noise floor: 25th-percentile tracker. Signal: fast attack on peaks,
        // slower tracking during marks, decay toward the noise floor.
        self.track_levels(x);
        if self.noise_q <= 0.0 {
            if vit {
                self.debounce(false);
                self.vit_frame(-1.0, 0.0);
            } else {
                self.slice(false, 0.0);
            }
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
        if let Some(d) = &mut self.dump {
            use std::io::Write;
            let _ = writeln!(d, "{:.5e},{:.5e},{:.5e},{:.5e},{}", x, nm, self.sig, thr, u8::from(self.state));
        }
        // Soft evidence: distance from the threshold in units of the noise
        // envelope's standard deviation (~0.52 x its mean for Rayleigh
        // noise), capped so a strong signal does not swamp the timing.
        // Computed even while the squelch or activity gate holds the slicer
        // off: a deep fade is exactly when a faded dit must still be found.
        let z = ((x - thr) / (0.52 * nm)).clamp(-8.0, 8.0);
        if vit {
            self.debounce(want);
            // Until the speed is known the squelch does not apply: the
            // search's run priors keep most noise out, and at low SNR (the
            // envelope still filtered wide) it would hide the marks the
            // speed estimate needs.
            let shut = !self.gate || (!self.squelch_open && self.beam.is_some() && self.params.vit_squelch > 0.0);
            let llr = if shut { -self.params.vit_cap } else { self.frame_llr(x, nm, thr) };
            if let Some(d) = &mut self.dump {
                use std::io::Write;
                let _ = writeln!(d, "L,{llr:.3},{},{}", u8::from(self.squelch_open), u8::from(self.gate));
            }
            self.vit_frame(llr, z * self.frame_ms);
        } else {
            self.slice(want, z * self.frame_ms);
        }
    }

    /// Log-likelihood ratio of one envelope sample, keyed carrier at the
    /// tracked signal level against noise alone, capped. The high-SNR
    /// (linear) form, centred on the slicing threshold: the exact
    /// Rician/Rayleigh ratio is convex in the amplitude, and on the smeared
    /// edges of each element (where neither model holds) that biases every
    /// boundary toward the mark.
    fn frame_llr(&self, x: f32, nm: f32, thr: f32) -> f32 {
        let s2 = (nm / 1.2533).powi(2);
        let a = (self.sig * self.sig - s2).max(0.25 * s2).sqrt();
        let cap = self.params.vit_cap;
        if self.params.viterbi >= 2.0 {
            let a2 = a * a;
            return (ln_i0(x * a / s2) - a2 / (2.0 * s2)).clamp(-cap, cap);
        }
        (a * (x - thr) / s2).clamp(-cap, cap)
    }

    /// The immediate (debounced threshold) key state, for level tracking
    /// and `is_mark` while the runs come from the Viterbi search.
    fn debounce(&mut self, want: bool) {
        if want != self.state {
            self.cand += 1;
        } else {
            self.cand = 0;
        }
        if self.cand >= self.confirm_frames() {
            self.state = want;
            self.cand = 0;
        }
        if self.state {
            self.frames_since_mark = 0;
        } else {
            self.frames_since_mark += 1;
        }
    }

    /// Feeds one frame to the Viterbi key search and builds runs from the
    /// frames it has decided.
    fn vit_frame(&mut self, llr: f32, ev: f32) {
        let dot = self.beam.as_ref().map(|b| b.timing.dot).unwrap_or(40.0);
        let fm = self.frame_ms;
        let k = self.confirm_frames() as usize;
        match &mut self.vit {
            Some(v) => v.set_k(k),
            None => self.vit = Some(KeyViterbi::new(k, self.vstate)),
        }
        let scale = self.params.vit_gain * fm * self.bw / 1000.0;
        let enter = [(fm / (2.5 * dot)).ln(), (fm / (1.6 * dot)).ln()];
        let lag = (self.params.vit_lag * dot / fm).round() as usize;
        if let Some(v) = &mut self.vit {
            v.step(llr * scale, ev, enter);
            // Trace back in batches: each traceback walks the whole lag.
            if v.back.len() >= lag + 16 {
                v.decide(lag, &mut self.vit_out);
            }
        }
        self.feed_vit();
    }

    fn feed_vit(&mut self) {
        let out = std::mem::take(&mut self.vit_out);
        for &(w, e) in &out {
            self.with_vit_state(|d| d.slice_n(w, e, 1));
        }
        self.vit_out = out;
        self.vit_out.clear();
    }

    /// Runs `f` with the run builder's key state in `state`/`cand`.
    fn with_vit_state(&mut self, f: impl FnOnce(&mut Self)) {
        std::mem::swap(&mut self.state, &mut self.vstate);
        std::mem::swap(&mut self.cand, &mut self.vcand);
        std::mem::swap(&mut self.frames_since_mark, &mut self.vfsm);
        f(self);
        std::mem::swap(&mut self.state, &mut self.vstate);
        std::mem::swap(&mut self.cand, &mut self.vcand);
        std::mem::swap(&mut self.frames_since_mark, &mut self.vfsm);
    }

    fn confirm_frames(&self) -> u32 {
        let dot = self.beam.as_ref().map(|b| b.timing.dot).unwrap_or(40.0);
        // Ignore flicker shorter than ~30 % of a dot (noise); real elements
        // are never that short. Before the speed is known, stay permissive.
        let frac = if self.beam.is_some() { self.params.debounce } else { 0.18 };
        ((frac * dot / self.frame_ms).round() as u32).clamp(1, 20)
    }

    fn slice(&mut self, want: bool, ev: f32) {
        let c = self.confirm_frames();
        self.slice_n(want, ev, c);
    }

    fn slice_n(&mut self, want: bool, ev: f32, confirm: u32) {
        self.run_z.push(ev);
        // A silence after a flush is never reported: keep only what a
        // pending transition could need.
        if self.flushed && self.run_z.len() > 256 {
            self.run_z.drain(..self.run_z.len() - 32);
        }
        self.run_frames += 1;
        self.held_frames = self.held_frames.saturating_add(1);
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
        if self.cand >= confirm {
            let dur = (self.run_frames - self.cand) as f32 * self.frame_ms;
            let tail = self.run_z.split_off(self.run_z.len().saturating_sub(self.cand as usize));
            let prof = std::mem::replace(&mut self.run_z, tail);
            let was_mark = self.state;
            if !was_mark && self.flushed {
                let wg = self.beam.as_ref().map(|b| b.timing.wgap()).unwrap_or(f32::MAX);
                let gap = self.frames_since_mark as f32 * self.frame_ms;
                if gap >= (3.0 * wg).max(2500.0) {
                    self.pause_pending = true;
                } else {
                    // The gap outlasted the end-of-word timeout but was no
                    // pause: let the spacing statistics see it, or a sender
                    // whose character gaps outlast the timeout never teaches
                    // the decoder his spacing.
                    self.note_long_gap(gap);
                }
            }
            self.state = want;
            self.run_frames = self.cand;
            self.cand = 0;
            if was_mark {
                self.total_marks += 1;
                self.on_run(Run::Mark(dur), Some(prof));
            } else if !self.flushed {
                self.on_run(Run::Space(dur), Some(prof));
            }
            self.flushed = false;
        } else if !self.state && !self.flushed {
            // Flush after a long silence so the last word is displayed.
            // While bootstrapping, wait longer: the gaps may be Farnsworth.
            let wg = self.beam.as_ref().map(|b| b.timing.wgap()).unwrap_or(1000.0);
            // Nor before the typical long gap of this sender has passed
            // well over: a slow hand with 7-dot character gaps was cut
            // into single letters before the spacing was learned.
            let typical = if self.long_gaps.len() >= 5 {
                let mut v: Vec<f32> = self.long_gaps.iter().copied().collect();
                v.sort_by(|a, b| a.total_cmp(b));
                v[v.len() / 2]
            } else {
                0.0
            };
            let dur = self.run_frames as f32 * self.frame_ms;
            if dur > (wg * 1.6).max(typical * 1.6).max(250.0) {
                self.run_z.clear();
                self.on_run(Run::Space(dur), None);
                self.on_run(Run::Flush, None);
                self.flushed = true;
            }
        }
    }

    fn on_run(&mut self, run: Run, prof: Option<Vec<f32>>) {
        // Start of an over after a long pause: hold its first elements.
        if self.beam.is_some() {
            if let Some(buf) = &mut self.over_buf {
                let flush = matches!(run, Run::Flush);
                buf.push((run, prof));
                let marks = buf.iter().filter(|(r, _)| matches!(r, Run::Mark(_))).count();
                if flush || marks >= 10 {
                    self.release_over();
                }
                return;
            }
            if std::mem::take(&mut self.pause_pending) && matches!(run, Run::Mark(_)) {
                self.over_buf = Some(vec![(run, prof)]);
                return;
            }
        }
        self.feed(run, prof);
    }

    /// The held start of an over: if its elements are explained clearly
    /// better by another speed (a different station), switch to it first,
    /// then decode them. Decoding them with the previous station's timing
    /// lost the first call of every over after a change of station.
    fn release_over(&mut self) {
        let Some(buf) = self.over_buf.take() else { return };
        let runs: Vec<Run> = buf.iter().map(|(r, _)| *r).filter(|r| !matches!(r, Run::Flush)).collect();
        if let (Some(tm), Some(b)) = (estimate_timing(&runs), self.beam.as_mut()) {
            let r = tm.dot / b.timing.dot;
            if !(0.8..=1.25).contains(&r) && better(&runs, b.timing.dot, tm.dot) {
                if std::env::var_os("CWSK_DEBUG").is_some() {
                    eprintln!("new over at another speed: {:?} -> {:?}", b.timing, tm);
                }
                // Gaps scale with the speed; the spacing statistics follow.
                let cgap = b.timing.cgap * r;
                b.timing = TimingModel { cgap, wscale: b.timing.wscale, ..tm };
                self.history.clear();
            }
        }
        for (run, prof) in buf {
            self.feed(run, prof);
        }
    }

    fn feed(&mut self, run: Run, prof: Option<Vec<f32>>) {
        if std::env::var_os("CWSK_RUNS").is_some() {
            let dot = self.beam.as_ref().map(|b| b.timing.dot).unwrap_or(0.0);
            eprintln!("RUN {:?} ev={:.1} dot={:.0} tent={:?}", run, prof.as_ref().map(|p| p.iter().sum::<f32>() / dot.max(1.0)).unwrap_or(f32::NAN), dot, self.beam.as_ref().map(|b| b.tentative()));
        }
        if !matches!(run, Run::Flush) {
            self.history.push_back(run);
            if self.history.len() > 64 {
                self.history.pop_front();
            }
        }
        if self.beam.is_none() {
            if !matches!(run, Run::Flush) {
                self.boot.push(run);
                self.boot_prof.push(prof);
            }
            let nmarks = self.boot.iter().filter(|r| matches!(r, Run::Mark(_))).count();
            // Wait for a few gaps longer than an element gap too, so the
            // character gap is measured rather than assumed (Farnsworth).
            if (nmarks >= 8 && (boot_long_gaps(&self.boot) >= 3 || nmarks >= 16)) || (matches!(run, Run::Flush) && nmarks >= 4) {
                if let Some(tm) = estimate_timing(&self.boot) {
                    if std::env::var_os("CWSK_DEBUG").is_some() {
                        eprintln!("boot: {:?} wpm={:.1} snr={:.1}dB runs={:?}", tm, tm.wpm(), self.snr_db(), &self.boot);
                    }
                    let mut b = BeamDecoder::new(self.params.clone(), self.lm.clone(), tm);
                    let profs = std::mem::take(&mut self.boot_prof);
                    for (r, p) in std::mem::take(&mut self.boot).into_iter().zip(profs) {
                        b.adapt(r);
                        b.push_soft(r, p.as_deref());
                    }
                    if matches!(run, Run::Flush) {
                        b.push(Run::Flush);
                    }
                    self.beam = Some(b);
                } else if matches!(run, Run::Flush) {
                    self.boot.clear();
                    self.boot_prof.clear();
                }
            } else if matches!(run, Run::Flush) && nmarks < 4 {
                // Too little to estimate speed; drop to avoid stale bootstrap.
                if self.boot.len() > 40 {
                    self.boot.clear();
                    self.boot_prof.clear();
                }
            }
            return;
        }
        if let Run::Mark(_) = run {
            self.marks_since_check += 1;
        }
        // Periodically re-estimate speed in case a different operator (or
        // a speed change) appears on this frequency.
        let poor_fit = self.beam.as_ref().is_some_and(|b| b.fit < 0.7);
        if self.marks_since_check >= if poor_fit { 8 } else { 16 } {
            self.marks_since_check = 0;
            let hist: Vec<Run> = self.history.iter().copied().collect();
            if let Some(tm) = estimate_timing(&hist) {
                let b = self.beam.as_mut().unwrap();
                let r = tm.dot / b.timing.dot;
                // Switch speed when the new estimate is clearly more likely,
                // or merely better when the current model fits poorly (e.g.
                // it was bootstrapped on noise before the station started).
                let poor = b.fit < 0.7;
                if !(0.7..=1.43).contains(&r) && (clearly_better(&hist, b.timing.dot, tm.dot) || (poor && better(&hist, b.timing.dot, tm.dot))) {
                    if std::env::var_os("CWSK_DEBUG").is_some() {
                        eprintln!("re-estimate: {:?} -> {:?}", b.timing, tm);
                    }
                    b.timing = tm;
                } else {
                    // Character gaps can be stuck classified as word gaps
                    // (Farnsworth); trust the spacing statistics.
                    let rc = tm.cgap / b.timing.cgap;
                    if !(0.75..=1.33).contains(&rc) {
                        b.timing.cgap = tm.cgap.clamp(b.timing.dot * 2.0, b.timing.dot * 25.0);
                    }
                }
                // The two clusters of recent long gaps know this sender's
                // character and word spacing better than a fresh estimate:
                // without this a slow sender's long character gaps were
                // read as word gaps after every re-estimate ("W 0 U").
                self.update_spacing();
            }
            if std::env::var_os("CWSK_TIMING").is_some() {
                let t = &self.beam.as_ref().unwrap().timing;
                eprintln!("TIMING dot {:.0} dash {:.0} egap {:.0} cgap {:.0} wgap {:.0} clusters {:?}", t.dot, t.dash, t.egap, t.cgap, t.wgap(), gap_clusters(&self.long_gaps.iter().copied().collect::<Vec<_>>()));
            }
        }
        if let Run::Space(sp) = run {
            self.note_long_gap(sp);
        }
        let b = self.beam.as_mut().unwrap();
        b.adapt(run);
        b.push_soft(run, prof.as_deref());
    }

    /// This sender's character and word spacing, from the two clusters of
    /// recent long gaps.
    /// Records a character or word gap (not a pause) for `gap_clusters`.
    fn note_long_gap(&mut self, sp: f32) {
        let Some(b) = &self.beam else { return };
        let t = &b.timing;
        let nominal = t.wgap() / t.wscale;
        if sp > 2.0 * t.egap.max(t.dot) && sp < 2.5 * nominal {
            self.long_gaps.push_back(sp);
            if self.long_gaps.len() > 48 {
                self.long_gaps.pop_front();
            }
            self.update_spacing();
        }
    }

    fn update_spacing(&mut self) {
        let lg: Vec<f32> = self.long_gaps.iter().copied().collect();
        if let Some((c, w)) = gap_clusters(&lg) {
            let Some(b) = self.beam.as_mut() else { return };
            let t = &mut b.timing;
            // The clusters are robust: trust them even when the tracked
            // character gap has drifted far off (it can absorb word gaps
            // with a heavy fist), as long as the lower cluster is not
            // just element gaps.
            if c > 1.5 * t.egap.max(0.7 * t.dot) {
                t.cgap = c.clamp(t.dot * 2.0, t.dot * 25.0);
            }
            let nominal = t.wgap() / t.wscale;
            t.wscale = (w / nominal).clamp(0.55, 1.6);
        }
    }

    /// Forces out any pending text (e.g. at end of a file).
    pub fn finish(&mut self) {
        if let Some(mut v) = self.vit.take() {
            v.decide(0, &mut self.vit_out);
            self.feed_vit();
            self.with_vit_state(|d| d.finish_runs());
        } else {
            self.finish_runs();
        }
        if let Some(b) = &mut self.beam {
            let c = b.take_committed();
            self.word.extend_from_slice(&c);
        }
        if !self.word.ends_with(b" ") && !self.word.is_empty() {
            self.word.push(b' ');
        }
        self.release_words();
        self.text.append(&mut self.held);
    }

    fn finish_runs(&mut self) {
        self.release_over();
        if self.beam.is_none() && !self.boot.is_empty() {
            self.on_run(Run::Flush, None);
        }
        if !self.flushed {
            let dur = self.run_frames as f32 * self.frame_ms;
            if self.state {
                self.on_run(Run::Mark(dur), None);
            } else {
                self.on_run(Run::Space(dur), None);
            }
            self.on_run(Run::Flush, None);
            self.flushed = true;
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

    /// Like `runs_for`, but `#` sends the error sign (8 dits) as one word.
    fn with_error(text: &str, dot: f32) -> Vec<Run> {
        let mut v = Vec::new();
        for (i, part) in text.split('#').enumerate() {
            if i > 0 {
                for k in 0..8 {
                    if k > 0 {
                        v.push(Run::Space(dot));
                    }
                    v.push(Run::Mark(dot));
                }
                v.push(Run::Space(7.0 * dot));
            }
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            let mut r = runs_for(part, dot);
            r.pop(); // Flush
            v.extend(r);
        }
        v.push(Run::Flush);
        v
    }

    /// Per-frame evidence profile for a run: `z` noise sigmas for its
    /// whole length, at `fm` ms per frame.
    fn flat(len: f32, z: f32, fm: f32) -> Vec<f32> {
        vec![z * fm; (len / fm).round() as usize]
    }

    #[test]
    fn faded_dit_is_recovered() {
        // "AN" with the dit of the A missed by the slicer: it sees a long
        // gap then a dash, i.e. "TN"-like timing; the gap holds a bump at
        // the right place.
        let lm = Arc::new(LanguageModel::uniform());
        let dot = 60.0;
        let fm = 2.0;
        let tm = TimingModel::from_dot(dot);
        let mut b = BeamDecoder::new(DecoderParams::default(), lm, tm);
        // Leading context: "E " (a dit and a word gap).
        b.push_soft(Run::Mark(dot), Some(&flat(dot, 3.0, fm)));
        // Word gap, a dit the squelch held off (z=+3 vs -1.5 noise),
        // element gap.
        let mut gap = flat(7.0 * dot, -1.5, fm);
        gap.extend(flat(dot, 3.0, fm));
        gap.extend(flat(dot, -1.5, fm));
        b.push_soft(Run::Space(9.0 * dot), Some(&gap));
        b.push_soft(Run::Mark(3.0 * dot), Some(&flat(3.0 * dot, 3.0, fm)));
        b.push_soft(Run::Space(3.0 * dot), Some(&flat(3.0 * dot, -1.5, fm)));
        b.push_soft(Run::Mark(3.0 * dot), Some(&flat(3.0 * dot, 3.0, fm)));
        b.push_soft(Run::Space(dot), Some(&flat(dot, -1.5, fm)));
        b.push_soft(Run::Mark(dot), Some(&flat(dot, 3.0, fm)));
        b.push(Run::Space(10.0 * dot));
        b.push(Run::Flush);
        let out = String::from_utf8(b.take_committed()).unwrap();
        assert_eq!(out.trim(), "E AN");
        // Without the bump (flat noise), no dit is invented.
        let mut b = BeamDecoder::new(DecoderParams::default(), Arc::new(LanguageModel::uniform()), TimingModel::from_dot(dot));
        b.push_soft(Run::Mark(dot), Some(&flat(dot, 3.0, fm)));
        b.push_soft(Run::Space(9.0 * dot), Some(&flat(9.0 * dot, -1.5, fm)));
        b.push_soft(Run::Mark(3.0 * dot), Some(&flat(3.0 * dot, 3.0, fm)));
        b.push_soft(Run::Space(3.0 * dot), Some(&flat(3.0 * dot, -1.5, fm)));
        b.push_soft(Run::Mark(3.0 * dot), Some(&flat(3.0 * dot, 3.0, fm)));
        b.push_soft(Run::Space(dot), Some(&flat(dot, -1.5, fm)));
        b.push_soft(Run::Mark(dot), Some(&flat(dot, 3.0, fm)));
        b.push(Run::Space(10.0 * dot));
        b.push(Run::Flush);
        let out = String::from_utf8(b.take_committed()).unwrap();
        assert_eq!(out.trim(), "E TN");
    }

    #[test]
    fn word_gap_clusters() {
        // Character gaps ~3 dots, squeezed word gaps ~4.5 dots.
        let mut g = Vec::new();
        for i in 0..30 {
            g.push(180.0 + (i % 5) as f32 * 8.0);
            if i % 4 == 0 {
                g.push(270.0 + (i % 3) as f32 * 10.0);
            }
        }
        let (c, w) = gap_clusters(&g).unwrap();
        assert!((c - 196.0).abs() < 15.0 && (w - 280.0).abs() < 20.0, "{c} {w}");
        // One cluster only: no split.
        assert!(gap_clusters(&[180.0; 30]).is_none());
    }

    #[test]
    fn run_together_bk() {
        let lm = Arc::new(LanguageModel::new());
        let dot = 60.0;
        let mut runs = runs_for("TNX", dot);
        runs.pop();
        runs.pop();
        runs.push(Run::Space(7.0 * dot));
        // -...-.- with element gaps only.
        for (i, e) in "-...-.-".bytes().enumerate() {
            if i > 0 {
                runs.push(Run::Space(dot));
            }
            runs.push(Run::Mark(if e == b'-' { 3.0 * dot } else { dot }));
        }
        runs.push(Run::Space(10.0 * dot));
        runs.push(Run::Flush);
        let mut b = BeamDecoder::new(DecoderParams::default(), lm, TimingModel::from_dot(dot));
        for r in runs {
            b.push(r);
        }
        assert_eq!(String::from_utf8(b.take_committed()).unwrap().trim(), "TNX BK");
    }

    #[test]
    fn error_sign_erases() {
        let lm = Arc::new(LanguageModel::new());
        let dot = 60.0;
        for (sent, want) in [
            ("DE W1AX # W1AW K", "DE W1AW K"),
            ("NAME IS JPHN # JOHN", "NAME IS JOHN"),
        ] {
            let runs = with_error(sent, dot);
            let mut d = CwDecoder::new(5.0, DecoderParams::default(), lm.clone());
            let tm = estimate_timing(&runs_for("CQ CQ DE W1AW K", dot)).unwrap();
            d.beam = Some(BeamDecoder::new(DecoderParams::default(), lm.clone(), tm));
            for r in runs {
                d.beam.as_mut().unwrap().push(r);
            }
            d.finish();
            assert_eq!(d.take_text().trim(), want, "{sent}");
        }
        assert_eq!(apply_errors(b"DE K1AB\x08 K1ABC"), b"DE K1ABC");
        assert_eq!(apply_errors(b"DE K1ABX \x08 K1ABC"), b"DE K1ABC");
    }
}
