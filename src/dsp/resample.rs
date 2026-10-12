//! Arbitrary-ratio resampler to the internal processing rate.
//!
//! A windowed-sinc low-pass is evaluated only at the input positions needed
//! for each output sample, followed by linear interpolation. CW audio lives
//! below ~3 kHz so this is more than accurate enough.

pub struct Resampler {
    in_rate: f64,
    out_rate: f64,
    taps: Vec<f32>,
    hist: Vec<f32>,
    /// Position of the next output sample, in input samples relative to the
    /// start of `hist`.
    pos: f64,
    passthrough: bool,
}

impl Resampler {
    pub fn new(in_rate: u32, out_rate: u32) -> Self {
        let in_rate = in_rate as f64;
        let out_rate = out_rate as f64;
        let passthrough = (in_rate - out_rate).abs() < 0.5;
        let cutoff = (out_rate * 0.42).min(in_rate * 0.45);
        // Transition width ~ 0.08 * out_rate.
        let tw = (out_rate * 0.08) / in_rate;
        let mut n = ((4.0 / tw) as usize) | 1;
        n = n.clamp(15, 1023);
        let fc = cutoff / in_rate;
        let mid = (n / 2) as f64;
        let mut taps: Vec<f32> = (0..n)
            .map(|i| {
                let x = i as f64 - mid;
                let sinc = if x == 0.0 { 2.0 * fc } else { (2.0 * std::f64::consts::PI * fc * x).sin() / (std::f64::consts::PI * x) };
                // Blackman window
                let w = 0.42 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()
                    + 0.08 * (4.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos();
                (sinc * w) as f32
            })
            .collect();
        let s: f32 = taps.iter().sum();
        for t in &mut taps {
            *t /= s;
        }
        Self { in_rate, out_rate, taps, hist: Vec::new(), pos: 0.0, passthrough }
    }

    #[inline]
    fn filt_at(&self, i: usize) -> f32 {
        // Output of the FIR whose newest sample is hist[i + taps-1]
        let n = self.taps.len();
        self.hist[i..i + n].iter().zip(&self.taps).map(|(a, b)| a * b).sum()
    }

    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        if self.passthrough {
            out.extend_from_slice(input);
            return;
        }
        let n = self.taps.len();
        self.hist.extend_from_slice(input);
        let step = self.in_rate / self.out_rate;
        loop {
            let i0 = self.pos.floor() as usize;
            if i0 + 1 + n > self.hist.len() {
                break;
            }
            let frac = (self.pos - i0 as f64) as f32;
            let a = self.filt_at(i0);
            let b = self.filt_at(i0 + 1);
            out.push(a + (b - a) * frac);
            self.pos += step;
        }
        let drop = (self.pos.floor() as usize).min(self.hist.len());
        if drop > 0 {
            self.hist.drain(..drop);
            self.pos -= drop as f64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratio_is_correct() {
        let mut r = Resampler::new(48000, 8000);
        let input: Vec<f32> = (0..48000).map(|i| (i as f32 * 0.05).sin()).collect();
        let mut out = Vec::new();
        for ch in input.chunks(1000) {
            r.process(ch, &mut out);
        }
        assert!((out.len() as i64 - 8000).abs() < 200, "{}", out.len());
        let mut r = Resampler::new(44100, 8000);
        let mut out = Vec::new();
        r.process(&vec![0.0; 44100], &mut out);
        assert!((out.len() as i64 - 8000).abs() < 200, "{}", out.len());
    }
}
