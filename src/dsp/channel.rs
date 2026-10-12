//! Narrow-band channel demodulator: complex mixer, 4th-order low-pass and
//! decimation to envelope frames, with automatic frequency control.

use super::Lowpass4;
use num_complex::Complex32;

pub struct ChannelDemod {
    fs: f32,
    pub freq: f64,
    phase: f64,
    lp_i: Lowpass4,
    lp_q: Lowpass4,
    pub bandwidth: f32,
    decim: usize,
    count: usize,
    prev: Complex32,
    afc_err: f32,
    afc_n: u32,
    pub afc_limit: (f64, f64),
}

impl ChannelDemod {
    /// `frame_samples` input samples are combined into one envelope frame.
    pub fn new(fs: f32, freq: f64, bandwidth: f32, frame_samples: usize) -> Self {
        Self {
            fs,
            freq,
            phase: 0.0,
            lp_i: Lowpass4::new(fs, bandwidth / 2.0),
            lp_q: Lowpass4::new(fs, bandwidth / 2.0),
            bandwidth,
            decim: frame_samples,
            count: 0,
            prev: Complex32::new(0.0, 0.0),
            afc_err: 0.0,
            afc_n: 0,
            afc_limit: (freq - 12.0, freq + 12.0),
        }
    }

    pub fn frame_ms(&self) -> f32 {
        self.decim as f32 * 1000.0 / self.fs
    }

    pub fn set_bandwidth(&mut self, bw: f32) {
        if (bw - self.bandwidth).abs() > 1.0 {
            self.bandwidth = bw;
            self.lp_i.retune(self.fs, bw / 2.0);
            self.lp_q.retune(self.fs, bw / 2.0);
        }
    }

    pub fn set_freq(&mut self, f: f64) {
        self.freq = f;
        self.afc_limit = (f - 12.0, f + 12.0);
    }

    /// Processes input; pushes complex baseband frames to `out`.
    pub fn process(&mut self, input: &[f32], out: &mut Vec<Complex32>) {
        let dphi = 2.0 * std::f64::consts::PI * self.freq / self.fs as f64;
        for &x in input {
            let (s, c) = self.phase.sin_cos();
            self.phase += dphi;
            if self.phase > std::f64::consts::TAU {
                self.phase -= std::f64::consts::TAU;
            }
            let i = self.lp_i.process(x * c as f32);
            let q = self.lp_q.process(-x * s as f32);
            self.count += 1;
            if self.count >= self.decim {
                self.count = 0;
                out.push(Complex32::new(i, q) * 2.0);
            }
        }
    }

    /// Feeds a frame for AFC while a mark is present.
    pub fn afc(&mut self, z: Complex32, is_mark: bool) {
        if is_mark && self.prev.norm_sqr() > 0.0 {
            let d = z * self.prev.conj();
            let hz = d.arg() / (std::f32::consts::TAU * self.frame_ms() / 1000.0);
            self.afc_err += hz;
            self.afc_n += 1;
            if self.afc_n >= 50 {
                let err = (self.afc_err / self.afc_n as f32) as f64;
                let nf = (self.freq + 0.3 * err).clamp(self.afc_limit.0, self.afc_limit.1);
                self.freq = nf;
                self.afc_err = 0.0;
                self.afc_n = 0;
            }
        }
        self.prev = if is_mark { z } else { Complex32::new(0.0, 0.0) };
    }
}
