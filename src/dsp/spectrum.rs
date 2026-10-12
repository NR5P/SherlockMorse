//! Short-time power spectrum for the waterfall and signal detection.

use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use std::sync::Arc;

pub struct Spectrum {
    pub size: usize,
    pub hop: usize,
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    buf: Vec<f32>,
    work: Vec<Complex32>,
    scratch: Vec<Complex32>,
}

impl Spectrum {
    pub fn new(size: usize, hop: usize) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(size);
        let window: Vec<f32> = (0..size)
            .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / size as f32).cos())
            .collect();
        let scratch = vec![Complex32::default(); fft.get_inplace_scratch_len()];
        Self { size, hop, fft, window, buf: Vec::with_capacity(size * 2), work: vec![Complex32::default(); size], scratch }
    }

    /// Pushes samples; calls `f` with each new power spectrum (size/2 bins).
    pub fn process(&mut self, input: &[f32], mut f: impl FnMut(&[f32])) {
        self.buf.extend_from_slice(input);
        let mut power = vec![0.0f32; self.size / 2];
        while self.buf.len() >= self.size {
            for (i, w) in self.work.iter_mut().enumerate() {
                *w = Complex32::new(self.buf[i] * self.window[i], 0.0);
            }
            self.fft.process_with_scratch(&mut self.work, &mut self.scratch);
            let norm = 1.0 / (self.size as f32 * self.size as f32);
            for (p, c) in power.iter_mut().zip(&self.work) {
                *p = c.norm_sqr() * norm;
            }
            f(&power);
            self.buf.drain(..self.hop);
        }
    }
}
