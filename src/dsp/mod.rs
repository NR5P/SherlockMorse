//! Signal processing building blocks.

pub mod channel;
pub mod resample;
pub mod spectrum;

/// Direct-form II transposed biquad.
#[derive(Clone, Debug, Default)]
pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    /// RBJ low-pass.
    pub fn lowpass(fs: f32, fc: f32, q: f32) -> Self {
        let w0 = 2.0 * std::f32::consts::PI * fc / fs;
        let (s, c) = w0.sin_cos();
        let alpha = s / (2.0 * q);
        let a0 = 1.0 + alpha;
        Self {
            b0: (1.0 - c) / 2.0 / a0,
            b1: (1.0 - c) / a0,
            b2: (1.0 - c) / 2.0 / a0,
            a1: -2.0 * c / a0,
            a2: (1.0 - alpha) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Updates coefficients, keeping state.
    pub fn retune(&mut self, other: &Biquad) {
        self.b0 = other.b0;
        self.b1 = other.b1;
        self.b2 = other.b2;
        self.a1 = other.a1;
        self.a2 = other.a2;
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

/// 4th-order Butterworth low-pass built from two biquads.
#[derive(Clone, Debug, Default)]
pub struct Lowpass4 {
    s1: Biquad,
    s2: Biquad,
}

impl Lowpass4 {
    pub fn new(fs: f32, fc: f32) -> Self {
        Self { s1: Biquad::lowpass(fs, fc, 0.541_196_1), s2: Biquad::lowpass(fs, fc, 1.306_563) }
    }
    pub fn retune(&mut self, fs: f32, fc: f32) {
        let n = Self::new(fs, fc);
        self.s1.retune(&n.s1);
        self.s2.retune(&n.s2);
    }
    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        self.s2.process(self.s1.process(x))
    }
}
