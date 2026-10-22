// Soundtrack for the intro, synthesized from the same timeline as the picture:
// fog and band hiss, a sneaking pizzicato bass, every CW signal (louder under the
// glass), SHERLOCK itself, a riser, the impact, a pad, and the "?" in the smoke.
const SR = 48000;

export function renderAudio(TL) {
  const T = TL.T;
  const I = T.impact;
  const N = Math.ceil(TL.DURATION * SR);
  const dry = [new Float32Array(N), new Float32Array(N)];
  const wet = [new Float32Array(N), new Float32Array(N)]; // sent through the reverb

  let seed = 0x5be1c0c;
  const rnd = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2147483648 - 1; };
  const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
  const seg = (t, a, b) => clamp((t - a) / (b - a));
  const coef = (fc) => 1 - Math.exp(-2 * Math.PI * fc / SR);
  const at = (t) => Math.round(t * SR);
  const midi = (m) => 440 * Math.pow(2, (m - 69) / 12);

  // band hiss and a low fog wind, a little different in each ear
  for (let ch = 0; ch < 2; ch++) {
    let lp1 = 0, lp2 = 0, hp = 0, w1 = 0, w2 = 0;
    const a1 = coef(2600), ah = coef(300);
    for (let n = 0; n < N; n++) {
      const t = n / SR;
      const x = rnd();
      lp1 += a1 * (x - lp1); lp2 += a1 * (lp1 - lp2);
      hp += ah * (lp2 - hp);
      const wc = coef(180 + 120 * Math.sin(t * 0.7 + ch));
      w1 += wc * (rnd() - w1); w2 += wc * (w1 - w2);
      let g = 0.11 * seg(t, 0, 0.5) * (1 - seg(t, T.riseStart, I));
      g += 0.03 * seg(t, I, I + 0.5) * (1 - seg(t, T.fadeOut - 0.6, T.fadeOut + 0.4));
      dry[ch][n] += (lp2 - hp) * g + w2 * 1.6 * (0.6 * seg(t, 0, 1) * (1 - 0.6 * seg(t, I, I + 1)));
    }
  }

  // every station on the band: faint, until the glass passes over it
  const RAMP = 0.005;
  for (const s of TL.signals) {
    const pan = 0.5 + (s.col - TL.MID) / 900;
    const gain = s === TL.target ? 0.34 : 0.22 * s.amp;
    for (const [a, b] of s.keys) {
      if (b < 0 || a > I) continue;
      let ph = 0;
      for (let n = Math.max(0, at(a)); n < at(b + RAMP) && n < N; n++) {
        const t = n / SR;
        const rise = t < a + RAMP ? 0.5 - 0.5 * Math.cos(Math.PI * (t - a) / RAMP) : 1;
        const fall = t > b ? 0.5 + 0.5 * Math.cos(Math.PI * clamp((t - b) / RAMP)) : 1;
        ph += 2 * Math.PI * s.pitch / SR;
        const v = gain * rise * fall * TL.hearing(s, t) * (Math.sin(ph) + 0.04 * Math.sin(2 * ph));
        dry[0][n] += v * (1 - pan) * 2 * 0.7; dry[1][n] += v * pan * 2 * 0.7;
        wet[0][n] += v * 0.3; wet[1][n] += v * 0.3;
      }
    }
  }

  // a breath of air whenever the glass moves quickly
  {
    const lp = [0, 0], hpS = [0, 0];
    for (let n = 0; n < at(T.settle + 0.3); n++) {
      const t = n / SR;
      const v = clamp(TL.lensSpeed(t) / 1800);
      const p = clamp(TL.pathAt(t).x / 1920);
      for (let ch = 0; ch < 2; ch++) {
        const r = rnd();
        lp[ch] += coef(1500 + 3500 * v) * (r - lp[ch]);
        hpS[ch] += coef(600) * (lp[ch] - hpS[ch]);
        const s = (lp[ch] - hpS[ch]) * 0.13 * v * v * (ch ? p : 1 - p) * 2;
        dry[ch][n] += s; wet[ch][n] += s * 0.5;
      }
    }
  }

  // pizzicato bass, sneaking about in D minor (Karplus-Strong)
  function pluck(t0, f, amp, pan, dur = 0.9) {
    const n0 = at(t0), len = Math.round(SR / f);
    const buf = new Float32Array(len);
    for (let i = 0; i < len; i++) buf[i] = rnd();
    // a softer attack than raw noise
    for (let k = 0; k < 3; k++) for (let i = 1; i < len; i++) buf[i] = 0.5 * (buf[i] + buf[i - 1]);
    let idx = 0;
    for (let n = 0; n < at(dur) && n0 + n < N; n++) {
      const next = (idx + 1) % len;
      const y = buf[idx];
      buf[idx] = 0.4985 * (y + buf[next]);
      idx = next;
      const env = Math.exp(-n / (0.35 * SR));
      const s = y * amp * env;
      dry[0][n0 + n] += s * (1 - pan); dry[1][n0 + n] += s * pan;
      wet[0][n0 + n] += s * 0.3; wet[1][n0 + n] += s * 0.3;
    }
  }
  {
    const E8 = 0.25; // eighths at 120 bpm
    const D2 = 38, F2 = 41, E2 = 40, A1 = 33, Bb1 = 34, Cs2 = 37;
    const line = [D2, 0, A1, 0, D2, F2, E2, 0, D2, 0, A1, 0, Bb1, 0, Cs2, 0];
    for (let k = 0; ; k++) {
      const t0 = 0.35 + k * E8;
      if (t0 > T.riseStart - 0.1) break;
      const m = line[k % line.length];
      if (!m) continue;
      const fade = 0.6 + 0.4 * seg(t0, 0.35, 1.2);
      pluck(t0, midi(m), 0.55 * fade, 0.42, 0.7);
      pluck(t0 + 0.004, midi(m + 12), 0.12 * fade, 0.6, 0.4);
    }
  }

  // a low string drone under the search, D and A
  {
    const vs = [midi(26), midi(33), midi(38)].flatMap((f) => [f * 0.998, f * 1.002]);
    const ph = vs.map(() => Math.abs(rnd()) * 6.28);
    let lp = [0, 0];
    for (let n = 0; n < at(I); n++) {
      const t = n / SR;
      let s = 0;
      vs.forEach((f, i) => { ph[i] += 2 * Math.PI * f / SR; s += ((ph[i] / (2 * Math.PI)) % 1) * 2 - 1; });
      const env = 0.035 * seg(t, 0.2, 2.5) * (1 - seg(t, T.riseStart, I));
      for (let ch = 0; ch < 2; ch++) {
        lp[ch] += coef(380 + 120 * Math.sin(t * 0.9 + ch)) * (s - lp[ch]);
        dry[ch][n] += lp[ch] * env;
        wet[ch][n] += lp[ch] * env * 0.5;
      }
    }
  }

  // riser into the impact
  {
    const t0 = T.riseStart - 0.1, t1 = I;
    const lp = [0, 0], hpS = [0, 0];
    let ph = 0;
    for (let n = at(t0); n < at(t1); n++) {
      const x = seg(n / SR, t0, t1);
      const a = coef(250 * Math.pow(36, x));
      ph += 2 * Math.PI * 146.83 * Math.pow(2, 2 * x) / SR;
      const sw = Math.sin(ph) * 0.1 * x * x;
      for (let ch = 0; ch < 2; ch++) {
        const r = rnd();
        lp[ch] += a * (r - lp[ch]);
        hpS[ch] += coef(3000) * (r - hpS[ch]);
        const s = lp[ch] * 0.5 * Math.pow(x, 2.2) + (r - hpS[ch]) * 0.16 * Math.pow(x, 3) + sw;
        dry[ch][n] += s;
        wet[ch][n] += s * 0.4;
      }
    }
  }

  // impact: sub drop, crash, a transient and a low bell
  {
    const n0 = at(I);
    let ph = 0;
    const lp = [0, 0], hpS = [0, 0];
    const bell = [[73.42, 1], [146.83 * 1.003, 0.6], [220.0 * 1.01, 0.35], [293.66 * 1.21, 0.22], [440 * 1.37, 0.12]];
    for (let n = 0; n < at(2.6) && n0 + n < N; n++) {
      const tau = n / SR;
      ph += 2 * Math.PI * (36 + 60 * Math.exp(-tau / 0.07)) / SR;
      const boom = Math.sin(ph) * 1.1 * Math.exp(-tau / 0.6) * (1 - Math.exp(-tau / 0.002));
      const click = tau < 0.004 ? rnd() * 0.6 * (1 - tau / 0.004) : 0;
      let b = 0;
      for (const [f, a] of bell) b += a * Math.sin(2 * Math.PI * f * tau) * Math.exp(-tau / (0.9 / Math.sqrt(a + 0.2)));
      b *= 0.16 * (1 - Math.exp(-tau / 0.003));
      for (let ch = 0; ch < 2; ch++) {
        const r = rnd();
        lp[ch] += coef(1200) * (r - lp[ch]);
        hpS[ch] += coef(2500) * (r - hpS[ch]);
        const crash = (r - hpS[ch]) * 0.35 * Math.exp(-tau / 0.3) + lp[ch] * 0.6 * Math.exp(-tau / 0.09);
        dry[ch][n0 + n] += boom + click + crash + b;
        wet[ch][n0 + n] += crash * 0.8 + boom * 0.15 + b * 0.8;
      }
    }
  }

  // pad: D major add9, bright on the hit then mellowing under the logo — case solved
  {
    const notes = [73.42, 110, 146.83, 185, 220, 329.63];
    const n0 = at(I), n1 = N;
    const voices = [];
    notes.forEach((f, i) => { for (const d of [-1, 1]) voices.push({ f: f * Math.pow(2, d * 4 / 1200), pan: 0.5 + d * 0.3 * ((i % 2) * 2 - 1), ph: Math.abs(rnd()) * 6.28 }); });
    for (let n = n0; n < n1; n++) {
      const t = n / SR, tau = t - I;
      const env = (0.55 * Math.exp(-tau / 0.35) + 0.45 * (1 - Math.exp(-tau / 0.8))) * (1 - seg(t, T.fadeOut - 0.3, TL.DURATION));
      const bright = 0.25 + 0.5 * Math.exp(-tau / 0.5);
      let l = 0, r = 0;
      for (const v of voices) {
        let s = 0;
        for (let h = 1; h <= 7; h++) s += Math.sin(v.ph * h) / h * Math.exp(-(h - 1) * (1 - bright) * 0.9);
        v.ph += 2 * Math.PI * v.f / SR;
        l += s * (1 - v.pan); r += s * v.pan;
      }
      const g = 0.055 * env;
      dry[0][n] += l * g; dry[1][n] += r * g;
      wet[0][n] += l * g * 0.6; wet[1][n] += r * g * 0.6;
    }
  }

  // "?" keyed in the smoke, soft and roomy
  for (const e of TL.smoke.chars[0].els) {
    let ph = 0;
    for (let n = at(e.a); n < at(e.b + RAMP) && n < N; n++) {
      const t = n / SR;
      const rise = t < e.a + RAMP ? 0.5 - 0.5 * Math.cos(Math.PI * (t - e.a) / RAMP) : 1;
      const fall = t > e.b ? 0.5 + 0.5 * Math.cos(Math.PI * clamp((t - e.b) / RAMP)) : 1;
      ph += 2 * Math.PI * 650 / SR;
      const s = 0.13 * rise * fall * Math.sin(ph);
      dry[0][n] += s * 0.8; dry[1][n] += s;
      wet[0][n] += s * 0.7; wet[1][n] += s * 0.7;
    }
  }

  // typewriter keys for the wordmark
  {
    const letters = 'Sherlock Morse';
    for (let i = 0; i < letters.length; i++) {
      if (letters[i] === ' ') continue;
      const n0 = at(T.wordStart + i * T.letterStep + 0.02);
      const pan = 0.3 + 0.4 * i / letters.length;
      let bp = 0, lpS = 0;
      const f = 1800 + 500 * Math.abs(rnd());
      for (let n = 0; n < at(0.09) && n0 + n < N; n++) {
        const tau = n / SR;
        const r = rnd();
        lpS += coef(f) * (r - lpS);
        bp += coef(400) * (lpS - bp);
        const thock = Math.sin(2 * Math.PI * 180 * tau) * Math.exp(-tau / 0.012);
        const s = ((lpS - bp) * 0.6 * Math.exp(-tau / 0.006) + thock * 0.12) * 0.55;
        dry[0][n0 + n] += s * (1 - pan); dry[1][n0 + n] += s * pan;
        wet[0][n0 + n] += s * 0.4; wet[1][n0 + n] += s * 0.4;
      }
    }
    // carriage bell once the name is in
    const n0 = at(T.wordStart + letters.length * T.letterStep + 0.12);
    for (let n = 0; n < at(1.4) && n0 + n < N; n++) {
      const tau = n / SR;
      const s = (Math.sin(2 * Math.PI * 2093 * tau) + 0.5 * Math.sin(2 * Math.PI * 2093 * 2.76 * tau) * Math.exp(-tau / 0.1)) * 0.05 * Math.exp(-tau / 0.5);
      dry[0][n0 + n] += s * 0.4; dry[1][n0 + n] += s * 0.6;
      wet[0][n0 + n] += s; wet[1][n0 + n] += s;
    }
  }

  // chip pops
  TL.chipLabels.forEach((_, i) => {
    const n0 = at(T.chips + i * T.chipStep + 0.05);
    for (let n = 0; n < at(0.06) && n0 + n < N; n++) {
      const tau = n / SR;
      const s = Math.sin(2 * Math.PI * (1300 + 300 * i) * tau * (1 - tau * 4)) * 0.05 * Math.exp(-tau / 0.015);
      dry[0][n0 + n] += s; dry[1][n0 + n] += s;
      wet[0][n0 + n] += s; wet[1][n0 + n] += s;
    }
  });

  // airy whoosh with the shine sweep
  {
    const t0 = T.shine - 0.05, t1 = T.shine + 0.9;
    const lp = [0, 0], hpS = [0, 0];
    for (let n = at(t0); n < at(t1); n++) {
      const x = seg(n / SR, t0, t1);
      const env = Math.sin(Math.PI * x) * 0.07;
      for (let ch = 0; ch < 2; ch++) {
        const r = rnd();
        lp[ch] += coef(9000) * (r - lp[ch]);
        hpS[ch] += coef(2500 + 3000 * x) * (lp[ch] - hpS[ch]);
        const s = (lp[ch] - hpS[ch]) * env * (ch ? x : 1 - x) * 2;
        dry[ch][n] += s; wet[ch][n] += s;
      }
    }
  }

  // small Schroeder reverb on the send
  const out = [new Float32Array(N), new Float32Array(N)];
  for (let ch = 0; ch < 2; ch++) {
    const combs = [1557, 1617, 1491, 1422, 1277, 1356].map((d) => ({ buf: new Float32Array(d + ch * 23), i: 0, lp: 0 }));
    const aps = [556, 441, 341].map((d) => ({ buf: new Float32Array(d + ch * 23), i: 0 }));
    for (let n = 0; n < N; n++) {
      const x = wet[ch][n] * 0.15;
      let y = 0;
      for (const c of combs) {
        const o = c.buf[c.i];
        c.lp = o * 0.7 + c.lp * 0.3;
        c.buf[c.i] = x + c.lp * 0.84;
        c.i = (c.i + 1) % c.buf.length;
        y += o;
      }
      for (const a of aps) {
        const o = a.buf[a.i];
        a.buf[a.i] = y + o * 0.5;
        a.i = (a.i + 1) % a.buf.length;
        y = o - y * 0.5;
      }
      out[ch][n] = dry[ch][n] + y * 0.9;
    }
  }

  // master: fade, soft clip, normalize
  let peak = 0;
  for (let ch = 0; ch < 2; ch++) {
    for (let n = 0; n < N; n++) {
      const t = n / SR;
      const fade = 1 - seg(t, TL.DURATION - 0.35, TL.DURATION);
      const v = Math.tanh(out[ch][n] * 1.1) * fade;
      out[ch][n] = v;
      peak = Math.max(peak, Math.abs(v));
    }
  }
  const norm = 0.9 / peak;
  const wav = Buffer.alloc(44 + N * 4);
  wav.write('RIFF', 0); wav.writeUInt32LE(36 + N * 4, 4); wav.write('WAVE', 8);
  wav.write('fmt ', 12); wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(2, 22);
  wav.writeUInt32LE(SR, 24); wav.writeUInt32LE(SR * 4, 28); wav.writeUInt16LE(4, 32); wav.writeUInt16LE(16, 34);
  wav.write('data', 36); wav.writeUInt32LE(N * 4, 40);
  for (let n = 0; n < N; n++) {
    wav.writeInt16LE(Math.round(clamp(out[0][n] * norm, -1, 1) * 32767), 44 + n * 4);
    wav.writeInt16LE(Math.round(clamp(out[1][n] * norm, -1, 1) * 32767), 46 + n * 4);
  }
  return wav;
}
