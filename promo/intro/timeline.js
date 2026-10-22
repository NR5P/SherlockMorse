// Shared timing for the Sherlock Morse intro: the page (intro.html) draws from it and
// render.mjs builds the soundtrack from it, so picture and sound stay in sync.
//
// A magnifying glass searches a foggy waterfall, decodes SHERLOCK, and becomes the cameo.
(function (g) {
  const FPS = 60;

  const MORSE = {
    A: '.-', B: '-...', C: '-.-.', D: '-..', E: '.', F: '..-.', G: '--.', H: '....', I: '..', J: '.---',
    K: '-.-', L: '.-..', M: '--', N: '-.', O: '---', P: '.--.', Q: '--.-', R: '.-.', S: '...', T: '-',
    U: '..-', V: '...-', W: '.--', X: '-..-', Y: '-.--', Z: '--..',
    0: '-----', 1: '.----', 2: '..---', 3: '...--', 4: '....-', 5: '.....', 6: '-....', 7: '--...', 8: '---..', 9: '----.',
    '?': '..--..', '/': '-..-.',
  };

  // Morse for `text` starting at `t`; returns key-down intervals and per-letter timing.
  function morse(text, t, unit) {
    const keys = [];
    const chars = [];
    for (let i = 0; i < text.length; i++) {
      if (text[i] === ' ') { t += 4 * unit; continue; }
      const code = MORSE[text[i]];
      const start = t;
      const els = [];
      for (let j = 0; j < code.length; j++) {
        const len = (code[j] === '.' ? 1 : 3) * unit;
        keys.push([t, t + len]);
        els.push({ sym: code[j], a: t, b: t + len });
        t += len;
        if (j < code.length - 1) t += unit;
      }
      chars.push({ ch: text[i], code, start, end: t, els });
      t += 3 * unit;
    }
    return { keys, chars, end: t - 3 * unit };
  }

  const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
  const seg = (t, a, b) => clamp((t - a) / (b - a));

  // ---------- the clue ----------
  const UNIT = 0.036; // 33 WPM
  const WORD = 'SHERLOCK';
  const main = morse(WORD, 2.75, UNIT);

  const I = main.end + 0.75; // impact
  const T = {
    lensIn: 0.15,
    settle: 2.6,
    morseStart: 2.75,
    morseEnd: main.end,
    riseStart: main.end + 0.15,
    impact: I,
    moveStart: I + 0.15,
    moveEnd: I + 0.9,
    wordStart: I + 0.7,
    letterStep: 0.045,
    smoke: I + 0.95,
    skimmer: I + 1.35,
    tagline: I + 1.75,
    charStep: 0.028,
    chips: I + 2.45,
    chipStep: 0.1,
    shine: I + 3.0,
    fadeOut: I + 4.55,
  };
  const DURATION = I + 5.15;

  // The pipe smoke rises as "?" (..--..), bottom puff first.
  const SMOKE_UNIT = 0.055;
  const smoke = morse('?', T.smoke, SMOKE_UNIT);

  // ---------- the band ----------
  // Waterfall columns (0..640, 3 screen px each); the target sits at MID.
  const WF_W = 640, WF_H = 360, MID = 320;
  const RPS = 2 * FPS; // waterfall rows per second
  const HISTORY = WF_H / RPS + 0.5; // the waterfall shows this much past
  const others = [
    { col: 58, text: 'CQ CQ DE K7EXQ K7EXQ K', wpm: 24, call: 'K7EXQ', amp: 0.62 },
    { col: 122, text: '5NN TU DE DL2ZZ 5NN', wpm: 31, call: 'DL2ZZ', amp: 0.5 },
    { col: 176, text: 'TEST G4KHS G4KHS TEST', wpm: 28, call: 'G4KHS', amp: 0.7 },
    { col: 238, text: 'R R TNX FER QSO 73', wpm: 18, call: null, amp: 0.45 },
    { col: 395, text: 'CQ DX DE JA1XQ JA1XQ', wpm: 26, call: 'JA1XQ', amp: 0.58 },
    { col: 452, text: 'QRL? QRL? QRL?', wpm: 22, call: null, amp: 0.4 },
    { col: 512, text: 'CQ CQ VE3QZ VE3QZ K', wpm: 29, call: 'VE3QZ', amp: 0.55 },
    { col: 584, text: '73 GL ES TU SK', wpm: 20, call: null, amp: 0.42 },
  ];
  // Each one has been sending (looping) since well before the video starts.
  others.forEach((s, i) => {
    const unit = 1.2 / s.wpm;
    const one = morse(s.text, 0, unit);
    const period = one.end + 7 * unit + 0.25 * i;
    const keys = [];
    for (let t0 = -HISTORY - period * (0.3 + 0.13 * i); t0 < 20; t0 += period) {
      for (const [a, b] of one.keys) keys.push([a + t0, b + t0]);
    }
    s.keys = keys;
    s.pitch = 650 + (s.col - MID) * 1.6;
  });
  const target = { col: MID, keys: main.keys, call: WORD, wpm: 33, amp: 0.85, pitch: 650 };
  const signals = others.concat([target]);

  // How much of [t0, t1) the key is down.
  function cover(keys, t0, t1) {
    let s = 0;
    for (const [a, b] of keys) {
      if (b <= t0) continue;
      if (a >= t1) break;
      s += Math.min(b, t1) - Math.max(a, t0);
    }
    return s;
  }

  // ---------- the lens ----------
  // Screen-space path of the lens centre (Catmull-Rom through timed points).
  const LENS_R = 230;
  const PATH = [
    [-0.4, -420, 980],
    [T.lensIn, -260, 860],
    [1.0, 470, 590],
    [1.75, 1330, 430],
    [2.3, 1060, 470],
    [T.settle, 960, 430],
    [T.riseStart, 960, 430],
  ];
  function pathAt(t) {
    const P = PATH;
    if (t <= P[0][0]) return { x: P[0][1], y: P[0][2] };
    if (t >= P[P.length - 1][0]) return { x: P[P.length - 1][1], y: P[P.length - 1][2] };
    let i = 0;
    while (t > P[i + 1][0]) i++;
    const p0 = P[Math.max(0, i - 1)], p1 = P[i], p2 = P[i + 1], p3 = P[Math.min(P.length - 1, i + 2)];
    const h = p2[0] - p1[0];
    const u = (t - p1[0]) / h;
    const s = u * u * (3 - 2 * u); // ease in and out of every waypoint
    const tan = (a, b, k) => ((b[k] - a[k]) / Math.max(1e-6, b[0] - a[0])) * h * 0.5;
    const out = [];
    for (const k of [1, 2]) {
      const m1 = i === 0 ? 0 : tan(p0, p2, k), m2 = i + 2 >= P.length ? 0 : tan(p1, p3, k);
      const s2 = s * s, s3 = s2 * s;
      out.push((2 * s3 - 3 * s2 + 1) * p1[k] + (s3 - 2 * s2 + s) * m1 + (-2 * s3 + 3 * s2) * p2[k] + (s3 - s2) * m2);
    }
    return { x: out[0], y: out[1] };
  }
  function lensSpeed(t) {
    const a = pathAt(t - 0.01), b = pathAt(t + 0.01);
    return Math.hypot(b.x - a.x, b.y - a.y) / 0.02;
  }

  // How loudly a signal is heard: everything is faint, the one under the glass is clear.
  function hearing(s, t) {
    if (s === target) return 1;
    const p = pathAt(t);
    const d = (s.col * 3 - p.x) / LENS_R;
    const dy = Math.max(0, Math.abs(p.y - 540) - 300) / 300;
    return (0.12 + 0.88 * Math.exp(-0.5 * (d * d) / 0.25) * Math.exp(-dy * dy)) * (1 - seg(t, T.riseStart, T.impact));
  }

  const chipLabels = ['Decodes the whole passband', 'Callsign spotting', 'CAT tuning', 'Written in Rust'];

  g.TIMELINE = {
    FPS, DURATION, UNIT, WORD, MORSE, morse, T, I, chars: main.chars, keys: main.keys, smoke,
    WF_W, WF_H, RPS, MID, LENS_R, signals, target, cover, pathAt, lensSpeed, hearing, chipLabels,
  };
})(typeof window !== 'undefined' ? window : globalThis);
