"""Draws one signal's keying as a strip picture and lists its mark/gap
timings, to read the Morse by eye (or have Claude read it) and compare with
the decoder's copy. Narrow (~60 Hz) envelope at the given audio frequency.
usage: python3 -I tools/keystrip.py file.wav HZ [out.png] [seconds-per-row]
       (WAV, 16-bit mono; find HZ with `cwtool keying` or `cwtool waterfall`)
Prints "t MARK|gap ms" lines; the picture has one row per 6 s (red tick = 1 s,
black = above the half-way level between noise and signal)."""
import sys, wave
import numpy as np
from PIL import Image, ImageDraw

fn, f0 = sys.argv[1], float(sys.argv[2])
out = sys.argv[3] if len(sys.argv) > 3 else None
row_s = float(sys.argv[4]) if len(sys.argv) > 4 else 6.0
w = wave.open(fn)
sr = w.getframerate()
x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(float)
z = x * np.exp(-2j * np.pi * f0 * np.arange(len(x)) / sr)
ker = np.ones(int(sr * 0.008)) / int(sr * 0.008)   # two 8 ms boxcars
z = np.convolve(np.convolve(z, ker, "same"), ker, "same")
e = 20 * np.log10(np.abs(z[:: sr // 1000]) + 1e-9)  # 1 ms per sample
lo, hi = np.percentile(e, 20), np.percentile(e, 99.5)
on = e > (lo + hi) / 2
start = 0
for i in np.append(np.flatnonzero(np.diff(on.astype(int))) + 1, len(on)):
    print(f"{start / 1000:8.3f} {'MARK' if on[start] else 'gap '} {i - start:5d} ms")
    start = i
if out:
    v = np.clip((e[::2] - lo) / (hi - lo), 0, 1)       # 2 ms per pixel
    px, H = int(row_s * 500), 70
    rows = int(np.ceil(len(v) / px))
    img = Image.new("RGB", (px, rows * (H + 14)), "white")
    d = ImageDraw.Draw(img)
    for r in range(rows):
        y0 = r * (H + 14) + 12
        d.text((2, y0 - 12), f"{r * row_s:.0f}s", fill="blue")
        for i, a in enumerate(v[r * px:(r + 1) * px]):
            d.line([(i, y0 + H), (i, y0 + H - int(a * H))], fill=(0, 0, 0) if a > 0.5 else (170, 170, 170))
        for s in range(int(row_s) + 1):
            d.line([(s * 500, y0 - 3), (s * 500, y0)], fill="red")
    img.save(out)
