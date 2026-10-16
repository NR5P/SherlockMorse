#!/usr/bin/env python3
# Element-level reading of one CW signal: narrow filter around a frequency,
# envelope at 2 ms, local hysteretic threshold, and the peak frequency of
# each mark (two stations on one channel show as two frequency clusters).
# usage: tools/elements.py <wav> <freq Hz> <t0 s> <t1 s> [bw Hz] [--detail]
import sys, wave, numpy as np
path, f0, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
bw = float(sys.argv[5]) if len(sys.argv) > 5 and not sys.argv[5].startswith('--') else 150
w = wave.open(path); sr = w.getframerate(); n = w.getnframes()
x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(float)
if w.getnchannels() > 1: x = x.reshape(-1, w.getnchannels())[:, 0]
x = x[int(t0 * sr):int(t1 * sr)]
X = np.fft.rfft(x); fr = np.fft.rfftfreq(len(x), 1 / sr); X[np.abs(fr - f0) > bw / 2] = 0
z = np.fft.irfft(X * 2, len(x)) + 1j * np.fft.irfft(-1j * X * 2, len(x))
hop = int(sr * 0.002); m = len(z) // hop
env = np.abs(z[:m * hop]).reshape(m, hop).mean(1)
win = int(1.5 / 0.002); ref = np.array([np.percentile(env[max(0, i - win // 2):i + win // 2], 95) for i in range(m)])
on = np.zeros(m, bool); state = False
for i in range(m):
    r = env[i] / max(ref[i], 1e-9); state = r > 0.5 if not state else r > 0.3; on[i] = state
runs = []; cur = on[0]; k = 0; start = 0
for i, v in enumerate(on):
    if v == cur: k += 1
    else: runs.append((cur, start, k * 2)); cur = v; start = i; k = 1
runs.append((cur, start, k * 2)); runs = [r for r in runs if r[2] >= 8]
marks = np.array([d for v, _, d in runs if v]); dit = np.median(marks[marks < np.percentile(marks, 60)]) if len(marks) > 3 else 40
out = []
for v, s, d in runs:
    tt = t0 + s * 0.002
    if v:
        seg = z[s * hop:s * hop + max(int(d / 1000 * sr), 64)]
        ph = np.unwrap(np.angle(seg)); fpk = np.mean(np.diff(ph)) * sr / (2 * np.pi) if len(seg) > 2 else 0.0
        out.append(('-' if d > 1.9 * dit else '.', tt, d, f0 + fpk))
    else: out.append((' / ' if d > 4.5 * dit else (' ' if d > 1.8 * dit else ''), tt, d, 0))
print(f"dit ~{dit:.0f} ms ({1200/dit:.0f} WPM); {len(marks)} marks"); print(''.join(o[0] for o in out))
if '--detail' in sys.argv:
    for o in out:
        if o[0] in '.-': print(f"{o[1]:7.2f} {o[0]} {o[2]:3.0f} ms  {o[3]:7.1f} Hz")
        elif o[0]: print(f"{o[1]:7.2f} {o[0].strip() or 'gap'} {o[2]:3.0f} ms")
