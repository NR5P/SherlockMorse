// Renders intro.html frame by frame in headless Chrome and muxes it with the
// synthesized soundtrack into an MP4.
//
//   node render.mjs                       -> sherlock-morse-intro.mp4 (1080p60)
//   node render.mjs --stills 1,4.6,7 DIR  -> PNG stills for checking frames
//   node render.mjs --audio-only          -> new soundtrack on the existing video
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { renderAudio } from './audio.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const opt = (name) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : undefined; };
const stills = opt('--stills');
const out = opt('--out') ?? path.join(here, 'sherlock-morse-intro.mp4');

const sandbox = {};
vm.runInNewContext(fs.readFileSync(path.join(here, 'timeline.js'), 'utf8'), sandbox);
const TL = sandbox.TIMELINE;

const wavPath = path.join(here, 'sherlock-morse-intro.wav');
if (!stills) {
  fs.writeFileSync(wavPath, renderAudio(TL));
  console.log('audio ->', wavPath);
}
if (args.includes('--audio-only')) {
  const tmp = out.replace(/\.mp4$/, '.tmp.mp4');
  const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-i', out, '-i', wavPath, '-map', '0:v', '-map', '1:a',
    '-c:v', 'copy', '-c:a', 'aac', '-b:a', '256k', '-movflags', '+faststart', '-shortest', tmp], { stdio: 'inherit' });
  const [code] = await once(ff, 'close');
  if (code !== 0) throw new Error(`ffmpeg exited with ${code}`);
  fs.renameSync(tmp, out);
  console.log('video ->', out);
  process.exit(0);
}

const CHROME = process.env.CHROME ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const port = 9300 + Math.floor(Math.random() * 500);
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'sherlock-morse-intro-'));
const pageUrl = pathToFileURL(path.join(here, 'intro.html')).href + '?render';
const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`,
  '--window-size=1920,1080', '--hide-scrollbars', '--force-device-scale-factor=1',
  '--no-first-run', '--no-default-browser-check', '--mute-audio', pageUrl,
], { stdio: 'ignore' });

let target;
for (let i = 0; i < 100 && !target; i++) {
  await new Promise((r) => setTimeout(r, 100));
  try {
    const list = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    target = list.find((t) => t.type === 'page' && t.url.startsWith('file:'));
  } catch {}
}
if (!target) throw new Error('Chrome did not come up');

const ws = new WebSocket(target.webSocketDebuggerUrl);
await once(ws, 'open');
let nextId = 0;
const pending = new Map();
ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  const p = pending.get(m.id);
  if (!p) return;
  pending.delete(m.id);
  m.error ? p.reject(new Error(m.error.message)) : p.resolve(m.result);
};
const send = (method, params = {}) => new Promise((resolve, reject) => {
  const id = ++nextId;
  pending.set(id, { resolve, reject });
  ws.send(JSON.stringify({ id, method, params }));
});
const evaluate = async (expression) => {
  const r = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description ?? r.exceptionDetails.text);
  return r.result.value;
};

for (let i = 0; i < 100; i++) {
  if (await evaluate('typeof window.renderAt === "function"').catch(() => false)) break;
  await new Promise((r) => setTimeout(r, 100));
}
await evaluate('document.fonts.ready.then(() => true)');
const frame = async (t) => {
  const url = await evaluate(`renderAt(${t}); document.getElementById('c').toDataURL('image/png')`);
  return Buffer.from(url.slice(url.indexOf(',') + 1), 'base64');
};

try {
  if (stills) {
    const dir = args[args.indexOf('--stills') + 2] ?? path.join(here, 'stills');
    fs.mkdirSync(dir, { recursive: true });
    // warm the waterfall's row cache from a little earlier so stills match the video
    for (const t of stills.split(',').map(Number)) {
      for (let w = Math.max(0, t - 0.2); w < t; w += 1 / TL.FPS) await evaluate(`renderAt(${w})`);
      const file = path.join(dir, `still-${t.toFixed(2)}.png`);
      fs.writeFileSync(file, await frame(t));
      console.log(file);
    }
  } else {
    const ff = spawn('ffmpeg', [
      '-y', '-loglevel', 'error',
      '-f', 'image2pipe', '-framerate', String(TL.FPS), '-c:v', 'png', '-i', '-',
      '-i', wavPath,
      '-c:v', 'libx264', '-preset', 'slow', '-crf', '15', '-pix_fmt', 'yuv420p', '-tune', 'film',
      '-c:a', 'aac', '-b:a', '256k', '-movflags', '+faststart', '-shortest', out,
    ], { stdio: ['pipe', 'inherit', 'inherit'] });
    const total = Math.round(TL.DURATION * TL.FPS);
    for (let f = 0; f < total; f++) {
      const png = await frame(f / TL.FPS);
      if (!ff.stdin.write(png)) await once(ff.stdin, 'drain');
      if (f % 60 === 0) process.stdout.write(`\rframe ${f}/${total}`);
    }
    ff.stdin.end();
    const [code] = await once(ff, 'close');
    if (code !== 0) throw new Error(`ffmpeg exited with ${code}`);
    console.log(`\nvideo -> ${out}`);
  }
} finally {
  ws.close();
  chrome.kill();
  await once(chrome, 'exit');
  fs.rmSync(profile, { recursive: true, force: true, maxRetries: 5 });
}
