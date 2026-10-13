//! Processing engine: audio source → resampler → skimmer, on a background
//! thread, publishing snapshots for the UI.

use crate::audio::{start_capture, AudioChunk, Capture, ChannelSel};
use crate::dsp::resample::Resampler;
use crate::lm::LanguageModel;
use crate::skimmer::{ChannelInfo, Skimmer, SkimmerConfig, Spot, TextEvent, RATE};
use anyhow::Result;
use crossbeam_channel::{Receiver, Sender};
use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub enum Source {
    Device { name: Option<String>, sel: ChannelSel },
    File { path: PathBuf, realtime: bool, looped: bool },
}

pub enum EngineCmd {
    AddChannel(f32),
    RemoveChannel(u64),
    Clear,
    Config(SkimmerConfig),
    /// Radio retuned; signals moved by this many Hz in the audio.
    Shift(f32),
    /// Start recording the input audio to a WAV file (None stops).
    Record(Option<PathBuf>),
    /// Write a JSON line per second describing every decoder (troubleshooting
    /// capture that accompanies an audio recording). None stops.
    DisplayLog(Option<PathBuf>),
    /// Radio state to include in the display log.
    RigInfo { vfo: Option<u64>, mode: Option<String>, pitch: f32 },
    /// Precisely measure the tone near `near` Hz (± `span`); the answer
    /// appears in `Snapshot::measurements` under `id`.
    Measure { id: u64, near: f32, span: f32 },
}

#[derive(Default, Clone)]
pub struct Snapshot {
    pub channels: Vec<ChannelInfo>,
    pub spots: Vec<Spot>,
    /// Waterfall rows (dB per FFT bin) produced since the UI last took them.
    pub new_rows: Vec<Vec<f32>>,
    pub events: Vec<TextEvent>,
    pub noise_floor_db: f32,
    /// Peak input level over the last ~60 ms, dBFS.
    pub input_db: f32,
    /// Average (RMS) input level over the last ~60 ms, dBFS.
    pub input_rms_db: f32,
    /// Set when any sample reached full scale; the UI clears it.
    pub input_clipped: bool,
    pub time: f64,
    pub source: String,
    pub error: Option<String>,
    pub running: bool,
    pub input_rate: u32,
    /// Answers to `EngineCmd::Measure`: (id, Some((Hz, dB)) or None).
    pub measurements: Vec<(u64, Option<(f32, f32)>)>,
}

pub struct Engine {
    cmd_tx: Sender<EngineCmd>,
    pub snap: Arc<Mutex<Snapshot>>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
    _capture: Option<Capture>,
}

impl Engine {
    pub fn start(source: Source, cfg: SkimmerConfig, lm: Arc<LanguageModel>) -> Result<Self> {
        let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded();
        let (audio_tx, audio_rx) = crossbeam_channel::bounded::<AudioChunk>(256);
        let snap = Arc::new(Mutex::new(Snapshot::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let mut capture = None;
        let desc;
        match &source {
            Source::Device { name, sel } => {
                let c = start_capture(name.as_deref(), *sel, audio_tx)?;
                desc = format!("{} ({} Hz, {} ch)", c.device, c.rate, c.channels);
                capture = Some(c);
            }
            Source::File { path, realtime, looped } => {
                desc = format!("file: {}", path.display());
                let (path, realtime, looped, stop2) = (path.clone(), *realtime, *looped, stop.clone());
                let snap2 = snap.clone();
                std::thread::Builder::new().name("file-src".into()).spawn(move || {
                    file_reader(path, realtime, looped, audio_tx, stop2, snap2);
                })?;
            }
        }
        snap.lock().source = desc;
        snap.lock().running = true;
        let (snap2, stop2) = (snap.clone(), stop.clone());
        let thread = std::thread::Builder::new()
            .name("dsp".into())
            .spawn(move || dsp_loop(cfg, lm, audio_rx, cmd_rx, snap2, stop2))?;
        Ok(Self { cmd_tx, snap, stop, thread: Some(thread), _capture: capture })
    }

    /// True when the live audio device stopped working (e.g. unplugged).
    pub fn audio_failed(&self) -> bool {
        self._capture.as_ref().is_some_and(|c| c.stalled())
    }

    pub fn send(&self, c: EngineCmd) {
        let _ = self.cmd_tx.send(c);
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn file_reader(path: PathBuf, realtime: bool, looped: bool, tx: Sender<AudioChunk>, stop: Arc<AtomicBool>, snap: Arc<Mutex<Snapshot>>) {
    let (samples, rate) = match crate::audiofile::read_audio(&path) {
        Ok(v) => v,
        Err(e) => {
            if std::env::var_os("CWSK_TRACE").is_some() {
                eprintln!("file_reader: {e:#}");
            }
            snap.lock().error = Some(format!("{e:#}"));
            return;
        }
    };
    let chunk = (rate as usize / 50).max(1); // 20 ms
    loop {
        let start = Instant::now();
        for (i, c) in samples.chunks(chunk).enumerate() {
            if stop.load(Ordering::Relaxed) {
                return;
            }
            if realtime {
                let due = start + Duration::from_secs_f64((i * chunk) as f64 / rate as f64);
                if let Some(d) = due.checked_duration_since(Instant::now()) {
                    std::thread::sleep(d);
                }
                if tx.send(AudioChunk { samples: c.to_vec(), rate }).is_err() {
                    return;
                }
            } else if tx.send(AudioChunk { samples: c.to_vec(), rate }).is_err() {
                return;
            }
        }
        if std::env::var_os("CWSK_TRACE").is_some() {
            eprintln!("file_reader: finished pass, stop={}", stop.load(Ordering::Relaxed));
        }
        if !looped {
            break;
        }
    }
    // Trailing silence so the last words are flushed.
    let _ = tx.send(AudioChunk { samples: vec![0.0; rate as usize * 2], rate });
}

fn dsp_loop(cfg: SkimmerConfig, lm: Arc<LanguageModel>, audio_rx: Receiver<AudioChunk>, cmd_rx: Receiver<EngineCmd>, snap: Arc<Mutex<Snapshot>>, stop: Arc<AtomicBool>) {
    let mut sk = Skimmer::new(cfg, lm);
    let mut resampler: Option<(u32, Resampler)> = None;
    let mut buf = Vec::new();
    let mut last_pub = Instant::now();
    let mut peak = 0.0f32;
    let mut sumsq = 0.0f64;
    let mut nsamp = 0u64;
    let mut clipped = false;
    let mut rate_seen = 0;
    let mut rec_path: Option<PathBuf> = None;
    let mut pending_measure: Vec<(u64, f32, f32, Instant)> = Vec::new();
    let mut display_log: Option<std::io::BufWriter<std::fs::File>> = None;
    let mut display_t0 = Instant::now();
    let mut last_display = Instant::now();
    let mut rig_info: (Option<u64>, Option<String>, f32) = (None, None, 0.0);
    let mut recorder: Option<(PathBuf, hound::WavWriter<std::io::BufWriter<std::fs::File>>)> = None;
    let mut last_rec_flush = Instant::now();
    while !stop.load(Ordering::Relaxed) {
        while let Ok(c) = cmd_rx.try_recv() {
            match c {
                EngineCmd::AddChannel(f) => {
                    sk.add_channel(f, true);
                }
                EngineCmd::RemoveChannel(id) => sk.remove_channel(id),
                EngineCmd::Clear => sk.clear(),
                EngineCmd::Config(c) => sk.cfg = c,
                EngineCmd::Shift(df) => sk.shift(df),
                EngineCmd::DisplayLog(p) => {
                    if std::env::var_os("CWSK_TRACE").is_some() {
                        eprintln!("engine: DisplayLog {:?}", p);
                    }
                    display_log = p.and_then(|p| std::fs::File::create(p).ok()).map(std::io::BufWriter::new);
                    display_t0 = Instant::now();
                    last_display = Instant::now() - Duration::from_secs(2);
                }
                EngineCmd::RigInfo { vfo, mode, pitch } => rig_info = (vfo, mode, pitch),
                EngineCmd::Measure { id, near, span } => pending_measure.push((id, near, span, Instant::now())),
                EngineCmd::Record(p) => {
                    if let Some((_, w)) = recorder.take() {
                        let _ = w.finalize();
                    }
                    rec_path = p;
                }
            }
        }
        match audio_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(chunk) => {
                rate_seen = chunk.rate;
                if let Some(p) = rec_path.take() {
                    let spec = hound::WavSpec { channels: 1, sample_rate: chunk.rate, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
                    match hound::WavWriter::create(&p, spec) {
                        Ok(w) => recorder = Some((p, w)),
                        Err(e) => snap.lock().error = Some(format!("recording failed: {e}")),
                    }
                }
                if let Some((_, w)) = &mut recorder {
                    for &x in &chunk.samples {
                        let _ = w.write_sample((x.clamp(-1.0, 1.0) * 32767.0) as i16);
                    }
                    // Keep the header's sizes current, so a recording is
                    // playable even if the program is killed mid-way.
                    if last_rec_flush.elapsed() > Duration::from_secs(5) {
                        let _ = w.flush();
                        last_rec_flush = Instant::now();
                    }
                }
                if resampler.as_ref().map(|r| r.0) != Some(chunk.rate) {
                    resampler = Some((chunk.rate, Resampler::new(chunk.rate, RATE)));
                }
                for &s in &chunk.samples {
                    let a = s.abs();
                    peak = peak.max(a);
                    sumsq += (s as f64) * (s as f64);
                    nsamp += 1;
                    if a >= 0.999 {
                        clipped = true;
                    }
                }
                buf.clear();
                resampler.as_mut().unwrap().1.process(&chunk.samples, &mut buf);
                sk.process(&buf);
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                if std::env::var_os("CWSK_TRACE").is_some() {
                    eprintln!("engine: audio source disconnected");
                }
                sk.finish();
                stop.store(true, Ordering::Relaxed);
            }
        }
        if let Some(f) = &mut display_log {
            if last_display.elapsed() >= Duration::from_secs(1) {
                last_display = Instant::now();
                let chans: Vec<serde_json::Value> = sk
                    .debug_channels()
                    .into_iter()
                    .map(|(freq, fit, runs, el, dash, shown, text)| {
                        let tail: String = text.chars().rev().take(120).collect::<Vec<_>>().into_iter().rev().collect();
                        serde_json::json!({"audio_hz": freq, "fit": fit, "runs": runs, "el_per_char": el, "dash_frac": dash, "shown": shown, "text": tail})
                    })
                    .collect();
                let infos: Vec<serde_json::Value> = sk
                    .channel_info()
                    .into_iter()
                    .map(|c| serde_json::json!({"audio_hz": c.freq, "wpm": c.wpm, "snr_db": c.snr_db, "call": c.callsign}))
                    .collect();
                let line = serde_json::json!({
                    "t": display_t0.elapsed().as_secs_f32(), "audio_t": sk.time,
                    "vfo_hz": rig_info.0, "mode": rig_info.1, "pitch": rig_info.2,
                    "noise_floor_db": 10.0 * sk.noise_floor.max(1e-15).log10(),
                    "shown": infos, "all_channels": chans,
                });
                use std::io::Write;
                let _ = writeln!(f, "{line}");
                let _ = f.flush();
            }
        }
        // Answer measurement requests once enough fresh audio is buffered.
        pending_measure.retain(|&(id, near, span, t0)| {
            let r = sk.measure_freq(near, span);
            if r.is_some() || t0.elapsed() > Duration::from_secs(4) {
                snap.lock().measurements.push((id, r));
                false
            } else {
                true
            }
        });
        if last_pub.elapsed() >= Duration::from_millis(60) || stop.load(Ordering::Relaxed) {
            last_pub = Instant::now();
            let mut s = snap.lock();
            s.channels = sk.channel_info();
            s.spots = sk.spots.clone();
            s.new_rows.extend(sk.waterfall.drain(..));
            if s.new_rows.len() > 400 {
                let n = s.new_rows.len() - 400;
                s.new_rows.drain(..n);
            }
            s.events.append(&mut sk.events);
            if s.events.len() > 2000 {
                let n = s.events.len() - 2000;
                s.events.drain(..n);
            }
            s.noise_floor_db = 10.0 * sk.noise_floor.max(1e-15).log10();
            s.input_db = 20.0 * peak.max(1e-9).log10();
            if nsamp > 0 {
                let rms = (sumsq / nsamp as f64).sqrt() as f32;
                s.input_rms_db = 20.0 * rms.max(1e-9).log10();
            }
            s.input_clipped |= clipped;
            sumsq = 0.0;
            nsamp = 0;
            clipped = false;
            s.time = sk.time;
            s.input_rate = rate_seen;
            peak = 0.0;
        }
    }
    if let Some((_, w)) = recorder.take() {
        let _ = w.finalize();
    }
    if std::env::var_os("CWSK_TRACE").is_some() {
        eprintln!("engine: dsp loop exit at audio t={:.1}", sk.time);
    }
    snap.lock().running = false;
}
