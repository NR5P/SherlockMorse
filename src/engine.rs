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
}

#[derive(Default, Clone)]
pub struct Snapshot {
    pub channels: Vec<ChannelInfo>,
    pub spots: Vec<Spot>,
    /// Waterfall rows (dB per FFT bin) produced since the UI last took them.
    pub new_rows: Vec<Vec<f32>>,
    pub events: Vec<TextEvent>,
    pub noise_floor_db: f32,
    /// Peak input level, dBFS.
    pub input_db: f32,
    pub time: f64,
    pub source: String,
    pub error: Option<String>,
    pub running: bool,
    pub input_rate: u32,
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
    let mut rate_seen = 0;
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
            }
        }
        match audio_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(chunk) => {
                rate_seen = chunk.rate;
                if resampler.as_ref().map(|r| r.0) != Some(chunk.rate) {
                    resampler = Some((chunk.rate, Resampler::new(chunk.rate, RATE)));
                }
                for &s in &chunk.samples {
                    peak = peak.max(s.abs());
                }
                buf.clear();
                resampler.as_mut().unwrap().1.process(&chunk.samples, &mut buf);
                sk.process(&buf);
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                sk.finish();
                stop.store(true, Ordering::Relaxed);
            }
        }
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
            s.time = sk.time;
            s.input_rate = rate_seen;
            peak *= 0.5;
        }
    }
    snap.lock().running = false;
}
