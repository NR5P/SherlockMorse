//! CW Skimmer – multi-channel Morse decoder with waterfall and CAT control.

use cwskimmer::activity::{self, Activity};
use cwskimmer::audio::{self, ChannelSel, DeviceInfo};
use cwskimmer::cat::controller::{RigController, RigState};
use cwskimmer::cat::{self, Mode, RigConfig, RigKind};
use cwskimmer::cty::{self, Country, Cty};
use cwskimmer::engine::{Engine, EngineCmd, Snapshot, Source};
use cwskimmer::lm::LanguageModel;
use cwskimmer::lookup::{self, Credentials, Lookup, LookupState};
use cwskimmer::morse::to_display;
use cwskimmer::skimmer::{bin_hz, ChannelInfo, SkimmerConfig, FFT_SIZE};
use eframe::egui::{self, Color32, ColorImage, Pos2, Rect, RichText, Sense, Stroke, TextureHandle, TextureOptions, Vec2};
use egui_extras::{Column, TableBuilder};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

const WF_ROWS: usize = 300;

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
struct Settings {
    audio_device: Option<String>,
    audio_auto: bool,
    channel_sel: ChannelSel,
    rig: RigConfig,
    rig_enabled: bool,
    /// CW pitch / sidetone (Hz). FTDX3000 default is 700 Hz.
    pitch: f32,
    min_freq: f32,
    max_freq: f32,
    detect_db: f32,
    max_channels: usize,
    squelch: f32,
    lm_weight: f32,
    show_rf: bool,
    /// Use the CW pitch reported by the radio over CAT when available.
    auto_pitch: bool,
    wf_contrast: f32,
    wf_brightness: f32,
    file_path: String,
    /// Print the English meaning of CW abbreviations in small text inline.
    explain_abbrev: bool,
    /// Logins for callsign lookups (optional; HamDB needs none).
    lookup: Credentials,
    /// Operator's own locator, for distance and bearing to stations.
    my_grid: String,
    /// Flags shown on stations heard in a contest or on-air program.
    activities: Vec<Activity>,
    /// Spots list filter: empty shows all, "*" any flagged spot, else one flag's name.
    spot_filter: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            audio_device: None,
            audio_auto: true,
            channel_sel: ChannelSel::Mix,
            rig: RigConfig::default(),
            rig_enabled: false,
            pitch: 700.0,
            min_freq: 250.0,
            max_freq: 2800.0,
            detect_db: 12.0,
            max_channels: 40,
            squelch: 2.3,
            lm_weight: 0.7,
            show_rf: true,
            auto_pitch: true,
            wf_contrast: 1.0,
            wf_brightness: 0.0,
            file_path: String::new(),
            explain_abbrev: true,
            lookup: Credentials::default(),
            my_grid: String::new(),
            activities: activity::defaults(),
            spot_filter: String::new(),
        }
    }
}

#[derive(Clone)]
struct SpotRow {
    call: String,
    rf: Option<u64>,
    audio: f32,
    wpm: f32,
    snr: f32,
    last: Instant,
    count: u32,
    /// Heard at least twice; a single sighting is shown dimmed.
    confirmed: bool,
    country: Option<Country>,
    /// Activities heard on its channel: (flag name, what was sent).
    tags: Vec<(String, String)>,
}

struct App {
    s: Settings,
    lm: Arc<LanguageModel>,
    engine: Option<Engine>,
    engine_err: Option<String>,
    devices: Vec<DeviceInfo>,
    ports: Vec<(String, String)>,
    rig: Option<RigController>,
    rig_state: RigState,
    last_vfo: Option<u64>,
    snap: Snapshot,
    wf_img: ColorImage,
    wf_tex: Option<TextureHandle>,
    wf_dirty: bool,
    wf_peak: f32,
    last_audio_retry: Instant,
    last_scan: Instant,
    /// Per-channel text not yet written to the log file: (freq, text, last update).
    lines: std::collections::HashMap<u64, (f32, String, Instant)>,
    recording: Option<(std::path::PathBuf, Instant)>,
    zb: Option<ZeroBeat>,
    rec_display: Option<RecDisplay>,
    qso: Option<Qso>,
    qso_history: Vec<Qso>,
    bottom_tab: u8,
    history_sel: Option<usize>,
    /// Playing an audio file rather than the sound card.
    file_mode: bool,
    last_rig_info: Instant,
    /// Smoothed average level and decaying peak-hold for the input meter.
    meter_avg: f32,
    meter_peak: f32,
    clip_until: Option<Instant>,
    zb_status: String,
    next_req: u64,
    selected: Option<u64>,
    spots: Vec<SpotRow>,
    log: Vec<(f32, String)>,
    freq_entry: String,
    probe_rx: Option<crossbeam_channel::Receiver<ProbeMsg>>,
    probe_status: String,
    show_settings: bool,
    /// Activities recognised on each decoder channel, kept while it lives.
    chan_tags: std::collections::HashMap<u64, Vec<(String, String)>>,
    lookup: Lookup,
    /// DXCC country file, and a download in progress.
    cty: Arc<Cty>,
    cty_status: String,
    cty_rx: Option<crossbeam_channel::Receiver<Result<Cty, String>>>,
    /// Callsign shown in the station-info window.
    info_call: Option<String>,
    /// `--screenshot PATH`: save a PNG of the window after a delay and exit.
    screenshot: Option<(std::path::PathBuf, Instant, bool)>,
}

/// The conversation on the operator's own frequency: everything decoded
/// within ±75 Hz of the CW pitch, saved to a file as it arrives.
#[derive(Clone)]
struct Qso {
    /// RF frequency of the conversation (None without CAT).
    rf: Option<u64>,
    started_utc: String,
    last_text: Instant,
    /// Time we moved away from the frequency (finalize after 30 s).
    away_since: Option<Instant>,
    text: String,
    calls: Vec<String>,
    path: std::path::PathBuf,
}

impl Qso {
    /// Loads the most recent saved conversations (newest first).
    fn load_recent(n: usize) -> Vec<Qso> {
        let Ok(rd) = std::fs::read_dir(log_dir()) else { return Vec::new() };
        let mut files: Vec<(std::time::SystemTime, std::path::PathBuf)> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.file_name().and_then(|f| f.to_str()).is_some_and(|f| f.starts_with("qso-") && f.ends_with(".txt")))
            .filter_map(|p| Some((std::fs::metadata(&p).ok()?.modified().ok()?, p)))
            .collect();
        files.sort_by(|a, b| b.0.cmp(&a.0));
        files
            .into_iter()
            .take(n)
            .filter_map(|(mtime, path)| {
                let content = std::fs::read_to_string(&path).ok()?;
                let started_utc = content.lines().find_map(|l| l.strip_prefix("Started ")).map(|l| l.trim_end_matches(" UTC").to_string())?;
                if started_utc.len() < 16 {
                    return None;
                }
                let text = content.splitn(2, "\n\n").nth(1).unwrap_or("").trim_end().to_string();
                // qso-YYYY-MM-DD-HHMMZ-14047.1-CALL….txt
                let stem = path.file_stem()?.to_str()?.to_string();
                let rf = stem.split('-').nth(5).and_then(|f| f.parse::<f64>().ok()).map(|k| (k * 1000.0).round() as u64);
                let calls = stem.split('-').skip(6).map(|c| c.replace('_', "/")).collect();
                let age = mtime.elapsed().unwrap_or_default();
                let last_text = Instant::now().checked_sub(age).unwrap_or_else(Instant::now);
                Some(Qso { rf, started_utc, last_text, away_since: None, text, calls, path })
            })
            .collect()
    }

    fn title(&self) -> String {
        let f = self.rf.map(|r| format!("{:.1} kHz", r as f64 / 1000.0)).unwrap_or_else(|| "on pitch".into());
        let calls = if self.calls.is_empty() { String::new() } else { format!("  {}", self.calls.join(" ")) };
        format!("{}Z  {f}{calls}", &self.started_utc[11..16])
    }

    fn save(&mut self) {
        // File name carries the callsigns heard; rename as they appear.
        let calls: Vec<&str> = self.calls.iter().take(3).map(|c| c.as_str()).collect();
        let f = self.rf.map(|r| format!("-{:.1}", r as f64 / 1000.0)).unwrap_or_default();
        let name = format!(
            "qso-{}-{}Z{f}{}{}.txt",
            &self.started_utc[..10],
            self.started_utc[11..16].replace(':', ""),
            if calls.is_empty() { "" } else { "-" },
            calls.join("-").replace('/', "_")
        );
        let path = log_dir().join(name);
        if path != self.path && self.path.exists() {
            let _ = std::fs::rename(&self.path, &path);
        }
        self.path = path;
        let _ = std::fs::create_dir_all(log_dir());
        let header = format!("Conversation {}\nStarted {} UTC\n\n", self.title(), self.started_utc);
        let _ = std::fs::write(&self.path, format!("{header}{}\n", self.text.trim_end()));
    }
}

/// Display capture while recording audio: a JSON line per second with every
/// decoder's state, and a screenshot every 10 s.
struct RecDisplay {
    log: Option<std::io::BufWriter<std::fs::File>>,
    shots: std::path::PathBuf,
    last_shot: Instant,
    shot_n: u32,
}

/// Click-to-tune with closed-loop correction ("zero-beat").
struct ZeroBeat {
    /// 0 = waiting for the initial measurement, 1.. = verification passes.
    stage: u8,
    req: Option<u64>,
    /// Don't measure before this (radio settling, fresh audio).
    wait_until: Instant,
    /// Where to look for the signal (audio Hz) on the next measurement.
    near: f32,
    span: f32,
}

enum ProbeMsg {
    Progress(String),
    Done(Vec<cat::probe::Found>),
}

fn wf_width() -> usize {
    FFT_SIZE / 2
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let s: Settings = cc.storage.and_then(|st| eframe::get_value(st, eframe::APP_KEY)).unwrap_or_default();
        let mut style = (*cc.egui_ctx.style()).clone();
        style.spacing.item_spacing = Vec2::new(6.0, 4.0);
        cc.egui_ctx.set_style(style);
        egui_extras::install_image_loaders(&cc.egui_ctx);
        let mut app = Self {
            lm: Arc::new(LanguageModel::new()),
            engine: None,
            engine_err: None,
            devices: audio::list_input_devices(),
            ports: cat::list_ports(),
            rig: None,
            rig_state: RigState::default(),
            last_vfo: None,
            snap: Snapshot::default(),
            wf_img: ColorImage::new([wf_width(), WF_ROWS], Color32::BLACK),
            wf_tex: None,
            wf_dirty: true,
            wf_peak: -200.0,
            last_audio_retry: Instant::now(),
            last_scan: Instant::now(),
            lines: Default::default(),
            recording: None,
            zb: None,
            rec_display: None,
            qso: None,
            qso_history: Qso::load_recent(20),
            bottom_tab: 0,
            history_sel: None,
            file_mode: false,
            last_rig_info: Instant::now(),
            meter_avg: -90.0,
            meter_peak: -90.0,
            clip_until: None,
            zb_status: String::new(),
            next_req: 1,
            selected: None,
            spots: Vec::new(),
            log: Vec::new(),
            freq_entry: String::new(),
            probe_rx: None,
            probe_status: String::new(),
            show_settings: false,
            chan_tags: Default::default(),
            lookup: Lookup::default(),
            cty: Arc::new(Cty::built_in()),
            cty_status: String::new(),
            cty_rx: None,
            info_call: None,
            screenshot: None,
            s,
        };
        app.load_cty();
        let args: Vec<String> = std::env::args().collect();
        let arg = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
        if let Some(p) = arg("--screenshot") {
            let secs: f32 = arg("--after").and_then(|v| v.parse().ok()).unwrap_or(10.0);
            app.screenshot = Some((p.into(), Instant::now() + Duration::from_secs_f32(secs), false));
        }
        if let Some(f) = arg("--file") {
            app.s.file_path = f;
            app.start_file();
        } else {
            app.start_audio();
        }
        if args.iter().any(|a| a == "--record") {
            app.toggle_recording();
        }
        // --no-rig: listen only, leaving the CAT port to another program
        // (e.g. a script tuning the radio while this one decodes).
        if app.s.rig_enabled && !args.iter().any(|a| a == "--no-rig") {
            app.connect_rig();
        }
        app
    }

    fn skimmer_config(&self) -> SkimmerConfig {
        let mut c = SkimmerConfig {
            min_freq: self.s.min_freq,
            max_freq: self.s.max_freq,
            max_channels: self.s.max_channels,
            detect_db: self.s.detect_db,
            ..Default::default()
        };
        c.params.squelch_open = self.s.squelch;
        c.params.squelch_close = self.s.squelch * 0.83;
        c.params.lm_weight = self.s.lm_weight;
        c
    }

    fn start_audio(&mut self) {
        self.file_mode = false;
        self.engine = None;
        if self.s.audio_auto || self.s.audio_device.is_none() {
            self.s.audio_device = audio::auto_select(&self.devices);
        }
        let src = Source::Device { name: self.s.audio_device.clone(), sel: self.s.channel_sel };
        self.start_source(src);
    }

    fn start_file(&mut self) {
        self.file_mode = true;
        self.engine = None;
        let path = std::path::PathBuf::from(self.s.file_path.trim());
        self.start_source(Source::File { path, realtime: true, looped: false });
    }

    fn start_source(&mut self, src: Source) {
        // A new engine starts without a recorder: keep the Rec button honest.
        self.recording = None;
        self.rec_display = None;
        match Engine::start(src, self.skimmer_config(), self.lm.clone()) {
            Ok(e) => {
                self.engine = Some(e);
                self.engine_err = None;
            }
            Err(e) => self.engine_err = Some(format!("{e:#}")),
        }
    }

    fn connect_rig(&mut self) {
        if self.s.rig.kind != RigKind::Hamlib && cat::is_keying_port(&self.s.rig.port) {
            self.probe_status = format!("{} is the radio's PTT/keying port, not CAT — pick the other port", self.s.rig.port);
            self.s.rig_enabled = false;
            return;
        }
        self.rig = Some(RigController::start(self.s.rig.clone()));
        self.s.rig_enabled = true;
    }

    fn disconnect_rig(&mut self) {
        self.rig = None;
        self.rig_state = RigState::default();
        self.s.rig_enabled = false;
        self.last_vfo = None;
    }

    fn upper(&self) -> bool {
        self.rig_state.mode.as_ref().map(|m| m.is_upper()).unwrap_or(true)
    }

    /// CW pitch in use: read from the radio when possible, else the setting.
    fn pitch(&self) -> f32 {
        match (self.s.auto_pitch, self.rig_state.pitch) {
            (true, Some(p)) if self.rig.is_some() => p as f32,
            _ => self.s.pitch,
        }
    }

    /// Audio frequency that corresponds to the displayed VFO frequency.
    fn audio_ref(&self) -> f64 {
        let cw = self.rig_state.mode.as_ref().map(|m| m.is_cw()).unwrap_or(true);
        if cw { self.pitch() as f64 } else { 0.0 }
    }

    /// RF frequency of a signal heard at `audio` Hz.
    fn rf_of(&self, audio: f32) -> Option<u64> {
        let vfo = self.rig_state.freq? as f64;
        let d = audio as f64 - self.audio_ref();
        Some((if self.upper() { vfo + d } else { vfo - d }).round() as u64)
    }

    /// Tunes the radio so the signal at `audio` Hz lands on the CW pitch.
    fn tune_to_audio(&mut self, audio: f32) {
        let Some(rf) = self.rf_of(audio) else { return };
        // VFO setting that puts this RF frequency at the pitch.
        let d = self.pitch() as f64 - self.audio_ref();
        let vfo = if self.upper() { rf as f64 - d } else { rf as f64 + d };
        self.set_vfo(vfo.round().max(0.0) as u64);
    }

    /// Starts a zero-beat on the signal near `audio` Hz: measure it
    /// precisely, tune it onto the pitch, then verify and correct.
    fn zero_beat(&mut self, audio: f32) {
        if self.rig.is_none() {
            self.zb_status = "connect the radio to tune".into();
            return;
        }
        if self.rig_state.freq.is_none() {
            self.zb_status = "waiting for the radio's frequency…".into();
            return;
        }
        // Move right away on the estimate, then measure and correct, so a
        // station between transmissions still gets tuned.
        self.tune_to_audio(audio);
        self.zero_beat_verify();
    }

    /// After tuning to a known RF frequency (spot), verify it lands on pitch.
    fn zero_beat_verify(&mut self) {
        if self.rig.is_none() {
            return;
        }
        let p = self.pitch();
        self.zb = Some(ZeroBeat { stage: 1, req: None, wait_until: Instant::now() + Duration::from_millis(1300), near: p, span: 40.0 });
        self.zb_status = "zero-beat: checking…".into();
    }

    fn step_zero_beat(&mut self, results: &[(u64, Option<(f32, f32)>)]) {
        let Some(zb) = &mut self.zb else { return };
        let Some(engine) = &self.engine else { return };
        if zb.req.is_none() {
            if Instant::now() >= zb.wait_until {
                let id = self.next_req;
                self.next_req += 1;
                engine.send(EngineCmd::Measure { id, near: zb.near, span: zb.span });
                zb.req = Some(id);
            }
            return;
        }
        let Some(&(_, res)) = results.iter().find(|(id, _)| Some(*id) == zb.req) else { return };
        let stage = zb.stage;
        let pitch = self.pitch();
        match res {
            None => {
                self.zb_status = if stage == 0 { "zero-beat: no clear tone there".into() } else { "zero-beat: signal faded, tuned by estimate".into() };
                self.zb = None;
            }
            Some((f, _)) => {
                let err = f - pitch;
                if stage > 0 && err.abs() <= 2.0 {
                    self.zb_status = format!("zero-beat ✓ {f:.0} Hz (±{:.0} Hz)", err.abs().max(1.0));
                    self.zb = None;
                } else if stage >= 3 {
                    self.zb_status = format!("zero-beat: {:+.0} Hz off (signal unstable)", err);
                    self.zb = None;
                } else {
                    self.tune_to_audio(f);
                    self.zb = Some(ZeroBeat { stage: stage + 1, req: None, wait_until: Instant::now() + Duration::from_millis(1300), near: pitch, span: 25.0 });
                    self.zb_status = format!("zero-beat: tuned, verifying… ({err:+.0} Hz)");
                }
            }
        }
    }

    fn set_vfo(&mut self, hz: u64) {
        if let Some(r) = &self.rig {
            r.set_freq(hz);
        }
    }

    fn fmt_freq(&self, audio: f32) -> String {
        if self.s.show_rf {
            if let Some(rf) = self.rf_of(audio) {
                return format!("{:.2}", rf as f64 / 1000.0);
            }
        }
        format!("{audio:.0} Hz")
    }

    fn poll(&mut self) {
        // In Auto mode, switch to the radio's codec when it (re)appears.
        if self.s.audio_auto && !self.file_mode && self.last_scan.elapsed() > Duration::from_secs(5) {
            self.last_scan = Instant::now();
            let cur_is_radio = self.s.audio_device.as_deref().is_some_and(|n| audio::radio_score(n) >= 50);
            if !cur_is_radio {
                let devs = audio::list_input_devices();
                if devs.iter().any(|d| d.score >= 50) {
                    self.devices = devs;
                    self.start_audio();
                }
            }
        }
        // Live audio lost (radio off / cable pulled): retry every 3 s.
        let failed = self.engine.as_ref().is_some_and(|e| e.audio_failed());
        let live = !self.file_mode;
        if (failed || (self.engine.is_none() && self.engine_err.is_some())) && live {
            if self.last_audio_retry.elapsed() > Duration::from_secs(3) {
                self.last_audio_retry = Instant::now();
                self.devices = audio::list_input_devices();
                let want = self.s.audio_device.clone();
                let present = want.as_ref().is_some_and(|w| self.devices.iter().any(|d| &d.name == w));
                if self.s.audio_auto || present {
                    self.start_audio();
                } else {
                    self.engine = None;
                }
                if self.engine.is_none() || self.engine.as_ref().is_some_and(|e| e.audio_failed()) {
                    self.engine_err = Some("audio device disconnected — waiting for it to come back…".into());
                }
            }
        }
        if let Some(e) = &self.engine {
            let mut s = e.snap.lock();
            let rows = std::mem::take(&mut s.new_rows);
            let (notes, events): (Vec<_>, Vec<_>) = std::mem::take(&mut s.events).into_iter().partition(|e| e.note.is_some());
            self.snap.channels = s.channels.clone();
            self.snap.spots = s.spots.clone();
            self.snap.noise_floor_db = s.noise_floor_db;
            self.snap.input_db = s.input_db;
            self.snap.time = s.time;
            self.snap.source = s.source.clone();
            self.snap.error = s.error.clone();
            self.snap.running = s.running;
            self.snap.input_rate = s.input_rate;
            let measurements = std::mem::take(&mut s.measurements);
            if std::mem::take(&mut s.input_clipped) {
                self.clip_until = Some(Instant::now() + Duration::from_secs(2));
            }
            // Meter ballistics: average smoothed (~0.3 s), peak holds and falls ~10 dB/s.
            self.meter_avg += 0.15 * (s.input_rms_db - self.meter_avg);
            self.meter_peak = (self.meter_peak - 0.6).max(s.input_db);
            drop(s);
            self.step_zero_beat(&measurements);
            for r in rows {
                self.push_wf_row(&r);
            }
            self.log_channel_notes(&notes);
            // A strong station near the pitch drags along weak phantom
            // channels (its clicks or sidebands) that decode as junk: when
            // a channel there reads well, leave the dimmed ones out of the
            // conversation. (Both sides of a real QSO normally read well.)
            let pitch = self.pitch();
            let near = |c: &&cwskimmer::skimmer::ChannelInfo| (c.freq - pitch).abs() <= 75.0;
            let clean_near = self.snap.channels.iter().filter(near).any(|c| !c.doubtful);
            for ev in &events {
                let doubtful = self.snap.channels.iter().find(|c| c.id == ev.channel).is_some_and(|c| c.doubtful);
                if clean_near && doubtful {
                    continue;
                }
                self.qso_feed(ev.freq, &ev.text);
            }
            self.tag_channels(events.iter().map(|e| e.channel).collect());
            for ev in events {
                let e = self.lines.entry(ev.channel).or_insert_with(|| (ev.freq, String::new(), Instant::now()));
                e.0 = ev.freq;
                e.1.push_str(&ev.text);
                e.2 = Instant::now();
                self.log.push((ev.freq, ev.text));
            }
            self.flush_log_lines(false);
            if self.log.len() > 500 {
                let n = self.log.len() - 500;
                self.log.drain(..n);
            }
        }
        if let Some(r) = &self.rig {
            self.rig_state = r.snapshot();
        }
        if self.recording.is_some() && self.last_rig_info.elapsed() > Duration::from_millis(500) {
            self.send_rig_info();
        }
        self.qso_tick();
        // Follow VFO changes: shift decoders with the signals.
        if let Some(f) = self.rig_state.freq {
            if let Some(prev) = self.last_vfo {
                if prev != f {
                    let d = f as f64 - prev as f64;
                    let df = if self.upper() { -d } else { d } as f32;
                    if let Some(e) = &self.engine {
                        if df.abs() < 3000.0 {
                            e.send(EngineCmd::Shift(df));
                        } else {
                            e.send(EngineCmd::Clear);
                        }
                    }
                }
            }
            self.last_vfo = Some(f);
        }
        // Spots: remember the RF frequency at the time they were heard.
        for sp in self.snap.spots.clone() {
            let rf = self.rf_of(sp.freq);
            match self.spots.iter_mut().find(|r| r.call == sp.call) {
                Some(r) => {
                    if sp.count != r.count || sp.confirmed != r.confirmed {
                        r.count = sp.count;
                        r.confirmed = sp.confirmed;
                        r.last = Instant::now();
                        r.audio = sp.freq;
                        r.wpm = sp.wpm;
                        r.snr = sp.snr_db;
                        if rf.is_some() {
                            r.rf = rf;
                        }
                    }
                }
                None => {
                    let country = self.cty.lookup(&sp.call);
                    self.spots.push(SpotRow { call: sp.call.clone(), rf, audio: sp.freq, wpm: sp.wpm, snr: sp.snr_db, last: Instant::now(), count: sp.count, confirmed: sp.confirmed, country, tags: vec![] })
                }
            }
            if let (Some(tags), Some(r)) = (self.chan_tags.get(&sp.channel), self.spots.iter_mut().find(|r| r.call == sp.call)) {
                for t in tags {
                    if !r.tags.iter().any(|(n, _)| n == &t.0) {
                        r.tags.push(t.clone());
                    }
                }
            }
        }
        self.spots.retain(|r| r.last.elapsed() < Duration::from_secs(1800));
        if let Some(r) = self.cty_rx.as_ref().and_then(|rx| rx.try_recv().ok()) {
            self.cty_rx = None;
            match r {
                Ok(c) => {
                    self.cty_status = format!("Updated: {} countries", c.len());
                    self.set_cty(c);
                }
                Err(e) => self.cty_status = format!("Update failed: {e}"),
            }
        }
        if let Some(rx) = &self.probe_rx {
            while let Ok(m) = rx.try_recv() {
                match m {
                    ProbeMsg::Progress(p) => self.probe_status = format!("Probing {p}…"),
                    ProbeMsg::Done(found) => {
                        if let Some(f) = found.first() {
                            self.probe_status = format!("Found {} on {} @ {}", f.model, f.config.port, f.config.baud);
                            self.s.rig = RigConfig { poll_ms: self.s.rig.poll_ms, ..f.config.clone() };
                            self.connect_rig();
                        } else {
                            self.probe_status = "No radio found on USB serial ports".into();
                        }
                        self.probe_rx = None;
                        break;
                    }
                }
            }
        }
    }

    /// Writes finished lines of decoded text to today's log file: a line ends
    /// after 8 s without new text or once it is long.
    fn flush_log_lines(&mut self, all: bool) {
        let ids: Vec<u64> = self.lines.keys().copied().collect();
        for id in ids {
            let (freq, text, last) = self.lines[&id].clone();
            let long = text.len() > 90 && text.ends_with(' ');
            if !(all || long || last.elapsed() > Duration::from_secs(8)) || text.trim().is_empty() {
                continue;
            }
            self.lines.remove(&id);
            let call = self.snap.channels.iter().find(|c| c.id == id).and_then(|c| c.callsign.clone()).unwrap_or_default();
            let f = match self.rf_of(freq) {
                Some(rf) => format!("{:>10.2} kHz", rf as f64 / 1000.0),
                None => format!("{freq:>7.0} Hz audio"),
            };
            let (date, time) = utc_now();
            let line = format!("{date} {time}Z  {f}  {call:<10}  {}\n", to_display(text.trim()));
            let dir = log_dir();
            let _ = std::fs::create_dir_all(&dir);
            let path = dir.join(format!("cw-{date}.txt"));
            use std::io::Write;
            if let Ok(mut fh) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
                let _ = fh.write_all(line.as_bytes());
            }
        }
    }

    /// Appends channel visibility changes (shown / hidden / removed, with
    /// the reason and measurements) to channels-<date>.txt, to diagnose
    /// stations disappearing from the list.
    fn log_channel_notes(&self, notes: &[cwskimmer::skimmer::TextEvent]) {
        if notes.is_empty() {
            return;
        }
        let (date, time) = utc_now();
        let dir = log_dir();
        let _ = std::fs::create_dir_all(&dir);
        let mut out = String::new();
        for n in notes {
            let f = match self.rf_of(n.freq) {
                Some(rf) => format!("{:>10.2} kHz", rf as f64 / 1000.0),
                None => format!("{:>7.0} Hz audio", n.freq),
            };
            out.push_str(&format!("{date} {time}Z  {f}  #{}  {}\n", n.channel, n.note.as_deref().unwrap_or("")));
        }
        use std::io::Write;
        if let Ok(mut fh) = std::fs::OpenOptions::new().create(true).append(true).open(dir.join(format!("channels-{date}.txt"))) {
            let _ = fh.write_all(out.as_bytes());
        }
    }

    /// Tunes so a station at RF frequency `rf` lands on the pitch, then
    /// fine-tunes on the live signal (zero-beat).
    fn tune_rf(&mut self, rf: u64) {
        if self.rig.is_none() {
            return;
        }
        let d = (self.pitch() as f64 - self.audio_ref()) as i64;
        let vfo = if self.upper() { rf as i64 - d } else { rf as i64 + d };
        self.set_vfo(vfo.max(0) as u64);
        self.zero_beat_verify();
    }

    /// Adds decoded text to the conversation on our frequency.
    fn qso_feed(&mut self, audio: f32, text: &str) {
        if (audio - self.pitch()).abs() > 75.0 {
            return;
        }
        let rf = self.rf_of(audio);
        let now = Instant::now();
        let same = self.qso.as_ref().is_some_and(|q| match (q.rf, rf) {
            (Some(a), Some(b)) => (a as i64 - b as i64).abs() <= 150,
            _ => true,
        });
        if !same {
            self.qso_finish();
        }
        if self.qso.is_none() {
            // Back on a recent conversation's frequency: resume it.
            if let Some(i) = self.qso_history.iter().position(|q| {
                matches!((q.rf, rf), (Some(a), Some(b)) if (a as i64 - b as i64).abs() <= 150) && q.last_text.elapsed() < Duration::from_secs(1800)
            }) {
                let q = self.qso_history.remove(i);
                self.history_sel = None;
                self.qso = Some(q);
            }
        }
        let q = self.qso.get_or_insert_with(|| {
            let (d, t) = utc_now();
            Qso { rf, started_utc: format!("{d} {t}"), last_text: now, away_since: None, text: String::new(), calls: Vec::new(), path: std::path::PathBuf::new() }
        });
        // A pause of more than 4 s means someone else (or a new over) is sending.
        if q.text.is_empty() || q.last_text.elapsed() > Duration::from_secs(4) {
            let (_, t) = utc_now();
            if !q.text.is_empty() {
                q.text.push('\n');
            }
            q.text.push_str(&format!("[{}Z] ", &t[..5]));
        }
        q.text.push_str(&to_display(text));
        q.last_text = now;
        q.away_since = None;
        // Only calls that are confirmed (spotted twice) or heard twice in
        // this conversation: a single garbled sighting is often noise.
        for w in text.split_whitespace() {
            if let Some(c) = cwskimmer::callsign::parse_callsign(w) {
                let spotted = self.spots.iter().any(|r| r.confirmed && r.call == c);
                let twice = q.text.split_whitespace().filter(|x| *x == c).count() >= 2;
                if !q.calls.contains(&c) && cwskimmer::callsign::valid_prefix(&c) && (spotted || twice) {
                    q.calls.push(c);
                }
            }
        }
        q.save();
    }

    /// Finalizes the current conversation into the history.
    fn qso_finish(&mut self) {
        if let Some(mut q) = self.qso.take() {
            if q.text.trim().len() >= 3 {
                q.save();
                self.qso_history.insert(0, q);
                self.qso_history.truncate(20);
            } else if q.path.exists() {
                let _ = std::fs::remove_file(&q.path);
            }
        }
    }

    /// Ends the conversation when we have moved off frequency for 30 s. While
    /// we stay on frequency it stays up; only without CAT (frequency unknown)
    /// does 10 minutes without text end it.
    fn qso_tick(&mut self) {
        let vfo_rf = self.rf_of(self.pitch());
        let Some(q) = &mut self.qso else { return };
        let away = matches!((q.rf, vfo_rf), (Some(a), Some(b)) if (a as i64 - b as i64).abs() > 300);
        if away {
            if q.away_since.get_or_insert_with(Instant::now).elapsed() > Duration::from_secs(30) {
                self.qso_finish();
                return;
            }
        } else {
            q.away_since = None;
        }
        if self.qso.as_ref().is_some_and(|q| (q.rf.is_none() || vfo_rf.is_none()) && q.last_text.elapsed() > Duration::from_secs(600)) {
            self.qso_finish();
        }
    }

    fn send_rig_info(&mut self) {
        self.last_rig_info = Instant::now();
        if let Some(e) = &self.engine {
            e.send(EngineCmd::RigInfo {
                vfo: self.rig_state.freq,
                mode: self.rig_state.mode.as_ref().map(|m| m.label()),
                pitch: self.pitch(),
            });
        }
    }

    fn toggle_recording(&mut self) {
        let Some(e) = &self.engine else { return };
        if self.recording.is_some() {
            e.send(EngineCmd::Record(None));
            e.send(EngineCmd::DisplayLog(None));
            self.recording = None;
            self.rec_display = None;
        } else {
            let (date, time) = utc_now();
            let dir = log_dir();
            let _ = std::fs::create_dir_all(&dir);
            let p = dir.join(format!("audio-{date}-{}.wav", time.replace(':', "")));
            e.send(EngineCmd::Record(Some(p.clone())));
            // Alongside the audio: what the display showed (for troubleshooting).
            let stem = p.with_extension("");
            let shots = std::path::PathBuf::from(format!("{}-screens", stem.display()));
            let _ = std::fs::create_dir_all(&shots);
            e.send(EngineCmd::DisplayLog(Some(std::path::PathBuf::from(format!("{}-display.jsonl", stem.display())))));
            self.send_rig_info();
            self.rec_display = Some(RecDisplay { log: None, shots, last_shot: Instant::now() - Duration::from_secs(60), shot_n: 0 });
            self.recording = Some((p, Instant::now()));
        }
    }

    fn push_wf_row(&mut self, row: &[f32]) {
        let w = wf_width();
        let px = &mut self.wf_img.pixels;
        px.copy_within(0..w * (WF_ROWS - 1), w);
        let nf = self.snap.noise_floor_db;
        // Auto-range between the noise floor and the strongest recent signal.
        let rowmax = row.iter().copied().fold(f32::MIN, f32::max);
        self.wf_peak = if rowmax > self.wf_peak { rowmax } else { self.wf_peak - 0.05 };
        let range = (self.wf_peak - nf).clamp(25.0, 80.0);
        for (i, p) in px.iter_mut().take(w).enumerate() {
            let db = row.get(i).copied().unwrap_or(-150.0);
            let v = ((db - nf + 3.0 - self.s.wf_brightness) / range * self.s.wf_contrast).clamp(0.0, 1.0);
            *p = colormap(v);
        }
        self.wf_dirty = true;
    }

    /// Looks for activity flags in the text of channels that just decoded
    /// something, and forgets channels that are gone.
    fn tag_channels(&mut self, mut ids: Vec<u64>) {
        ids.sort_unstable();
        ids.dedup();
        for ch in self.snap.channels.iter().filter(|c| ids.binary_search(&c.id).is_ok()) {
            let mut from = ch.text.len().saturating_sub(300);
            while !ch.text.is_char_boundary(from) {
                from += 1;
            }
            let tail = &ch.text[from..];
            let found = activity::detect(&self.s.activities, &activity::words(&to_display(tail)));
            let tags = self.chan_tags.entry(ch.id).or_default();
            for t in found {
                if !tags.iter().any(|(n, _)| n == &t.0) {
                    tags.push(t);
                }
            }
        }
        let live: std::collections::HashSet<u64> = self.snap.channels.iter().map(|c| c.id).collect();
        self.chan_tags.retain(|id, _| live.contains(id));
    }

    /// Flags to draw for `tags`: only activities still enabled, in settings order.
    fn flags<'a>(&'a self, tags: &'a [(String, String)]) -> impl Iterator<Item = (&'a Activity, &'a str)> + 'a {
        self.s.activities.iter().filter(|a| a.enabled).filter_map(move |a| tags.iter().find(|(n, _)| *n == a.name).map(|(_, p)| (a, p.as_str())))
    }

    /// Uses a downloaded country file if there is a good one, else the built-in copy.
    fn load_cty(&mut self) {
        self.cty_status = match cty_path().filter(|p| p.exists()) {
            Some(p) => match Cty::load(&p) {
                Ok(c) => {
                    let date = std::fs::metadata(&p).and_then(|m| m.modified()).ok().map(date_of).unwrap_or_default();
                    self.cty = Arc::new(c);
                    format!("Downloaded {date}")
                }
                Err(e) => format!("Built-in (downloaded file unreadable: {e})"),
            },
            None => "Built-in".into(),
        };
    }

    fn set_cty(&mut self, c: Cty) {
        for r in &mut self.spots {
            r.country = c.lookup(&r.call);
        }
        self.cty = Arc::new(c);
    }

    fn start_cty_update(&mut self) {
        let Some(dest) = cty_path() else {
            self.cty_status = "No settings directory to save to".into();
            return;
        };
        let (tx, rx) = crossbeam_channel::bounded(1);
        std::thread::spawn(move || {
            let _ = tx.send(Cty::download(&dest).map_err(|e| e.to_string()));
        });
        self.cty_rx = Some(rx);
        self.cty_status = "Downloading…".into();
    }

    fn start_probe(&mut self) {
        let (tx, rx) = crossbeam_channel::unbounded();
        self.rig = None;
        std::thread::spawn(move || {
            let tx2 = tx.clone();
            let found = cat::probe::probe(move |p| {
                let _ = tx2.send(ProbeMsg::Progress(p.to_string()));
            });
            let _ = tx.send(ProbeMsg::Done(found));
        });
        self.probe_rx = Some(rx);
        self.probe_status = "Probing…".into();
    }

    // ---------------------------------------------------------------- UI

    fn ui_top(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new("Audio").strong());
            let cur = self.s.audio_device.clone().unwrap_or_else(|| "(default)".into());
            let mut chosen: Option<Option<String>> = None;
            egui::ComboBox::from_id_salt("audio_dev")
                .width(260.0)
                .selected_text(short(&cur, 40))
                .show_ui(ui, |ui| {
                    for d in &self.devices {
                        let label = format!("{}{}{}", d.name, if d.is_default { "  (default)" } else { "" }, if d.score >= 50 { "  ★ radio" } else { "" });
                        if ui.selectable_label(Some(&d.name) == self.s.audio_device.as_ref(), label).clicked() {
                            chosen = Some(Some(d.name.clone()));
                        }
                    }
                });
            if let Some(c) = chosen {
                self.s.audio_device = c;
                self.s.audio_auto = false;
                self.start_audio();
            }
            if ui.button("Auto").on_hover_text("Pick the radio's USB audio codec automatically").clicked() {
                self.s.audio_auto = true;
                self.start_audio();
            }
            if ui.button("⟳").on_hover_text("Rescan audio devices").clicked() {
                self.devices = audio::list_input_devices();
            }
            let before = self.s.channel_sel;
            egui::ComboBox::from_id_salt("chsel").width(60.0).selected_text(format!("{:?}", self.s.channel_sel)).show_ui(ui, |ui| {
                ui.selectable_value(&mut self.s.channel_sel, ChannelSel::Mix, "Mix");
                ui.selectable_value(&mut self.s.channel_sel, ChannelSel::Left, "Left");
                ui.selectable_value(&mut self.s.channel_sel, ChannelSel::Right, "Right");
            });
            if before != self.s.channel_sel {
                self.start_audio();
            }
            // Input level meter: average level bar, peak-hold tick, clip light.
            let to_x = |db: f32| ((db + 60.0) / 60.0).clamp(0.0, 1.0);
            let (r, resp) = ui.allocate_exact_size(Vec2::new(90.0, 12.0), Sense::hover());
            let p = ui.painter();
            p.rect_filled(r, 2.0, Color32::from_gray(40));
            let avg = self.meter_avg;
            let col = if avg > -10.0 { Color32::RED } else if avg > -18.0 { Color32::YELLOW } else { Color32::from_rgb(70, 200, 90) };
            p.rect_filled(Rect::from_min_size(r.min, Vec2::new(r.width() * to_x(avg), r.height())), 2.0, col);
            let px = r.left() + r.width() * to_x(self.meter_peak);
            p.line_segment([Pos2::new(px, r.top()), Pos2::new(px, r.bottom())], Stroke::new(2.0f32, Color32::WHITE));
            resp.on_hover_text("Bar: average input level. White tick: recent peak. CLIP lights only when the sound card actually overloads.\nAim for the bar in the green (about −40 to −18 dBFS) with no CLIP. On the FTDX3000, Menu 077 sets the USB audio level.");
            ui.label(RichText::new(format!("{avg:.0} dB")).small());
            let clip_on = self.clip_until.is_some_and(|t| Instant::now() < t);
            ui.label(RichText::new(" CLIP ").small().strong().color(if clip_on { Color32::WHITE } else { Color32::from_gray(70) }).background_color(if clip_on {
                Color32::from_rgb(200, 30, 30)
            } else {
                Color32::from_gray(35)
            }));
            ui.separator();
            ui.label("File");
            ui.add(egui::TextEdit::singleline(&mut self.s.file_path).desired_width(160.0).hint_text("path to .wav/.mp3"));
            if ui.button("▶ Play").on_hover_text("Decode an audio file instead of the sound card").clicked() {
                self.start_file();
            }
            if ui.button("Live").on_hover_text("Back to the sound card").clicked() {
                self.start_audio();
            }
            ui.separator();
            if ui.button("Clear").clicked() {
                if let Some(e) = &self.engine {
                    e.send(EngineCmd::Clear);
                }
                self.log.clear();
                self.spots.clear();
            }
            let rec_label = match &self.recording {
                Some((_, t)) => RichText::new(format!("■ Stop rec {}s", t.elapsed().as_secs())).color(Color32::LIGHT_RED),
                None => RichText::new("● Rec audio"),
            };
            if ui.button(rec_label).on_hover_text(format!("Record the radio audio to a WAV file in {}", log_dir().display())).clicked() {
                self.toggle_recording();
            }
            if ui.button("Logs").on_hover_text(format!("Decoded text from every station is saved automatically to {}", log_dir().display())).clicked() {
                let _ = std::fs::create_dir_all(log_dir());
                let opener = if cfg!(windows) { "explorer" } else if cfg!(target_os = "macos") { "open" } else { "xdg-open" };
                let _ = std::process::Command::new(opener).arg(log_dir()).spawn();
            }
            ui.toggle_value(&mut self.show_settings, "⚙ Settings");
        });
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new("Radio").strong());
            let before = self.s.rig.kind;
            egui::ComboBox::from_id_salt("rigkind").width(230.0).selected_text(short(self.s.rig.kind.label(), 34)).show_ui(ui, |ui| {
                for k in RigKind::ALL {
                    ui.selectable_value(&mut self.s.rig.kind, k, k.label());
                }
            });
            if before != self.s.rig.kind {
                // Sensible defaults per protocol.
                match self.s.rig.kind {
                    RigKind::Yaesu => self.s.rig.baud = 38400,
                    RigKind::Kenwood => self.s.rig.baud = 38400,
                    RigKind::Icom => self.s.rig.baud = 19200,
                    RigKind::YaesuLegacy => self.s.rig.baud = 9600,
                    RigKind::Hamlib => {}
                }
            }
            if self.s.rig.kind == RigKind::Hamlib {
                ui.add(egui::TextEdit::singleline(&mut self.s.rig.host).desired_width(140.0));
            } else {
                egui::ComboBox::from_id_salt("port").width(150.0).selected_text(self.s.rig.port.clone()).show_ui(ui, |ui| {
                    for (p, d) in &self.ports {
                        let label = if d.is_empty() { p.clone() } else { format!("{p}  {d}") };
                        ui.selectable_value(&mut self.s.rig.port, p.clone(), label);
                    }
                });
                if ui.small_button("⟳").on_hover_text("Rescan serial ports").clicked() {
                    self.ports = cat::list_ports();
                }
                egui::ComboBox::from_id_salt("baud").width(70.0).selected_text(self.s.rig.baud.to_string()).show_ui(ui, |ui| {
                    for b in [4800, 9600, 19200, 38400, 57600, 115200] {
                        ui.selectable_value(&mut self.s.rig.baud, b, b.to_string());
                    }
                });
                if self.s.rig.kind == RigKind::Icom {
                    ui.label("CI-V");
                    let mut a = format!("{:02X}", self.s.rig.civ_addr);
                    if ui.add(egui::TextEdit::singleline(&mut a).desired_width(28.0)).changed() {
                        if let Ok(v) = u8::from_str_radix(a.trim(), 16) {
                            self.s.rig.civ_addr = v;
                        }
                    }
                }
            }
            if self.rig.is_none() {
                if ui.button("Connect").clicked() {
                    self.connect_rig();
                }
                if ui.button("Auto-detect").on_hover_text("Probe USB serial ports for a radio").clicked() && self.probe_rx.is_none() {
                    self.start_probe();
                }
            } else if ui.button("Disconnect").clicked() {
                self.disconnect_rig();
            }
            let st = &self.rig_state;
            if self.rig.is_some() {
                if st.connected {
                    ui.label(RichText::new(format!("● {}", st.name)).color(Color32::LIGHT_GREEN));
                } else {
                    ui.label(RichText::new("● connecting").color(Color32::YELLOW));
                }
                if let Some(e) = &st.error {
                    ui.label(RichText::new(short(e, 60)).color(Color32::LIGHT_RED).small());
                }
            }
            if !self.probe_status.is_empty() {
                ui.label(RichText::new(&self.probe_status).small());
            }
        });
        if self.rig.is_some() {
            ui.horizontal_wrapped(|ui| self.ui_vfo(ui));
        }
    }

    fn ui_vfo(&mut self, ui: &mut egui::Ui) {
        let vfo = self.rig_state.freq;
        let txt = vfo.map(fmt_vfo).unwrap_or_else(|| "--.---.---".into());
        ui.label(RichText::new(txt).monospace().size(26.0).color(Color32::from_rgb(255, 200, 80)));
        let mode = self.rig_state.mode.as_ref().map(|m| m.label()).unwrap_or_default();
        ui.label(RichText::new(mode).strong());
        if !self.zb_status.is_empty() {
            let col = if self.zb_status.contains('✓') { Color32::LIGHT_GREEN } else { Color32::LIGHT_GRAY };
            ui.label(RichText::new(&self.zb_status).small().color(col));
        }
        ui.label(RichText::new(format!("pitch {:.0} Hz", self.pitch())).small())
            .on_hover_text(if self.rig_state.pitch.is_some() && self.s.auto_pitch { "Read from the radio" } else { "From Settings" });
        if let Some(f) = vfo {
            for (lbl, d) in [("-1k", -1000i64), ("-100", -100), ("-10", -10), ("+10", 10), ("+100", 100), ("+1k", 1000)] {
                if ui.button(lbl).clicked() {
                    self.set_vfo((f as i64 + d).max(0) as u64);
                }
            }
        }
        ui.separator();
        for (b, f) in [("160", 1_820_000u64), ("80", 3_530_000), ("40", 7_030_000), ("30", 10_115_000), ("20", 14_030_000), ("17", 18_080_000), ("15", 21_030_000), ("12", 24_900_000), ("10", 28_030_000)] {
            if ui.small_button(b).on_hover_text(format!("{} kHz", f / 1000)).clicked() {
                self.set_vfo(f);
            }
        }
        ui.separator();
        let r = ui.add(egui::TextEdit::singleline(&mut self.freq_entry).desired_width(80.0).hint_text("kHz"));
        if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
            if let Ok(k) = self.freq_entry.trim().parse::<f64>() {
                self.set_vfo((k * 1000.0).round() as u64);
            }
        }
        if ui.small_button("CW").clicked() {
            if let Some(r) = &self.rig {
                r.set_mode(Mode::Cw);
            }
        }
        if ui.small_button("CW-R").clicked() {
            if let Some(r) = &self.rig {
                r.set_mode(Mode::CwR);
            }
        }
    }

    fn ui_settings(&mut self, ui: &mut egui::Ui) {
        let mut changed = false;
        egui::Grid::new("settings").num_columns(2).spacing([12.0, 6.0]).show(ui, |ui| {
            ui.label("CW pitch (Hz)");
            ui.horizontal(|ui| {
                let from_radio = self.s.auto_pitch && self.rig.is_some() && self.rig_state.pitch.is_some();
                if from_radio {
                    ui.label(RichText::new(format!("{:.0} Hz (read from radio)", self.pitch())).strong());
                } else {
                    ui.add(egui::DragValue::new(&mut self.s.pitch).range(300.0..=1200.0).speed(10.0))
                        .on_hover_text("Must match the radio's CW pitch (FTDX3000 default 700 Hz)");
                }
                ui.checkbox(&mut self.s.auto_pitch, "auto").on_hover_text("Read the CW pitch from the radio over CAT (Yaesu)");
            });
            ui.end_row();
            ui.label("Passband (Hz)");
            ui.horizontal(|ui| {
                changed |= ui.add(egui::DragValue::new(&mut self.s.min_freq).range(100.0..=1500.0).speed(10.0)).changed();
                ui.label("–");
                changed |= ui.add(egui::DragValue::new(&mut self.s.max_freq).range(800.0..=3900.0).speed(10.0)).changed();
            });
            ui.end_row();
            ui.label("Detection threshold (dB)");
            changed |= ui.add(egui::Slider::new(&mut self.s.detect_db, 4.0..=20.0)).changed();
            ui.end_row();
            ui.label("Max decoders");
            changed |= ui.add(egui::Slider::new(&mut self.s.max_channels, 1..=80)).changed();
            ui.end_row();
            ui.label("Squelch");
            changed |= ui.add(egui::Slider::new(&mut self.s.squelch, 1.6..=4.0)).on_hover_text("Lower decodes weaker signals but more noise garbage").changed();
            ui.end_row();
            ui.label("Text prediction (LM weight)");
            changed |= ui.add(egui::Slider::new(&mut self.s.lm_weight, 0.0..=2.0)).on_hover_text("0 = timing only; higher trusts the language model more").changed();
            ui.end_row();
            ui.label("Show RF frequency");
            ui.checkbox(&mut self.s.show_rf, "");
            ui.end_row();
            ui.label("Waterfall brightness / contrast");
            ui.horizontal(|ui| {
                ui.add(egui::Slider::new(&mut self.s.wf_brightness, -20.0..=20.0).show_value(false));
                ui.add(egui::Slider::new(&mut self.s.wf_contrast, 0.3..=3.0).show_value(false));
            });
            ui.end_row();
            ui.label("Always drive RTS / DTR high");
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.s.rig.assert_rts, "RTS")
                    .on_hover_text("Normally leave OFF: the RTS handshake is raised automatically on the CAT port if the radio needs it (CAT RTS = ON), and never on a PTT/keying port.");
                ui.checkbox(&mut self.s.rig.assert_dtr, "DTR").on_hover_text("Leave OFF: DTR can key the transmitter.");
            });
            ui.end_row();
        });
        ui.add_space(6.0);
        ui.strong("Station lookup");
        ui.label(
            RichText::new("Click a callsign for name, location and photo. US/Canada calls work without an account (HamDB). Add a free QRZ.com and/or HamQTH.com login for DX stations and photos.")
                .small()
                .weak(),
        );
        let before = self.s.lookup.clone();
        egui::Grid::new("lookup").num_columns(3).spacing([8.0, 4.0]).show(ui, |ui| {
            ui.label("My grid");
            ui.add(egui::TextEdit::singleline(&mut self.s.my_grid).desired_width(70.0).hint_text("FN31pr"))
                .on_hover_text("Your Maidenhead locator, to show distance and beam heading");
            ui.end_row();
            for (label, user, pass) in [
                ("QRZ.com", &mut self.s.lookup.qrz_user, &mut self.s.lookup.qrz_pass),
                ("HamQTH", &mut self.s.lookup.hamqth_user, &mut self.s.lookup.hamqth_pass),
            ] {
                ui.label(label);
                ui.add(egui::TextEdit::singleline(user).desired_width(90.0).hint_text("username"));
                ui.add(egui::TextEdit::singleline(pass).desired_width(90.0).password(true).hint_text("password"));
                ui.end_row();
            }
        });
        ui.label(RichText::new("Passwords are stored unencrypted in the app's settings file.").small().weak());
        ui.add_space(6.0);
        ui.strong("Country file");
        ui.horizontal(|ui| {
            ui.label(format!("{} countries", self.cty.len()));
            let busy = self.cty_rx.is_some();
            if ui.add_enabled(!busy, egui::Button::new("Update")).on_hover_text(format!("Download the latest AD1C cty.dat from {}", cty::URL)).clicked() {
                self.start_cty_update();
            }
            if busy {
                ui.spinner();
            }
        });
        ui.label(RichText::new(&self.cty_status).small().weak());
        ui.add_space(6.0);
        self.ui_activity_settings(ui);
        if self.s.lookup != before {
            self.lookup.clear();
        }
        if changed {
            if let Some(e) = &self.engine {
                e.send(EngineCmd::Config(self.skimmer_config()));
            }
        }
        ui.label(RichText::new("Decoder parameters apply to newly detected signals.").small().weak());
    }

    /// Editable list of activity flags.
    fn ui_activity_settings(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new(RichText::new("Activity flags").strong()).id_salt("activity_flags").show(ui, |ui| {
            ui.label(
                RichText::new("Stations sending these words get a flag. Separate patterns with commas; a pattern can be several words (\"CQ POTA\") and * matches anything (\"K-*\").")
                    .small()
                    .weak(),
            );
            let mut remove = None;
            egui::Grid::new("activities").num_columns(4).spacing([6.0, 4.0]).show(ui, |ui| {
                for (i, a) in self.s.activities.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut a.enabled, "").on_hover_text("Show this flag");
                        ui.color_edit_button_srgb(&mut a.color);
                    });
                    ui.add(egui::TextEdit::singleline(&mut a.name).desired_width(56.0).hint_text("name"));
                    ui.add(egui::TextEdit::singleline(&mut a.patterns).desired_width(170.0).hint_text("CQ XYZ, XYZ"));
                    if ui.small_button("✖").on_hover_text("Delete this flag").clicked() {
                        remove = Some(i);
                    }
                    ui.end_row();
                }
            });
            if let Some(i) = remove {
                self.s.activities.remove(i);
            }
            ui.horizontal(|ui| {
                if ui.button("Add").clicked() {
                    self.s.activities.push(Activity::default());
                }
                if ui.button("Restore defaults").on_hover_text("Replace this list with the built-in flags").clicked() {
                    self.s.activities = activity::defaults();
                }
            });
        });
    }

    fn ui_waterfall(&mut self, ui: &mut egui::Ui, height: f32) {
        let tex = self.wf_tex.get_or_insert_with(|| ui.ctx().load_texture("waterfall", self.wf_img.clone(), TextureOptions::LINEAR));
        if self.wf_dirty {
            tex.set(self.wf_img.clone(), TextureOptions::LINEAR);
            self.wf_dirty = false;
        }
        let tex_id = tex.id();
        let width = ui.available_width();
        let (rect, resp) = ui.allocate_exact_size(Vec2::new(width, height), Sense::click());
        let painter = ui.painter_at(rect);
        let total_bins = wf_width() as f32;
        let hz = bin_hz();
        let (f0, f1) = (self.s.min_freq, self.s.max_freq);
        let u0 = f0 / hz / total_bins;
        let u1 = f1 / hz / total_bins;
        let axis_h = 16.0;
        let img_rect = Rect::from_min_max(rect.min + Vec2::new(0.0, axis_h), rect.max);
        painter.image(tex_id, img_rect, Rect::from_min_max(Pos2::new(u0, 0.0), Pos2::new(u1, 1.0)), Color32::WHITE);
        let x_of = |f: f32| rect.left() + (f - f0) / (f1 - f0) * rect.width();
        let f_of = |x: f32| f0 + (x - rect.left()) / rect.width() * (f1 - f0);
        // Axis
        painter.rect_filled(Rect::from_min_max(rect.min, Pos2::new(rect.right(), rect.top() + axis_h)), 0.0, Color32::from_gray(25));
        let step = if self.s.show_rf && self.rig_state.freq.is_some() { 200.0 } else { 250.0 };
        let mut f = (f0 / step).ceil() * step;
        while f < f1 {
            let x = x_of(f);
            painter.line_segment([Pos2::new(x, rect.top() + axis_h - 4.0), Pos2::new(x, rect.top() + axis_h)], Stroke::new(1.0f32, Color32::GRAY));
            let label = if self.s.show_rf {
                self.rf_of(f).map(|rf| format!("{:.1}", rf as f64 / 1000.0)).unwrap_or(format!("{f:.0}"))
            } else {
                format!("{f:.0}")
            };
            painter.text(Pos2::new(x, rect.top() + 1.0), egui::Align2::CENTER_TOP, label, egui::FontId::proportional(10.0), Color32::LIGHT_GRAY);
            f += step;
        }
        // Pitch marker
        let px = x_of(self.pitch());
        painter.line_segment([Pos2::new(px, img_rect.top()), Pos2::new(px, img_rect.bottom())], Stroke::new(1.0f32, Color32::from_rgba_unmultiplied(255, 80, 80, 120)));
        // Channel markers with callsigns
        for ch in &self.snap.channels {
            let x = x_of(ch.freq);
            let sel = Some(ch.id) == self.selected;
            let col = if sel { Color32::YELLOW } else if ch.is_mark { Color32::from_rgb(120, 255, 120) } else { Color32::from_rgb(80, 160, 255) };
            painter.line_segment([Pos2::new(x, img_rect.top()), Pos2::new(x, img_rect.top() + 10.0)], Stroke::new(2.0f32, col));
            // Until a callsign is decoded, show the measured code speed.
            let label = ch.callsign.clone().unwrap_or_else(|| ch.wpm.map(|w| format!("{w:.0}wpm")).unwrap_or_default());
            if !label.is_empty() {
                // Dark backing box so the label stands out from the waterfall.
                let font = if ch.callsign.is_some() { egui::FontId::monospace(13.0) } else { egui::FontId::monospace(11.0) };
                // Steady colour (the tick above blinks with the keying, the label doesn't).
                let col = if sel { Color32::YELLOW } else if ch.callsign.is_some() { Color32::from_rgb(255, 210, 90) } else { Color32::from_rgb(170, 200, 255) };
                let galley = painter.layout_no_wrap(label, font, col);
                let pos = Pos2::new(x + 3.0, img_rect.top() + 10.0);
                let bg = Rect::from_min_size(pos, galley.size()).expand2(Vec2::new(3.0, 1.0));
                painter.rect_filled(bg, 3.0, Color32::from_black_alpha(200));
                painter.rect_stroke(bg, 3.0, Stroke::new(1.0f32, col.gamma_multiply(0.6)), egui::StrokeKind::Inside);
                painter.galley(pos, galley, col);
            }
        }
        // Hover readout
        if let Some(p) = resp.hover_pos() {
            let f = f_of(p.x);
            painter.line_segment([Pos2::new(p.x, img_rect.top()), Pos2::new(p.x, img_rect.bottom())], Stroke::new(1.0f32, Color32::from_white_alpha(60)));
            let t = match self.rf_of(f) {
                Some(rf) => format!("{:.2} kHz ({f:.0} Hz)", rf as f64 / 1000.0),
                None => format!("{f:.0} Hz"),
            };
            painter.text(p + Vec2::new(8.0, -14.0), egui::Align2::LEFT_BOTTOM, t, egui::FontId::proportional(12.0), Color32::WHITE);
        }
        if resp.clicked() {
            if let Some(p) = resp.interact_pointer_pos() {
                let f = f_of(p.x);
                // Select the nearest channel, or start a decoder there.
                let near = self.snap.channels.iter().filter(|c| (c.freq - f).abs() < 30.0).min_by(|a, b| (a.freq - f).abs().total_cmp(&(b.freq - f).abs())).map(|c| c.id);
                match near {
                    Some(id) => {
                        self.selected = Some(id);
                        self.bottom_tab = 1;
                    }
                    None => {
                        if let Some(e) = &self.engine {
                            e.send(EngineCmd::AddChannel(f));
                        }
                    }
                }
            }
        }
        if resp.double_clicked() || resp.secondary_clicked() {
            if let Some(p) = resp.interact_pointer_pos() {
                let f = f_of(p.x);
                let snapped = self.snap.channels.iter().filter(|c| (c.freq - f).abs() < 30.0).min_by(|a, b| (a.freq - f).abs().total_cmp(&(b.freq - f).abs())).map(|c| c.freq).unwrap_or(f);
                self.zero_beat(snapped);
            }
        }
        // Mouse wheel over the waterfall tunes the radio.
        if resp.hovered() {
            let scroll = ui.input(|i| i.raw_scroll_delta.y);
            if scroll.abs() > 0.5 {
                if let Some(f) = self.rig_state.freq {
                    let step = if scroll > 0.0 { 50i64 } else { -50 };
                    self.set_vfo((f as i64 + step).max(0) as u64);
                }
            }
        }
        resp.on_hover_text_at_pointer("Click: decode / select   Double-click or right-click: tune radio   Wheel: tune ±50 Hz");
    }

    fn ui_channels(&mut self, ui: &mut egui::Ui) {
        let chans: Vec<ChannelInfo> = self.snap.channels.clone();
        let mut tune: Option<f32> = None;
        let mut remove: Option<u64> = None;
        let text_w = ui.available_width() - 330.0;
        let char_w = 7.6;
        let max_chars = ((text_w / char_w) as usize).max(10);
        let ui_ctx = ui.ctx().clone();
        TableBuilder::new(ui)
            .striped(true)
            .sense(Sense::click())
            .stick_to_bottom(false)
            .column(Column::exact(86.0))
            .column(Column::exact(80.0))
            .column(Column::exact(34.0))
            .column(Column::exact(44.0))
            .column(Column::remainder().clip(true))
            .column(Column::exact(22.0))
            .header(18.0, |mut h| {
                for t in ["Freq", "Call", "WPM", "SNR", "Decoded text", ""] {
                    h.col(|ui| {
                        ui.strong(t);
                    });
                }
            })
            .body(|mut body| {
                for ch in &chans {
                    let sel = Some(ch.id) == self.selected;
                    body.row(20.0, |mut row| {
                        row.set_selected(sel);
                        row.col(|ui| {
                            let dot = if ch.is_mark { "●" } else { "○" };
                            ui.add(egui::Label::new(RichText::new(format!("{dot} {}", self.fmt_freq(ch.freq))).monospace()).selectable(false));
                        });
                        row.col(|ui| {
                            if let Some(c) = &ch.callsign {
                                let r = call_link(ui, c, 13.0);
                                if r.clicked() {
                                    self.info_call = Some(c.clone());
                                }
                                r.on_hover_text("Click for station info");
                            }
                        });
                        row.col(|ui| {
                            ui.label(ch.wpm.map(|w| format!("{w:.0}")).unwrap_or_else(|| "–".into()));
                        });
                        row.col(|ui| {
                            ui.label(format!("{:.0}", ch.snr_db));
                        });
                        row.col(|ui| {
                            let tags = self.chan_tags.get(&ch.id).map(Vec::as_slice).unwrap_or_default();
                            let flags: Vec<_> = self.flags(tags).collect();
                            let flag_chars: usize = flags.iter().map(|(a, _)| a.name.chars().count() + 3).sum();
                            let full = to_display(&ch.text);
                            let tent = to_display(&ch.tentative);
                            let total = full.len() + tent.len();
                            let skip = total.saturating_sub(max_chars.saturating_sub(flag_chars).max(10));
                            let (a, b) = if skip >= full.len() {
                                (String::new(), tent[skip - full.len()..].to_string())
                            } else {
                                (full[skip..].to_string(), tent)
                            };
                            let age = ch.last_mark_age;
                            // Text that reads poorly (weak or garbled) is dimmed, not hidden.
                            let col = if ch.doubtful { Color32::from_gray(150) } else if age < 5.0 { Color32::WHITE } else { Color32::GRAY };
                            ui.horizontal(|ui| {
                                for (act, heard) in &flags {
                                    flag_badge(ui, act, heard);
                                }
                                ui.spacing_mut().item_spacing.x = 0.0;
                                ui.add(egui::Label::new(RichText::new(a).monospace().color(col)).selectable(false));
                                ui.add(egui::Label::new(RichText::new(b).monospace().color(Color32::from_gray(130))).selectable(false));
                            });
                        });
                        row.col(|ui| {
                            if ui.small_button("✖").on_hover_text("Remove decoder").clicked() {
                                remove = Some(ch.id);
                            }
                        });
                        // Clicking anywhere else on the row follows and tunes to it.
                        let r = row.response();
                        if r.clicked() && remove.is_none() {
                            self.selected = Some(ch.id);
                            self.bottom_tab = 1;
                            tune = Some(ch.freq);
                        }
                        if r.hovered() {
                            ui_ctx.set_cursor_icon(egui::CursorIcon::PointingHand);
                        }
                        r.on_hover_text("Click to tune the radio to this station");
                    });
                }
            });
        if let Some(f) = tune {
            self.zero_beat(f);
        }
        if let (Some(id), Some(e)) = (remove, &self.engine) {
            e.send(EngineCmd::RemoveChannel(id));
        }
    }

    fn ui_spots(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("Spots");
            let label = |f: &str| match f {
                "" => "All".to_string(),
                "*" => "Any flag".to_string(),
                n => n.to_string(),
            };
            let names: Vec<String> = self.s.activities.iter().filter(|a| a.enabled && !a.name.trim().is_empty()).map(|a| a.name.clone()).collect();
            // A flag that was renamed or turned off no longer filters anything.
            if !matches!(self.s.spot_filter.as_str(), "" | "*") && !names.contains(&self.s.spot_filter) {
                self.s.spot_filter.clear();
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                egui::ComboBox::from_id_salt("spot_filter").selected_text(label(&self.s.spot_filter)).show_ui(ui, |ui| {
                    for f in [String::new(), "*".to_string()].into_iter().chain(names) {
                        let l = label(&f);
                        ui.selectable_value(&mut self.s.spot_filter, f, l);
                    }
                });
                ui.label("Show");
            });
        });
        let mut spots = self.spots.clone();
        let filter = self.s.spot_filter.clone();
        if !filter.is_empty() {
            spots.retain(|r| self.flags(&r.tags).any(|(a, _)| filter == "*" || a.name == filter));
        }
        spots.sort_by(|a, b| a.last.cmp(&b.last).reverse());
        let mut tune_rf: Option<u64> = None;
        let mut tune_audio: Option<f32> = None;
        TableBuilder::new(ui)
            .striped(true)
            .column(Column::exact(82.0))
            .column(Column::exact(72.0))
            .column(Column::exact(30.0))
            .column(Column::exact(30.0))
            .column(Column::exact(34.0))
            .column(Column::remainder().at_least(40.0).clip(true))
            .header(18.0, |mut h| {
                for t in ["Call", "Freq", "WPM", "SNR", "Age", "Country"] {
                    h.col(|ui| {
                        ui.strong(t);
                    });
                }
            })
            .body(|mut body| {
                for s in &spots {
                    let dim = if s.confirmed { 1.0 } else { 0.5 };
                    body.row(18.0, |mut row| {
                        row.col(|ui| {
                            ui.multiply_opacity(dim);
                            let r = call_link(ui, &s.call, 13.0);
                            if r.clicked() {
                                self.info_call = Some(s.call.clone());
                            }
                            if r.double_clicked() {
                                match s.rf {
                                    Some(rf) => tune_rf = Some(rf),
                                    None => tune_audio = Some(s.audio),
                                }
                            }
                            let hint = if s.confirmed { "" } else { "Heard once so far (may be a decoding error)\n" };
                            r.on_hover_text(format!("{hint}Click: station info   Double-click: tune"));
                        });
                        row.col(|ui| {
                            let f = match s.rf {
                                Some(rf) => format!("{:.2}", rf as f64 / 1000.0),
                                None => format!("{:.0} Hz", s.audio),
                            };
                            let r = ui.add(egui::Label::new(f).sense(Sense::click()));
                            if r.clicked() {
                                match s.rf {
                                    Some(rf) => tune_rf = Some(rf),
                                    None => tune_audio = Some(s.audio),
                                }
                            }
                            r.on_hover_text("Click to tune");
                        });
                        row.col(|ui| {
                            ui.label(format!("{:.0}", s.wpm));
                        });
                        row.col(|ui| {
                            ui.label(format!("{:.0}", s.snr));
                        });
                        row.col(|ui| {
                            let a = s.last.elapsed().as_secs();
                            ui.label(if a < 60 { format!("{a}s") } else { format!("{}m", a / 60) });
                        });
                        row.col(|ui| {
                            ui.horizontal(|ui| {
                                for (act, heard) in self.flags(&s.tags) {
                                    flag_badge(ui, act, heard);
                                }
                                if let Some(c) = &s.country {
                                    ui.add(egui::Label::new(&c.name).truncate()).on_hover_text(self.country_summary(c));
                                }
                            });
                        });
                    });
                }
            });
        if let Some(rf) = tune_rf {
            self.tune_rf(rf);
        }
        if let Some(a) = tune_audio {
            self.zero_beat(a);
        }
    }

    fn ui_bottom(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let conv = match &self.qso {
                Some(q) => format!("Conversation: {}", q.title()),
                None => "Conversation".into(),
            };
            ui.selectable_value(&mut self.bottom_tab, 0, conv);
            ui.selectable_value(&mut self.bottom_tab, 1, "Selected signal");
            ui.selectable_value(&mut self.bottom_tab, 2, format!("History ({})", self.qso_history.len()));
            if self.bottom_tab == 0 && self.qso.is_some() {
                let rf = self.qso.as_ref().and_then(|q| q.rf);
                if let (Some(rf), true) = (rf, self.rig.is_some()) {
                    if ui.small_button("Tune back").on_hover_text(format!("Return to {:.2} kHz", rf as f64 / 1000.0)).clicked() {
                        self.tune_rf(rf);
                    }
                }
                if ui.small_button("End / new").on_hover_text("Save and close this conversation now").clicked() {
                    self.qso_finish();
                }
            }
            if self.bottom_tab == 2 {
                let rf = self.history_sel.and_then(|i| self.qso_history.get(i)).and_then(|q| q.rf);
                if let (Some(rf), true) = (rf, self.rig.is_some()) {
                    if ui.small_button("Tune").on_hover_text(format!("Go to {:.2} kHz", rf as f64 / 1000.0)).clicked() {
                        self.tune_rf(rf);
                    }
                }
            }
            ui.checkbox(&mut self.s.explain_abbrev, "Explain abbreviations")
                .on_hover_text("Show the meaning of CW abbreviations (TNX, QTH, 73…) in small print. Highlighted words always explain themselves on hover.");
            if ui.small_button("Open folder").on_hover_text(format!("Conversations are saved in {}", log_dir().display())).clicked() {
                let opener = if cfg!(windows) { "explorer" } else if cfg!(target_os = "macos") { "open" } else { "xdg-open" };
                let _ = std::process::Command::new(opener).arg(log_dir()).spawn();
            }
        });
        ui.separator();
        let gloss = self.s.explain_abbrev;
        match self.bottom_tab {
            0 => match &self.qso {
                Some(q) => {
                    let text = q.text.clone();
                    let rf = q.rf;
                    let mut tune = false;
                    egui::ScrollArea::vertical().stick_to_bottom(true).auto_shrink([false, false]).show(ui, |ui| {
                        let (r, call) = cw_text(ui, &text, 15.0, gloss, Sense::click(), "Double-click to tune back to this conversation");
                        tune = r.double_clicked();
                        if call.is_some() {
                            self.info_call = call;
                        }
                    });
                    if let (true, Some(rf)) = (tune, rf) {
                        self.tune_rf(rf);
                    }
                }
                None => {
                    ui.label(RichText::new("Tune to a station (double-click it). Everything decoded on your frequency is collected here and saved to a file automatically; it is archived when you move away.").weak());
                }
            },
            1 => self.ui_selected(ui),
            _ => {
                let mut sel = self.history_sel;
                let mut tune: Option<u64> = None;
                ui.horizontal_top(|ui| {
                    egui::ScrollArea::vertical().id_salt("hist_list").max_width(300.0).auto_shrink([true, false]).show(ui, |ui| {
                        for (i, q) in self.qso_history.iter().enumerate() {
                            let r = ui.selectable_label(sel == Some(i), q.title());
                            if r.clicked() {
                                sel = Some(i);
                            }
                            if r.double_clicked() {
                                tune = q.rf;
                            }
                            r.on_hover_text("Click to read · double-click to tune there");
                        }
                        if self.qso_history.is_empty() {
                            ui.label(RichText::new("Finished conversations appear here.").weak());
                        }
                    });
                    ui.separator();
                    if let Some(q) = sel.and_then(|i| self.qso_history.get(i)) {
                        let text = q.text.clone();
                        let rf = q.rf;
                        egui::ScrollArea::vertical().id_salt("hist_text").auto_shrink([false, false]).show(ui, |ui| {
                            let (r, call) = cw_text(ui, &text, 14.0, gloss, Sense::click(), "Double-click to tune to this conversation");
                            if r.double_clicked() {
                                tune = rf;
                            }
                            if call.is_some() {
                                self.info_call = call;
                            }
                        });
                    }
                });
                self.history_sel = sel;
                if let Some(rf) = tune {
                    self.tune_rf(rf);
                }
            }
        }
    }

    fn ui_selected(&mut self, ui: &mut egui::Ui) {
        let Some(id) = self.selected else {
            ui.label(RichText::new("Click a signal on the waterfall or a row above to follow it here.").weak());
            return;
        };
        let Some(ch) = self.snap.channels.iter().find(|c| c.id == id).cloned() else {
            ui.label(RichText::new("Selected signal is gone.").weak());
            return;
        };
        ui.horizontal(|ui| {
            ui.label(RichText::new(self.fmt_freq(ch.freq)).monospace().strong());
            if let Some(c) = &ch.callsign {
                if call_link(ui, c, 14.0).on_hover_text("Station info").clicked() {
                    self.info_call = Some(c.clone());
                }
            }
            ui.label(format!("{} WPM  SNR {:.0} dB", ch.wpm.map(|w| format!("{w:.0}")).unwrap_or("–".into()), ch.snr_db));
            if self.rig.is_some() && ui.button("Tune radio here").clicked() {
                self.zero_beat(ch.freq);
            }
        });
        let gloss = self.s.explain_abbrev;
        if ch.doubtful {
            ui.label(RichText::new("Text reads poorly (weak or garbled signal): shown dimmed").small().weak());
        }
        egui::ScrollArea::vertical().stick_to_bottom(true).auto_shrink([false, false]).show(ui, |ui| {
            if ch.doubtful {
                ui.multiply_opacity(0.6);
            }
            let (_, call) = cw_text(ui, &format!("{}{}", to_display(&ch.text), to_display(&ch.tentative)), 15.0, gloss, Sense::hover(), "");
            if call.is_some() {
                self.info_call = call;
            }
        });
    }

    /// "Fed. Rep. of Germany (DL)  EU  CQ 14  ITU 28", plus the local time and,
    /// with the operator's grid set, the beam heading.
    fn country_summary(&self, c: &Country) -> String {
        let mut t = format!("{} ({})   {}   CQ {}   ITU {}\nLocal time {}", c.name, c.prefix, c.cont, c.cq, c.itu, time_at(c.utc_offset));
        if let Some((km, brg)) = self.path_to((c.lat, c.lon)) {
            t += &format!("\nAbout {km:.0} km, beam {brg:.0}°");
        }
        if !c.dxcc {
            t += "\nCounts for WAE only, not DXCC";
        }
        t
    }

    /// Distance (km) and beam heading from the operator's grid.
    fn path_to(&self, pos: (f64, f64)) -> Option<(f64, f64)> {
        lookup::grid_pos(&self.s.my_grid).map(|me| lookup::distance_bearing(me, pos))
    }

    /// Station-info window for the clicked callsign.
    fn ui_call_info(&mut self, ctx: &egui::Context) {
        let Some(call) = self.info_call.clone() else { return };
        let state = self.lookup.get(&call, &self.s.lookup);
        let country = self.cty.lookup(&call);
        let mut open = true;
        egui::Window::new(RichText::new(&call).monospace().strong().color(CALL_COLOR))
            .id(egui::Id::new("call_info"))
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .default_width(380.0)
            .show(ctx, |ui| {
                if let Some(c) = &country {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(RichText::new(&c.name).strong());
                        ui.label(RichText::new(format!("{}   {}   CQ {}   ITU {}   local {}", c.prefix, c.cont, c.cq, c.itu, time_at(c.utc_offset))).weak());
                    });
                    if !c.dxcc {
                        ui.label(RichText::new("Counts for WAE only, not DXCC").small().weak());
                    }
                    ui.add_space(2.0);
                }
                match &state {
                    LookupState::Pending => {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label("Looking up…");
                        });
                    }
                    LookupState::Failed(e) => {
                        ui.label("No information found.");
                        ui.label(RichText::new(e).small().weak());
                        if self.s.lookup.qrz_user.is_empty() && self.s.lookup.hamqth_user.is_empty() {
                            ui.label(RichText::new("Without a login only US and Canadian calls can be looked up. Add a free QRZ.com or HamQTH login in Settings.").small());
                        }
                    }
                    LookupState::Done(i) => self.ui_info_body(ui, i, country.as_ref()),
                }
                // Without a station position, aim at the country instead.
                let located = matches!(&state, LookupState::Done(i) if i.lat_lon.is_some() || lookup::grid_pos(&i.grid).is_some());
                if let (false, Some((km, brg))) = (located, country.as_ref().and_then(|c| self.path_to((c.lat, c.lon)))) {
                    ui.label(format!("Country centre: about {km:.0} km / {:.0} mi, beam {brg:.0}° (long path {:.0}°)", km * 0.621371, (brg + 180.0) % 360.0));
                }
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Open on QRZ.com").clicked() {
                        open_in_browser(&lookup::qrz_url(&call));
                    }
                    if ui.button("HamQTH").clicked() {
                        open_in_browser(&lookup::hamqth_url(&call));
                    }
                    if let LookupState::Done(i) = &state {
                        ui.label(RichText::new(format!("via {}", i.sources.join(", "))).small().weak());
                    }
                });
            });
        if !open {
            self.info_call = None;
        }
    }

    fn ui_info_body(&self, ui: &mut egui::Ui, i: &lookup::CallInfo, country: Option<&Country>) {
        ui.horizontal_top(|ui| {
            if let Some(img) = &i.image {
                let uri = format!("bytes://photo/{}", i.call);
                ui.add(egui::Image::from_bytes(uri, egui::load::Bytes::Shared(img.clone())).max_size(Vec2::new(160.0, 200.0)).corner_radius(4.0));
            }
            ui.vertical(|ui| {
                if !i.name.is_empty() {
                    ui.label(RichText::new(&i.name).size(16.0).strong());
                }
                for l in &i.address {
                    ui.label(l);
                }
                // Shown already from the country file unless the lookup says otherwise.
                if !i.country.is_empty() && country.is_none_or(|c| !c.name.eq_ignore_ascii_case(&i.country)) {
                    ui.label(&i.country);
                }
                ui.add_space(4.0);
                egui::Grid::new("info_grid").num_columns(2).spacing([10.0, 2.0]).show(ui, |ui| {
                    let mut row = |k: &str, v: String| {
                        if !v.is_empty() {
                            ui.label(RichText::new(k).weak());
                            ui.label(v);
                            ui.end_row();
                        }
                    };
                    let pos = i.lat_lon.or_else(|| lookup::grid_pos(&i.grid));
                    let grid = if i.grid.is_empty() { pos.map(|(a, b)| lookup::grid_of(a, b)).unwrap_or_default() } else { i.grid.clone() };
                    row("Grid", grid);
                    if let Some((km, brg)) = pos.and_then(|p| self.path_to(p)) {
                        row("Distance", format!("{km:.0} km / {:.0} mi", km * 0.621371));
                        row("Beam heading", format!("{brg:.0}°  (long path {:.0}°)", (brg + 180.0) % 360.0));
                    }
                    row("Class", i.class.clone());
                    row("QSL", i.qsl.clone());
                    row("LoTW", if i.lotw { "yes".into() } else { String::new() });
                    row("Email", i.email.clone());
                });
                if !i.web.is_empty() {
                    let url = if i.web.starts_with("http") { i.web.clone() } else { format!("https://{}", i.web) };
                    if ui.link(&i.web).clicked() {
                        open_in_browser(&url);
                    }
                }
            });
        });
    }
}

/// Decoded CW text with abbreviations, Q-codes and prosigns highlighted.
/// Hovering one shows its meaning; with `gloss` the meaning is also printed
/// in small text underneath it (rows holding one get taller to make room).
/// Callsigns are shown as links; the one clicked is returned.
/// `hint` is the hover text elsewhere.
fn cw_text(ui: &mut egui::Ui, text: &str, size: f32, gloss: bool, sense: Sense, hint: &str) -> (egui::Response, Option<String>) {
    use egui::text::{LayoutJob, TextFormat, TextWrapping};
    let mono = egui::FontId::monospace(size);
    let small_font = egui::FontId::proportional((size * 0.62).max(9.0));
    let (text_h, small_h) = ui.fonts(|f| (f.row_height(&mono), f.row_height(&small_font)));
    // Top-aligned so a taller abbreviation leaves the rest of its row in place.
    let normal = TextFormat { font_id: mono, color: ui.visuals().text_color(), valign: egui::Align::TOP, ..Default::default() };
    let abbr = TextFormat {
        color: if ui.visuals().dark_mode { Color32::from_rgb(110, 205, 255) } else { Color32::from_rgb(0, 95, 170) },
        line_height: gloss.then_some(text_h + small_h + 1.0),
        ..normal.clone()
    };
    let link = TextFormat { color: CALL_COLOR, underline: Stroke::new(1.0f32, CALL_COLOR.gamma_multiply(0.5)), ..normal.clone() };
    let mut job = LayoutJob::default();
    job.wrap.max_width = ui.available_width();
    // Highlighted words (byte ranges): abbreviations with their meaning, and callsigns.
    let mut marks: Vec<(std::ops::Range<usize>, Option<String>)> = cwskimmer::abbrev::find(text).into_iter().map(|(r, m)| (r, Some(m))).collect();
    let mut pos = 0;
    for w in text.split_inclusive(char::is_whitespace) {
        let word = w.trim_end();
        let r = pos..pos + word.len();
        pos += w.len();
        if !word.is_empty() && cwskimmer::callsign::parse_callsign(word).is_some() && !marks.iter().any(|(m, _)| m.start < r.end && r.start < m.end) {
            marks.push((r, None));
        }
    }
    marks.sort_by_key(|(r, _)| r.start);
    // Char ranges of the abbreviations with their meanings, and of the callsigns.
    let mut spans: Vec<(std::ops::Range<usize>, String)> = Vec::new();
    let mut calls: Vec<(std::ops::Range<usize>, String)> = Vec::new();
    let mut chars = 0;
    let mut last = 0;
    for (r, m) in marks {
        job.append(&text[last..r.start], 0.0, normal.clone());
        chars += text[last..r.start].chars().count();
        let n = text[r.clone()].chars().count();
        match m {
            Some(m) => {
                job.append(&text[r.clone()], 0.0, abbr.clone());
                spans.push((chars..chars + n, m));
            }
            None => {
                job.append(&text[r.clone()], 0.0, link.clone());
                calls.push((chars..chars + n, text[r.clone()].to_string()));
            }
        }
        chars += n;
        last = r.end;
    }
    job.append(&text[last..], 0.0, normal);
    let galley = ui.fonts(|f| f.layout_job(job));
    let r = ui.add(egui::Label::new(galley.clone()).sense(sense.union(Sense::click())));
    let char_at = |p: Pos2| galley.cursor_from_pos(p - r.rect.min).ccursor.index;
    let on_call = |p: Pos2| {
        let c = char_at(p);
        calls.iter().find(|(s, _)| s.contains(&c)).map(|(_, call)| call.clone())
    };
    let clicked = if r.clicked() { r.interact_pointer_pos().and_then(on_call) } else { None };
    if r.hover_pos().and_then(on_call).is_some() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    let origin = r.rect.min.to_vec2();
    // Word boxes (galley coordinates): x range and row top.
    let boxes: Vec<(f32, f32, f32)> = spans
        .iter()
        .map(|(s, _)| {
            let a = galley.pos_from_ccursor(egui::text::CCursor::new(s.start));
            let b = galley.pos_from_ccursor(egui::text::CCursor::new(s.end));
            (a.left(), if (b.top() - a.top()).abs() < 1.0 { b.left() } else { galley.size().x }, a.top())
        })
        .collect();
    let painter = ui.painter();
    let underline = Stroke::new(1.0f32, Color32::from_rgba_unmultiplied(110, 205, 255, 70));
    let mut prev_end = f32::NEG_INFINITY;
    let mut prev_top = f32::NAN;
    for (i, ((x0, x1, top), (_, m))) in boxes.iter().zip(&spans).enumerate() {
        let y = top + text_h - 1.0;
        if !gloss {
            painter.line_segment([Pos2::new(*x0, y) + origin, Pos2::new(*x1, y) + origin], underline);
            continue;
        }
        if *top != prev_top {
            prev_end = f32::NEG_INFINITY;
            prev_top = *top;
        }
        // Left-aligned under the word, shortened (…) to stop before the next one.
        let x = x0.max(prev_end + 6.0);
        let limit = match boxes.get(i + 1) {
            Some((nx, _, nt)) if nt == top => nx - 6.0,
            _ => galley.size().x.max(r.rect.width()),
        };
        let w = (limit - x).max(x1 - x0).max(8.0);
        let mut gj = LayoutJob::simple_singleline(m.clone(), small_font.clone(), ui.visuals().weak_text_color());
        gj.wrap = TextWrapping::truncate_at_width(w);
        let g = ui.fonts(|f| f.layout_job(gj));
        prev_end = x + g.size().x;
        painter.galley(Pos2::new(x, top + text_h) + origin, g, Color32::GRAY);
    }
    let hit = r.hover_pos().and_then(|p| {
        let c = char_at(p);
        spans.iter().find(|(s, _)| s.contains(&c) || s.end == c)
    });
    let hover_call = r.hover_pos().and_then(on_call);
    let r = match (hit, hover_call) {
        (_, Some(call)) => r.clone().on_hover_text(format!("{call} — click for station info")),
        (Some((s, m)), _) => {
            let word: String = galley.job.text.chars().skip(s.start).take(s.len()).collect();
            r.clone().on_hover_text(RichText::new(format!("{word} — {m}")).strong())
        }
        _ if !hint.is_empty() => r.clone().on_hover_text(hint),
        _ => r.clone(),
    };
    (r, clicked)
}

const CALL_COLOR: Color32 = Color32::from_rgb(255, 210, 90);

/// A callsign drawn as a clickable link.
/// A small coloured tag such as "POTA"; hovering shows what was heard.
fn flag_badge(ui: &mut egui::Ui, a: &Activity, heard: &str) -> egui::Response {
    let [r, g, b] = a.color;
    let text = RichText::new(format!(" {} ", a.name)).small().strong().color(Color32::BLACK).background_color(Color32::from_rgb(r, g, b));
    ui.add(egui::Label::new(text).selectable(false)).on_hover_text(format!("{}: heard \"{heard}\"", a.name))
}

fn call_link(ui: &mut egui::Ui, call: &str, size: f32) -> egui::Response {
    let r = ui.add(egui::Label::new(RichText::new(call).monospace().size(size).strong().color(CALL_COLOR)).sense(Sense::click()));
    if r.hovered() {
        ui.painter().line_segment([r.rect.left_bottom(), r.rect.right_bottom()], Stroke::new(1.0f32, CALL_COLOR));
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    r
}

fn open_in_browser(url: &str) {
    let r = if cfg!(windows) {
        std::process::Command::new("cmd").args(["/C", "start", "", url]).spawn()
    } else {
        std::process::Command::new(if cfg!(target_os = "macos") { "open" } else { "xdg-open" }).arg(url).spawn()
    };
    if let Err(e) = r {
        eprintln!("could not open {url}: {e}");
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll();
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.add_space(2.0);
            self.ui_top(ui);
            ui.add_space(2.0);
        });
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if let Some(e) = &self.engine_err {
                    ui.label(RichText::new(format!("Audio error: {e}")).color(Color32::LIGHT_RED));
                } else if let Some(e) = &self.snap.error {
                    ui.label(RichText::new(e).color(Color32::LIGHT_RED));
                } else {
                    ui.label(RichText::new(&self.snap.source).small());
                }
                ui.separator();
                ui.label(RichText::new(format!("{} decoders  {} spots", self.snap.channels.len(), self.spots.len())).small());
            });
        });
        if self.show_settings {
            egui::SidePanel::left("settings").resizable(false).show(ctx, |ui| {
                ui.heading("Settings");
                egui::ScrollArea::vertical().show(ui, |ui| self.ui_settings(ui));
            });
        }
        egui::SidePanel::right("spots").default_width(370.0).resizable(true).show(ctx, |ui| self.ui_spots(ui));
        egui::TopBottomPanel::bottom("selected").resizable(true).default_height(150.0).show(ctx, |ui| self.ui_bottom(ui));
        egui::CentralPanel::default().show(ctx, |ui| {
            let h = (ui.available_height() * 0.42).clamp(120.0, 420.0);
            self.ui_waterfall(ui, h);
            ui.add_space(4.0);
            self.ui_channels(ui);
        });
        self.ui_call_info(ctx);
        if let (Some(rd), Some((_, started))) = (&mut self.rec_display, &self.recording) {
            let _ = &rd.log;
            let t = started.elapsed().as_secs_f32();
            if rd.last_shot.elapsed() >= Duration::from_secs(10) {
                rd.last_shot = Instant::now();
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::new(format!("rec:{t:06.1}"))));
            }
            let shots: Vec<(Arc<ColorImage>, String)> = ctx.input(|i| {
                i.raw
                    .events
                    .iter()
                    .filter_map(|e| match e {
                        egui::Event::Screenshot { image, user_data, .. } => {
                            let tag = user_data.data.as_ref().and_then(|d| d.downcast_ref::<String>()).cloned()?;
                            tag.starts_with("rec:").then(|| (image.clone(), tag))
                        }
                        _ => None,
                    })
                    .collect()
            });
            for (img, tag) in shots {
                rd.shot_n += 1;
                let _ = save_png(&rd.shots.join(format!("{}s.png", &tag[4..])), &img);
            }
        }
        if let Some((path, at, sent)) = &mut self.screenshot {
            if !*sent && Instant::now() >= *at {
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
                *sent = true;
            }
            let path = path.clone();
            let shots: Vec<Arc<ColorImage>> = ctx.input(|i| {
                i.raw
                    .events
                    .iter()
                    .filter_map(|e| match e {
                        // Ignore the periodic screenshots taken while recording.
                        egui::Event::Screenshot { image, user_data, .. } if user_data.data.is_none() => Some(image.clone()),
                        _ => None,
                    })
                    .collect()
            });
            if let (true, Some(img)) = (*sent, shots.first()) {
                if let Err(e) = save_png(&path, img) {
                    eprintln!("screenshot failed: {e}");
                }
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
        ctx.request_repaint_after(Duration::from_millis(50));
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.flush_log_lines(true);
        self.qso_finish();
        if let Some(e) = &self.engine {
            e.send(EngineCmd::Record(None));
        }
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.s);
    }
}

fn save_png(path: &std::path::Path, img: &ColorImage) -> anyhow::Result<()> {
    let f = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut enc = png::Encoder::new(f, img.size[0] as u32, img.size[1] as u32);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut w = enc.write_header()?;
    let data: Vec<u8> = img.pixels.iter().flat_map(|p| p.to_array()).collect();
    w.write_image_data(&data)?;
    Ok(())
}

/// Where an updated country file is kept (next to the settings).
fn cty_path() -> Option<std::path::PathBuf> {
    eframe::storage_dir("cwskimmer").map(|d| d.join("cty.dat"))
}

/// "YYYY-MM-DD" of a file time.
fn date_of(t: std::time::SystemTime) -> String {
    let days = t.duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64 / 86400;
    let (y, m, d) = civil(days);
    format!("{y:04}-{m:02}-{d:02}")
}

/// Local time "HH:MM" at a UTC offset in hours.
fn time_at(offset_h: f32) -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64;
    let t = (secs + (offset_h * 3600.0) as i64).rem_euclid(86400);
    format!("{:02}:{:02}", t / 3600, t % 3600 / 60)
}

fn log_dir() -> std::path::PathBuf {
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).unwrap_or_else(|| ".".into());
    std::path::PathBuf::from(home).join("CwSkimmer-logs")
}

/// Current UTC date ("YYYY-MM-DD") and time ("HH:MM:SS").
fn utc_now() -> (String, String) {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64;
    let (days, rem) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    let (y, m, d) = civil(days);
    (format!("{y:04}-{m:02}-{d:02}"), format!("{:02}:{:02}:{:02}", rem / 3600, rem % 3600 / 60, rem % 60))
}

/// (year, month, day) of a day count since 1970-01-01 (civil-from-days, H. Hinnant).
fn civil(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

fn fmt_vfo(hz: u64) -> String {
    let mhz = hz / 1_000_000;
    let khz = (hz / 1000) % 1000;
    let h = hz % 1000;
    format!("{mhz:>2}.{khz:03}.{h:03}")
}

fn short(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n).collect::<String>())
    }
}

/// Black → blue → cyan → yellow → red → white.
fn colormap(v: f32) -> Color32 {
    const STOPS: [(f32, [f32; 3]); 6] = [
        (0.0, [0.0, 0.0, 0.0]),
        (0.25, [0.0, 0.1, 0.6]),
        (0.45, [0.0, 0.7, 0.9]),
        (0.65, [1.0, 0.95, 0.2]),
        (0.85, [1.0, 0.2, 0.1]),
        (1.0, [1.0, 1.0, 1.0]),
    ];
    for w in STOPS.windows(2) {
        let (a, ca) = w[0];
        let (b, cb) = w[1];
        if v <= b {
            let t = (v - a) / (b - a);
            let c = |i: usize| ((ca[i] + (cb[i] - ca[i]) * t) * 255.0) as u8;
            return Color32::from_rgb(c(0), c(1), c(2));
        }
    }
    Color32::WHITE
}

/// The Sherlock Morse window icon (rendered from docs/logo/sherlock-morse-icon-strong.svg).
fn app_icon() -> Option<egui::IconData> {
    let mut reader = png::Decoder::new(&include_bytes!("../assets/icon-256.png")[..]).read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    if info.color_type != png::ColorType::Rgba || info.bit_depth != png::BitDepth::Eight {
        return None;
    }
    buf.truncate(info.buffer_size());
    Some(egui::IconData { rgba: buf, width: info.width, height: info.height })
}

fn main() -> eframe::Result<()> {
    let mut viewport = egui::ViewportBuilder::default().with_inner_size([1280.0, 820.0]).with_min_inner_size([800.0, 500.0]).with_title("Sherlock Morse")
        // Wayland ignores window icons and takes the one from cwskimmer.desktop, matched by
        // this id. It is also eframe's settings key; "cwskimmer" keeps the existing settings.
        .with_app_id("cwskimmer");
    if let Some(icon) = app_icon() {
        viewport = viewport.with_icon(icon);
    }
    let options = eframe::NativeOptions { viewport, ..Default::default() };
    eframe::run_native("CW Skimmer", options, Box::new(|cc| Ok(Box::new(App::new(cc)))))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Renders `cw_text` offscreen and returns all text that was painted.
    fn painted(gloss: bool) -> String {
        let ctx = egui::Context::default();
        let mut out = String::new();
        for _ in 0..2 {
            let o = ctx.run(egui::RawInput::default(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    cw_text(ui, "[12:00Z] W1AW DE K1ABC TNX FER CALL UR 5NN QTH? 73", 15.0, gloss, Sense::click(), "");
                });
            });
            out.clear();
            for s in o.shapes {
                if let egui::epaint::Shape::Text(t) = s.shape {
                    out.push_str(&t.galley.job.text);
                    out.push('|');
                }
            }
        }
        out
    }

    #[test]
    fn glosses_are_painted() {
        let t = painted(true);
        assert!(t.contains("thanks") && t.contains("best regards") && t.contains("location?"), "{t}");
        assert!(!painted(false).contains("thanks"));
    }
}
