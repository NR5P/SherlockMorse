//! CW Skimmer – multi-channel Morse decoder with waterfall and CAT control.

use cwskimmer::audio::{self, ChannelSel, DeviceInfo};
use cwskimmer::cat::controller::{RigController, RigState};
use cwskimmer::cat::{self, Mode, RigConfig, RigKind};
use cwskimmer::engine::{Engine, EngineCmd, Snapshot, Source};
use cwskimmer::lm::LanguageModel;
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
    wf_contrast: f32,
    wf_brightness: f32,
    file_path: String,
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
            wf_contrast: 1.0,
            wf_brightness: 0.0,
            file_path: String::new(),
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
    selected: Option<u64>,
    spots: Vec<SpotRow>,
    log: Vec<(f32, String)>,
    freq_entry: String,
    probe_rx: Option<crossbeam_channel::Receiver<ProbeMsg>>,
    probe_status: String,
    show_settings: bool,
    /// `--screenshot PATH`: save a PNG of the window after a delay and exit.
    screenshot: Option<(std::path::PathBuf, Instant, bool)>,
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
            selected: None,
            spots: Vec::new(),
            log: Vec::new(),
            freq_entry: String::new(),
            probe_rx: None,
            probe_status: String::new(),
            show_settings: false,
            screenshot: None,
            s,
        };
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
        if app.s.rig_enabled {
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
        self.engine = None;
        if self.s.audio_auto || self.s.audio_device.is_none() {
            self.s.audio_device = audio::auto_select(&self.devices);
        }
        let src = Source::Device { name: self.s.audio_device.clone(), sel: self.s.channel_sel };
        self.start_source(src);
    }

    fn start_file(&mut self) {
        self.engine = None;
        let path = std::path::PathBuf::from(self.s.file_path.trim());
        self.start_source(Source::File { path, realtime: true, looped: false });
    }

    fn start_source(&mut self, src: Source) {
        match Engine::start(src, self.skimmer_config(), self.lm.clone()) {
            Ok(e) => {
                self.engine = Some(e);
                self.engine_err = None;
            }
            Err(e) => self.engine_err = Some(format!("{e:#}")),
        }
    }

    fn connect_rig(&mut self) {
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

    /// RF frequency of a signal heard at `audio` Hz.
    fn rf_of(&self, audio: f32) -> Option<u64> {
        let vfo = self.rig_state.freq? as f64;
        let d = (audio - self.s.pitch) as f64;
        Some((if self.upper() { vfo + d } else { vfo - d }).round() as u64)
    }

    /// Tunes the radio so the signal at `audio` Hz lands on the CW pitch.
    fn tune_to_audio(&mut self, audio: f32) {
        if let Some(rf) = self.rf_of(audio) {
            self.set_vfo(rf);
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
        if let Some(e) = &self.engine {
            let mut s = e.snap.lock();
            let rows = std::mem::take(&mut s.new_rows);
            let events = std::mem::take(&mut s.events);
            self.snap.channels = s.channels.clone();
            self.snap.spots = s.spots.clone();
            self.snap.noise_floor_db = s.noise_floor_db;
            self.snap.input_db = s.input_db;
            self.snap.time = s.time;
            self.snap.source = s.source.clone();
            self.snap.error = s.error.clone();
            self.snap.running = s.running;
            self.snap.input_rate = s.input_rate;
            drop(s);
            for r in rows {
                self.push_wf_row(&r);
            }
            for ev in events {
                self.log.push((ev.freq, ev.text));
            }
            if self.log.len() > 500 {
                let n = self.log.len() - 500;
                self.log.drain(..n);
            }
        }
        if let Some(r) = &self.rig {
            self.rig_state = r.snapshot();
        }
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
                    if sp.count != r.count {
                        r.count = sp.count;
                        r.last = Instant::now();
                        r.audio = sp.freq;
                        r.wpm = sp.wpm;
                        r.snr = sp.snr_db;
                        if rf.is_some() {
                            r.rf = rf;
                        }
                    }
                }
                None => self.spots.push(SpotRow { call: sp.call, rf, audio: sp.freq, wpm: sp.wpm, snr: sp.snr_db, last: Instant::now(), count: sp.count }),
            }
        }
        self.spots.retain(|r| r.last.elapsed() < Duration::from_secs(1800));
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
            // Input level meter.
            let lvl = ((self.snap.input_db + 60.0) / 60.0).clamp(0.0, 1.0);
            let (r, _) = ui.allocate_exact_size(Vec2::new(70.0, 12.0), Sense::hover());
            ui.painter().rect_filled(r, 2.0, Color32::from_gray(40));
            let col = if self.snap.input_db > -3.0 { Color32::RED } else if self.snap.input_db > -12.0 { Color32::YELLOW } else { Color32::GREEN };
            ui.painter().rect_filled(Rect::from_min_size(r.min, Vec2::new(r.width() * lvl, r.height())), 2.0, col);
            ui.label(RichText::new(format!("{:.0} dBFS", self.snap.input_db)).small());
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
                    RigKind::Yaesu => {
                        self.s.rig.baud = 38400;
                        self.s.rig.rts = true;
                    }
                    RigKind::Kenwood => self.s.rig.baud = 38400,
                    RigKind::Icom => {
                        self.s.rig.baud = 19200;
                        self.s.rig.rts = false;
                    }
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
            ui.add(egui::DragValue::new(&mut self.s.pitch).range(300.0..=1200.0).speed(10.0))
                .on_hover_text("Must match the radio's CW pitch setting (FTDX3000: menu CW PITCH, default 700)");
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
            ui.label("Assert RTS / DTR");
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.s.rig.rts, "RTS").on_hover_text("Yaesu radios with CAT RTS = ON need this");
                ui.checkbox(&mut self.s.rig.dtr, "DTR");
            });
            ui.end_row();
        });
        if changed {
            if let Some(e) = &self.engine {
                e.send(EngineCmd::Config(self.skimmer_config()));
            }
        }
        ui.label(RichText::new("Decoder parameters apply to newly detected signals.").small().weak());
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
        let px = x_of(self.s.pitch);
        painter.line_segment([Pos2::new(px, img_rect.top()), Pos2::new(px, img_rect.bottom())], Stroke::new(1.0f32, Color32::from_rgba_unmultiplied(255, 80, 80, 120)));
        // Channel markers with callsigns
        for ch in &self.snap.channels {
            let x = x_of(ch.freq);
            let sel = Some(ch.id) == self.selected;
            let col = if sel { Color32::YELLOW } else if ch.is_mark { Color32::from_rgb(120, 255, 120) } else { Color32::from_rgb(80, 160, 255) };
            painter.line_segment([Pos2::new(x, img_rect.top()), Pos2::new(x, img_rect.top() + 10.0)], Stroke::new(2.0f32, col));
            let label = ch.callsign.clone().unwrap_or_else(|| ch.wpm.map(|w| format!("{w:.0}")).unwrap_or_default());
            if !label.is_empty() {
                painter.text(Pos2::new(x + 2.0, img_rect.top() + 10.0), egui::Align2::LEFT_TOP, label, egui::FontId::monospace(11.0), col);
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
                    Some(id) => self.selected = Some(id),
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
                self.tune_to_audio(snapped);
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
        TableBuilder::new(ui)
            .striped(true)
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
                            let r = ui.add(egui::Label::new(RichText::new(format!("{dot} {}", self.fmt_freq(ch.freq))).monospace()).sense(Sense::click()));
                            if r.clicked() {
                                self.selected = Some(ch.id);
                            }
                            if r.double_clicked() {
                                tune = Some(ch.freq);
                            }
                            r.on_hover_text("Double-click to tune the radio here");
                        });
                        row.col(|ui| {
                            if let Some(c) = &ch.callsign {
                                let r = ui.add(egui::Label::new(RichText::new(c).monospace().strong().color(Color32::from_rgb(255, 210, 90))).sense(Sense::click()));
                                if r.double_clicked() {
                                    tune = Some(ch.freq);
                                }
                            }
                        });
                        row.col(|ui| {
                            ui.label(ch.wpm.map(|w| format!("{w:.0}")).unwrap_or_else(|| "–".into()));
                        });
                        row.col(|ui| {
                            ui.label(format!("{:.0}", ch.snr_db));
                        });
                        row.col(|ui| {
                            let full = to_display(&ch.text);
                            let tent = to_display(&ch.tentative);
                            let total = full.len() + tent.len();
                            let skip = total.saturating_sub(max_chars);
                            let (a, b) = if skip >= full.len() {
                                (String::new(), tent[skip - full.len()..].to_string())
                            } else {
                                (full[skip..].to_string(), tent)
                            };
                            let age = ch.last_mark_age;
                            let col = if age < 5.0 { Color32::WHITE } else { Color32::GRAY };
                            ui.horizontal(|ui| {
                                ui.spacing_mut().item_spacing.x = 0.0;
                                ui.label(RichText::new(a).monospace().color(col));
                                ui.label(RichText::new(b).monospace().color(Color32::from_gray(130)));
                            });
                        });
                        row.col(|ui| {
                            if ui.small_button("✖").on_hover_text("Remove decoder").clicked() {
                                remove = Some(ch.id);
                            }
                        });
                    });
                }
            });
        if let Some(f) = tune {
            self.tune_to_audio(f);
        }
        if let (Some(id), Some(e)) = (remove, &self.engine) {
            e.send(EngineCmd::RemoveChannel(id));
        }
    }

    fn ui_spots(&mut self, ui: &mut egui::Ui) {
        ui.heading("Spots");
        let mut spots = self.spots.clone();
        spots.sort_by(|a, b| a.last.cmp(&b.last).reverse());
        let mut tune_rf: Option<u64> = None;
        let mut tune_audio: Option<f32> = None;
        TableBuilder::new(ui)
            .striped(true)
            .column(Column::exact(82.0))
            .column(Column::exact(72.0))
            .column(Column::exact(30.0))
            .column(Column::exact(30.0))
            .column(Column::remainder())
            .header(18.0, |mut h| {
                for t in ["Call", "Freq", "WPM", "SNR", "Age"] {
                    h.col(|ui| {
                        ui.strong(t);
                    });
                }
            })
            .body(|mut body| {
                for s in &spots {
                    body.row(18.0, |mut row| {
                        row.col(|ui| {
                            let r = ui.add(egui::Label::new(RichText::new(&s.call).monospace().strong().color(Color32::from_rgb(255, 210, 90))).sense(Sense::click()));
                            if r.clicked() {
                                match s.rf {
                                    Some(rf) => tune_rf = Some(rf),
                                    None => tune_audio = Some(s.audio),
                                }
                            }
                            r.on_hover_text("Click to tune");
                        });
                        row.col(|ui| {
                            ui.label(match s.rf {
                                Some(rf) => format!("{:.2}", rf as f64 / 1000.0),
                                None => format!("{:.0} Hz", s.audio),
                            });
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
                    });
                }
            });
        if let Some(rf) = tune_rf {
            self.set_vfo(rf);
        }
        if let Some(a) = tune_audio {
            self.tune_to_audio(a);
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
                ui.label(RichText::new(c).strong().color(Color32::from_rgb(255, 210, 90)));
            }
            ui.label(format!("{} WPM  SNR {:.0} dB", ch.wpm.map(|w| format!("{w:.0}")).unwrap_or("–".into()), ch.snr_db));
            if self.rig.is_some() && ui.button("Tune radio here").clicked() {
                self.tune_to_audio(ch.freq);
            }
        });
        egui::ScrollArea::vertical().stick_to_bottom(true).auto_shrink([false, false]).show(ui, |ui| {
            ui.add(
                egui::Label::new(RichText::new(format!("{}{}", to_display(&ch.text), to_display(&ch.tentative))).monospace().size(15.0)).wrap(),
            );
        });
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
                self.ui_settings(ui);
            });
        }
        egui::SidePanel::right("spots").default_width(280.0).resizable(true).show(ctx, |ui| self.ui_spots(ui));
        egui::TopBottomPanel::bottom("selected").resizable(true).default_height(110.0).show(ctx, |ui| self.ui_selected(ui));
        egui::CentralPanel::default().show(ctx, |ui| {
            let h = (ui.available_height() * 0.42).clamp(120.0, 420.0);
            self.ui_waterfall(ui, h);
            ui.add_space(4.0);
            self.ui_channels(ui);
        });
        if let Some((path, at, sent)) = &mut self.screenshot {
            if !*sent && Instant::now() >= *at {
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
                *sent = true;
            }
            let path = path.clone();
            let shots: Vec<Arc<ColorImage>> = ctx.input(|i| {
                i.raw.events.iter().filter_map(|e| if let egui::Event::Screenshot { image, .. } = e { Some(image.clone()) } else { None }).collect()
            });
            if let Some(img) = shots.first() {
                if let Err(e) = save_png(&path, img) {
                    eprintln!("screenshot failed: {e}");
                }
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
        ctx.request_repaint_after(Duration::from_millis(50));
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

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 820.0]).with_min_inner_size([800.0, 500.0]).with_title("CW Skimmer"),
        ..Default::default()
    };
    eframe::run_native("CW Skimmer", options, Box::new(|cc| Ok(Box::new(App::new(cc)))))
}
