//! Command-line companion: decode audio files, synthesize test signals,
//! benchmark the decoder, list audio devices and talk to radios.

use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use cwskimmer::cat::{self, RigConfig, RigKind};
use cwskimmer::cty::Cty;
use cwskimmer::decoder::DecoderParams;
use cwskimmer::dsp::resample::Resampler;
use cwskimmer::lm::LanguageModel;
use cwskimmer::skimmer::{Skimmer, SkimmerConfig, RATE};
use cwskimmer::synth::{self, Scene, SignalSpec};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "cwtool", about = "CW skimmer command-line tools")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Decode every CW signal in an audio file (wav/mp3/flac/ogg/m4a).
    Decode {
        file: PathBuf,
        /// Only decode a single signal at this audio frequency (Hz).
        #[arg(long)]
        freq: Option<f32>,
        /// Disable the language model.
        #[arg(long)]
        no_lm: bool,
        #[arg(long, default_value_t = 200.0)]
        min_freq: f32,
        #[arg(long, default_value_t = 3000.0)]
        max_freq: f32,
        /// Compare against a reference transcript and print the CER.
        #[arg(long)]
        reference: Option<PathBuf>,
        /// Seconds to skip from the start of the file.
        #[arg(long, default_value_t = 0.0)]
        skip: f32,
        /// Max seconds to process.
        #[arg(long)]
        duration: Option<f32>,
    },
    /// Generate a test signal WAV.
    Synth {
        #[arg(long, default_value = "CQ CQ DE W1AW W1AW K")]
        text: String,
        #[arg(long, default_value_t = 20.0)]
        wpm: f32,
        #[arg(long, default_value_t = 700.0)]
        freq: f32,
        #[arg(long, default_value_t = 10.0)]
        snr: f32,
        #[arg(long, default_value_t = 0.0)]
        jitter: f32,
        #[arg(long, default_value_t = 0.0)]
        qsb: f32,
        #[arg(long, default_value_t = 0.0)]
        qrn: f32,
        #[arg(short, long, default_value = "test.wav")]
        out: PathBuf,
        /// Generate a multi-station band scene instead of a single signal.
        #[arg(long)]
        pileup: bool,
    },
    /// Run the synthetic benchmark suite.
    Bench {
        /// Trials per scenario.
        #[arg(long, default_value_t = 3)]
        trials: u64,
        /// Also run every scenario with the language model disabled.
        #[arg(long)]
        compare_lm: bool,
        /// Only run scenarios whose name contains this.
        #[arg(long)]
        only: Option<String>,
        #[arg(long)]
        verbose: bool,
    },
    /// Render a waterfall PNG of an audio file (for inspecting recordings).
    Waterfall {
        file: PathBuf,
        #[arg(short, long, default_value = "waterfall.png")]
        out: PathBuf,
        #[arg(long, default_value_t = 0.0)]
        skip: f32,
        #[arg(long, default_value_t = 60.0)]
        duration: f32,
    },
    /// Weak-signal test from a real recording: fade a strong, cleanly copied
    /// signal into real band noise at several SNRs and score the copy against
    /// the clean decode (or a reference transcript).
    Weak {
        /// Recording with one strong signal.
        signal: PathBuf,
        /// Audio frequency of that signal (Hz).
        #[arg(long)]
        freq: f32,
        /// Band-noise recordings (no strong signals), looped as needed.
        #[arg(long, num_args = 1.., required = true)]
        noise: Vec<PathBuf>,
        /// SNRs to test, dB in 2500 Hz.
        #[arg(long, value_delimiter = ',', default_value = "6,0,-4,-8,-11")]
        snr: Vec<f32>,
        #[arg(long)]
        reference: Option<PathBuf>,
        #[arg(long, default_value_t = 0.0)]
        skip: f32,
        #[arg(long)]
        duration: Option<f32>,
        /// Print the decoded text at each SNR.
        #[arg(long)]
        verbose: bool,
        /// Runs per SNR, each with a different stretch of the noise.
        #[arg(long, default_value_t = 1)]
        trials: usize,
    },
    /// Write the benchmark scenes as WAV files with their transcripts, for
    /// decoding by another program. Callsigns are replaced by random ones,
    /// unique per scene, so a skimmer that suppresses repeat spots is not
    /// penalised for calls the benchmark reuses.
    ExportBench {
        #[arg(long, default_value_t = 2)]
        trials: u64,
        #[arg(long)]
        only: Option<String>,
        #[arg(long, default_value = "bench-export")]
        out: PathBuf,
    },
    /// Quick check for on/off keyed tones (CW) in a short clip, for band scanning.
    /// Prints one line per keyed tone, strongest first.
    Keying {
        file: PathBuf,
        /// Seconds to check from the start of the file.
        #[arg(long, default_value_t = 8.0)]
        duration: f32,
    },
    /// List audio input devices (best radio match marked).
    Devices,
    /// List serial ports.
    Ports,
    /// Auto-detect radios on USB serial ports.
    Probe,
    /// Read or set a radio's frequency.
    Rig {
        #[arg(long, value_enum, default_value = "yaesu")]
        kind: KindArg,
        #[arg(long, default_value = "/dev/ttyUSB0")]
        port: String,
        #[arg(long, default_value_t = 38400)]
        baud: u32,
        #[arg(long, default_value = "127.0.0.1:4532")]
        host: String,
        #[arg(long, default_value = "94")]
        civ: String,
        /// Frequency to set, Hz.
        #[arg(long)]
        set: Option<u64>,
    },
    /// Look up the DXCC country of callsigns (built-in cty.dat, or --file).
    Cty {
        calls: Vec<String>,
        #[arg(long)]
        file: Option<PathBuf>,
    },
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum KindArg {
    Yaesu,
    Kenwood,
    Icom,
    YaesuLegacy,
    Hamlib,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Decode { file, freq, no_lm, min_freq, max_freq, reference, skip, duration } => {
            let (mut samples, rate) = cwskimmer::audiofile::read_audio(&file)?;
            let s0 = ((skip * rate as f32) as usize).min(samples.len());
            samples.drain(..s0);
            if let Some(d) = duration {
                samples.truncate((d * rate as f32) as usize);
            }
            eprintln!("{}: {:.1} s at {} Hz", file.display(), samples.len() as f32 / rate as f32, rate);
            let lm = Arc::new(if no_lm { LanguageModel::uniform() } else { LanguageModel::new() });
            let detect_db = std::env::var("CWSK_DETECT_DB").ok().and_then(|v| v.parse().ok()).unwrap_or(SkimmerConfig::default().detect_db);
            let cfg = SkimmerConfig { min_freq, max_freq, auto_detect: freq.is_none(), params: DecoderParams::from_env(), detect_db, ..Default::default() };
            let mut sk = Skimmer::new(cfg, lm);
            if let Some(f) = freq {
                sk.add_channel(f, true);
            }
            let mut rs = Resampler::new(rate, RATE);
            let mut buf = Vec::new();
            let mut log: std::collections::BTreeMap<u64, (f32, String)> = Default::default();
            let mut feed = |sk: &mut Skimmer, chunk: &[f32]| {
                buf.clear();
                rs.process(chunk, &mut buf);
                sk.process(&buf);
            };
            let chunk_div: usize = std::env::var("CWSK_CHUNKDIV").ok().and_then(|v| v.parse().ok()).unwrap_or(10);
            // Every spot ever listed (the live list forgets spots after
            // 15 minutes): call -> (count, confirmed).
            let mut all_spots: std::collections::BTreeMap<String, (u32, bool)> = Default::default();
            // (A spot the skimmer withdraws as a misread is forgotten; one
            // that merely ages out of the live list is kept.)
            let mut last_seen: std::collections::HashMap<String, f64> = Default::default();
            for chunk in samples.chunks(rate as usize / chunk_div) {
                feed(&mut sk, chunk);
                for sp in &sk.spots {
                    let e = all_spots.entry(sp.call.clone()).or_insert((0, false));
                    e.0 = e.0.max(sp.count);
                    e.1 |= sp.confirmed;
                    last_seen.insert(sp.call.clone(), sp.time);
                }
                let now = sk.time;
                let live: std::collections::HashSet<&str> = sk.spots.iter().map(|s| s.call.as_str()).collect();
                all_spots.retain(|c, _| live.contains(c.as_str()) || last_seen.get(c).is_some_and(|&t| now - t >= 899.0));
            }
            if std::env::var_os("CWSK_CHANNELS").is_some() {
                for (f, fit, n, q, d, ok, t) in sk.debug_channels() {
                    eprintln!("CH {f:7.1} fit={fit:.2} n={n:3} el/ch={q:.2} dash={d:.2} shown={ok} | {}", t.trim());
                }
            }
            feed(&mut sk, &vec![0.0; rate as usize * 3]);
            sk.finish();
            if std::env::var_os("CWSK_WPM").is_some() {
                for c in sk.channel_info() {
                    let t = sk.channels.iter().find(|ch| ch.id == c.id).and_then(|ch| ch.decoder.timing().cloned());
                    if let Some(t) = t {
                        eprintln!("WPM {:7.1} Hz: {:.1} wpm (dot {:.0} ms, egap {:.0} ms, dash/dot {:.2}, char gap {:.1} dots, word gap {:.1} dots)", c.freq, t.wpm(), t.dot, t.egap, t.dash / t.dot, t.cgap / t.dot, t.wgap() / t.dot);
                    }
                }
            }
            for e in sk.events.drain(..) {
                if let Some(n) = &e.note {
                    if std::env::var_os("CWSK_NOTES").is_some() {
                        eprintln!("NOTE {:7.1} Hz #{}: {n}", e.freq, e.channel);
                    }
                    continue;
                }
                let ent = log.entry(e.channel).or_insert((e.freq, String::new()));
                ent.0 = e.freq;
                ent.1.push_str(&e.text);
            }
            let mut best = String::new();
            for (_, (f, t)) in &log {
                let t = cwskimmer::morse::to_display(t.trim());
                if t.len() > best.len() {
                    best = t.clone();
                }
                println!("{f:7.1} Hz: {t}");
            }
            for sp in &sk.spots {
                let e = all_spots.entry(sp.call.clone()).or_insert((0, false));
                e.0 = e.0.max(sp.count);
                e.1 |= sp.confirmed;
            }
            if std::env::var_os("CWSK_ALLSPOTS").is_some() {
                println!("\nAll spots:");
                for (c, (n, conf)) in &all_spots {
                    println!("  {c:<10} x{n}{}", if *conf { "" } else { " (unconfirmed)" });
                }
            }
            if !sk.spots.is_empty() {
                println!("\nSpots:");
                for s in &sk.spots {
                    println!("  {:<10} {:7.1} Hz {:4.0} WPM {:5.1} dB x{}", s.call, s.freq, s.wpm, s.snr_db, s.count);
                }
            }
            if let Some(r) = reference {
                // ARRL transcripts are Windows-1252 with 0x89 for the "=" (BT).
                let r: String = std::fs::read(r)?.iter().map(|&b| if b == 0x89 { '=' } else { b as char }).collect();
                let r = normalize_ref(&r);
                let c = synth::cer(&r, &best);
                println!("\nCER vs reference (longest channel): {:.1}%", c * 100.0);
            }
        }
        Cmd::Synth { text, wpm, freq, snr, jitter, qsb, qrn, out, pileup } => {
            let signals = if pileup {
                let st = [
                    ("CQ CQ CQ DE K1ABC K1ABC K1ABC K CQ CQ DE K1ABC K", 420.0, 22.0, 15.0, 0.0),
                    ("CQ TEST DL5XYZ DL5XYZ TEST", 540.0, 30.0, 8.0, 0.0),
                    ("W3LPL DE G4ABC TNX FER CALL UR 599 599 NAME JOHN QTH LONDON HW? W3LPL DE G4ABC KN", 700.0, 20.0, 20.0, 0.08),
                    ("CQ CQ DE JA1XYZ JA1XYZ K", 880.0, 18.0, 3.0, 0.0),
                    ("R R TU 5NN MA TU DE N2IC", 1150.0, 35.0, 12.0, 0.0),
                    ("VE3NEA DE OH2BH GM OM UR RST 579 579 QSB HR BK", 1420.0, 16.0, 10.0, 0.15),
                ];
                st.iter()
                    .enumerate()
                    .map(|(i, &(t, f, w, s, j))| SignalSpec {
                        text: t.into(),
                        freq: f,
                        wpm: w,
                        snr_db: s,
                        jitter: j,
                        qsb_depth: if i == 5 { 0.6 } else { 0.0 },
                        start: 0.4 * i as f32,
                        ..Default::default()
                    })
                    .collect()
            } else {
                vec![SignalSpec { text, wpm, freq, snr_db: snr, jitter, qsb_depth: qsb, ..Default::default() }]
            };
            let scene = Scene { signals, qrn_rate: qrn, noise_sigma: pileup.then_some(0.05), ..Default::default() };
            let s = synth::render(&scene);
            cwskimmer::audiofile::write_wav(&out, &s, scene.rate)?;
            println!("wrote {} ({:.1} s)", out.display(), s.len() as f32 / scene.rate as f32);
        }
        Cmd::Bench { trials, compare_lm, only, verbose } => bench(trials, compare_lm, only, verbose),
        Cmd::Weak { signal, freq, noise, snr, reference, skip, duration, verbose, trials } => {
            let load = |p: &PathBuf, skip: f32, dur: Option<f32>| -> Result<Vec<f32>> {
                let (x, rate) = cwskimmer::audiofile::read_audio(p)?;
                let s0 = ((skip * rate as f32) as usize).min(x.len());
                let s1 = dur.map_or(x.len(), |d| (s0 + (d * rate as f32) as usize).min(x.len()));
                let mut rs = Resampler::new(rate, RATE);
                let mut out = Vec::new();
                rs.process(&x[s0..s1], &mut out);
                Ok(out)
            };
            let sig = load(&signal, skip, duration)?;
            let mut nz = Vec::new();
            for p in &noise {
                nz.extend(load(p, 0.0, None)?);
            }
            anyhow::ensure!(!nz.is_empty(), "no noise audio");
            let lm = Arc::new(LanguageModel::new());
            let r = match reference {
                Some(r) => {
                    let r: String = std::fs::read(r)?.iter().map(|&b| if b == 0x89 { '=' } else { b as char }).collect();
                    normalize_ref(&r)
                }
                None => normalize_ref(&decode_near(&sig, freq, lm.clone())),
            };
            // Power in a 20 ms box filter (about 50 Hz wide) at `freq`.
            let w = (RATE as f32 * 0.02) as usize;
            let envelope = |x: &[f32]| {
                let (mut acc, mut q) = (num_complex::Complex32::new(0.0, 0.0), std::collections::VecDeque::new());
                let mut env: Vec<f32> = Vec::with_capacity(x.len() / 8);
                for (i, &x) in x.iter().enumerate() {
                    let ph = 2.0 * std::f32::consts::PI * freq * ((i % RATE as usize) as f32 / RATE as f32);
                    let z = num_complex::Complex32::new(x * ph.cos(), -x * ph.sin());
                    acc += z;
                    q.push_back(z);
                    if q.len() > w {
                        acc -= q.pop_front().unwrap();
                    }
                    if i % 8 == 0 {
                        env.push((acc / w as f32).norm_sqr());
                    }
                }
                env
            };
            // Tone power while keyed: 95th percentile of the envelope power.
            let mut env = envelope(&sig);
            env.sort_by(|a, b| a.total_cmp(b));
            let ps = 2.0 * env[env.len() * 95 / 100];
            // Noise in 2500 Hz at the density found at `freq` (the radio's
            // filter shapes band noise; this keeps SNRs comparable with the
            // white-noise benchmark).
            let en = envelope(&nz);
            let pn = en.iter().sum::<f32>() / en.len() as f32 * w as f32 * 2500.0 / (RATE as f32 / 2.0);
            println!("{}: {:.0} s, {} ref chars: {}", signal.display(), sig.len() as f32 / RATE as f32, r.len(), r.chars().take(80).collect::<String>());
            let jobs: Vec<(f32, usize)> = snr.iter().flat_map(|&db| (0..trials).map(move |k| (db, k))).collect();
            let res: Vec<(f32, f32, String)> = std::thread::scope(|sc| {
                let hs: Vec<_> = jobs
                    .iter()
                    .map(|&(db, k)| {
                        let (sig, nz, lm, r) = (&sig, &nz, lm.clone(), &r);
                        sc.spawn(move || {
                            let g = (pn * 10f32.powf(db / 10.0) / ps).sqrt();
                            let off = k * nz.len() / trials;
                            let mix: Vec<f32> = sig.iter().enumerate().map(|(i, &x)| g * x + nz[(i + off) % nz.len()]).collect();
                            if let Ok(p) = std::env::var("CWSK_SAVEMIX") {
                                let spec = hound::WavSpec { channels: 1, sample_rate: RATE, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
                                if let Ok(mut w) = hound::WavWriter::create(format!("{p}-{db}-{k}.wav"), spec) {
                                    let peak = mix.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-9);
                                    for v in &mix {
                                        let _ = w.write_sample((v / peak * 30000.0) as i16);
                                    }
                                }
                            }
                            let hyp = normalize_ref(&decode_near(&mix, freq, lm));
                            (db, synth::cer(r, &hyp), hyp)
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            for &db in &snr {
                let v: Vec<&(f32, f32, String)> = res.iter().filter(|x| x.0 == db).collect();
                let mean = v.iter().map(|x| x.1).sum::<f32>() / v.len() as f32;
                let each: Vec<String> = v.iter().map(|x| format!("{:.1}", x.1 * 100.0)).collect();
                println!("  snr {db:5.1} dB: CER {:5.1}%   ({})", mean * 100.0, each.join(" "));
                if verbose {
                    for x in v {
                        println!("      {}", x.2);
                    }
                }
            }
        }
        Cmd::ExportBench { trials, only, out } => {
            use rand::{Rng, SeedableRng};
            std::fs::create_dir_all(&out)?;
            const PFX: &[&str] = &[
                "K", "W", "N", "AA", "AB", "KB", "KC", "KD", "WA", "WB", "VE", "VA", "DL", "DJ", "G", "M", "F", "I", "EA", "ON", "PA",
                "OK", "OM", "SP", "HA", "YU", "S5", "9A", "OH", "SM", "LA", "OZ", "UA", "UR", "JA", "JH", "VK", "ZL", "LU", "PY", "CE",
                "XE", "CT", "EI", "OE", "ES", "LY", "YL", "ZS",
            ];
            for (ci, case) in cases().iter().enumerate() {
                if only.as_ref().is_some_and(|o| !case.name.contains(o.as_str())) {
                    continue;
                }
                for t in 0..trials {
                    let mut scene = (case.make)(t + 1);
                    let mut rng = rand::rngs::StdRng::seed_from_u64(1000 * ci as u64 + t);
                    let mut map: std::collections::HashMap<String, String> = Default::default();
                    for sig in &mut scene.signals {
                        let words: Vec<String> = sig
                            .text
                            .split(' ')
                            .map(|w| {
                                if cwskimmer::callsign::parse_callsign(w).is_none() || w.contains('/') {
                                    return w.to_string();
                                }
                                map.entry(w.to_string())
                                    .or_insert_with(|| loop {
                                        let p = PFX[rng.gen_range(0..PFX.len())];
                                        let n = rng.gen_range(if p.len() == 1 { 2..=3 } else { 1..=3 });
                                        let suf: String = (0..n).map(|_| (b'A' + rng.gen_range(0..26u8)) as char).collect();
                                        let c = format!("{p}{}{suf}", rng.gen_range(0..10));
                                        if cwskimmer::callsign::parse_callsign(&c).is_some() && cwskimmer::callsign::valid_prefix(&c) {
                                            break c;
                                        }
                                    })
                                    .clone()
                            })
                            .collect();
                        sig.text = words.join(" ");
                    }
                    let samples = synth::render(&scene);
                    let peak = samples.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-9);
                    let name = format!("{}-{}", case.name.replace('/', "_"), t + 1);
                    let spec = hound::WavSpec { channels: 1, sample_rate: scene.rate, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
                    let mut w = hound::WavWriter::create(out.join(format!("{name}.wav")), spec)?;
                    for v in &samples {
                        w.write_sample((v / peak * 30000.0) as i16)?;
                    }
                    w.finalize()?;
                    let mut truth: Vec<String> = scene.signals.iter().map(|s| format!("{:.0}|{:.0}|{:.1}|{}", s.freq, s.wpm, s.snr_db, s.text)).collect();
                    let mut calls: Vec<&String> = map.values().collect();
                    calls.sort();
                    truth.push(format!("#CALLS {}", calls.iter().map(|c| c.as_str()).collect::<Vec<_>>().join(" ")));
                    std::fs::write(out.join(format!("{name}.txt")), truth.join("\n") + "\n")?;
                }
            }
        }
        Cmd::Keying { file, duration } => {
            let (mut samples, rate) = cwskimmer::audiofile::read_audio(&file)?;
            samples.truncate((duration * rate as f32) as usize);
            let mut rs = Resampler::new(rate, RATE);
            let mut buf = Vec::new();
            rs.process(&samples, &mut buf);
            for (f, ons, duty, snr) in keyed_tones(&buf) {
                println!("{f:7.1} Hz  {ons:3} marks  duty {duty:.2}  {snr:4.1} dB");
            }
        }
        Cmd::Waterfall { file, out, skip, duration } => {
            let (samples, rate) = cwskimmer::audiofile::read_audio(&file)?;
            let s0 = ((skip * rate as f32) as usize).min(samples.len());
            let s1 = (s0 + (duration * rate as f32) as usize).min(samples.len());
            let mut rs = Resampler::new(rate, RATE);
            let mut buf = Vec::new();
            rs.process(&samples[s0..s1], &mut buf);
            let mut sp = cwskimmer::dsp::spectrum::Spectrum::new(1024, 160);
            let mut rows: Vec<Vec<f32>> = Vec::new();
            sp.process(&buf, |p| rows.push(p.iter().map(|v| 10.0 * (v + 1e-14).log10()).collect()));
            let w = 1024 / 2 * 3000 / 4000; // up to 3 kHz
            let mut all: Vec<f32> = rows.iter().flat_map(|r| r[..w].iter().copied()).collect();
            all.sort_by(|a, b| a.total_cmp(b));
            let (lo, hi) = (all[all.len() / 2] - 3.0, all[all.len() * 999 / 1000]);
            let f = std::io::BufWriter::new(std::fs::File::create(&out)?);
            let mut enc = png::Encoder::new(f, w as u32, rows.len() as u32);
            enc.set_color(png::ColorType::Grayscale);
            let mut wr = enc.write_header()?;
            let data: Vec<u8> = rows.iter().flat_map(|r| r[..w].iter().map(|&v| (((v - lo) / (hi - lo)).clamp(0.0, 1.0) * 255.0) as u8).collect::<Vec<_>>()).collect();
            wr.write_image_data(&data)?;
            println!("wrote {} ({} x {}, 0-3000 Hz left to right, time downward, {:.0} ms/row)", out.display(), w, rows.len(), 160.0 / 8.0);
        }
        Cmd::Devices => {
            let devs = cwskimmer::audio::list_input_devices();
            let pick = cwskimmer::audio::auto_select(&devs);
            for d in &devs {
                let mark = if Some(&d.name) == pick.as_ref() { "*" } else { " " };
                println!("{mark} [{:4}] {}{}", d.score, d.name, if d.is_default { " (default)" } else { "" });
            }
        }
        Cmd::Cty { calls, file } => {
            let cty = match file {
                Some(f) => Cty::load(&f)?,
                None => Cty::built_in(),
            };
            for call in calls {
                match cty.lookup(&call) {
                    Some(c) => println!(
                        "{call:<12} {} ({})  {}  CQ {}  ITU {}  {:.2},{:.2}  UTC{:+}{}",
                        c.name, c.prefix, c.cont, c.cq, c.itu, c.lat, c.lon, c.utc_offset, if c.dxcc { "" } else { "  (WAE only)" }
                    ),
                    None => println!("{call:<12} unknown"),
                }
            }
        }
        Cmd::Ports => {
            for (p, d) in cat::list_ports() {
                println!("{p}  {d}");
            }
        }
        Cmd::Probe => {
            let found = cat::probe::probe(|s| eprintln!("  probing {s}"));
            if found.is_empty() {
                println!("no radios found");
            }
            for f in found {
                println!("{} on {} @ {} baud ({:?})", f.model, f.config.port, f.config.baud, f.config.kind);
            }
        }
        Cmd::Rig { kind, port, baud, host, civ, set } => {
            let kind = match kind {
                KindArg::Yaesu => RigKind::Yaesu,
                KindArg::Kenwood => RigKind::Kenwood,
                KindArg::Icom => RigKind::Icom,
                KindArg::YaesuLegacy => RigKind::YaesuLegacy,
                KindArg::Hamlib => RigKind::Hamlib,
            };
            let civ_addr = u8::from_str_radix(civ.trim_start_matches("0x"), 16).map_err(|_| anyhow!("bad CI-V address"))?;
            let cfg = RigConfig { kind, port, baud, host, civ_addr, ..Default::default() };
            let mut rig = cat::open(&cfg)?;
            println!("connected: {}", rig.name());
            if let Some(f) = set {
                rig.set_freq(f)?;
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
            println!("frequency: {} Hz", rig.get_freq()?);
            if let Ok(m) = rig.get_mode() {
                println!("mode: {}", m.label());
            }
            match rig.get_pitch() {
                Ok(Some(p)) => println!("CW pitch: {p} Hz"),
                Ok(None) => println!("CW pitch: not reported by this radio"),
                Err(e) => println!("CW pitch: error {e}"),
            }
        }
    }
    Ok(())
}

/// Runs the skimmer (auto-detect, as live) over 8 kHz audio and returns the
/// text of the channels within 40 Hz of `freq`, in channel order.
fn decode_near(audio: &[f32], freq: f32, lm: Arc<LanguageModel>) -> String {
    let mut sk = Skimmer::new(SkimmerConfig { params: DecoderParams::from_env(), ..Default::default() }, lm);
    for c in audio.chunks(RATE as usize / 10) {
        sk.process(c);
    }
    sk.process(&vec![0.0; RATE as usize * 3]);
    sk.finish();
    let mut by: std::collections::BTreeMap<u64, String> = Default::default();
    for e in sk.events.drain(..) {
        if e.note.is_none() && (e.freq - freq).abs() < 40.0 {
            by.entry(e.channel).or_default().push_str(&e.text);
        }
    }
    by.into_values().map(|t| cwskimmer::morse::to_display(t.trim())).collect::<Vec<_>>().join(" ")
}

/// Tones that key on and off like CW: (audio Hz, marks, duty cycle, mark level over the
/// noise floor in dB). A bin counts only if, while it is off, its neighbours within
/// 150 Hz are quiet too, so tone-hopping digital modes (RTTY, MFSK, Olivia) and steady
/// carriers (FT8 tones, carriers) do not count.
fn keyed_tones(audio: &[f32]) -> Vec<(f32, usize, f32, f32)> {
    let mut sp = cwskimmer::dsp::spectrum::Spectrum::new(512, 80);
    let mut rows: Vec<Vec<f32>> = Vec::new();
    sp.process(audio, |p| rows.push(p.to_vec()));
    if rows.len() < 100 {
        return Vec::new();
    }
    // Smooth power over 4 frames (40 ms) so single-bin noise does not chop weak marks.
    let rows: Vec<Vec<f32>> = (0..rows.len())
        .map(|t| {
            let w = &rows[t.saturating_sub(1)..(t + 3).min(rows.len())];
            (0..rows[t].len()).map(|b| 10.0 * (w.iter().map(|r| r[b]).sum::<f32>() / w.len() as f32 + 1e-14).log10()).collect()
        })
        .collect();
    let hz = RATE as f32 / 512.0;
    let (lo, hi) = ((250.0 / hz) as usize, (2600.0 / hz) as usize);
    // Noise floor per bin: 20th percentile over time, then the median over a 300 Hz span.
    let pct = |mut v: Vec<f32>, q: f32| {
        v.sort_by(|a, b| a.total_cmp(b));
        v[((v.len() - 1) as f32 * q) as usize]
    };
    let low: Vec<f32> = (0..rows[0].len()).map(|b| pct(rows.iter().map(|r| r[b]).collect(), 0.2)).collect();
    let floor: Vec<f32> = (0..low.len())
        .map(|b| pct(low[b.saturating_sub(10)..(b + 10).min(low.len())].to_vec(), 0.5))
        .collect();
    let mut found: Vec<(f32, usize, f32, f32)> = Vec::new();
    for b in lo..hi {
        let lvl: Vec<f32> = rows.iter().map(|r| r[b] - floor[b]).collect();
        let peak = pct(lvl.clone(), 0.9);
        if peak < 6.0 {
            continue;
        }
        // Local maximum in frequency (by the 90th percentile level).
        let p90 = |bb: usize| pct(rows.iter().map(|r| r[bb] - floor[bb]).collect(), 0.9);
        if p90(b - 1) > peak || p90(b + 1) > peak {
            continue;
        }
        let thr = peak / 2.0;
        let on: Vec<bool> = lvl.iter().map(|&v| v > thr).collect();
        let (mut marks, mut good_off, mut offs, mut on_frames) = (0usize, 0usize, 0usize, 0usize);
        let mut i = 0;
        while i < on.len() {
            let j = (i..on.len()).find(|&k| on[k] != on[i]).unwrap_or(on.len());
            let len = j - i;
            if on[i] {
                on_frames += len;
                if (3..=60).contains(&len) {
                    marks += 1;
                }
            } else if len >= 3 && i > 0 && j < on.len() {
                offs += 1;
                // Neighbours (30-150 Hz away) quiet while this bin is off.
                let near = |r: &Vec<f32>| {
                    (2..=10usize).flat_map(|d| [b - d, b + d]).filter(|&x| x < r.len()).map(|x| r[x] - floor[x]).fold(f32::MIN, f32::max)
                };
                let busy = rows[i..j].iter().filter(|r| near(r) > thr).count();
                if busy * 2 < len {
                    good_off += 1;
                }
            }
            i = j;
        }
        let duty = on_frames as f32 / on.len() as f32;
        if std::env::var_os("CWSK_KEYDBG").is_some() {
            eprintln!("{:6.1} peak {peak:4.1} marks {marks} offs {offs} good {good_off} duty {duty:.2}", b as f32 * hz);
        }
        if marks >= 8 && offs > 0 && good_off * 10 >= offs * 7 && (0.15..0.8).contains(&duty) {
            if found.last().is_some_and(|l| (b as f32 * hz - l.0) < 40.0) {
                continue;
            }
            found.push((b as f32 * hz, marks, duty, peak));
        }
    }
    found.sort_by(|a, b| b.3.total_cmp(&a.3));
    found
}

fn normalize_ref(s: &str) -> String {
    let up = cwskimmer::morse::from_display(s);
    up.chars()
        .map(|c| if c.is_ascii_alphanumeric() || "/?.,=+(%&-':@".contains(c) { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

const TEXTS: &[&str] = &[
    "CQ CQ CQ DE N4XYZ N4XYZ N4XYZ K",
    "WB2TQE DE G3VMW GE OM TNX FER RPRT UR 579 579 IN SURREY NAME IS ALAN HW? WB2TQE DE G3VMW KN",
    "THE AMERICAN RADIO RELAY LEAGUE IS THE NATIONAL ASSOCIATION FOR AMATEUR RADIO IN THE UNITED STATES",
    "R R FB ALAN RIG IS FTDX3000 ES ANT IS DIPOLE UP 40 FT WX HR CLOUDY ES COOL 73 ES TU SK",
    "TEST K5ZD K5ZD TU 5NN MA",
    "QRZ? DE VE3NEA PSE AGN UR CALL",
    "IT WAS A BRIGHT COLD DAY IN APRIL AND THE CLOCKS WERE STRIKING THIRTEEN",
    "DL1ABC DE JA1XYZ UR 559 559 QSB QSB NAME TARO QTH TOKYO BK",
];

struct Case {
    name: &'static str,
    make: fn(seed: u64) -> Scene,
}

fn one(spec: SignalSpec, seed: u64) -> Scene {
    Scene { signals: vec![spec], seed, ..Default::default() }
}

fn text(seed: u64) -> String {
    TEXTS[(seed as usize) % TEXTS.len()].to_string()
}

fn groups(seed: u64) -> String {
    use rand::{Rng, SeedableRng};
    let mut r = rand::rngs::StdRng::seed_from_u64(seed * 7919);
    let a = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    (0..10).map(|_| (0..5).map(|_| a[r.gen_range(0..a.len())] as char).collect::<String>()).collect::<Vec<_>>().join(" ")
}

fn cases() -> Vec<Case> {
    vec![
        Case { name: "clean-20wpm", make: |s| one(SignalSpec { text: text(s), wpm: 20.0, snr_db: 25.0, ..Default::default() }, s) },
        Case { name: "snr+5-25wpm", make: |s| one(SignalSpec { text: text(s), wpm: 25.0, snr_db: 5.0, ..Default::default() }, s) },
        Case { name: "snr0-22wpm", make: |s| one(SignalSpec { text: text(s), wpm: 22.0, snr_db: 0.0, ..Default::default() }, s) },
        Case { name: "snr-4-18wpm", make: |s| one(SignalSpec { text: text(s), wpm: 18.0, snr_db: -4.0, ..Default::default() }, s) },
        Case { name: "snr-8-18wpm", make: |s| one(SignalSpec { text: text(s), wpm: 18.0, snr_db: -8.0, ..Default::default() }, s) },
        Case {
            name: "handkey-jitter15",
            make: |s| one(SignalSpec { text: text(s), wpm: 18.0, snr_db: 12.0, jitter: 0.15, dash_ratio: 3.4, ..Default::default() }, s),
        },
        Case {
            name: "bad-fist-jitter25",
            make: |s| one(SignalSpec { text: text(s), wpm: 16.0, snr_db: 12.0, jitter: 0.25, dash_ratio: 3.8, weight: 1.3, ..Default::default() }, s),
        },
        Case {
            name: "farnsworth-18/8",
            make: |s| one(SignalSpec { text: text(s), wpm: 18.0, farnsworth_wpm: Some(8.0), snr_db: 10.0, ..Default::default() }, s),
        },
        Case {
            name: "qsb-deep",
            make: |s| one(SignalSpec { text: text(s), wpm: 22.0, snr_db: 10.0, qsb_depth: 0.85, qsb_hz: 0.3, ..Default::default() }, s),
        },
        Case {
            name: "qrn-crashes",
            make: |s| Scene { qrn_rate: 4.0, qrn_amp: 15.0, ..one(SignalSpec { text: text(s), wpm: 20.0, snr_db: 8.0, ..Default::default() }, s) },
        },
        Case {
            name: "rayleigh-snr0-16wpm",
            make: |s| one(SignalSpec { text: text(s), wpm: 16.0, snr_db: 0.0, jitter: 0.08, fade_hz: 1.0, qsb_depth: 0.9, ..Default::default() }, s),
        },
        Case {
            name: "rayleigh-snr-3-20wpm",
            make: |s| one(SignalSpec { text: text(s), wpm: 20.0, snr_db: -3.0, fade_hz: 0.5, qsb_depth: 0.7, ..Default::default() }, s),
        },
        Case {
            name: "rayleigh-handkey",
            make: |s| one(SignalSpec { text: text(s), wpm: 16.0, snr_db: 2.0, jitter: 0.15, dash_ratio: 3.4, fade_hz: 2.0, qsb_depth: 0.8, ..Default::default() }, s),
        },
        Case {
            name: "tight-wordgaps",
            make: |s| one(SignalSpec { text: text(s), wpm: 20.0, snr_db: 8.0, jitter: 0.12, word_gap: 4.5, ..Default::default() }, s),
        },
        Case {
            name: "wide-chargaps",
            make: |s| one(SignalSpec { text: text(s), wpm: 22.0, snr_db: 8.0, jitter: 0.12, char_gap: 4.5, word_gap: 9.0, ..Default::default() }, s),
        },
        Case {
            name: "freq-drift-40hz",
            make: |s| one(SignalSpec { text: text(s), wpm: 16.0, snr_db: 3.0, jitter: 0.08, freq_drift: 40.0, ..Default::default() }, s),
        },
        Case { name: "drift+40%", make: |s| one(SignalSpec { text: text(s), wpm: 18.0, drift: 0.4, snr_db: 10.0, ..Default::default() }, s) },
        Case { name: "fast-40wpm", make: |s| one(SignalSpec { text: text(s), wpm: 40.0, snr_db: 10.0, ..Default::default() }, s) },
        Case { name: "slow-10wpm", make: |s| one(SignalSpec { text: text(s), wpm: 10.0, snr_db: 5.0, ..Default::default() }, s) },
        Case { name: "groups-snr+3", make: |s| one(SignalSpec { text: groups(s), wpm: 20.0, snr_db: 3.0, ..Default::default() }, s) },
        Case {
            name: "pileup-5-signals",
            make: |s| Scene {
                signals: (0..5)
                    .map(|i| SignalSpec {
                        text: text(s + i),
                        freq: [480.0, 560.0, 700.0, 940.0, 1500.0][i as usize],
                        wpm: [18.0, 26.0, 22.0, 30.0, 15.0][i as usize],
                        snr_db: [8.0, 12.0, 15.0, 5.0, 10.0][i as usize],
                        jitter: [0.05, 0.0, 0.1, 0.0, 0.15][i as usize],
                        start: 0.3 * i as f32,
                        ..Default::default()
                    })
                    .collect(),
                seed: s,
                ..Default::default()
            },
        },
    ]
}

/// Runs a scene through the full skimmer (auto-detection on) and returns the
/// CER for each signal: decoded text is gathered from channels within 25 Hz.
/// Spot accounting across a bench run: callsigns in the sent texts, how
/// many were spotted (confirmed), and spots that were not sent at all.
static SPOT_TRUE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static SPOT_FOUND: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static SPOT_ANY: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static SPOT_MISSED_TWICE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static SPOT_TWICE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static SPOT_WRONG: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static SPOT_WRONG_CONF: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn count_spots(scene: &Scene, sk: &Skimmer, verbose: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    let truth: std::collections::HashSet<String> =
        scene.signals.iter().flat_map(|s| s.text.split_whitespace()).filter_map(cwskimmer::callsign::parse_callsign).collect();
    SPOT_TRUE.fetch_add(truth.len(), Relaxed);
    // Calls sent at least twice should end up confirmed.
    for c in &truth {
        let sent = scene.signals.iter().flat_map(|s| s.text.split_whitespace()).filter(|w| w == c).count();
        if sent >= 2 {
            SPOT_TWICE.fetch_add(1, Relaxed);
            if !sk.spots.iter().any(|sp| &sp.call == c && sp.confirmed) {
                SPOT_MISSED_TWICE.fetch_add(1, Relaxed);
                if verbose {
                    println!("    missed (sent {sent}x): {c}");
                }
            }
        }
    }
    for sp in &sk.spots {
        if truth.contains(&sp.call) {
            SPOT_ANY.fetch_add(1, Relaxed);
            if sp.confirmed {
                SPOT_FOUND.fetch_add(1, Relaxed);
            }
        } else {
            SPOT_WRONG.fetch_add(1, Relaxed);
            if sp.confirmed {
                SPOT_WRONG_CONF.fetch_add(1, Relaxed);
            }
            if verbose {
                println!("    wrong spot: {}{}", sp.call, if sp.confirmed { " (confirmed)" } else { "" });
            }
        }
    }
}

fn run_scene(scene: &Scene, lm: Arc<LanguageModel>, verbose: bool) -> Vec<f32> {
    let samples = synth::render(scene);
    let detect_db = std::env::var("CWSK_DETECT_DB").ok().and_then(|v| v.parse().ok()).unwrap_or(SkimmerConfig::default().detect_db);
    let mut sk = Skimmer::new(SkimmerConfig { params: DecoderParams::from_env(), detect_db, ..Default::default() }, lm);
    // 20 ms chunks, like live sound-card capture.
    for c in samples.chunks(160) {
        sk.process(c);
    }
    sk.finish();
    count_spots(scene, &sk, verbose);
    let events: Vec<_> = sk.events.drain(..).collect();
    scene
        .signals
        .iter()
        .map(|sig| {
            let mut txt = String::new();
            // Use the channel (within 25 Hz, plus any drift) that produced
            // the most text.
            let mut per: std::collections::HashMap<u64, String> = Default::default();
            for e in &events {
                if (e.freq - sig.freq).abs() < 25.0 + sig.freq_drift.abs() {
                    per.entry(e.channel).or_default().push_str(&e.text);
                }
            }
            if let Some(t) = per.values().max_by_key(|t| t.len()) {
                txt = t.clone();
            }
            let c = synth::cer(&sig.text, &txt);
            if verbose {
                println!("    ref: {}\n    got: {}  [{:.1}%]", sig.text, txt.trim(), c * 100.0);
            }
            c
        })
        .collect()
}

fn bench(trials: u64, compare_lm: bool, only: Option<String>, verbose: bool) {
    let lm = Arc::new(LanguageModel::new());
    let uni = Arc::new(LanguageModel::uniform());
    println!("{:<20} {:>10} {}", "scenario", "CER", if compare_lm { "  (no LM)" } else { "" });
    let mut total = 0.0;
    let mut n = 0;
    for case in cases() {
        if let Some(o) = &only {
            if !case.name.contains(o.as_str()) {
                continue;
            }
        }
        let mut cers = Vec::new();
        let mut cers_nolm = Vec::new();
        for t in 0..trials {
            let scene = (case.make)(t + 1);
            if verbose {
                println!("  {} trial {}", case.name, t + 1);
            }
            cers.extend(run_scene(&scene, lm.clone(), verbose));
            if compare_lm {
                cers_nolm.extend(run_scene(&scene, uni.clone(), false));
            }
        }
        let m = cers.iter().sum::<f32>() / cers.len() as f32;
        total += m;
        n += 1;
        if compare_lm {
            let m2 = cers_nolm.iter().sum::<f32>() / cers_nolm.len() as f32;
            println!("{:<20} {:>9.1}%   {:>6.1}%", case.name, m * 100.0, m2 * 100.0);
        } else {
            println!("{:<20} {:>9.1}%", case.name, m * 100.0);
        }
    }
    println!("{:<20} {:>9.1}%", "MEAN", total / n.max(1) as f32 * 100.0);
    use std::sync::atomic::Ordering::Relaxed;
    let (t, f, w, wc) = (SPOT_TRUE.load(Relaxed), SPOT_FOUND.load(Relaxed), SPOT_WRONG.load(Relaxed), SPOT_WRONG_CONF.load(Relaxed));
    let (a, tw, mt) = (SPOT_ANY.load(Relaxed), SPOT_TWICE.load(Relaxed), SPOT_MISSED_TWICE.load(Relaxed));
    println!(
        "SPOTS {a}/{t} calls spotted ({:.0}%), {f} confirmed; sent 2+ times: {}/{tw} confirmed; {w} wrong spots ({wc} confirmed)",
        100.0 * a as f32 / t.max(1) as f32,
        tw - mt
    );
}
