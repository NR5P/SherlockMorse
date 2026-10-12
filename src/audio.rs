//! Sound card capture (cpal) with device enumeration and auto-detection of
//! radio USB audio interfaces.

use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::Sender;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ChannelSel {
    #[default]
    Mix,
    Left,
    Right,
}

#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub is_default: bool,
    /// Higher means more likely to be a radio's USB audio codec.
    pub score: i32,
}

/// Scores a device name for "is this the radio?".
pub fn radio_score(name: &str) -> i32 {
    let n = name.to_ascii_lowercase();
    let mut s = 0;
    // Yaesu, Icom, Kenwood and many interfaces use the TI/Burr-Brown
    // "USB Audio CODEC" (ALSA card id "CODEC").
    if n.contains("codec") {
        s += 100;
    }
    for k in ["yaesu", "icom", "kenwood", "ftdx", "ic-7", "ic7", "ts-", "elecraft", "flex", "signalink", "rigblaster", "digirig", "c-media", "cm108", "cm119"] {
        if n.contains(k) {
            s += 60;
        }
    }
    if n.contains("usb") {
        s += 20;
    }
    for k in ["dock", "webcam", "camera", "headset", "hdmi", "displayport", "monitor of"] {
        if n.contains(k) {
            s -= 40;
        }
    }
    // On ALSA prefer the plug/sysdefault layers which convert formats.
    if n.starts_with("plughw:") || n.starts_with("sysdefault:") {
        s += 10;
    } else if n.starts_with("hw:") {
        s += 2;
    } else if n.starts_with("front:") || n.starts_with("surround") || n.starts_with("dsnoop:") {
        s -= 5;
    }
    s
}

pub fn list_input_devices() -> Vec<DeviceInfo> {
    let host = cpal::default_host();
    let default = host.default_input_device().and_then(|d| d.name().ok());
    let mut v = Vec::new();
    if let Ok(devs) = host.input_devices() {
        for d in devs {
            if let Ok(name) = d.name() {
                // Skip devices that cannot capture at all.
                if d.default_input_config().is_err() {
                    continue;
                }
                let is_default = default.as_deref() == Some(name.as_str());
                v.push(DeviceInfo { score: radio_score(&name), name, is_default });
            }
        }
    }
    v
}

/// Picks the most likely radio audio device, falling back to the default.
pub fn auto_select(devs: &[DeviceInfo]) -> Option<String> {
    let best = devs.iter().max_by_key(|d| d.score)?;
    if best.score >= 50 {
        return Some(best.name.clone());
    }
    devs.iter().find(|d| d.is_default).or(devs.first()).map(|d| d.name.clone())
}

pub struct AudioChunk {
    pub samples: Vec<f32>,
    pub rate: u32,
}

/// A running capture; stops when dropped.
pub struct Capture {
    stop: Arc<AtomicBool>,
    /// Set when the stream reports an error (e.g. the device was unplugged).
    pub failed: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
    pub device: String,
    pub rate: u32,
    pub channels: u16,
}

impl Drop for Capture {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Starts capturing from the named device (or default) into `tx`.
pub fn start_capture(name: Option<&str>, sel: ChannelSel, tx: Sender<AudioChunk>) -> Result<Capture> {
    let stop = Arc::new(AtomicBool::new(false));
    let (ready_tx, ready_rx) = crossbeam_channel::bounded::<Result<(String, u32, u16)>>(1);
    let name = name.map(|s| s.to_string());
    let stop2 = stop.clone();
    let failed = Arc::new(AtomicBool::new(false));
    let failed2 = failed.clone();
    let thread = std::thread::Builder::new()
        .name("audio".into())
        .spawn(move || {
            // The stream must live on this thread (it is !Send on some hosts).
            let res = build_stream(name.as_deref(), sel, tx, failed2.clone());
            match res {
                Ok((stream, info)) => {
                    let _ = ready_tx.send(Ok(info));
                    while !stop2.load(Ordering::Relaxed) && !failed2.load(Ordering::Relaxed) {
                        std::thread::sleep(std::time::Duration::from_millis(50));
                    }
                    drop(stream);
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                }
            }
        })?;
    let (device, rate, channels) = ready_rx
        .recv_timeout(std::time::Duration::from_secs(10))
        .map_err(|_| anyhow!("audio device did not start"))??;
    Ok(Capture { stop, failed, thread: Some(thread), device, rate, channels })
}

fn build_stream(name: Option<&str>, sel: ChannelSel, tx: Sender<AudioChunk>, failed: Arc<AtomicBool>) -> Result<(cpal::Stream, (String, u32, u16))> {
    let host = cpal::default_host();
    let device = match name {
        Some(n) => host
            .input_devices()?
            .find(|d| d.name().map(|x| x == n).unwrap_or(false))
            .ok_or_else(|| anyhow!("audio device '{n}' not found"))?,
        None => host.default_input_device().ok_or_else(|| anyhow!("no default input device"))?,
    };
    let dev_name = device.name().unwrap_or_default();
    let default_cfg = device.default_input_config().context("querying input config")?;
    // Prefer 48 kHz (radio codecs run natively at 48k), else the default.
    let mut cfg = default_cfg.clone();
    if let Ok(ranges) = device.supported_input_configs() {
        for r in ranges {
            if r.sample_format() == default_cfg.sample_format()
                && r.min_sample_rate().0 <= 48000
                && r.max_sample_rate().0 >= 48000
                && r.channels() <= 2
            {
                cfg = r.with_sample_rate(cpal::SampleRate(48000));
                break;
            }
        }
    }
    let rate = cfg.sample_rate().0;
    let channels = cfg.channels();
    let fmt = cfg.sample_format();
    let config: cpal::StreamConfig = cfg.into();
    // Report once; the owner notices `failed` and reconnects.
    let err_fn = move |e| {
        if !failed.swap(true, Ordering::Relaxed) {
            eprintln!("audio stream error: {e}");
        }
    };
    macro_rules! build {
        ($t:ty, $conv:expr) => {{
            let tx = tx.clone();
            device.build_input_stream(
                &config,
                move |data: &[$t], _: &cpal::InputCallbackInfo| {
                    let ch = channels as usize;
                    let conv = $conv;
                    let mut out = Vec::with_capacity(data.len() / ch.max(1));
                    for frame in data.chunks(ch.max(1)) {
                        let v = match (sel, ch) {
                            (_, 1) => conv(frame[0]),
                            (ChannelSel::Left, _) => conv(frame[0]),
                            (ChannelSel::Right, _) => conv(frame[1]),
                            (ChannelSel::Mix, _) => frame.iter().map(|&x| conv(x)).sum::<f32>() / ch as f32,
                        };
                        out.push(v);
                    }
                    let _ = tx.try_send(AudioChunk { samples: out, rate });
                },
                err_fn,
                None,
            )?
        }};
    }
    let stream = match fmt {
        cpal::SampleFormat::F32 => build!(f32, |x: f32| x),
        cpal::SampleFormat::I16 => build!(i16, |x: i16| x as f32 / 32768.0),
        cpal::SampleFormat::U16 => build!(u16, |x: u16| (x as f32 - 32768.0) / 32768.0),
        cpal::SampleFormat::I32 => build!(i32, |x: i32| x as f32 / 2147483648.0),
        cpal::SampleFormat::I8 => build!(i8, |x: i8| x as f32 / 128.0),
        cpal::SampleFormat::U8 => build!(u8, |x: u8| (x as f32 - 128.0) / 128.0),
        other => return Err(anyhow!("unsupported sample format {other:?}")),
    };
    stream.play()?;
    Ok((stream, (dev_name, rate, channels)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scores_radio_codec_highest() {
        let devs = vec![
            DeviceInfo { name: "default".into(), is_default: true, score: radio_score("default") },
            DeviceInfo { name: "sysdefault:CARD=Audio".into(), is_default: false, score: radio_score("sysdefault:CARD=Audio ThinkPad Dock USB Audio") },
            DeviceInfo { name: "plughw:CARD=CODEC,DEV=0".into(), is_default: false, score: radio_score("plughw:CARD=CODEC,DEV=0") },
            DeviceInfo { name: "Microphone (USB Audio CODEC )".into(), is_default: false, score: radio_score("Microphone (USB Audio CODEC )") },
        ];
        let pick = auto_select(&devs).unwrap();
        assert!(pick.contains("CODEC"));
    }
}
