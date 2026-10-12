//! CAT (computer aided transceiver) radio control.
//!
//! Supported protocols:
//! * Yaesu "new CAT" ASCII protocol – FTDX3000, FTDX5000, FTDX1200, FT-950,
//!   FT-2000, FT-991(A), FT-891, FT-710, FTDX10, FTDX101 …
//! * Kenwood / Elecraft ASCII protocol – TS-480/590/890/990, K3, K4, KX2/3 …
//! * Icom CI-V – IC-7300, IC-7610, IC-705, IC-9700, IC-7100 …
//! * Yaesu legacy 5-byte binary protocol – FT-817/818/857/897.
//! * Hamlib `rigctld` network daemon – any radio Hamlib supports.

pub mod controller;
pub mod hamlib;
pub mod icom;
pub mod probe;
pub mod textcat;
pub mod yaesu_legacy;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RigKind {
    /// Yaesu ASCII CAT (FTDX3000 etc.)
    Yaesu,
    /// Kenwood / Elecraft ASCII CAT.
    Kenwood,
    /// Icom CI-V.
    Icom,
    /// Yaesu FT-817/818/857/897 binary CAT.
    YaesuLegacy,
    /// Hamlib rigctld over TCP.
    Hamlib,
}

impl RigKind {
    pub const ALL: [RigKind; 5] = [RigKind::Yaesu, RigKind::Kenwood, RigKind::Icom, RigKind::YaesuLegacy, RigKind::Hamlib];
    pub fn label(self) -> &'static str {
        match self {
            RigKind::Yaesu => "Yaesu CAT (FTDX3000/FTDX101/FT-991/FT-891/FT-710…)",
            RigKind::Kenwood => "Kenwood / Elecraft (TS-590/890, K3, K4, KX3…)",
            RigKind::Icom => "Icom CI-V (IC-7300/7610/705/9700…)",
            RigKind::YaesuLegacy => "Yaesu legacy (FT-817/818/857/897)",
            RigKind::Hamlib => "Hamlib rigctld (network, any radio)",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mode {
    Cw,
    CwR,
    Usb,
    Lsb,
    Am,
    Fm,
    Rtty,
    Data,
    Other(String),
}

impl Mode {
    pub fn label(&self) -> String {
        match self {
            Mode::Cw => "CW".into(),
            Mode::CwR => "CW-R".into(),
            Mode::Usb => "USB".into(),
            Mode::Lsb => "LSB".into(),
            Mode::Am => "AM".into(),
            Mode::Fm => "FM".into(),
            Mode::Rtty => "RTTY".into(),
            Mode::Data => "DATA".into(),
            Mode::Other(s) => s.clone(),
        }
    }

    /// Whether audio frequency increases with RF frequency in this mode.
    /// (CW on Yaesu/Kenwood/Icom is upper sideband; CW-R is lower.)
    pub fn is_upper(&self) -> bool {
        match self {
            Mode::CwR | Mode::Lsb => false,
            Mode::Other(s) => !s.ends_with("-L"),
            _ => true,
        }
    }

    /// In CW modes the displayed frequency is the carrier of a signal heard
    /// at the CW pitch; in SSB/DATA modes it is the suppressed carrier
    /// (audio 0 Hz).
    pub fn is_cw(&self) -> bool {
        matches!(self, Mode::Cw | Mode::CwR)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RigConfig {
    pub kind: RigKind,
    pub port: String,
    pub baud: u32,
    /// Icom CI-V address (e.g. 0x94 for IC-7300).
    pub civ_addr: u8,
    /// rigctld address, host:port.
    pub host: String,
    /// Drive RTS high. OFF by default: on many radios RTS/DTR key PTT or CW
    /// (e.g. the FTDX3000 "Standard" COM port). Only enable this for a CAT
    /// port whose radio menu requires it ("CAT RTS" = ON).
    #[serde(default)]
    pub assert_rts: bool,
    #[serde(default)]
    pub assert_dtr: bool,
    pub poll_ms: u64,
}

impl Default for RigConfig {
    fn default() -> Self {
        Self {
            kind: RigKind::Yaesu,
            port: default_port(),
            baud: 38400,
            civ_addr: 0x94,
            host: "127.0.0.1:4532".into(),
            assert_rts: false,
            assert_dtr: false,
            poll_ms: 300,
        }
    }
}

fn default_port() -> String {
    if cfg!(windows) {
        "COM3".into()
    } else if cfg!(target_os = "macos") {
        "/dev/cu.SLAB_USBtoUART".into()
    } else {
        "/dev/ttyUSB0".into()
    }
}

pub trait Rig: Send {
    fn name(&self) -> String;
    fn get_freq(&mut self) -> Result<u64>;
    fn set_freq(&mut self, hz: u64) -> Result<()>;
    fn get_mode(&mut self) -> Result<Mode>;
    fn set_mode(&mut self, mode: &Mode) -> Result<()>;
}

pub fn open_serial(cfg: &RigConfig, stop_bits_two: bool) -> Result<Box<dyn serialport::SerialPort>> {
    let mut p = serialport::new(&cfg.port, cfg.baud)
        .timeout(Duration::from_millis(250))
        .data_bits(serialport::DataBits::Eight)
        .parity(serialport::Parity::None)
        .stop_bits(if stop_bits_two { serialport::StopBits::Two } else { serialport::StopBits::One })
        .flow_control(serialport::FlowControl::None)
        .open()
        .map_err(|e| anyhow!("cannot open {}: {e}", cfg.port))?;
    // Set the control lines immediately after opening (the OS may raise
    // them on open). Low unless explicitly enabled, so we never key PTT/CW.
    let _ = p.write_request_to_send(cfg.assert_rts);
    let _ = p.write_data_terminal_ready(cfg.assert_dtr);
    let _ = p.clear(serialport::ClearBuffer::All);
    Ok(p)
}

/// Opens a radio according to the configuration.
pub fn open(cfg: &RigConfig) -> Result<Box<dyn Rig>> {
    Ok(match cfg.kind {
        RigKind::Yaesu => Box::new(textcat::TextCat::open(cfg, textcat::Dialect::Yaesu)?),
        RigKind::Kenwood => Box::new(textcat::TextCat::open(cfg, textcat::Dialect::Kenwood)?),
        RigKind::Icom => Box::new(icom::Icom::open(cfg)?),
        RigKind::YaesuLegacy => Box::new(yaesu_legacy::YaesuLegacy::open(cfg)?),
        RigKind::Hamlib => Box::new(hamlib::Rigctld::open(cfg)?),
    })
}

/// USB interface number of a serial port (Linux sysfs), if known.
pub fn usb_interface(port: &str) -> Option<u8> {
    let name = std::path::Path::new(port).file_name()?.to_str()?;
    for rel in ["device/../bInterfaceNumber", "device/bInterfaceNumber"] {
        let p = format!("/sys/class/tty/{name}/{rel}");
        if let Ok(v) = std::fs::read_to_string(&p) {
            if let Ok(n) = u8::from_str_radix(v.trim(), 16) {
                return Some(n);
            }
        }
    }
    None
}

/// True for the secondary port of dual-port USB serial chips, which radios
/// use for PTT / CW keying via RTS/DTR (FTDX3000 "Standard" COM port on the
/// CP2105 interface 1). Such ports must never be probed.
pub fn is_keying_port(port: &str) -> bool {
    usb_interface(port).is_some_and(|i| i != 0)
}

/// Whether RTS may be raised automatically as a CAT handshake on this port.
/// Only for a port positively identified as the CAT port (USB interface 0),
/// or, where interface info is unavailable, one not named a "Standard"
/// (PTT/keying) port. Many radios (e.g. FTDX3000 with CAT RTS = ON) ignore
/// CAT until the computer raises RTS, as WSJT-X/Hamlib do.
pub fn rts_handshake_allowed(port: &str) -> bool {
    match usb_interface(port) {
        Some(i) => i == 0,
        None => {
            let named_standard = serialport::available_ports().unwrap_or_default().into_iter().any(|p| {
                p.port_name == port
                    && matches!(&p.port_type, serialport::SerialPortType::UsbPort(u)
                        if u.product.as_deref().is_some_and(|x| x.to_ascii_lowercase().contains("standard")))
            });
            !named_standard
        }
    }
}

/// Runs `attempt` with RTS low; if the radio does not answer and the port is
/// a confirmed CAT port, raises RTS (hardware handshake) and tries again.
pub fn with_rts_fallback<T>(
    cfg: &RigConfig,
    two_stop: bool,
    mut attempt: impl FnMut(Box<dyn serialport::SerialPort>) -> Result<T>,
) -> Result<T> {
    let first = attempt(open_serial(cfg, two_stop)?);
    if first.is_ok() || cfg.assert_rts || !rts_handshake_allowed(&cfg.port) {
        return first;
    }
    let cfg2 = RigConfig { assert_rts: true, ..cfg.clone() };
    attempt(open_serial(&cfg2, two_stop)?).map_err(|e| anyhow!("{e} (also tried with RTS handshake)"))
}

/// Lists serial ports, USB ones first, with a description.
pub fn list_ports() -> Vec<(String, String)> {
    let mut v: Vec<(String, String, bool)> = serialport::available_ports()
        .unwrap_or_default()
        .into_iter()
        .map(|p| match &p.port_type {
            serialport::SerialPortType::UsbPort(u) => {
                let desc = format!(
                    "{} {} [{:04x}:{:04x}]",
                    u.manufacturer.clone().unwrap_or_default(),
                    u.product.clone().unwrap_or_default(),
                    u.vid,
                    u.pid,
                );
                let mut desc = desc.trim().to_string();
                match usb_interface(&p.port_name) {
                    Some(0) if u.vid == 0x10c4 && u.pid == 0xea70 => desc = format!("CAT (Enhanced port) — {desc}"),
                    Some(i) if i > 0 => desc = format!("⚠ secondary port (PTT/keying, not CAT) — {desc}"),
                    _ => {}
                }
                (p.port_name.clone(), desc, true)
            }
            _ => (p.port_name.clone(), String::new(), false),
        })
        .collect();
    v.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)));
    v.into_iter().map(|(a, b, _)| (a, b)).collect()
}
