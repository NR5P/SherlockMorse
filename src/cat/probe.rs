//! Auto-detection of radios on USB serial ports.
//!
//! Only USB serial ports are probed (so we never poke unrelated hardware).
//! Each port is tried with the ASCII `ID;` command (Yaesu / Kenwood /
//! Elecraft) at common baud rates, then with a CI-V transceiver-ID request
//! broadcast (Icom).

use super::icom::{model_name as icom_name, Icom};
use super::textcat::{model_name, Dialect, TextCat};
use super::{RigConfig, RigKind};

#[derive(Clone, Debug)]
pub struct Found {
    pub config: RigConfig,
    pub model: String,
}

fn usb_ports() -> Vec<String> {
    serialport::available_ports()
        .unwrap_or_default()
        .into_iter()
        .filter(|p| matches!(p.port_type, serialport::SerialPortType::UsbPort(_)))
        .map(|p| p.port_name)
        .collect()
}

/// Probes all USB serial ports. Stops at the first radio found per port.
pub fn probe(mut progress: impl FnMut(&str)) -> Vec<Found> {
    let mut found = Vec::new();
    for port in usb_ports() {
        if let Some(f) = probe_port(&port, &mut progress) {
            found.push(f);
        }
    }
    found
}

pub fn probe_port(port: &str, progress: &mut impl FnMut(&str)) -> Option<Found> {
    // ASCII protocols. FTDX3000 default CAT rate is 4800; most people set 38400.
    for baud in [38400, 4800, 9600, 19200, 57600, 115200] {
        progress(&format!("{port} @ {baud}: ID;"));
        let cfg = RigConfig { port: port.into(), baud, rts: true, ..Default::default() };
        let Ok(p) = super::open_serial(&cfg, false) else { break };
        let mut t = TextCat::with_port(p, Dialect::Yaesu);
        if let Ok(r) = t.query("ID;") {
            let id = r.trim_start_matches("ID").to_string();
            let yaesu = id.len() == 4;
            let model = model_name(&id).map(|s| s.to_string()).unwrap_or_else(|| format!("ID {id}"));
            let kind = if yaesu { RigKind::Yaesu } else { RigKind::Kenwood };
            return Some(Found { config: RigConfig { kind, ..cfg }, model });
        }
    }
    // Icom CI-V: broadcast "read transceiver ID" (19 00).
    for baud in [19200, 115200, 9600, 4800] {
        progress(&format!("{port} @ {baud}: CI-V"));
        let cfg = RigConfig { port: port.into(), baud, rts: false, dtr: false, kind: RigKind::Icom, ..Default::default() };
        let Ok(p) = super::open_serial(&cfg, false) else { break };
        let mut ic = Icom::with_port(p, 0x00);
        if let Ok(r) = ic.transact(&[0x19, 0x00]) {
            if let Some(&addr) = r.get(1).or(r.first()) {
                return Some(Found { config: RigConfig { civ_addr: addr, ..cfg }, model: icom_name(addr) });
            }
        }
    }
    None
}
