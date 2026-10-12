//! Icom CI-V protocol.

use super::{open_serial, Mode, Rig, RigConfig};
use anyhow::{anyhow, Result};
use std::io::{Read, Write};
use std::time::{Duration, Instant};

pub const CONTROLLER: u8 = 0xE0;

pub struct Icom {
    port: Box<dyn serialport::SerialPort>,
    addr: u8,
}

pub fn model_name(addr: u8) -> String {
    match addr {
        0x94 => "Icom IC-7300".into(),
        0x98 => "Icom IC-7610".into(),
        0xA4 => "Icom IC-705".into(),
        0xA2 => "Icom IC-9700".into(),
        0x88 => "Icom IC-7100".into(),
        0x8C => "Icom IC-7410".into(),
        0x7A => "Icom IC-7600".into(),
        0x80 => "Icom IC-7000".into(),
        0x70 => "Icom IC-7200".into(),
        0xA6 => "Icom IC-905".into(),
        0xAC => "Icom IC-7760".into(),
        a => format!("Icom (CI-V {a:02X}h)"),
    }
}

pub fn bcd_to_hz(b: &[u8]) -> u64 {
    // Little-endian packed BCD, two digits per byte.
    let mut hz = 0u64;
    for &x in b.iter().rev() {
        hz = hz * 100 + ((x >> 4) as u64) * 10 + (x & 0x0f) as u64;
    }
    hz
}

pub fn hz_to_bcd(mut hz: u64) -> [u8; 5] {
    let mut out = [0u8; 5];
    for o in out.iter_mut() {
        let lo = (hz % 10) as u8;
        hz /= 10;
        let hi = (hz % 10) as u8;
        hz /= 10;
        *o = (hi << 4) | lo;
    }
    out
}

impl Icom {
    pub fn open(cfg: &RigConfig) -> Result<Self> {
        let port = open_serial(cfg, false)?;
        let mut s = Self { port, addr: cfg.civ_addr };
        s.get_freq()?;
        Ok(s)
    }

    pub fn with_port(port: Box<dyn serialport::SerialPort>, addr: u8) -> Self {
        Self { port, addr }
    }

    /// Sends a command and returns the payload of the radio's reply
    /// (bytes after the command byte).
    pub fn transact(&mut self, cmd: &[u8]) -> Result<Vec<u8>> {
        let mut frame = vec![0xFE, 0xFE, self.addr, CONTROLLER];
        frame.extend_from_slice(cmd);
        frame.push(0xFD);
        let _ = self.port.clear(serialport::ClearBuffer::Input);
        self.port.write_all(&frame)?;
        self.port.flush()?;
        let start = Instant::now();
        let mut buf: Vec<u8> = Vec::new();
        let mut b = [0u8; 64];
        while start.elapsed() < Duration::from_millis(600) {
            match self.port.read(&mut b) {
                Ok(n) if n > 0 => buf.extend_from_slice(&b[..n]),
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::TimedOut => {}
                Err(e) => return Err(e.into()),
            }
            // Parse complete frames.
            while let Some(end) = buf.iter().position(|&x| x == 0xFD) {
                let f: Vec<u8> = buf.drain(..=end).collect();
                let Some(st) = f.windows(2).position(|w| w == [0xFE, 0xFE]) else { continue };
                let f = &f[st..];
                if f.len() < 6 {
                    continue;
                }
                let (to, from) = (f[2], f[3]);
                // Ignore our own echo on the CI-V bus.
                if to != CONTROLLER || (from != self.addr && self.addr != 0x00) {
                    continue;
                }
                let body = &f[4..f.len() - 1];
                match body.first() {
                    Some(0xFB) => return Ok(vec![]),
                    Some(0xFA) => return Err(anyhow!("radio rejected CI-V command")),
                    Some(&c) if c == cmd[0] => return Ok(body[1..].to_vec()),
                    _ => continue,
                }
            }
        }
        Err(anyhow!("no CI-V reply"))
    }
}

impl Rig for Icom {
    fn name(&self) -> String {
        model_name(self.addr)
    }

    fn get_freq(&mut self) -> Result<u64> {
        let r = self.transact(&[0x03])?;
        if r.len() < 5 {
            return Err(anyhow!("short frequency reply"));
        }
        Ok(bcd_to_hz(&r[..5]))
    }

    fn set_freq(&mut self, hz: u64) -> Result<()> {
        let mut cmd = vec![0x05];
        cmd.extend_from_slice(&hz_to_bcd(hz));
        self.transact(&cmd).map(|_| ())
    }

    fn get_mode(&mut self) -> Result<Mode> {
        let r = self.transact(&[0x04])?;
        Ok(match r.first() {
            Some(0x00) => Mode::Lsb,
            Some(0x01) => Mode::Usb,
            Some(0x02) => Mode::Am,
            Some(0x03) => Mode::Cw,
            Some(0x04) | Some(0x08) => Mode::Rtty,
            Some(0x05) => Mode::Fm,
            Some(0x07) => Mode::CwR,
            Some(x) => Mode::Other(format!("{x:02X}")),
            None => return Err(anyhow!("empty mode reply")),
        })
    }

    fn set_mode(&mut self, mode: &Mode) -> Result<()> {
        let m = match mode {
            Mode::Lsb => 0x00,
            Mode::Usb => 0x01,
            Mode::Am => 0x02,
            Mode::Cw => 0x03,
            Mode::Fm => 0x05,
            Mode::CwR => 0x07,
            _ => return Err(anyhow!("mode not supported")),
        };
        self.transact(&[0x06, m]).map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bcd() {
        let b = hz_to_bcd(14_025_120);
        assert_eq!(b, [0x20, 0x51, 0x02, 0x14, 0x00]);
        assert_eq!(bcd_to_hz(&b), 14_025_120);
    }
}
