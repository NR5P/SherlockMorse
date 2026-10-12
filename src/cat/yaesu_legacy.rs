//! Yaesu FT-817/818/857/897 five-byte binary CAT.

use super::{open_serial, Mode, Rig, RigConfig};
use anyhow::{anyhow, Result};
use std::io::{Read, Write};
use std::time::{Duration, Instant};

pub struct YaesuLegacy {
    port: Box<dyn serialport::SerialPort>,
}

impl YaesuLegacy {
    pub fn open(cfg: &RigConfig) -> Result<Self> {
        let port = open_serial(cfg, true)?;
        let mut s = Self { port };
        s.read_status()?;
        Ok(s)
    }

    fn cmd(&mut self, c: [u8; 5], reply: usize) -> Result<Vec<u8>> {
        let _ = self.port.clear(serialport::ClearBuffer::Input);
        self.port.write_all(&c)?;
        self.port.flush()?;
        let mut out = Vec::new();
        let start = Instant::now();
        let mut b = [0u8; 8];
        while out.len() < reply && start.elapsed() < Duration::from_millis(500) {
            match self.port.read(&mut b) {
                Ok(n) => out.extend_from_slice(&b[..n]),
                Err(e) if e.kind() == std::io::ErrorKind::TimedOut => {}
                Err(e) => return Err(e.into()),
            }
        }
        if out.len() < reply {
            return Err(anyhow!("no reply from radio"));
        }
        Ok(out)
    }

    fn read_status(&mut self) -> Result<(u64, u8)> {
        let r = self.cmd([0, 0, 0, 0, 0x03], 5)?;
        let mut f = 0u64;
        for &x in &r[..4] {
            f = f * 100 + ((x >> 4) as u64) * 10 + (x & 0xf) as u64;
        }
        Ok((f * 10, r[4]))
    }
}

impl Rig for YaesuLegacy {
    fn name(&self) -> String {
        "Yaesu FT-817/857/897".into()
    }
    fn get_freq(&mut self) -> Result<u64> {
        Ok(self.read_status()?.0)
    }
    fn set_freq(&mut self, hz: u64) -> Result<()> {
        let mut v = hz / 10;
        let mut b = [0u8; 5];
        for i in (0..4).rev() {
            let lo = (v % 10) as u8;
            v /= 10;
            let hi = (v % 10) as u8;
            v /= 10;
            b[i] = (hi << 4) | lo;
        }
        b[4] = 0x01;
        self.port.write_all(&b)?;
        Ok(())
    }
    fn get_mode(&mut self) -> Result<Mode> {
        Ok(match self.read_status()?.1 {
            0x00 => Mode::Lsb,
            0x01 => Mode::Usb,
            0x02 => Mode::Cw,
            0x03 => Mode::CwR,
            0x04 => Mode::Am,
            0x06 | 0x08 | 0x88 => Mode::Fm,
            0x0A | 0x0C => Mode::Data,
            x => Mode::Other(format!("{x:02X}")),
        })
    }
    fn set_mode(&mut self, mode: &Mode) -> Result<()> {
        let m = match mode {
            Mode::Lsb => 0x00,
            Mode::Usb => 0x01,
            Mode::Cw => 0x02,
            Mode::CwR => 0x03,
            Mode::Am => 0x04,
            Mode::Fm => 0x08,
            _ => return Err(anyhow!("mode not supported")),
        };
        self.port.write_all(&[m, 0, 0, 0, 0x07])?;
        Ok(())
    }
}
