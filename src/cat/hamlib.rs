//! Hamlib `rigctld` network protocol. Lets the skimmer control any radio
//! Hamlib supports: run e.g. `rigctld -m 1036 -r /dev/ttyUSB0 -s 38400`.

use super::{Mode, Rig, RigConfig};
use anyhow::{anyhow, Result};
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::time::Duration;

pub struct Rigctld {
    reader: BufReader<TcpStream>,
    writer: TcpStream,
    host: String,
}

impl Rigctld {
    pub fn open(cfg: &RigConfig) -> Result<Self> {
        let addr = cfg.host.parse().or_else(|_| {
            use std::net::ToSocketAddrs;
            cfg.host.to_socket_addrs()?.next().ok_or_else(|| anyhow!("bad host"))
        })?;
        let s = TcpStream::connect_timeout(&addr, Duration::from_secs(2))?;
        s.set_read_timeout(Some(Duration::from_secs(2)))?;
        s.set_nodelay(true)?;
        let writer = s.try_clone()?;
        Ok(Self { reader: BufReader::new(s), writer, host: cfg.host.clone() })
    }

    fn line(&mut self) -> Result<String> {
        let mut l = String::new();
        self.reader.read_line(&mut l)?;
        if l.is_empty() {
            return Err(anyhow!("rigctld closed connection"));
        }
        Ok(l.trim().to_string())
    }

    fn cmd(&mut self, c: &str) -> Result<()> {
        self.writer.write_all(c.as_bytes())?;
        self.writer.write_all(b"\n")?;
        let r = self.line()?;
        if r.starts_with("RPRT 0") {
            Ok(())
        } else {
            Err(anyhow!("rigctld: {r}"))
        }
    }
}

impl Rig for Rigctld {
    fn name(&self) -> String {
        format!("rigctld @ {}", self.host)
    }
    fn get_freq(&mut self) -> Result<u64> {
        self.writer.write_all(b"f\n")?;
        let r = self.line()?;
        if r.starts_with("RPRT") {
            return Err(anyhow!("rigctld: {r}"));
        }
        Ok(r.parse::<f64>().map_err(|_| anyhow!("bad reply {r}"))? as u64)
    }
    fn set_freq(&mut self, hz: u64) -> Result<()> {
        self.cmd(&format!("F {hz}"))
    }
    fn get_mode(&mut self) -> Result<Mode> {
        self.writer.write_all(b"m\n")?;
        let m = self.line()?;
        let _passband = self.line()?;
        Ok(match m.as_str() {
            "CW" => Mode::Cw,
            "CWR" => Mode::CwR,
            "USB" => Mode::Usb,
            "LSB" => Mode::Lsb,
            "AM" => Mode::Am,
            "FM" => Mode::Fm,
            "RTTY" | "RTTYR" => Mode::Rtty,
            "PKTUSB" | "PKTLSB" => Mode::Data,
            o => Mode::Other(o.into()),
        })
    }
    fn set_mode(&mut self, mode: &Mode) -> Result<()> {
        let m = match mode {
            Mode::Cw => "CW",
            Mode::CwR => "CWR",
            Mode::Usb => "USB",
            Mode::Lsb => "LSB",
            Mode::Am => "AM",
            Mode::Fm => "FM",
            _ => return Err(anyhow!("mode not supported")),
        };
        self.cmd(&format!("M {m} 0"))
    }
}
