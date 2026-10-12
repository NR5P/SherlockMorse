//! Yaesu and Kenwood/Elecraft ASCII CAT ("FA;", "MD;" …).

use super::{Mode, Rig, RigConfig};
use anyhow::{anyhow, Result};
use std::io::{Read, Write};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dialect {
    Yaesu,
    Kenwood,
}

pub struct TextCat {
    port: Box<dyn serialport::SerialPort>,
    dialect: Dialect,
    /// Number of digits the radio uses in FA (8 on FTDX3000, 9 on FT-991,
    /// 11 on Kenwood). Learned from the first FA reply.
    digits: usize,
    model: String,
}

/// Maps Yaesu/Kenwood ID replies to model names.
pub fn model_name(id: &str) -> Option<&'static str> {
    Some(match id {
        "0462" => "Yaesu FTDX3000",
        "0362" => "Yaesu FTDX5000",
        "0583" => "Yaesu FTDX1200",
        "0310" => "Yaesu FT-950",
        "0251" => "Yaesu FT-2000",
        "0570" => "Yaesu FT-991(A)",
        "0650" => "Yaesu FT-891",
        "0681" => "Yaesu FTDX101D",
        "0682" => "Yaesu FTDX101MP",
        "0761" => "Yaesu FTDX10",
        "0800" => "Yaesu FT-710",
        "0241" => "Yaesu FT-450",
        "017" => "Elecraft K3/K4/KX",
        "019" => "Kenwood TS-2000",
        "020" => "Kenwood TS-480",
        "021" => "Kenwood TS-590S",
        "023" => "Kenwood TS-590SG",
        "024" => "Kenwood TS-890S",
        "025" => "Kenwood TS-990S",
        _ => return None,
    })
}

impl TextCat {
    pub fn open(cfg: &RigConfig, dialect: Dialect) -> Result<Self> {
        super::with_rts_fallback(cfg, false, |port| Self::init(port, dialect))
    }

    fn init(port: Box<dyn serialport::SerialPort>, dialect: Dialect) -> Result<Self> {
        let mut s = Self { port, dialect, digits: if dialect == Dialect::Yaesu { 9 } else { 11 }, model: String::new() };
        // Wake up / identify.
        // The first command after (re)opening is sometimes dropped.
        let id = s.query("ID;").or_else(|_| s.query("ID;")).unwrap_or_default();
        let id = id.trim_start_matches("ID").to_string();
        s.model = model_name(&id).map(|m| m.to_string()).unwrap_or_else(|| {
            if id.is_empty() {
                format!("{dialect:?}")
            } else {
                format!("{dialect:?} (ID {id})")
            }
        });
        // Learn the frequency digit count.
        let fa = s.query("FA;")?;
        let digits = fa.trim_start_matches("FA").len();
        if (6..=12).contains(&digits) {
            s.digits = digits;
        }
        Ok(s)
    }

    pub fn with_port(port: Box<dyn serialport::SerialPort>, dialect: Dialect) -> Self {
        Self { port, dialect, digits: if dialect == Dialect::Yaesu { 9 } else { 11 }, model: String::new() }
    }

    /// Sends a command and returns the reply without the trailing ';'.
    pub fn query(&mut self, cmd: &str) -> Result<String> {
        let _ = self.port.clear(serialport::ClearBuffer::Input);
        self.port.write_all(cmd.as_bytes())?;
        self.port.flush()?;
        let start = Instant::now();
        let mut buf = Vec::new();
        let mut byte = [0u8; 1];
        while start.elapsed() < Duration::from_millis(600) {
            match self.port.read(&mut byte) {
                Ok(1) => {
                    if byte[0] == b';' {
                        let s = String::from_utf8_lossy(&buf).trim().to_string();
                        if s == "?" {
                            return Err(anyhow!("radio rejected command {cmd}"));
                        }
                        // Skip unrelated auto-information replies.
                        let want = &cmd[..2];
                        if s.starts_with(want) {
                            return Ok(s);
                        }
                        buf.clear();
                    } else {
                        buf.push(byte[0]);
                    }
                }
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::TimedOut => {}
                Err(e) => return Err(e.into()),
            }
        }
        Err(anyhow!("no reply to {cmd}"))
    }

    fn send(&mut self, cmd: &str) -> Result<()> {
        self.port.write_all(cmd.as_bytes())?;
        self.port.flush()?;
        Ok(())
    }
}

pub fn parse_mode_code(dialect: Dialect, c: char) -> Mode {
    match (dialect, c) {
        (_, '1') => Mode::Lsb,
        (_, '2') => Mode::Usb,
        (_, '3') => Mode::Cw,
        (_, '4') => Mode::Fm,
        (_, '5') => Mode::Am,
        (_, '7') => Mode::CwR,
        (Dialect::Yaesu, '6') | (Dialect::Yaesu, '9') | (Dialect::Kenwood, '6') | (Dialect::Kenwood, '9') => Mode::Rtty,
        (Dialect::Yaesu, '8') => Mode::Other("DATA-L".into()),
        (Dialect::Yaesu, 'C') | (Dialect::Yaesu, 'A') => Mode::Data,
        (_, 'B') => Mode::Fm,
        (_, 'D') => Mode::Am,
        (_, x) => Mode::Other(format!("MD{x}")),
    }
}

fn mode_code(m: &Mode) -> Option<char> {
    Some(match m {
        Mode::Lsb => '1',
        Mode::Usb => '2',
        Mode::Cw => '3',
        Mode::Fm => '4',
        Mode::Am => '5',
        Mode::CwR => '7',
        _ => return None,
    })
}

impl Rig for TextCat {
    fn name(&self) -> String {
        self.model.clone()
    }

    fn get_freq(&mut self) -> Result<u64> {
        let r = self.query("FA;")?;
        let d = r.trim_start_matches("FA");
        let hz: u64 = d.parse().map_err(|_| anyhow!("bad FA reply '{r}'"))?;
        if (6..=12).contains(&d.len()) {
            self.digits = d.len();
        }
        Ok(hz)
    }

    fn set_freq(&mut self, hz: u64) -> Result<()> {
        let cmd = format!("FA{:0width$};", hz, width = self.digits);
        self.send(&cmd)
    }

    fn get_mode(&mut self) -> Result<Mode> {
        let r = match self.dialect {
            Dialect::Yaesu => self.query("MD0;")?,
            Dialect::Kenwood => self.query("MD;")?,
        };
        let c = r.chars().last().ok_or_else(|| anyhow!("bad MD reply"))?;
        Ok(parse_mode_code(self.dialect, c))
    }

    fn set_mode(&mut self, mode: &Mode) -> Result<()> {
        let c = mode_code(mode).ok_or_else(|| anyhow!("mode not supported"))?;
        match self.dialect {
            Dialect::Yaesu => self.send(&format!("MD0{c};")),
            Dialect::Kenwood => self.send(&format!("MD{c};")),
        }
    }
}
