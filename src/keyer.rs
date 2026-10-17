//! CW transmit keyer: turns text into key-down / key-up timing and keys the
//! transmitter through a serial port's DTR or RTS line, the way most radios
//! accept CW from a computer (FTDX3000: Menu 065 PC KEYING on the
//! "Standard" COM port, with break-in on).
//!
//! The timing is made here, not by the radio's keyer, so speed, weight and
//! letter/word spacing are all adjustable. With no port chosen the keyer
//! runs in test mode: the same timing, but nothing is keyed.
//!
//! The port is opened only when there is something to send, and kept open
//! until the keyer is dropped. Every exit path (abort, error, drop) lowers
//! the key and PTT lines.

use anyhow::{anyhow, Result};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Speed and shape of the sent code. Gaps are in dits at the set speed;
/// standard Morse is 3-dit letter gaps and 7-dit word gaps.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct KeyTiming {
    pub wpm: f32,
    /// Mark weight in percent: 50 is standard; more lengthens every mark
    /// and shortens the gap after it by the same time.
    pub weight: f32,
    /// Dah length in dits.
    pub dah: f32,
    pub letter_gap: f32,
    pub word_gap: f32,
    /// Added to every mark (and taken from the following gap), for radios
    /// whose break-in clips the start of each element.
    pub comp_ms: f32,
}

impl Default for KeyTiming {
    fn default() -> Self {
        Self { wpm: 20.0, weight: 50.0, dah: 3.0, letter_gap: 3.0, word_gap: 7.0, comp_ms: 0.0 }
    }
}

impl KeyTiming {
    pub fn dit_ms(&self) -> f64 {
        1200.0 / self.wpm.clamp(5.0, 60.0) as f64
    }

    /// Time added to marks and taken from gaps (ms).
    fn shift_ms(&self) -> f64 {
        let dit = self.dit_ms();
        (self.weight.clamp(25.0, 75.0) as f64 - 50.0) / 50.0 * dit + self.comp_ms.clamp(0.0, 30.0) as f64
    }

    /// Gaps never shrink below a fifth of a dit, whatever the weight.
    fn gap_ms(&self, dits: f32) -> f64 {
        let dit = self.dit_ms();
        (dits as f64 * dit - self.shift_ms()).max(0.2 * dit)
    }

    /// Key-down/key-up durations (ms) of one character from its pattern
    /// (".-"), ending with the letter gap.
    pub fn elements(&self, pattern: &str) -> Vec<(bool, f64)> {
        let dit = self.dit_ms();
        let shift = self.shift_ms();
        let n = pattern.len();
        let mut v = Vec::with_capacity(2 * n);
        for (i, e) in pattern.bytes().enumerate() {
            let base = if e == b'-' { self.dah.clamp(2.0, 6.0) as f64 * dit } else { dit };
            v.push((true, base + shift));
            v.push((false, if i + 1 == n { self.gap_ms(self.letter_gap.max(1.0)) } else { self.gap_ms(1.0) }));
        }
        v
    }

    /// Extra time a space adds after a character's letter gap (ms); a
    /// second space in a row adds a whole word gap.
    pub fn space_ms(&self, after_space: bool) -> f64 {
        let dit = self.dit_ms();
        if after_space {
            self.word_gap as f64 * dit
        } else {
            ((self.word_gap - self.letter_gap.max(1.0)).max(0.0)) as f64 * dit
        }
    }

    /// Whole-text timeline as the keyer would send it without pauses.
    pub fn timeline(&self, text: &str) -> Vec<(bool, f64)> {
        let mut out: Vec<(bool, f64)> = Vec::new();
        let mut rest = text;
        let mut after_space = true;
        while let Next::Sym { len, sym } = next_symbol(rest, true) {
            match sym {
                Sym::Space => {
                    let d = self.space_ms(after_space);
                    match out.last_mut() {
                        Some((false, g)) => *g += d,
                        _ => out.push((false, d)),
                    }
                    after_space = true;
                }
                Sym::Code(p) => {
                    out.extend(self.elements(&p));
                    after_space = false;
                }
                Sym::Skip => {}
            }
            rest = &rest[len..];
        }
        out
    }

    /// Speed in WPM counted the PARIS way, including the letter and word
    /// gaps (lower than `wpm` with Farnsworth spacing).
    pub fn effective_wpm(&self) -> f64 {
        let ms: f64 = self.timeline("PARIS ").iter().map(|e| e.1).sum();
        60_000.0 / ms
    }
}

/// One unit of text to send.
#[derive(Clone, Debug, PartialEq)]
pub enum Sym {
    /// Element pattern, e.g. ".-." (prosigns are one pattern).
    Code(String),
    Space,
    /// A character that has no Morse code.
    Skip,
}

#[derive(Debug, PartialEq)]
pub enum Next {
    Empty,
    /// An opened prosign ("<A") whose closing '>' may still be typed.
    Wait,
    Sym { len: usize, sym: Sym },
}

/// The next symbol at the start of `text` (uppercase) and its length in
/// bytes. `<AR>`, `<SK>`, `<BT>`, `<KN>` … are sent as one character
/// (letters run together); `=`, `+` and the other punctuation in the
/// Morse table are sent as themselves. With `complete`, an unclosed '<' is
/// not waited for.
pub fn next_symbol(text: &str, complete: bool) -> Next {
    let Some(c) = text.chars().next() else { return Next::Empty };
    let one = c.len_utf8();
    if c.is_whitespace() {
        return Next::Sym { len: one, sym: Sym::Space };
    }
    if c == '<' {
        let inner: String = text[1..].chars().take_while(|c| c.is_ascii_alphanumeric()).collect();
        let after = &text[1 + inner.len()..];
        if after.starts_with('>') && (1..=6).contains(&inner.len()) {
            let pattern: Option<String> = inner.bytes().map(crate::morse::pattern_for).collect();
            if let Some(p) = pattern {
                return Next::Sym { len: inner.len() + 2, sym: Sym::Code(p) };
            }
        } else if after.is_empty() && inner.len() < 6 && !complete {
            return Next::Wait;
        }
        return Next::Sym { len: one, sym: Sym::Skip };
    }
    match u8::try_from(c).ok().filter(|b| b.is_ascii_graphic()).and_then(crate::morse::pattern_for) {
        Some(p) => Next::Sym { len: one, sym: Sym::Code(p.to_string()) },
        None => Next::Sym { len: one, sym: Sym::Skip },
    }
}

/// Which serial control line keys the radio.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Line {
    Dtr,
    Rts,
}

impl Line {
    pub fn label(self) -> &'static str {
        match self {
            Line::Dtr => "DTR",
            Line::Rts => "RTS",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct KeyerConfig {
    /// Serial port whose control line keys the transmitter. Empty: test
    /// mode, nothing is keyed.
    pub port: String,
    pub line: Line,
    /// Hold this line (the other one) as PTT around each transmission,
    /// for radios without break-in.
    pub ptt: Option<Line>,
    pub ptt_lead_ms: f32,
    pub ptt_tail_ms: f32,
}

impl Default for KeyerConfig {
    fn default() -> Self {
        Self { port: String::new(), line: Line::Dtr, ptt: None, ptt_lead_ms: 50.0, ptt_tail_ms: 200.0 }
    }
}

impl KeyerConfig {
    pub fn test_mode(&self) -> bool {
        self.port.trim().is_empty()
    }
}

/// Something the keyer can key: a serial port, nothing (test mode), or a
/// recorder (tests).
pub trait KeyOutput: Send {
    fn set_key(&mut self, down: bool) -> Result<()>;
    fn set_ptt(&mut self, on: bool) -> Result<()>;
}

/// Keys DTR or RTS of a serial port.
pub struct SerialKey {
    port: port::Port,
    key: Line,
    ptt: Option<Line>,
}

impl SerialKey {
    pub fn open(cfg: &KeyerConfig) -> Result<Self> {
        if cfg.ptt == Some(cfg.line) {
            return Err(anyhow!("the PTT line must differ from the key line"));
        }
        let port = port::Port::open(cfg.port.trim()).map_err(|e| anyhow!("cannot open keying port {}: {e:#}", cfg.port.trim()))?;
        Ok(Self { port, key: cfg.line, ptt: cfg.ptt })
    }
}

impl KeyOutput for SerialKey {
    fn set_key(&mut self, down: bool) -> Result<()> {
        self.port.set(self.key, down)
    }
    fn set_ptt(&mut self, on: bool) -> Result<()> {
        match self.ptt {
            Some(l) => self.port.set(l, on),
            None => Ok(()),
        }
    }
}

/// A serial port used only for its DTR and RTS lines.
#[cfg(unix)]
mod port {
    use super::Line;
    use anyhow::{anyhow, Result};
    use std::os::raw::c_int;

    pub struct Port {
        fd: c_int,
    }

    fn check(r: c_int, what: &str) -> Result<()> {
        if r < 0 {
            Err(anyhow!("{what}: {}", std::io::Error::last_os_error()))
        } else {
            Ok(())
        }
    }

    impl Port {
        /// Opening a serial port raises DTR and RTS (the kernel does it,
        /// unless the port was left at 0 baud), and either line may key the
        /// radio: both are dropped in the very next call, before anything
        /// else is configured. Nothing else is configured: no data is sent.
        pub fn open(path: &str) -> Result<Self> {
            let c = std::ffi::CString::new(path)?;
            let fd = unsafe { libc::open(c.as_ptr(), libc::O_RDWR | libc::O_NOCTTY | libc::O_NONBLOCK | libc::O_CLOEXEC) };
            check(fd, "open")?;
            let port = Self { fd };
            let both: c_int = libc::TIOCM_DTR | libc::TIOCM_RTS;
            check(unsafe { libc::ioctl(fd, libc::TIOCMBIC as _, &both) }, "cannot lower DTR/RTS")?;
            // Keep other programs off the port while we hold it.
            unsafe { libc::ioctl(fd, libc::TIOCEXCL as _) };
            Ok(port)
        }

        pub fn set(&mut self, line: Line, on: bool) -> Result<()> {
            let bit: c_int = match line {
                Line::Dtr => libc::TIOCM_DTR,
                Line::Rts => libc::TIOCM_RTS,
            };
            let req = if on { libc::TIOCMBIS } else { libc::TIOCMBIC };
            check(unsafe { libc::ioctl(self.fd, req as _, &bit) }, &format!("cannot set {}", line.label()))
        }
    }

    impl Drop for Port {
        /// Lines low, and the port left at 0 baud: the next open (by this
        /// program or another) then does not raise DTR/RTS.
        fn drop(&mut self) {
            unsafe {
                let both: c_int = libc::TIOCM_DTR | libc::TIOCM_RTS;
                libc::ioctl(self.fd, libc::TIOCMBIC as _, &both);
                let mut t: libc::termios = std::mem::zeroed();
                if libc::tcgetattr(self.fd, &mut t) == 0 {
                    libc::cfsetospeed(&mut t, libc::B0);
                    libc::cfsetispeed(&mut t, libc::B0);
                    libc::tcsetattr(self.fd, libc::TCSANOW, &t);
                }
                libc::ioctl(self.fd, libc::TIOCNXCL as _);
                libc::close(self.fd);
            }
        }
    }
}

#[cfg(not(unix))]
mod port {
    use super::Line;
    use anyhow::{anyhow, Result};
    use std::time::Duration;

    pub struct Port(Box<dyn serialport::SerialPort>);

    impl Port {
        pub fn open(path: &str) -> Result<Self> {
            let mut p = serialport::new(path, 9600).timeout(Duration::from_millis(100)).flow_control(serialport::FlowControl::None).open()?;
            p.write_data_terminal_ready(false)?;
            p.write_request_to_send(false)?;
            Ok(Self(p))
        }

        pub fn set(&mut self, line: Line, on: bool) -> Result<()> {
            match line {
                Line::Dtr => self.0.write_data_terminal_ready(on),
                Line::Rts => self.0.write_request_to_send(on),
            }
            .map_err(|e| anyhow!("cannot set {}: {e}", line.label()))
        }
    }

    impl Drop for Port {
        fn drop(&mut self) {
            let _ = self.0.write_data_terminal_ready(false);
            let _ = self.0.write_request_to_send(false);
        }
    }
}

/// Test mode: keys nothing.
pub struct NoOutput;

impl KeyOutput for NoOutput {
    fn set_key(&mut self, _: bool) -> Result<()> {
        Ok(())
    }
    fn set_ptt(&mut self, _: bool) -> Result<()> {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyEvent {
    Key(bool),
    Ptt(bool),
}

/// Records every line change with its time (tests, `cwtool keyer`).
#[derive(Clone, Default)]
pub struct Recorder(pub Arc<Mutex<Vec<(Instant, KeyEvent)>>>);

impl KeyOutput for Recorder {
    fn set_key(&mut self, down: bool) -> Result<()> {
        self.0.lock().push((Instant::now(), KeyEvent::Key(down)));
        Ok(())
    }
    fn set_ptt(&mut self, on: bool) -> Result<()> {
        self.0.lock().push((Instant::now(), KeyEvent::Ptt(on)));
        Ok(())
    }
}

/// What the keyer is doing, for the display.
#[derive(Clone, Debug)]
pub struct KeyerState {
    /// Text of the current transmission: sent, being sent and queued.
    pub text: String,
    /// Bytes of `text` already sent.
    pub sent: usize,
    /// Start of the symbol being sent (`sent..next` is on the air).
    pub next: usize,
    pub key_down: bool,
    pub ptt: bool,
    pub port_open: bool,
    pub error: Option<String>,
    /// Text sent since the last `take_sent` (for the conversation log).
    sent_new: String,
    /// The log ends in a space (so a cut-off transmission is separated
    /// from the next).
    logged_space: bool,
    abort: bool,
}

impl Default for KeyerState {
    fn default() -> Self {
        Self { text: String::new(), sent: 0, next: 0, key_down: false, ptt: false, port_open: false, error: None, sent_new: String::new(), logged_space: true, abort: false }
    }
}

impl KeyerState {
    /// Text queued and not yet started.
    pub fn pending(&self) -> &str {
        &self.text[self.next..]
    }
    pub fn busy(&self) -> bool {
        self.next < self.text.len() || self.sent < self.next || self.ptt || self.key_down
    }
}

enum Cmd {
    Wake,
    Stop,
}

type Opener = Box<dyn FnMut() -> Result<Box<dyn KeyOutput>> + Send>;

pub struct Keyer {
    tx: Sender<Cmd>,
    state: Arc<Mutex<KeyerState>>,
    timing: Arc<Mutex<KeyTiming>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Keyer {
    /// A keyer for this configuration: the serial port, or test mode. The
    /// port is not opened until there is something to send.
    pub fn start(cfg: KeyerConfig, timing: KeyTiming) -> Self {
        let c = cfg.clone();
        let opener: Opener = if cfg.test_mode() {
            Box::new(|| Ok(Box::new(NoOutput) as Box<dyn KeyOutput>))
        } else {
            Box::new(move || Ok(Box::new(SerialKey::open(&c)?) as Box<dyn KeyOutput>))
        };
        Self::with_output(opener, cfg, timing)
    }

    pub fn with_output(opener: Opener, cfg: KeyerConfig, timing: KeyTiming) -> Self {
        let (tx, rx) = crossbeam_channel::unbounded();
        let state = Arc::new(Mutex::new(KeyerState::default()));
        let timing = Arc::new(Mutex::new(timing));
        let (st, tm) = (state.clone(), timing.clone());
        let thread = std::thread::Builder::new().name("keyer".into()).spawn(move || run(opener, cfg, rx, st, tm)).ok();
        Self { tx, state, timing, thread }
    }

    /// Queues text (sent after anything already queued).
    pub fn send(&self, text: &str) {
        let text: String = text.chars().map(|c| if c.is_whitespace() { ' ' } else { c.to_ascii_uppercase() }).collect();
        if text.is_empty() {
            return;
        }
        {
            let mut s = self.state.lock();
            if !s.busy() {
                s.text.clear();
                s.sent = 0;
                s.next = 0;
            }
            s.text.push_str(&text);
            s.error = None;
        }
        let _ = self.tx.send(Cmd::Wake);
    }

    /// Removes up to `chars` characters from the end of the queue, if not
    /// yet started. Returns how many were removed.
    pub fn retract(&self, chars: usize) -> usize {
        let mut s = self.state.lock();
        let mut n = 0;
        while n < chars && s.text.len() > s.next {
            s.text.pop();
            n += 1;
        }
        n
    }

    /// Stops sending at once (key up) and drops the queue.
    pub fn abort(&self) {
        self.state.lock().abort = true;
        let _ = self.tx.send(Cmd::Wake);
    }

    /// New timing, used from the next character on.
    pub fn set_timing(&self, t: KeyTiming) {
        *self.timing.lock() = t;
    }

    pub fn snapshot(&self) -> KeyerState {
        self.state.lock().clone()
    }

    /// Text sent since the last call.
    pub fn take_sent(&self) -> String {
        std::mem::take(&mut self.state.lock().sent_new)
    }
}

impl Drop for Keyer {
    fn drop(&mut self) {
        self.state.lock().abort = true;
        let _ = self.tx.send(Cmd::Stop);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

struct Stopped;

/// Line changes later than this re-time the rest of the character from
/// now (the thread was held up), so one delay cannot squeeze the elements
/// after it.
const LATE: Duration = Duration::from_millis(8);

struct Run {
    rx: Receiver<Cmd>,
    st: Arc<Mutex<KeyerState>>,
    out: Option<Box<dyn KeyOutput>>,
    key_down: bool,
    ptt_on: bool,
    /// The keyer is being dropped.
    stopping: bool,
}

impl Run {
    /// Waits until `t`; Err when stopping or aborting. Sleeps until just
    /// before the deadline, then spins for the last stretch for accuracy.
    fn wait_until(&mut self, t: Instant) -> Result<(), Stopped> {
        loop {
            if self.st.lock().abort {
                return Err(Stopped);
            }
            let now = Instant::now();
            if now >= t {
                return Ok(());
            }
            let left = t - now;
            if left > Duration::from_micros(1500) {
                match self.rx.recv_timeout(left - Duration::from_micros(1000)) {
                    Ok(Cmd::Stop) | Err(RecvTimeoutError::Disconnected) => {
                        self.stopping = true;
                        return Err(Stopped);
                    }
                    Ok(Cmd::Wake) | Err(RecvTimeoutError::Timeout) => {}
                }
            } else {
                std::thread::yield_now();
            }
        }
    }

    fn key(&mut self, down: bool) -> Result<()> {
        if let Some(o) = &mut self.out {
            o.set_key(down)?;
        }
        self.key_down = down;
        self.st.lock().key_down = down;
        Ok(())
    }

    fn ptt(&mut self, on: bool) -> Result<()> {
        if let Some(o) = &mut self.out {
            o.set_ptt(on)?;
        }
        self.ptt_on = on;
        self.st.lock().ptt = on;
        Ok(())
    }

    /// Key and PTT up, queue dropped. If the port fails, it is closed
    /// (closing lowers the lines too).
    fn release(&mut self, err: Option<String>) {
        let ok = self.key(false).is_ok() & self.ptt(false).is_ok();
        if !ok || err.is_some() {
            self.out = None;
        }
        self.key_down = false;
        self.ptt_on = false;
        let mut s = self.st.lock();
        s.key_down = false;
        s.ptt = false;
        s.abort = false;
        let n = s.next;
        s.text.truncate(n);
        s.sent = n;
        if !s.logged_space {
            s.sent_new.push(' ');
            s.logged_space = true;
        }
        s.port_open = self.out.is_some();
        if err.is_some() {
            s.error = err;
        }
    }
}

fn run(mut opener: Opener, cfg: KeyerConfig, rx: Receiver<Cmd>, st: Arc<Mutex<KeyerState>>, timing: Arc<Mutex<KeyTiming>>) {
    let mut r = Run { rx, st, out: None, key_down: false, ptt_on: false, stopping: false };
    let use_ptt = cfg.ptt.is_some();
    let lead = Duration::from_secs_f32(cfg.ptt_lead_ms.clamp(0.0, 1000.0) / 1000.0);
    let tail = Duration::from_secs_f32(cfg.ptt_tail_ms.clamp(0.0, 3000.0) / 1000.0);
    // Earliest start of the next mark (end of the last gap), and the end of
    // the last mark.
    let mut free_at = Instant::now();
    let mut last_up = Instant::now();
    let mut after_space = true;
    loop {
        if r.stopping {
            break;
        }
        if r.st.lock().abort {
            r.release(None);
            free_at = Instant::now();
            after_space = true;
            continue;
        }
        let next = {
            let s = r.st.lock();
            next_symbol(&s.text[s.next..], false)
        };
        let (len, sym) = match next {
            Next::Sym { len, sym } => (len, sym),
            Next::Empty | Next::Wait => {
                // Idle: drop PTT after the tail, then wait for text.
                let release_at = r.ptt_on.then(|| last_up + tail);
                let msg = match release_at {
                    Some(t) if Instant::now() >= t => {
                        if let Err(e) = r.ptt(false) {
                            r.release(Some(format!("{e:#}")));
                        }
                        continue;
                    }
                    Some(t) => r.rx.recv_deadline(t).map_err(|e| e == RecvTimeoutError::Disconnected),
                    None => r.rx.recv().map_err(|_| true),
                };
                match msg {
                    Ok(Cmd::Stop) | Err(true) => break,
                    _ => continue,
                }
            }
        };
        // Something to send: open the port now.
        if r.out.is_none() && !matches!(sym, Sym::Skip) {
            match opener() {
                Ok(o) => {
                    r.out = Some(o);
                    r.st.lock().port_open = true;
                }
                Err(e) => {
                    r.release(Some(format!("{e:#}")));
                    continue;
                }
            }
        }
        let start = {
            let mut s = r.st.lock();
            let a = s.next;
            s.next += len;
            a
        };
        let t = timing.lock().clone();
        let result = match &sym {
            Sym::Skip => Ok(()),
            Sym::Space => {
                let d = Duration::from_secs_f64(t.space_ms(after_space) / 1000.0);
                free_at = (free_at + d).max(Instant::now());
                after_space = true;
                Ok(())
            }
            Sym::Code(p) => {
                after_space = false;
                let mut res = Ok(());
                if use_ptt && !r.ptt_on {
                    match r.ptt(true) {
                        Ok(()) => free_at = free_at.max(Instant::now() + lead),
                        Err(e) => res = Err(Some(format!("{e:#}"))),
                    }
                }
                let mut at = free_at.max(Instant::now());
                for (down, ms) in t.elements(p) {
                    if res.is_err() {
                        break;
                    }
                    if let Err(Stopped) = r.wait_until(at) {
                        res = Err(None);
                        break;
                    }
                    let now = Instant::now();
                    if now > at + LATE {
                        at = now;
                    }
                    if down != r.key_down {
                        if let Err(e) = r.key(down) {
                            res = Err(Some(format!("{e:#}")));
                            break;
                        }
                    }
                    if !down {
                        last_up = at;
                    }
                    at += Duration::from_secs_f64(ms / 1000.0);
                }
                free_at = at;
                res
            }
        };
        match result {
            Ok(()) => {
                let mut s = r.st.lock();
                let end = start + len;
                let done = s.text.get(start..end).unwrap_or("").to_string();
                s.sent = end;
                if !matches!(sym, Sym::Skip) {
                    s.logged_space = matches!(sym, Sym::Space);
                    s.sent_new.push_str(&done);
                }
            }
            Err(e) => {
                // Aborted or failed mid-character: the character is lost.
                r.st.lock().next = start;
                r.release(e);
                free_at = Instant::now();
                after_space = true;
            }
        }
    }
    r.release(None);
    r.out = None;
    r.st.lock().port_open = false;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn total(v: &[(bool, f64)]) -> f64 {
        v.iter().map(|e| e.1).sum()
    }

    #[test]
    fn paris_is_fifty_dits() {
        let t = KeyTiming::default();
        let ms = total(&t.timeline("PARIS "));
        assert!((ms - 50.0 * 60.0).abs() < 1e-6, "{ms}");
        assert!((t.effective_wpm() - 20.0).abs() < 1e-9);
    }

    #[test]
    fn weight_and_comp_keep_the_character_length() {
        let base = KeyTiming::default();
        for t in [KeyTiming { weight: 60.0, ..base.clone() }, KeyTiming { comp_ms: 8.0, ..base.clone() }] {
            let a = t.elements(".-.");
            let b = base.elements(".-.");
            assert!((total(&a) - total(&b)).abs() < 1e-9);
            assert!(a[0].1 > b[0].1 && a[1].1 < b[1].1);
        }
    }

    #[test]
    fn spacing_stretches_only_the_gaps() {
        let t = KeyTiming { letter_gap: 6.0, word_gap: 14.0, ..Default::default() };
        let v = t.timeline("EE E");
        let dit = t.dit_ms();
        assert_eq!(v, vec![(true, dit), (false, 6.0 * dit), (true, dit), (false, 14.0 * dit), (true, dit), (false, 6.0 * dit)]);
        assert!(t.effective_wpm() < 15.0);
    }

    #[test]
    fn symbols_and_prosigns() {
        let s = |t: &str| next_symbol(t, false);
        assert_eq!(s("<AR> K"), Next::Sym { len: 4, sym: Sym::Code(".-.-.".into()) });
        assert_eq!(s("<SK>"), Next::Sym { len: 4, sym: Sym::Code("...-.-".into()) });
        assert_eq!(s("=X"), Next::Sym { len: 1, sym: Sym::Code("-...-".into()) });
        assert_eq!(s("<A"), Next::Wait);
        assert_eq!(next_symbol("<A", true), Next::Sym { len: 1, sym: Sym::Skip });
        assert_eq!(s("<A B>"), Next::Sym { len: 1, sym: Sym::Skip });
        assert_eq!(s("!"), Next::Sym { len: 1, sym: Sym::Skip });
        assert_eq!(s("é"), Next::Sym { len: 2, sym: Sym::Skip });
        assert_eq!(s(" "), Next::Sym { len: 1, sym: Sym::Space });
        assert_eq!(s(""), Next::Empty);
    }

    fn recorded(timing: KeyTiming, cfg: KeyerConfig) -> (Keyer, Recorder) {
        let rec = Recorder::default();
        let r2 = rec.clone();
        let k = Keyer::with_output(Box::new(move || Ok(Box::new(r2.clone()) as Box<dyn KeyOutput>)), cfg, timing);
        (k, rec)
    }

    fn wait_idle(k: &Keyer, max: Duration) {
        let t0 = Instant::now();
        while k.snapshot().busy() && t0.elapsed() < max {
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// The real-time thread keys what the timeline says, within a few ms.
    #[test]
    fn keys_on_time() {
        let timing = KeyTiming { wpm: 30.0, ..Default::default() };
        let (k, rec) = recorded(timing.clone(), KeyerConfig::default());
        k.send("cq de nr5p");
        wait_idle(&k, Duration::from_secs(10));
        let ev = rec.0.lock().clone();
        let keys: Vec<(Instant, bool)> = ev.iter().filter_map(|(t, e)| match e {
            KeyEvent::Key(d) => Some((*t, *d)),
            _ => None,
        }).collect();
        // Expected line changes: merge the timeline into mark/space runs.
        let tl = timing.timeline("CQ DE NR5P");
        assert_eq!(keys.len(), tl.len(), "one change per element edge");
        for ((_, down), (edown, _)) in keys.iter().zip(&tl) {
            assert_eq!(down, edown);
        }
        // Each mark and gap as long as planned. Compared element by element
        // (not from the start) so one scheduling stall on a busy machine,
        // which the keyer re-times around, fails at most two of them.
        let mut errs: Vec<f64> = keys.windows(2).zip(&tl).map(|(w, (_, ms))| ((w[1].0 - w[0].0).as_secs_f64() * 1000.0 - ms).abs()).collect();
        errs.sort_by(|a, b| a.total_cmp(b));
        let median = errs[errs.len() / 2];
        let p95 = errs[errs.len() * 95 / 100];
        assert!(median < 0.5 && p95 < 2.0, "element length error: median {median:.2} ms, 95% {p95:.2} ms");
        assert_eq!(k.take_sent(), "CQ DE NR5P");
        assert!(!k.snapshot().key_down);
    }

    #[test]
    fn abort_keys_up_and_drops_the_queue() {
        let (k, rec) = recorded(KeyTiming { wpm: 10.0, ..Default::default() }, KeyerConfig::default());
        k.send("TTTTTTTT");
        std::thread::sleep(Duration::from_millis(60));
        assert!(k.snapshot().key_down, "a 360 ms dah is on");
        k.abort();
        std::thread::sleep(Duration::from_millis(30));
        let s = k.snapshot();
        assert!(!s.key_down && !s.busy(), "{s:?}");
        assert_eq!(s.pending(), "");
        let last_key = rec.0.lock().iter().rev().find(|e| matches!(e.1, KeyEvent::Key(_))).map(|e| e.1);
        assert_eq!(last_key, Some(KeyEvent::Key(false)));
        // The aborted character is not logged as sent.
        assert_eq!(k.take_sent(), "");
        // Sending works again afterwards.
        k.send("E");
        wait_idle(&k, Duration::from_secs(2));
        assert_eq!(k.take_sent(), "E");
    }

    #[test]
    fn retract_only_removes_unsent_text() {
        let (k, _rec) = recorded(KeyTiming { wpm: 10.0, ..Default::default() }, KeyerConfig::default());
        k.send("TEST");
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(k.retract(10), 3, "the T on the air stays");
        wait_idle(&k, Duration::from_secs(2));
        assert_eq!(k.take_sent(), "T");
        k.abort();
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(k.take_sent(), " ", "an abort separates the next transmission in the log");
        k.abort();
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(k.take_sent(), "", "once");
    }

    #[test]
    fn ptt_wraps_the_transmission() {
        let cfg = KeyerConfig { ptt: Some(Line::Rts), ptt_lead_ms: 40.0, ptt_tail_ms: 100.0, ..Default::default() };
        let (k, rec) = recorded(KeyTiming { wpm: 30.0, ..Default::default() }, cfg);
        k.send("EE");
        wait_idle(&k, Duration::from_secs(3));
        let ev = rec.0.lock().clone();
        let kinds: Vec<KeyEvent> = ev.iter().map(|e| e.1).collect();
        assert_eq!(
            kinds,
            vec![KeyEvent::Ptt(true), KeyEvent::Key(true), KeyEvent::Key(false), KeyEvent::Key(true), KeyEvent::Key(false), KeyEvent::Ptt(false)]
        );
        let ms = |a: usize, b: usize| (ev[b].0 - ev[a].0).as_secs_f64() * 1000.0;
        assert!((ms(0, 1) - 40.0).abs() < 5.0, "lead {}", ms(0, 1));
        assert!((ms(4, 5) - 100.0).abs() < 8.0, "tail {}", ms(4, 5));
    }

    #[test]
    fn open_failure_is_reported_and_nothing_queued() {
        let k = Keyer::with_output(Box::new(|| Err(anyhow!("no such port"))), KeyerConfig::default(), KeyTiming::default());
        k.send("CQ");
        wait_idle(&k, Duration::from_secs(1));
        let s = k.snapshot();
        assert!(s.error.as_deref().is_some_and(|e| e.contains("no such port")), "{s:?}");
        assert!(!s.busy() && !s.port_open);
    }

    /// The real serial-port code against a pseudo-terminal (no radio
    /// involved): a pty has no modem lines, so opening it must fail
    /// cleanly with the reason, and a missing port must too.
    #[cfg(unix)]
    #[test]
    fn serial_port_errors() {
        let mut master: std::os::raw::c_int = 0;
        let mut slave: std::os::raw::c_int = 0;
        let mut name = [0 as std::os::raw::c_char; 128];
        let r = unsafe { libc::openpty(&mut master, &mut slave, name.as_mut_ptr(), std::ptr::null(), std::ptr::null()) };
        assert_eq!(r, 0);
        let path = unsafe { std::ffi::CStr::from_ptr(name.as_ptr()) }.to_str().unwrap().to_string();
        assert!(path.starts_with("/dev/pts/"), "{path}");
        let cfg = KeyerConfig { port: path.clone(), ..Default::default() };
        match SerialKey::open(&cfg) {
            Err(e) => assert!(e.to_string().contains("DTR/RTS"), "{e}"),
            Ok(_) => {} // a pty that accepts modem-line ioctls is fine too
        }
        let k = Keyer::start(cfg, KeyTiming::default());
        k.send("E");
        wait_idle(&k, Duration::from_secs(1));
        drop(k);
        unsafe {
            libc::close(slave);
            libc::close(master);
        }
        let e = SerialKey::open(&KeyerConfig { port: "/dev/does-not-exist".into(), ..Default::default() }).err().unwrap();
        assert!(e.to_string().contains("No such file"), "{e}");
        let e = SerialKey::open(&KeyerConfig { port: path, ptt: Some(Line::Dtr), ..Default::default() }).err().unwrap();
        assert!(e.to_string().contains("PTT line must differ"), "{e}");
    }

    /// A port whose line cannot be set stops with an error and closes,
    /// rather than hanging or retrying.
    #[cfg(unix)]
    #[test]
    fn line_failure_closes_the_port() {
        struct Broken;
        impl KeyOutput for Broken {
            fn set_key(&mut self, _: bool) -> Result<()> {
                Err(anyhow!("cannot set DTR"))
            }
            fn set_ptt(&mut self, _: bool) -> Result<()> {
                Ok(())
            }
        }
        let k = Keyer::with_output(Box::new(|| Ok(Box::new(Broken) as Box<dyn KeyOutput>)), KeyerConfig::default(), KeyTiming::default());
        k.send("TEST");
        wait_idle(&k, Duration::from_secs(1));
        let s = k.snapshot();
        assert!(s.error.as_deref().is_some_and(|e| e.contains("DTR")), "{s:?}");
        assert!(!s.port_open && !s.busy());
    }
}
