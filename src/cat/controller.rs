//! Background thread that owns the radio connection, polls frequency/mode
//! and executes tuning commands.

use super::{open, Mode, RigConfig};
use crossbeam_channel::{Receiver, Sender};
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Default)]
pub struct RigState {
    pub connected: bool,
    pub name: String,
    pub freq: Option<u64>,
    pub mode: Option<Mode>,
    /// CW pitch reported by the radio (Hz).
    pub pitch: Option<u32>,
    pub error: Option<String>,
}

pub enum RigCmd {
    SetFreq(u64),
    SetMode(Mode),
    Stop,
}

pub struct RigController {
    tx: Sender<RigCmd>,
    pub state: Arc<Mutex<RigState>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl RigController {
    pub fn start(cfg: RigConfig) -> Self {
        let (tx, rx) = crossbeam_channel::unbounded();
        let state = Arc::new(Mutex::new(RigState::default()));
        let st = state.clone();
        let thread = std::thread::Builder::new().name("rig".into()).spawn(move || run(cfg, rx, st)).ok();
        Self { tx, state, thread }
    }

    pub fn set_freq(&self, hz: u64) {
        let _ = self.tx.send(RigCmd::SetFreq(hz));
        // Optimistic update so the display reacts immediately.
        self.state.lock().freq = Some(hz);
    }

    pub fn set_mode(&self, m: Mode) {
        let _ = self.tx.send(RigCmd::SetMode(m));
    }

    pub fn snapshot(&self) -> RigState {
        self.state.lock().clone()
    }
}

impl Drop for RigController {
    fn drop(&mut self) {
        let _ = self.tx.send(RigCmd::Stop);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn run(cfg: RigConfig, rx: Receiver<RigCmd>, state: Arc<Mutex<RigState>>) {
    let poll = Duration::from_millis(cfg.poll_ms.max(100));
    loop {
        // (Re)connect.
        let mut rig = match open(&cfg) {
            Ok(r) => {
                let mut s = state.lock();
                s.connected = true;
                s.name = r.name();
                s.error = None;
                r
            }
            Err(e) => {
                {
                    let mut s = state.lock();
                    s.connected = false;
                    s.error = Some(format!("{e:#}"));
                }
                // Wait before retrying, but stay responsive to Stop.
                match rx.recv_timeout(Duration::from_secs(3)) {
                    Ok(RigCmd::Stop) | Err(crossbeam_channel::RecvTimeoutError::Disconnected) => return,
                    _ => continue,
                }
            }
        };
        let mut failures = 0;
        let mut last_poll = Instant::now() - poll;
        let mut mode_counter = 0u32;
        loop {
            let wait = poll.saturating_sub(last_poll.elapsed());
            match rx.recv_timeout(wait) {
                Ok(RigCmd::Stop) | Err(crossbeam_channel::RecvTimeoutError::Disconnected) => return,
                Ok(RigCmd::SetFreq(mut hz)) => {
                    // Coalesce rapid tuning (e.g. mouse wheel).
                    while let Ok(c) = rx.try_recv() {
                        match c {
                            RigCmd::SetFreq(h) => hz = h,
                            RigCmd::SetMode(m) => {
                                let _ = rig.set_mode(&m);
                            }
                            RigCmd::Stop => return,
                        }
                    }
                    if let Err(e) = rig.set_freq(hz) {
                        state.lock().error = Some(format!("{e:#}"));
                    }
                    continue;
                }
                Ok(RigCmd::SetMode(m)) => {
                    if let Err(e) = rig.set_mode(&m) {
                        state.lock().error = Some(format!("{e:#}"));
                    }
                    mode_counter = 0;
                    continue;
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            }
            last_poll = Instant::now();
            match rig.get_freq() {
                Ok(f) => {
                    failures = 0;
                    let mut s = state.lock();
                    s.freq = Some(f);
                    s.connected = true;
                    s.error = None;
                }
                Err(e) => {
                    failures += 1;
                    state.lock().error = Some(format!("{e:#}"));
                    if failures >= 5 {
                        state.lock().connected = false;
                        break; // reconnect
                    }
                }
            }
            if mode_counter % 5 == 0 {
                if let Ok(m) = rig.get_mode() {
                    state.lock().mode = Some(m);
                }
            }
            // Pitch rarely changes; read it at connect and every ~3 s.
            if mode_counter % 10 == 0 {
                if let Ok(Some(p)) = rig.get_pitch() {
                    state.lock().pitch = Some(p);
                }
            }
            mode_counter += 1;
        }
    }
}
