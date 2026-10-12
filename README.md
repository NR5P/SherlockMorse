# CW Skimmer (Rust)

A cross-platform, multi-channel CW (Morse code) skimmer in the spirit of
VE3NEA's CW Skimmer. It finds **every CW signal in your receiver's audio
passband**, decodes them all at the same time, pulls out callsigns, and can
retune your radio over CAT with a click.

Runs on Linux, Windows and macOS (single native binary, no runtime needed).

## Features

- **Decodes the whole passband at once.** A spectrum detector finds keyed
  carriers (it ignores steady carriers and birdies) and starts an
  independent decoder on each one. A 1.5 s audio pre-roll means the first
  characters of a transmission aren't lost.
- **Probabilistic decoder.** Dots and dashes are never perfect, so nothing is
  decided by fixed thresholds:
  - Element timing is scored with log-normal likelihoods. The speed comes from
    a robust maximum-likelihood search and is then tracked adaptively, which
    handles hand-key "fists", odd weighting, Farnsworth spacing and speed drift.
  - A **beam search** keeps dozens of competing interpretations alive. It
    includes hypotheses that a mark was a static crash, or that a gap was a
    QSB dropout in the middle of a dash.
  - A **character trigram language model** of amateur traffic (QSOs, contest
    exchanges, callsigns, plain English) picks the most plausible text. Text
    is only committed once all surviving hypotheses agree on it.
- **Waterfall** with callsign labels on each signal. Click a signal to follow
  it; double-click or right-click to tune the radio to it; the mouse wheel
  tunes ±50 Hz.
- **Spots list** of callsigns heard (after `DE`/`CQ`, or seen twice), with RF
  frequency, speed and SNR. Click a spot to tune to it.
- **Decoders follow the radio.** When you QSY, decoders move with their
  signals in the passband instead of being lost.
- **Sound card selection with auto-detect.** The radio's USB audio codec
  (Yaesu/Icom/Kenwood "USB Audio CODEC", SignaLink, Digirig, CM108…) is
  picked automatically. Any input works (USB, line-in, microphone), and you
  can use the left, right or mixed channel.
- **CAT control for many radios**:
  | Protocol | Radios |
  |---|---|
  | Yaesu CAT | **FTDX3000**, FTDX5000, FTDX1200, FT-950, FT-2000, FT-991/A, FT-891, FT-710, FTDX10, FTDX101 |
  | Kenwood / Elecraft | TS-480, TS-590S/SG, TS-890, TS-990, TS-2000, K3, K4, KX2, KX3 |
  | Icom CI-V | IC-7300, IC-7610, IC-705, IC-9700, IC-7100, IC-7410, IC-7600, IC-7760 … |
  | Yaesu legacy | FT-817/818, FT-857, FT-897 |
  | Hamlib `rigctld` | anything Hamlib supports (hundreds of radios) |

  **Auto-detect** probes USB serial ports, identifies the radio model (for
  example `ID0462;` means FTDX3000), and picks the right protocol and baud
  rate.
- **Audio files** (WAV/MP3/FLAC/OGG/M4A) can be played through the skimmer
  for practice or testing.

## Yaesu FTDX3000 setup

1. Connect the radio's rear **USB** port to the computer. That one cable
   carries both the sound card and the CAT port.
   - **Windows**: install the Yaesu "Virtual COM Port" (Silicon Labs
     CP210x) driver. Two COM ports appear; use the **Enhanced** COM port
     for CAT.
   - **Linux**: you get `/dev/ttyUSB0` (Enhanced/CAT) and `/dev/ttyUSB1`.
     Add yourself to the `dialout` group (`sudo usermod -aG dialout $USER`,
     then log out and back in).
2. Radio menus (press and hold MENU):
   - **CAT RATE**: 38400 (the default is 4800; any rate works, as long as the
     program uses the same one).
   - **CAT RTS**: either setting works; the program asserts RTS.
   - **CW PITCH**: note the value (default 700 Hz) and enter it under
     ⚙ Settings → CW pitch so that click-to-tune lands signals exactly.
   - USB audio output level (**USB OUT LEVEL** in the CW/DATA menus) about
     50, so the input meter peaks well below 0 dBFS.
3. In the program, under **Radio**, choose *Yaesu CAT*, select the port, set
   38400, and press **Connect**. Or just press **Auto-detect**.
4. Under **Audio**, press **Auto**. The radio's "USB Audio CODEC" is
   selected automatically.
5. Put the radio in CW with a wide filter (2.4–3 kHz) so the skimmer sees
   the whole passband.

## Building

Install Rust from <https://rustup.rs>, then:

```bash
# Linux only: system libraries
sudo apt install build-essential pkg-config libasound2-dev libudev-dev libxkbcommon-dev libwayland-dev libgl1-mesa-dev

cargo build --release
./target/release/cwskimmer
```

On Windows and macOS, `cargo build --release` is all you need.

## Command-line tool

`cwtool` handles testing and scripting:

```bash
cwtool devices                      # list sound cards (★ = detected radio)
cwtool ports                        # list serial ports
cwtool probe                        # auto-detect radios on USB serial
cwtool rig --kind yaesu --port /dev/ttyUSB0 --baud 38400            # read VFO
cwtool rig --kind yaesu --port /dev/ttyUSB0 --set 14025000          # QSY
cwtool decode recording.mp3         # decode every signal in a recording
cwtool decode w1aw.mp3 --reference w1aw.txt   # measure character error rate
cwtool synth --pileup -o band.wav   # generate a 6-station test scene
cwtool bench --compare-lm           # synthetic benchmark suite
```

## How well it decodes

`cwtool bench` runs the full skimmer (detection plus decoding) on synthetic
signals with realistic impairments. The score is character error rate
(Levenshtein distance); SNR is measured in a 2500 Hz bandwidth.

| Scenario | CER |
|---|---|
| Clean, 20 WPM | 0.0 % |
| SNR +5 dB, 25 WPM | 0.0 % |
| SNR 0 dB, 22 WPM | 0.8 % |
| SNR −4 dB, 18 WPM | ~16 % |
| Hand key, 15 % timing jitter | 0.7 % |
| Bad fist: 25 % jitter, 3.8:1 dashes, heavy weighting | 7.3 % |
| Farnsworth 18/8 WPM | 3.3 % |
| Deep QSB (−16 dB fades) | 22 % |
| Static crashes (QRN) | 1.6 % |
| Speed drifting +40 % | 0.0 % |
| 40 WPM / 10 WPM | 0.0 % / 0.0 % |
| Random 5-letter groups (LM can't help) | 0.0 % |
| 5-station pileup, 80–240 Hz apart | 0.4 % |

Turning the language model off (`--compare-lm`) raises the error rate on clean
signals from about 0 % to about 20 %, so the text prediction matters.

## Architecture

```
sound card / file ─► resampler (8 kHz) ─► FFT 512 ─► waterfall + signal detector
                                     │
                                     └─► per signal: mixer ─► 4th-order LPF (matched to speed, AFC)
                                              ─► envelope ─► adaptive slicer ─► mark/space runs
                                              ─► timing model + beam search + trigram LM ─► text ─► callsigns
```

| Module | Purpose |
|---|---|
| `src/decoder.rs` | slicer, timing model, beam-search decoder |
| `src/lm.rs` | trigram language model (trained at start-up from a generated corpus) |
| `src/skimmer.rs` | signal detection, channel lifecycle, spots |
| `src/dsp/` | resampler, channel demodulator, spectrum |
| `src/cat/` | radio protocols, auto-detect, polling thread |
| `src/audio.rs` | sound card capture and auto-selection |
| `src/synth.rs` | impaired-CW synthesizer used by tests and the benchmark |
| `src/main.rs` | egui user interface |

## License

MIT
