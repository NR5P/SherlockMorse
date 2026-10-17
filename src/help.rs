//! Built-in user guide, shown in the Help window.
//!
//! Each topic is plain text: "## " starts a subheading, "- " a bullet, a
//! blank line a new paragraph; other lines run on into the paragraph.

pub const TOPICS: &[(&str, &str)] = &[
    (
        "Getting started",
        "Sherlock Morse finds every CW signal in your receiver's audio, decodes them all at once, picks out callsigns and can tune your radio with a click.

## Quick start
- Connect the radio's USB cable (or any audio interface) to the computer.
- Under Audio, press Auto. The radio's USB sound card is picked automatically; the level bar should sit in the green with no CLIP.
- Under Radio, press Auto-detect (or choose the protocol, port and baud rate and press Connect) so the program can read and set the frequency.
- Put the radio in CW with a wide filter (2.4 to 3 kHz) so the whole passband reaches the program.
- Check ⚙ Settings > CW pitch matches the radio's CW pitch (read automatically from Yaesu radios).

Signals appear on the waterfall with their speed, and each one gets its own decoder row below it. Callsigns go to the Spots list on the right.

## Without a radio
Type the path of a recording (WAV, MP3, FLAC, OGG or M4A) in the File box and press ▶ Play. Live goes back to the sound card. The radio connection is optional: without it everything works except tuning, and frequencies are shown as audio Hz.",
    ),
    (
        "Waterfall and decoders",
        "The waterfall shows the audio passband: low pitches on the left, high on the right. Keyed signals are labelled with their speed (or callsign once known).

## Mouse
- Click a signal: start a decoder on it (if there is none) and follow it in the Selected signal tab.
- Double-click or right-click: tune the radio to that signal (it is zero-beated: moved to your CW pitch, then checked and corrected).
- Mouse wheel: tune the radio up or down 50 Hz.

## Decoder rows
Each signal has a row with its frequency, callsign, speed (WPM), signal-to-noise ratio and the text decoded so far. Text still being decided is shown grey; it turns white once all the decoder's guesses agree.
- Click a row to tune the radio to that station.
- Click a callsign for station information.
- ✖ removes the decoder.
- A dimmed row reads poorly (weak or garbled signal).

Rows stay visible through a fade for the Row dropout time (Settings) and are hidden after that.",
    ),
    (
        "Spots and station info",
        "The Spots list collects the callsigns heard: after CQ or DE, in an exchange, or heard more than once. Each shows its frequency, speed, SNR, how long ago it was heard and its country.

- Calls heard only once are dimmed: they may be decoding errors. They brighten when heard again.
- Click a call for station information; double-click it or click its frequency to tune there.
- Hover the country for continent, CQ and ITU zones, local time and (with your grid set) distance and beam heading.
- Coloured flags mark stations in a contest or on-air programme (POTA, SOTA, SKCC, CWT …). The Show menu filters the list by flag.

## Station information
Clicking any callsign (spots, decoder rows, decoded text) opens a window with the station's name, location, grid, distance and photo when available. US and Canadian calls work without an account (HamDB). A free QRZ.com or HamQTH.com login, entered in Settings, adds stations worldwide and photos. Buttons open the call on QRZ.com or HamQTH.

## Countries
The country comes from the built-in AD1C country file (cty.dat), so it works offline. Settings > Country file > Update downloads the newest copy.",
    ),
    (
        "Conversation and history",
        "The panel under the decoder rows has three tabs.

## Conversation
Everything decoded on your own frequency (within 75 Hz of the CW pitch) is collected here, one line per over, with the UTC time. If you transmit with the program, your own text appears as TX: lines. The conversation is saved to a text file as it grows and is archived when you move away from the frequency for 30 seconds. End / new closes it now; Tune back returns to its frequency.

## Selected signal
The full text of the signal you clicked on the waterfall or in the decoder list.

## History
The last 20 conversations. Click one to read it, double-click to tune to its frequency.

Abbreviations (QTH, RST, TNX …) are highlighted: hover them for their meaning. Callsigns in the text are links to station information. The Text size slider sets the size of this text.",
    ),
    (
        "Recording and logs",
        "- ● Rec audio records the radio audio to a WAV file, together with a log of what was on the screen. Press again to stop.
- Logs opens the folder with the recordings, the decoded text of every station and the saved conversations (CwSkimmer-logs in your home folder).
- Clear empties the decoders and the spots list.",
    ),
    (
        "Radio connection",
        "Supported protocols: Yaesu CAT (FTDX3000, FTDX101, FT-991, FT-891, FT-710 …), Kenwood and Elecraft, Icom CI-V, Yaesu legacy (FT-817/818/857/897), and Hamlib rigctld for any radio Hamlib supports.

- Auto-detect tries the USB serial ports and identifies the radio and baud rate. It never touches a radio's second (PTT/keying) port.
- The frequency buttons step the VFO; the band buttons jump to the CW end of each band; type kHz in the box and press Enter to go to a frequency.
- When you change frequency, decoders move with their signals instead of being lost.
- Without CAT, frequencies are shown as audio Hz and tuning is not available.

## Yaesu FTDX3000
- One USB cable carries the sound card and two serial ports. Use the first (Enhanced) port, /dev/ttyUSB0 on Linux, for CAT, at the CAT RATE set in the radio's menu (38400 is a good choice).
- The second (Standard) port can key the transmitter: the program labels it and will not use it for CAT.
- Set the radio's USB audio level so the input bar sits in the green.",
    ),
    (
        "Settings",
        "⚙ Settings opens the settings panel on the left. Decoder settings apply to signals detected after the change.

- CW pitch: must match the radio's CW pitch so clicking tunes exactly. With auto ticked it is read from the radio.
- Passband: the audio range searched for signals.
- Detection threshold: how far above the noise a keyed signal must be to get a decoder. Lower finds weaker signals and more junk.
- Max decoders: the most signals decoded at once.
- Row dropout: how long a station's row stays through a fade.
- Sensitive slicer (experimental): copies more of weak and fluttering signals, but shows more junk rows.
- Squelch: lower decodes weaker signals but more noise garbage.
- Text prediction: how much the language model may correct the timing. 0 is timing only.
- Show RF frequency: show signals as radio frequencies instead of audio Hz.
- Waterfall brightness and contrast.
- Always drive RTS / DTR high: normally leave both off.
- Station lookup: your grid square, and optional QRZ.com and HamQTH logins.
- Country file: download the newest cty.dat.
- Activity flags: the words that mark a station as POTA, SKCC, a contest and so on. Patterns are separated by commas, may be several words, and * matches anything.
- Transmit: shows the transmit panel and its settings (see Transmitting).",
    ),
    (
        "Transmitting",
        "Sending CW is off and hidden by default. Tick ⚙ Settings > Transmit > Show the transmit panel to use it.

## The transmit panel
- Type a message and press Enter (or Send) to send it.
- Tick Send as I type to send each letter as you type it. Backspace takes back letters not yet sent; Enter adds a word space.
- Esc or ■ Stop stops sending at once.
- The TX light is red while the key is down (orange in test mode).
- The line below shows what was sent (dim), the letter on the air (highlighted) and what is still queued.
- The WPM box changes the speed, also during a message.
- The Call box holds the other station's call. Clicking any callsign fills it.
- Prosigns: <AR> <SK> <BT> <KN> <BK>, or + for AR and = for BT.

## Keying the radio
The program keys the transmitter through the DTR or RTS line of a serial port, which most radios accept. Choose it under Settings > Transmit > Key with. Until you do, the panel runs in test mode: everything works but nothing is keyed. The port is opened only when you first send something, and the line is held low at all other times.

## FTDX3000
- Key with: the radio's second (Standard / keying) port, /dev/ttyUSB1 on Linux. Key line: DTR.
- Radio Menu 065 PC KEYING: DTR.
- The radio transmits only with BK-IN on. With BK-IN off it just plays its sidetone, which is a safe way to check the keying.

## Radios without break-in
Choose the other line as PTT. It is held through each transmission, with adjustable lead and tail times.

## Timing
- Speed (WPM).
- Weight: 50% is standard; more lengthens every dit and dah and shortens the gaps by the same time.
- Dah length: 3 dits is standard.
- Letter gap and word gap: 3 and 7 dits are standard. Longer gaps at the same speed is Farnsworth spacing; the overall speed is shown.
- Key compensation: added to every element for a radio whose break-in shortens them. Usually 0.
- Default timing puts all of these back to standard.",
    ),
    (
        "Memory keyer",
        "The memory buttons in the transmit panel send stored messages: click one or press its function key (F1 to F12).

## Editing memories
Right-click a memory button, or click ✏, to open the Keyer memories window (also Settings > Transmit > Edit keyer memories). There you can change each memory's label and text, add memories (up to 12), delete them, move them up or down, or restore the defaults.

## Placeholders
- {MYCALL}: your call, set in Settings > Transmit.
- {CALL}: the Call box. Clicking a callsign anywhere fills it.

A memory that needs a call that is not set is not sent, and the panel says why.

## Repeating (calling CQ)
Give a memory a repeat time in seconds and it is sent again that many seconds after it ends, until you stop it. Its button shows ⟳ and the panel counts down to the next call. Repeating stops with Esc, ■ Stop, typing in the text box, sending another message, and, with Stop repeating when someone answers ticked, when a station is decoded on your frequency between calls.",
    ),
    (
        "Keyboard",
        "- Esc: stop transmitting (and stop a repeating memory).
- F1 to F12: send keyer memories (when the transmit panel is shown).
- Enter in the transmit box: send the message.
- Enter in the frequency box: tune to the kHz typed.
- Mouse wheel over the waterfall: tune ±50 Hz.",
    ),
    (
        "Troubleshooting",
        "## Nothing on the waterfall
- Check the Audio device (Auto picks the radio's USB codec) and that the level bar moves.
- On the radio, raise the USB audio output level.

## CLIP lights or the bar is red
The sound card is overloaded: lower the radio's USB audio level.

## Signals decoded but clicking tunes to the wrong place
The CW pitch in Settings must match the radio's. Tick auto to read it from the radio.

## The radio does not connect
- Pick the radio's first (CAT) port and the baud rate set in its menu, or press Auto-detect.
- On Linux, your user must be in the dialout group (sudo usermod -aG dialout $USER, then log out and in).
- Another program (logger, WSJT-X) may already have the port open.

## Lots of garbage rows
Raise the Detection threshold or the Squelch, or turn off the Sensitive slicer.

## Transmit does nothing
- The panel says test mode: choose a keying port in Settings > Transmit.
- FTDX3000: Menu 065 PC KEYING must match the key line (DTR), and BK-IN must be on to transmit.
- A red message under the transmit box gives the reason a port could not be opened.",
    ),
];
