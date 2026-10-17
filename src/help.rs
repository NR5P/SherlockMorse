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
        "⚙ Settings opens the settings panel on the left. Every setting has a small circled ? next to its name: click it to see what the setting does and how to adjust it. All of them are listed below.

Decoder settings apply to signals detected after the change; press Clear to restart every decoder with the new values.

Restore default settings, at the bottom of the panel, puts every setting back to its default after asking. Your call, grid, logins, radio and audio choices, keying port and keyer memories are kept.",
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

/// One setting's explanation: what it does and how to adjust it. Shown
/// by the (?) next to the setting and listed in the Settings help topic.
pub struct Setting {
    pub key: &'static str,
    pub section: &'static str,
    pub label: &'static str,
    pub text: &'static str,
}

pub fn setting(key: &str) -> Option<&'static Setting> {
    SETTINGS.iter().find(|s| s.key == key)
}

pub const SETTINGS: &[Setting] = &[
    Setting {
        key: "pitch",
        section: "Receiving",
        label: "CW pitch",
        text: "The audio tone a signal has when the radio is tuned exactly to it (the radio's CW pitch or sidetone). Clicking a signal to tune moves it to this pitch, and the conversation follows the station on this pitch.

- Set it to the same value as the radio (FTDX3000: Menu 054 CW PITCH, default 700 Hz).
- Tick auto to read it from the radio over CAT (Yaesu); the box is then not needed.
- If clicked stations end up slightly off, this does not match the radio.",
    },
    Setting {
        key: "passband",
        section: "Receiving",
        label: "Passband",
        text: "The range of audio frequencies searched for signals (default 250 to 2800 Hz).

- Match it to the radio's filter: with a 2.4 kHz filter there is nothing useful above about 2700 Hz.
- Raise the lower edge if hum or low-frequency noise makes junk decoders.
- Narrow it to decode only part of the passband.",
    },
    Setting {
        key: "detect_db",
        section: "Receiving",
        label: "Detection threshold",
        text: "How far a keyed signal must rise above the noise (in dB) before a decoder is started on it. Default 12 dB.

- Lower it (8 to 10) to catch weaker signals; you also get more rows of noise and junk.
- Raise it (14 to 18) on a noisy or crowded band to show only the clearer signals.",
    },
    Setting {
        key: "max_channels",
        section: "Receiving",
        label: "Max decoders",
        text: "The most signals decoded at the same time (default 40). Each decoder uses some processor time.

- Lower it on a slow computer, or if the program falls behind the audio.
- Raise it for a crowded contest band.",
    },
    Setting {
        key: "dropout",
        section: "Receiving",
        label: "Row dropout",
        text: "How many seconds a station's row stays on screen while its signal is too weak to read (a fade), before it is hidden. Default 30 s.

- Raise it to keep stations visible through long fades (QSB).
- Lower it (or 0) for a list that only shows what is readable right now.",
    },
    Setting {
        key: "viterbi",
        section: "Receiving",
        label: "Sensitive slicer (experimental)",
        text: "A different way of deciding where each dit and dah starts and ends: it weighs each whole element against the noise instead of deciding moment by moment. Off by default.

- Turn it on for weak or fluttering signals: it copies more of them and spots more calls correctly.
- It also shows more rows of junk, so leave it off on a busy or noisy band.",
    },
    Setting {
        key: "squelch",
        section: "Receiving",
        label: "Squelch",
        text: "How sure the decoder must be that it hears keying before it prints text (default 2.3).

- Lower it (toward 1.6) to read weaker signals; more noise turns into stray letters (E, T, I).
- Raise it (toward 3 to 4) if noise prints garbage between transmissions.",
    },
    Setting {
        key: "lm_weight",
        section: "Receiving",
        label: "Text prediction",
        text: "How much the built-in language model (which knows typical CW words, abbreviations and callsigns) may override the timing when the code is unclear. Default 0.5.

- 0 decodes from timing alone: best for random text such as code groups or unusual languages.
- Higher values clean up poor copy into likely words, but can also turn real oddities into the expected word. Above 1 is rarely better.",
    },
    Setting {
        key: "show_rf",
        section: "Receiving",
        label: "Show RF frequency",
        text: "Show signals by their radio frequency (kHz) when the radio is connected, instead of by their audio tone (Hz). On by default; without CAT, audio Hz is always shown.",
    },
    Setting {
        key: "waterfall",
        section: "Receiving",
        label: "Waterfall brightness / contrast",
        text: "Only the look of the waterfall; decoding is not affected.

- Brightness (left slider): move it right if the waterfall is too dark, left if the noise is too bright.
- Contrast (right slider): move it right to make signals stand out from the noise more sharply.",
    },
    Setting {
        key: "rts_dtr",
        section: "Receiving",
        label: "Always drive RTS / DTR high",
        text: "Keeps the RTS or DTR line of the CAT port switched on. Leave both off.

- Some radios ignore CAT until RTS is on (FTDX3000 with CAT RTS = ENABLE); the program already raises RTS by itself in that case, on the CAT port only.
- On many radios these lines key the transmitter, so only tick one if the radio's manual requires it for CAT.",
    },
    Setting {
        key: "my_grid",
        section: "Station lookup",
        label: "My grid",
        text: "Your Maidenhead grid locator (for example EM12ab). With it set, station information and country hovers show the distance and beam heading to the other station. Four or six characters.",
    },
    Setting {
        key: "logins",
        section: "Station lookup",
        label: "QRZ.com and HamQTH logins",
        text: "Optional accounts for the station information window. US and Canadian calls work without any account (HamDB).

- A free QRZ.com account adds name and country for calls worldwide; a QRZ subscription adds full details and photos.
- A free HamQTH.com account adds worldwide data and photos.
- Enter the username and password of the account; leave them empty if you have none. Passwords are stored unencrypted in the settings file.",
    },
    Setting {
        key: "cty",
        section: "Country file",
        label: "Country file",
        text: "The list that turns a callsign into its country (AD1C's cty.dat, as used by loggers). A copy is built in, so it works offline.

- Press Update now and then (new DXpeditions, changed prefixes) to download the newest copy from country-files.com.",
    },
    Setting {
        key: "activities",
        section: "Activity flags",
        label: "Activity flags",
        text: "Coloured flags on stations that send certain words, such as POTA, SKCC or a contest name. The spots list can be filtered by flag.

- Each flag has a name, a colour and patterns. Patterns are separated by commas, may be several words (CQ POTA), and * matches anything (K-*).
- Untick a flag to hide it, ✖ deletes it, Add makes a new one, Restore defaults brings back the built-in list.",
    },
    Setting {
        key: "tx_enabled",
        section: "Transmit",
        label: "Show the transmit panel",
        text: "Shows the panel for sending CW from the keyboard, with the memory keyer. Off by default: most people only listen. While it is off the panel is hidden and no keying port is ever opened.",
    },
    Setting {
        key: "my_call",
        section: "Transmit",
        label: "My call",
        text: "Your callsign. It replaces {MYCALL} in keyer memories and typed text. Memories that use {MYCALL} are not sent until it is set.",
    },
    Setting {
        key: "key_with",
        section: "Transmit",
        label: "Key with",
        text: "The serial port whose DTR or RTS line keys the transmitter.

- Nothing (test mode): everything works but nothing is keyed. Good for trying the panel.
- FTDX3000: its second (Standard) port, /dev/ttyUSB1 on Linux, marked keying/PTT port. Not the CAT port.
- Other radios: the port of their USB keying interface, or the radio's own port if its manual allows CW keying by DTR/RTS there.
- The port is opened only when you first send something.",
    },
    Setting {
        key: "key_line",
        section: "Transmit",
        label: "Key line",
        text: "Which line of the port keys the CW: DTR or RTS. It must match the radio.

- FTDX3000: DTR, with Menu 065 PC KEYING set to DTR.
- Most interfaces key on DTR; check the interface's manual if nothing happens.",
    },
    Setting {
        key: "ptt_line",
        section: "Transmit",
        label: "PTT line",
        text: "For radios without CW break-in: the other line is held as PTT (transmit) through each transmission.

- none (break-in): the radio switches to transmit by itself when keyed (FTDX3000 with BK-IN on). Use this when the radio has break-in.
- DTR / RTS: choose this if the radio only transmits while its PTT is held.",
    },
    Setting {
        key: "ptt_times",
        section: "Transmit",
        label: "PTT lead / tail",
        text: "With a PTT line: lead is how long PTT is held before the first dit (default 50 ms), tail how long after the last one (default 200 ms).

- Raise the lead if the first dit is clipped (the radio or amplifier relay is still switching).
- Raise the tail to stay in transmit through short pauses between words.",
    },
    Setting {
        key: "wpm",
        section: "Transmit",
        label: "Speed (WPM)",
        text: "Sending speed in words per minute (default 20). Also in the transmit panel, and changeable during a message. A good rule is to send no faster than the station you are working.",
    },
    Setting {
        key: "weight",
        section: "Transmit",
        label: "Weight",
        text: "How heavy the code sounds. 50% is standard: a dit is as long as the gap after it.

- Raise it (55 to 60) to make every dit and dah longer and the gaps shorter by the same time: heavier, punchier code that cuts through noise.
- Lower it for lighter code. The speed stays the same either way.",
    },
    Setting {
        key: "dah",
        section: "Transmit",
        label: "Dah length",
        text: "Length of a dah in dits (default 3, the standard). Some operators like a slightly longer dah (3.2 to 3.5) for a more distinct rhythm. Changing it changes the speed slightly.",
    },
    Setting {
        key: "letter_gap",
        section: "Transmit",
        label: "Letter gap",
        text: "The gap between letters, in dits (default 3, the standard).

- Raise it to space letters out while each letter keeps its full speed (Farnsworth spacing). This helps slower readers and practice; the overall speed is shown below.
- Below 3 runs letters together and makes the code harder to read.",
    },
    Setting {
        key: "word_gap",
        section: "Transmit",
        label: "Word gap",
        text: "The gap between words, in dits (default 7, the standard). Raise it along with the letter gap for Farnsworth spacing, or on its own to make words stand apart. It is never shorter than the letter gap.",
    },
    Setting {
        key: "comp",
        section: "Transmit",
        label: "Key compensation",
        text: "Milliseconds added to every dit and dah (and taken from the gap after it). Usually 0.

- Some radios' break-in (QSK) clips the start of each element, making dits sound thin at high speed; add a few ms (2 to 8) to make up for it.
- Check with the radio's monitor or another receiver: dits should sound as long as the gaps between them.",
    },
    Setting {
        key: "memories",
        section: "Transmit",
        label: "Keyer memories",
        text: "Opens the editor for the messages on F1 to F12 (also: right-click a memory button in the transmit panel). See the Memory keyer help topic.",
    },
];
