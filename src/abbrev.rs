//! Common CW abbreviations, Q-codes and prosigns, with plain-English
//! meanings, for annotating decoded conversations.

/// (abbreviation, meaning). Matched against whole words, case-insensitive.
const TABLE: &[(&str, &str)] = &[
    // Prosigns (as shown by `morse::to_display`, plus raw AR/BT bytes)
    ("<KN>", "over, to the named station only"),
    ("<SK>", "end of contact"),
    ("<AS>", "wait"),
    ("<AR>", "end of message"),
    ("<BT>", "pause / new paragraph"),
    ("+", "end of message (AR)"),
    ("=", "pause / new paragraph (BT)"),
    ("K", "over, go ahead"),
    ("KN", "over, to the named station only"),
    ("SK", "end of contact"),
    ("AR", "end of message"),
    ("BT", "pause / new paragraph"),
    ("BK", "back to you (break)"),
    ("R", "received / roger"),
    // Q-codes
    ("QRG", "your exact frequency"),
    ("QRL", "is this frequency in use"),
    ("QRM", "interference from other stations"),
    ("QRN", "static / atmospheric noise"),
    ("QRO", "high power"),
    ("QRP", "low power"),
    ("QRQ", "send faster"),
    ("QRS", "send slower"),
    ("QRT", "stopping / closing station"),
    ("QRU", "nothing more for you"),
    ("QRV", "ready"),
    ("QRX", "wait / stand by"),
    ("QRZ", "who is calling me"),
    ("QSB", "signal fading"),
    ("QSK", "full break-in"),
    ("QSL", "confirmed / I acknowledge"),
    ("QSO", "contact / conversation"),
    ("QSY", "change frequency"),
    ("QTH", "location"),
    ("QTR", "time"),
    // Numbers
    ("73", "best regards"),
    ("72", "best regards (QRP)"),
    ("88", "love and kisses"),
    ("55", "good luck / success"),
    // Abbreviations
    ("ABT", "about"),
    ("ADR", "address"),
    ("AGN", "again"),
    ("ANT", "antenna"),
    ("B4", "before"),
    ("BCNU", "be seeing you"),
    ("BURO", "QSL card bureau"),
    ("CFM", "confirm"),
    ("CL", "closing station"),
    ("CONDX", "conditions"),
    ("CPY", "copy"),
    ("CQ", "calling any station"),
    ("CU", "see you"),
    ("CUAGN", "see you again"),
    ("CUL", "see you later"),
    ("CW", "Morse code"),
    ("DE", "from / this is"),
    ("DR", "dear"),
    ("DX", "distant station"),
    ("ES", "and"),
    ("FB", "fine business (excellent)"),
    ("FER", "for"),
    ("FM", "from"),
    ("FREQ", "frequency"),
    ("GA", "good afternoon / go ahead"),
    ("GE", "good evening"),
    ("GL", "good luck"),
    ("GM", "good morning"),
    ("GN", "good night"),
    ("GUD", "good"),
    ("HI", "laughter"),
    ("HPE", "hope"),
    ("HR", "here"),
    ("HV", "have"),
    ("HVE", "have"),
    ("HW", "how (do you copy)"),
    ("INFO", "information"),
    ("LTR", "later / letter"),
    ("MNI", "many"),
    ("MSG", "message"),
    ("NR", "number / near"),
    ("NW", "now"),
    ("OB", "old boy (friend)"),
    ("OM", "old man (friend)"),
    ("OP", "operator / name"),
    ("OT", "old timer"),
    ("PSE", "please"),
    ("PWR", "power"),
    ("RCVR", "receiver"),
    ("RIG", "radio equipment"),
    ("RPT", "report / repeat"),
    ("RPRT", "report"),
    ("RST", "signal report"),
    ("RX", "receiver"),
    ("SED", "said"),
    ("SIG", "signal"),
    ("SKED", "schedule"),
    ("SN", "soon"),
    ("SRI", "sorry"),
    ("STN", "station"),
    ("TEMP", "temperature"),
    ("TKS", "thanks"),
    ("TMW", "tomorrow"),
    ("TNX", "thanks"),
    ("THX", "thanks"),
    ("TU", "thank you"),
    ("TX", "transmitter"),
    ("U", "you"),
    ("UFB", "ultra fine business (excellent)"),
    ("UR", "your / you're"),
    ("VERT", "vertical antenna"),
    ("VY", "very"),
    ("WID", "with"),
    ("WKD", "worked"),
    ("WKG", "working"),
    ("WL", "will"),
    ("WPM", "words per minute"),
    ("WUD", "would"),
    ("WX", "weather"),
    ("XCVR", "transceiver"),
    ("XMTR", "transmitter"),
    ("XYL", "wife"),
    ("YL", "young lady"),
    ("YRS", "years"),
    ("POTA", "Parks on the Air"),
    ("SOTA", "Summits on the Air"),
];

/// Meaning of a single word, if it is a known abbreviation, Q-code, prosign
/// or signal report. A trailing `?` turns it into a question.
pub fn meaning(word: &str) -> Option<String> {
    let (core, q) = match word.strip_suffix('?') {
        Some(c) if !c.is_empty() => (c, true),
        _ => (word, false),
    };
    let up = core.to_ascii_uppercase();
    let m = TABLE.iter().find(|(k, _)| *k == up).map(|(_, m)| m.to_string()).or_else(|| rst(&up))?;
    Some(if q { format!("{m}?") } else { m })
}

/// RST signal reports such as 599, 5NN, 579 (N is a cut 9, T a cut 0).
fn rst(w: &str) -> Option<String> {
    let d: Vec<u8> = w.bytes().map(|b| match b {
        b'N' => b'9',
        b'T' => b'0',
        b'A' => b'1',
        b'E' => b'5',
        _ => b,
    }).collect();
    let ok = d.len() == 3 && (b'1'..=b'5').contains(&d[0]) && (b'1'..=b'9').contains(&d[1]) && (b'1'..=b'9').contains(&d[2]);
    // Need at least one digit so plain words ("NNN", "EAT") don't match.
    if !ok || !w.bytes().any(|b| b.is_ascii_digit()) {
        return None;
    }
    Some(format!("signal report: readability {}, strength {}, tone {}", d[0] as char, d[1] as char, d[2] as char))
}

/// Byte ranges of the annotatable words in `text`, with their meanings.
pub fn find(text: &str) -> Vec<(std::ops::Range<usize>, String)> {
    let mut out = Vec::new();
    let mut start = None;
    for (i, c) in text.char_indices().chain(std::iter::once((text.len(), ' '))) {
        if c.is_whitespace() {
            if let Some(s) = start.take() {
                if let Some(m) = meaning(&text[s..i]) {
                    out.push((s..i, m));
                }
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_words() {
        let t = "[12:34Z] W1AW DE K1ABC TNX FER CALL UR 5NN QTH? <SK>";
        let f = find(t);
        let words: Vec<&str> = f.iter().map(|(r, _)| &t[r.clone()]).collect();
        assert_eq!(words, ["DE", "TNX", "FER", "UR", "5NN", "QTH?", "<SK>"]);
        assert_eq!(f[5].1, "location?");
        assert!(f[4].1.contains("readability 5, strength 9, tone 9"));
    }

    #[test]
    fn no_false_rst() {
        assert!(meaning("NNN").is_none());
        assert!(meaning("EAT").is_none());
        assert!(meaning("699").is_none());
        assert!(meaning("599").is_some());
        assert_eq!(meaning("tnx").as_deref(), Some("thanks"));
    }
}
