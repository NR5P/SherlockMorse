//! Callsign recognition in decoded text.

/// Returns the base callsign if `w` looks like an amateur callsign.
pub fn parse_callsign(w: &str) -> Option<String> {
    let w = w.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '/');
    if w.len() < 3 || w.len() > 14 {
        return None;
    }
    // Pick the longest '/'-separated part that is a valid base call
    // (handles DL/W1ABC, W1ABC/P, W1ABC/3).
    let mut best: Option<&str> = None;
    for part in w.split('/') {
        if is_base_call(part) && best.is_none_or(|b| part.len() > b.len()) {
            best = Some(part);
        }
    }
    best.map(|_| w.to_string())
}

fn is_base_call(s: &str) -> bool {
    let b = s.as_bytes();
    if b.len() < 3 || b.len() > 7 || !b.iter().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit()) {
        return false;
    }
    // Last digit separates prefix and suffix.
    let Some(d) = b.iter().rposition(|c| c.is_ascii_digit()) else { return false };
    let (pfx, suf) = (&b[..d], &b[d + 1..]);
    if suf.is_empty() || suf.len() > 4 || !suf.iter().all(|c| c.is_ascii_uppercase()) {
        return false;
    }
    if pfx.is_empty() || pfx.len() > 3 {
        return false;
    }
    // Prefix must contain a letter (rules out things like "599" or "73").
    if !pfx.iter().any(|c| c.is_ascii_uppercase()) {
        return false;
    }
    // Reject common non-call tokens.
    !matches!(s, "5NN" | "TU5NN" | "R5NN" | "QRP5W" | "TEMP")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calls() {
        for c in ["W1AW", "K1ABC", "DL1XYZ", "VE3ABC", "G4ABC", "9A1AA", "JA1XYZ", "W1ABC/P", "DL/W1ABC", "2E0ABC"] {
            assert!(parse_callsign(c).is_some(), "{c}");
        }
        for c in ["CQ", "599", "5NN", "TNX", "73", "UR", "QTH", "ANT", "100W", "1A"] {
            assert!(parse_callsign(c).is_none(), "{c}");
        }
    }
}
