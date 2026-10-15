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

/// True if the callsign has a known DXCC prefix (built-in cty.dat). Decoding
/// errors often produce call-shaped strings with impossible prefixes.
pub fn valid_prefix(call: &str) -> bool {
    static CTY: std::sync::OnceLock<crate::cty::Cty> = std::sync::OnceLock::new();
    CTY.get_or_init(crate::cty::Cty::built_in).lookup(call).is_some()
}

/// Edit distance between the Morse renderings of two strings (elements and
/// character breaks), so a dit lost or gained in a fade (N8AE read as
/// T8AE, R8AI as N8AI) is distance 1.
pub fn morse_distance(a: &str, b: &str) -> usize {
    let render = |s: &str| -> Vec<u8> {
        let mut v = Vec::new();
        for c in s.bytes() {
            if !v.is_empty() {
                v.push(b' ');
            }
            v.extend_from_slice(crate::morse::pattern_for(c).unwrap_or("#").as_bytes());
        }
        v
    };
    let (x, y) = (render(a), render(b));
    let mut prev: Vec<usize> = (0..=y.len()).collect();
    for i in 1..=x.len() {
        let mut cur = vec![i; y.len() + 1];
        for j in 1..=y.len() {
            let sub = prev[j - 1] + usize::from(x[i - 1] != y[j - 1]);
            cur[j] = sub.min(prev[j] + 1).min(cur[j - 1] + 1);
        }
        prev = cur;
    }
    prev[y.len()]
}

/// True if the call has the shape most real calls have: a prefix of one or
/// two characters (at most one of them a digit), one area digit, and a
/// suffix of one to three letters; two or three letters after a one-letter
/// K/N/W prefix (US 1x1 calls are rare special-event calls). Portable calls
/// are judged by their home call. Decoding errors often break the shape
/// (K56KB, E7WKDX, W7R).
pub fn standard_shape(call: &str) -> bool {
    let base = call.split('/').max_by_key(|p| p.len()).unwrap_or(call);
    let b = base.as_bytes();
    let Some(d) = b.iter().rposition(|c| c.is_ascii_digit()) else { return false };
    let (pfx, suf) = (&b[..d], &b[d + 1..]);
    if !(1..=3).contains(&suf.len()) || !suf.iter().all(|c| c.is_ascii_uppercase()) {
        return false;
    }
    if pfx.is_empty() || pfx.len() > 2 || pfx.iter().filter(|c| c.is_ascii_digit()).count() > 1 || !pfx.iter().any(|c| c.is_ascii_uppercase()) {
        return false;
    }
    // US prefixes (K, N, W) never contain a digit: K56KB is K5 + noise.
    let us = matches!(pfx[0], b'K' | b'N' | b'W');
    !(us && (pfx.len() == 2 && pfx[1].is_ascii_digit() || pfx.len() == 1 && suf.len() < 2))
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
    fn standard_shapes() {
        for c in ["E71A", "K9UIY", "W2GLN", "KH6KB", "TI5/VA3RA", "9A1K", "S57WJ", "N5KV", "GW0SGL", "4X1AB"] {
            assert!(standard_shape(c), "{c}");
        }
        for c in ["K56KB", "E7WKDX", "W7R", "DL1BMEA"] {
            assert!(!standard_shape(c), "{c}");
        }
    }

    #[test]
    fn calls() {
        for c in ["W1AW", "K1ABC", "DL1XYZ", "VE3ABC", "G4ABC", "9A1AA", "JA1XYZ", "W1ABC/P", "DL/W1ABC", "2E0ABC"] {
            assert!(parse_callsign(c).is_some(), "{c}");
        }
        for c in ["CQ", "599", "5NN", "TNX", "73", "UR", "QTH", "ANT", "100W", "1A"] {
            assert!(parse_callsign(c).is_none(), "{c}");
        }
    }

    #[test]
    fn near_misses() {
        assert_eq!(morse_distance("N8AE", "T8AE"), 1);
        assert_eq!(morse_distance("N8AI", "R8AI"), 1);
        assert_eq!(morse_distance("W1AW", "W1AW"), 0);
        assert!(morse_distance("K5ZD", "W1AW") > 3);
        assert!(valid_prefix("N8AI") && valid_prefix("WA5MUL"));
        assert!(!valid_prefix("Q5ABC"));
    }
}
