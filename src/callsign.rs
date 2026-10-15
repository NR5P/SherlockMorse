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

/// The Morse rendering of a string: element patterns, a space between
/// characters.
fn render(s: &str) -> Vec<u8> {
    let mut v = Vec::new();
    for c in s.bytes() {
        if !v.is_empty() {
            v.push(b' ');
        }
        v.extend_from_slice(crate::morse::pattern_for(c).unwrap_or("#").as_bytes());
    }
    v
}

/// Weighted edit distance of two renderings: an element gained, lost or
/// changed costs 2, a character break gained or lost 1 (a fade splits or
/// joins characters more often than it loses an element).
fn copy_distance(x: &[u8], y: &[u8]) -> usize {
    let w = |c: u8| if c == b' ' { 1 } else { 2 };
    let mut prev: Vec<usize> = std::iter::once(0).chain(y.iter().scan(0, |a, &c| { *a += w(c); Some(*a) })).collect();
    for &a in x {
        let mut cur = vec![prev[0] + w(a); y.len() + 1];
        for (j, &b) in y.iter().enumerate() {
            let sub = prev[j] + if a == b { 0 } else if a == b' ' || b == b' ' { 3 } else { 2 };
            cur[j + 1] = sub.min(prev[j + 1] + w(a)).min(cur[j] + w(b));
        }
        prev = cur;
    }
    prev[y.len()]
}

/// Element-only edit distance (character breaks ignored): how many dits
/// and dahs two copies differ by.
pub fn element_distance(a: &str, b: &str) -> usize {
    let strip = |s: &str| render(s).into_iter().filter(|&c| c != b' ').collect::<Vec<u8>>();
    copy_distance(&strip(a), &strip(b)) / 2
}

fn char_for(pat: &[u8]) -> Option<u8> {
    crate::morse::TABLE.iter().find(|(c, p)| p.as_bytes() == pat && (c.is_ascii_alphanumeric() || *c == b'/')).map(|(c, _)| *c)
}

/// Strings one copying error away from `w`: an element gained, lost or
/// flipped in one character, a character split in two, or two joined.
fn neighbors(w: &str) -> Vec<String> {
    let b = w.as_bytes();
    let mut out = Vec::new();
    let with = |i: usize, n: usize, rep: &[u8]| -> String {
        let mut v = b[..i].to_vec();
        v.extend_from_slice(rep);
        v.extend_from_slice(&b[i + n..]);
        String::from_utf8(v).unwrap_or_default()
    };
    for i in 0..b.len() {
        let Some(p) = crate::morse::pattern_for(b[i]).map(|p| p.as_bytes()) else { continue };
        let mut pats: Vec<Vec<u8>> = Vec::new();
        for k in 0..=p.len() {
            for &e in b".-" {
                pats.push([&p[..k], &[e][..], &p[k..]].concat());
            }
            if k < p.len() {
                pats.push([&p[..k], &p[k + 1..]].concat());
                pats.push([&p[..k], &[if p[k] == b'.' { b'-' } else { b'.' }][..], &p[k + 1..]].concat());
            }
        }
        for q in pats {
            if let Some(c) = char_for(&q) {
                out.push(with(i, 1, &[c]));
            }
        }
        for k in 1..p.len() {
            if let (Some(x), Some(y)) = (char_for(&p[..k]), char_for(&p[k..])) {
                out.push(with(i, 1, &[x, y]));
            }
        }
        if i + 1 < b.len() {
            if let Some(q) = crate::morse::pattern_for(b[i + 1]) {
                if let Some(c) = char_for(&[p, q.as_bytes()].concat()) {
                    out.push(with(i, 2, &[c]));
                }
            }
        }
    }
    out
}

/// The call several differing copies of one transmission agree on best,
/// as an operator would piece it together over an over or two of a weak,
/// fading signal (DL1DGR, DLEJBGR, DLEWBGR, DL1BMR: DL1BGR). Candidates are
/// the copies and everything one copying error from them, of real-call
/// shape; the one closest to all copies wins if it is a clear winner and
/// at least half the copies (three or more) are within an element of it.
pub fn consensus(copies: &[&str]) -> Option<String> {
    if copies.len() < 3 {
        return None;
    }
    let rendered: Vec<Vec<u8>> = copies.iter().map(|c| render(c)).collect();
    let mut cands: Vec<String> = copies.iter().flat_map(|c| std::iter::once(c.to_string()).chain(neighbors(c))).collect();
    cands.sort();
    cands.dedup();
    let mut scored: Vec<(usize, String)> = cands
        .into_iter()
        .filter(|c| parse_callsign(c).as_deref() == Some(c.as_str()) && standard_shape(c) && valid_prefix(c))
        .map(|c| {
            let r = render(&c);
            (rendered.iter().map(|x| copy_distance(&r, x)).sum(), c)
        })
        .collect();
    scored.sort();
    let (best, call) = scored.first()?;
    // Abstain when a copy that is no call fits the copies as well: they
    // are copies of a word (THE, not T4E).
    if copies.iter().any(|c| parse_callsign(c).is_none() && rendered.iter().map(|x| copy_distance(&render(c), x)).sum::<usize>() <= *best) {
        return None;
    }
    let close = copies.iter().filter(|c| element_distance(c, call) <= 1).count();
    if scored.get(1).is_some_and(|(s, _)| s == best) || close < 3 || 2 * close < copies.len() {
        return None;
    }
    Some(call.clone())
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
    fn consensus_of_copies() {
        assert_eq!(element_distance("DLEJBGR", "DL1BGR"), 1);
        assert_eq!(element_distance("DL1DGR", "DL1BGR"), 1);
        // A weak DL1BGR over two overs (40 m, 2026-10-14 06:12Z).
        assert_eq!(consensus(&["DL1DGR", "DLEJBGR", "DLEWBGR", "DL1BMR", "DL1DGR"]).as_deref(), Some("DL1BGR"));
        assert_eq!(consensus(&["K9UIY", "K9UIY", "K9UI", "K9UIY"]).as_deref(), Some("K9UIY"));
        assert_eq!(consensus(&["K9UIY", "K9UIY"]), None);
        // Three loose copies are not enough (HA8WZ really).
        assert_eq!(consensus(&["DS8WP", "HA8WG", "HA7EMP"]), None);
        assert_eq!(consensus(&["THE", "THE", "T4E", "THE"]), None);
        // Both ends of a QSO on one channel stay apart ("K9UR DE K9UIY").
        assert!(element_distance("K9UR", "K9UIY") > 2);
        let qso = ["K9UR", "DE", "K9UIY", "K9UIY", "K9UR", "DE", "K9UIY", "K9UR", "DE", "K9UIY"];
        for call in ["K9UR", "K9UIY"] {
            let copies: Vec<&str> = qso.iter().copied().filter(|w| element_distance(w, call) <= 2).collect();
            assert_eq!(consensus(&copies).as_deref(), Some(call));
        }
        assert_eq!(consensus(&["K9UIY", "K9UIY", "K9UIY", "K9UI"]).as_deref(), Some("K9UIY"));
    }

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
