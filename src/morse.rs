//! Morse code tables.
//!
//! A Morse character is stored as `(bits, len)` where the most significant of
//! the `len` bits is the first element and `1` means dash.
//!
//! Prosigns are mapped to single bytes so the decoder and language model can
//! treat them as ordinary symbols; [`display_char`] expands them for display.

/// (character, pattern)
pub const TABLE: &[(u8, &str)] = &[
    (b'A', ".-"),
    (b'B', "-..."),
    (b'C', "-.-."),
    (b'D', "-.."),
    (b'E', "."),
    (b'F', "..-."),
    (b'G', "--."),
    (b'H', "...."),
    (b'I', ".."),
    (b'J', ".---"),
    (b'K', "-.-"),
    (b'L', ".-.."),
    (b'M', "--"),
    (b'N', "-."),
    (b'O', "---"),
    (b'P', ".--."),
    (b'Q', "--.-"),
    (b'R', ".-."),
    (b'S', "..."),
    (b'T', "-"),
    (b'U', "..-"),
    (b'V', "...-"),
    (b'W', ".--"),
    (b'X', "-..-"),
    (b'Y', "-.--"),
    (b'Z', "--.."),
    (b'0', "-----"),
    (b'1', ".----"),
    (b'2', "..---"),
    (b'3', "...--"),
    (b'4', "....-"),
    (b'5', "....."),
    (b'6', "-...."),
    (b'7', "--..."),
    (b'8', "---.."),
    (b'9', "----."),
    (b'/', "-..-."),
    (b'?', "..--.."),
    (b'.', ".-.-.-"),
    (b',', "--..--"),
    (b'=', "-...-"),  // BT
    (b'+', ".-.-."),  // AR
    (b'(', "-.--."),  // KN
    (b'%', "...-.-"), // SK
    (b'&', ".-..."),  // AS
    (b'-', "-....-"),
    (b'\'', ".----."),
    (b':', "---..."),
    (b'@', ".--.-."),
];

/// Maximum number of elements in any character we decode.
pub const MAX_ELEMS: u8 = 7;

/// The error sign (a string of dits, nominally 8, "HH"): the sender is
/// erasing what they just sent. Stored as the ASCII backspace byte.
pub const ERROR: u8 = 0x08;
/// Fewest dits read as the error sign (operators rarely count exactly 8).
pub const ERROR_MIN_DITS: u8 = 7;
/// Longest run of dits accepted as one error sign.
pub const ERROR_MAX_DITS: u8 = 16;

/// True when another element may follow `len` elements: within the normal
/// character length, or a run of dits that may still be an error sign.
#[inline]
pub fn can_extend(bits: u8, len: u8) -> bool {
    len < MAX_ELEMS || (bits == 0 && len < ERROR_MAX_DITS)
}

/// Lookup table indexed by `(1 << len) | bits` (a sentinel-bit encoding).
pub struct MorseLookup {
    table: [u8; 256],
}

impl Default for MorseLookup {
    fn default() -> Self {
        Self::new()
    }
}

impl MorseLookup {
    pub fn new() -> Self {
        let mut table = [0u8; 256];
        for &(c, pat) in TABLE {
            let (bits, len) = pattern_bits(pat);
            table[key(bits, len)] = c;
        }
        Self { table }
    }

    /// Returns the character for an element sequence, or `None`.
    #[inline]
    pub fn get(&self, bits: u8, len: u8) -> Option<u8> {
        if bits == 0 && (ERROR_MIN_DITS..=ERROR_MAX_DITS).contains(&len) {
            return Some(ERROR);
        }
        if len == 0 || len > MAX_ELEMS {
            return None;
        }
        let c = self.table[key(bits, len)];
        (c != 0).then_some(c)
    }

    /// True when some valid character begins with this element prefix.
    pub fn is_prefix(&self, bits: u8, len: u8) -> bool {
        if len == 0 || (bits == 0 && len <= ERROR_MAX_DITS) {
            return true;
        }
        if len > MAX_ELEMS {
            return false;
        }
        for extra in 0..=(MAX_ELEMS - len) {
            for tail in 0..(1u16 << extra) {
                let b = ((bits as u16) << extra) | tail;
                if b < 256 && self.get(b as u8, len + extra).is_some() {
                    return true;
                }
            }
        }
        false
    }
}

#[inline]
fn key(bits: u8, len: u8) -> usize {
    ((1usize << len) | bits as usize) & 0xff
}

pub fn pattern_bits(pat: &str) -> (u8, u8) {
    let mut bits = 0u8;
    let mut len = 0u8;
    for ch in pat.bytes() {
        bits = (bits << 1) | u8::from(ch == b'-');
        len += 1;
    }
    (bits, len)
}

/// Morse pattern for a character (uppercase), if known.
pub fn pattern_for(c: u8) -> Option<&'static str> {
    if c == ERROR {
        return Some("........");
    }
    let c = c.to_ascii_uppercase();
    TABLE.iter().find(|(k, _)| *k == c).map(|(_, p)| *p)
}

/// Expands internal prosign bytes for display.
pub fn display_char(c: u8) -> &'static str {
    match c {
        b'(' => "<KN>",
        b'%' => "<SK>",
        b'&' => "<AS>",
        ERROR => "<HH>",
        _ => "",
    }
}

/// Converts decoder text to display text (prosigns expanded).
pub fn to_display(text: &str) -> String {
    let mut s = String::with_capacity(text.len() + 8);
    for b in text.bytes() {
        let d = display_char(b);
        if d.is_empty() {
            s.push(b as char);
        } else {
            s.push_str(d);
        }
    }
    s
}

/// Converts user text (which may contain `<SK>`-style prosigns) into the
/// internal single-byte representation.
pub fn from_display(text: &str) -> String {
    text.to_ascii_uppercase()
        .replace("<KN>", "(")
        .replace("<SK>", "%")
        .replace("<AS>", "&")
        .replace("<AR>", "+")
        .replace("<BT>", "=")
        .replace("<HH>", "\x08")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_roundtrip() {
        let m = MorseLookup::new();
        for &(c, pat) in TABLE {
            let (b, l) = pattern_bits(pat);
            assert_eq!(m.get(b, l), Some(c), "{}", c as char);
        }
        assert!(m.is_prefix(0, 1));
        assert!(m.get(0b111111, 6).is_none());
        assert_eq!(m.get(0, 8), Some(ERROR));
        assert_eq!(m.get(0, 6), None);
        assert!(m.is_prefix(0, 7));
    }
}
