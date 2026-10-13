//! DXCC country lookup from AD1C's country file (`cty.dat`,
//! <https://www.country-files.com>), the same file most loggers use.
//!
//! A copy is built into the program; a newer one can be downloaded into the
//! settings directory and is preferred when present.

use std::collections::HashMap;
use std::path::Path;

pub const URL: &str = "https://www.country-files.com/cty/cty.dat";
const BUILT_IN: &str = include_str!("../data/cty.dat");

/// The country of a callsign, with any zone or position overrides applied.
#[derive(Clone, Debug, PartialEq)]
pub struct Country {
    pub name: String,
    /// Primary prefix of the entity, e.g. "DL".
    pub prefix: String,
    /// Continent: NA, SA, EU, AF, AS, OC or AN.
    pub cont: String,
    pub cq: u8,
    pub itu: u8,
    /// Approximate position (degrees, east positive).
    pub lat: f64,
    pub lon: f64,
    /// Local time offset from UTC in hours (east positive).
    pub utc_offset: f32,
    /// False for entities that only count for WAE (Sicily, Shetland…), not DXCC.
    pub dxcc: bool,
}

struct Entity {
    name: String,
    prefix: String,
    dxcc: bool,
}

#[derive(Clone, Copy)]
struct Entry {
    entity: u16,
    cq: u8,
    itu: u8,
    cont: [u8; 2],
    lat: f32,
    lon: f32,
    tz: f32,
}

pub struct Cty {
    entities: Vec<Entity>,
    /// Whole callsigns ("=CALL" entries).
    exact: HashMap<String, Entry>,
    prefixes: HashMap<String, Entry>,
    max_len: usize,
}

impl Cty {
    /// The copy compiled into the program.
    pub fn built_in() -> Self {
        Self::parse(BUILT_IN).expect("built-in cty.dat parses")
    }

    /// Reads a country file; fails unless it looks complete.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        Self::parse(&std::fs::read_to_string(path)?)
    }

    /// Fetches the latest file from country-files.com and saves it to `dest`
    /// once it has parsed successfully.
    pub fn download(dest: &Path) -> anyhow::Result<Self> {
        let text = crate::lookup::agent().get(URL).call()?.into_string()?;
        let c = Self::parse(&text)?;
        if let Some(d) = dest.parent() {
            std::fs::create_dir_all(d)?;
        }
        std::fs::write(dest, text)?;
        Ok(c)
    }

    pub fn parse(text: &str) -> anyhow::Result<Self> {
        let mut c = Cty { entities: Vec::new(), exact: HashMap::new(), prefixes: HashMap::new(), max_len: 0 };
        let mut base: Option<Entry> = None;
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            if !line.starts_with(char::is_whitespace) {
                // "Name: CQ: ITU: Cont: Lat: Lon(west +): TZ(west +): Prefix:"
                let f: Vec<&str> = line.split(':').map(str::trim).collect();
                anyhow::ensure!(f.len() >= 8, "bad country line: {line}");
                let num = |s: &str| s.parse::<f32>().map_err(|_| anyhow::anyhow!("bad number {s:?} in: {line}"));
                let cont = f[3].as_bytes();
                anyhow::ensure!(cont.len() == 2, "bad continent in: {line}");
                let (dxcc, prefix) = match f[7].strip_prefix('*') {
                    Some(p) => (false, p),
                    None => (true, f[7]),
                };
                c.entities.push(Entity { name: f[0].to_string(), prefix: prefix.to_string(), dxcc });
                base = Some(Entry {
                    entity: (c.entities.len() - 1) as u16,
                    cq: num(f[1])? as u8,
                    itu: num(f[2])? as u8,
                    cont: [cont[0], cont[1]],
                    lat: num(f[4])?,
                    lon: -num(f[5])?,
                    tz: -num(f[6])?,
                });
                continue;
            }
            let Some(base) = base else { anyhow::bail!("prefix list before any country") };
            for tok in line.split([',', ';']).map(str::trim).filter(|t| !t.is_empty()) {
                let (exact, tok) = match tok.strip_prefix('=') {
                    Some(t) => (true, t),
                    None => (false, tok),
                };
                let (key, e) = parse_token(tok, base);
                if exact {
                    c.exact.insert(key, e);
                } else {
                    c.max_len = c.max_len.max(key.len());
                    c.prefixes.insert(key, e);
                }
            }
        }
        anyhow::ensure!(c.entities.len() > 300 && c.prefixes.len() > 1000, "incomplete country file ({} countries)", c.entities.len());
        Ok(c)
    }

    pub fn len(&self) -> usize {
        self.entities.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Country of a callsign, including portable forms such as "DL/W1ABC",
    /// "W1ABC/KH6" and "W6ABC/4". None for maritime/aeronautical mobile and
    /// unknown prefixes.
    pub fn lookup(&self, call: &str) -> Option<Country> {
        let call = call.trim().to_ascii_uppercase();
        if let Some(e) = self.exact.get(&call) {
            return Some(self.country(e));
        }
        let mut parts: Vec<&str> = call.split('/').filter(|p| !p.is_empty()).collect();
        if parts.iter().skip(1).any(|p| matches!(*p, "MM" | "AM")) {
            return None;
        }
        // Operating-condition suffixes say nothing about location.
        while parts.len() > 1 && is_condition(parts[parts.len() - 1]) {
            parts.pop();
        }
        let e = match parts.as_slice() {
            [] => None,
            [c] => self.exact.get(*c).or_else(|| self.longest(c)),
            [c, d] if d.len() == 1 && d.as_bytes()[0].is_ascii_digit() => {
                // W6ABC/4: the call district changes, which matters for
                // countries like Russia (UA1 vs UA9).
                let mut b = c.as_bytes().to_vec();
                match b.iter().rposition(|x| x.is_ascii_digit()) {
                    Some(i) => {
                        b[i] = d.as_bytes()[0];
                        self.longest(&String::from_utf8_lossy(&b)).or_else(|| self.longest(c))
                    }
                    None => self.longest(c),
                }
            }
            [a, b, ..] => {
                // DL/W1ABC or W1ABC/DL: the shorter part is the location prefix.
                let (pfx, home) = if b.len() < a.len() { (b, a) } else { (a, b) };
                self.longest(pfx).or_else(|| self.exact.get(*home)).or_else(|| self.longest(home))
            }
        };
        e.map(|e| self.country(e))
    }

    fn longest(&self, s: &str) -> Option<&Entry> {
        (1..=s.len().min(self.max_len)).rev().find_map(|n| s.get(..n).and_then(|p| self.prefixes.get(p)))
    }

    fn country(&self, e: &Entry) -> Country {
        let ent = &self.entities[e.entity as usize];
        Country {
            name: ent.name.clone(),
            prefix: ent.prefix.clone(),
            cont: String::from_utf8_lossy(&e.cont).into_owned(),
            cq: e.cq,
            itu: e.itu,
            lat: e.lat as f64,
            lon: e.lon as f64,
            utc_offset: e.tz,
            dxcc: ent.dxcc,
        }
    }
}

fn is_condition(s: &str) -> bool {
    matches!(s, "P" | "M" | "A" | "R" | "B" | "J" | "E" | "N" | "QRP" | "QRPP" | "LH" | "LGT" | "AE" | "AG" | "KT")
}

/// "AA0BY(3)[6]" → ("AA0BY", entry with CQ 3, ITU 6). Overrides are
/// (CQ) [ITU] <lat/lon> {continent} ~tz~.
fn parse_token(tok: &str, mut e: Entry) -> (String, Entry) {
    let end = tok.find(['(', '[', '<', '{', '~']).unwrap_or(tok.len());
    let mut rest = &tok[end..];
    while let Some(open) = rest.chars().next() {
        let close = match open {
            '(' => ')',
            '[' => ']',
            '<' => '>',
            '{' => '}',
            '~' => '~',
            _ => break,
        };
        let Some(n) = rest[1..].find(close) else { break };
        let v = &rest[1..1 + n];
        match open {
            '(' => e.cq = v.parse().unwrap_or(e.cq),
            '[' => e.itu = v.parse().unwrap_or(e.itu),
            '<' => {
                if let Some((la, lo)) = v.split_once('/') {
                    if let (Ok(la), Ok(lo)) = (la.parse::<f32>(), lo.parse::<f32>()) {
                        e.lat = la;
                        e.lon = -lo;
                    }
                }
            }
            '{' if v.len() == 2 => e.cont = [v.as_bytes()[0], v.as_bytes()[1]],
            '~' => e.tz = v.parse::<f32>().map(|t| -t).unwrap_or(e.tz),
            _ => {}
        }
        rest = &rest[2 + n..];
    }
    (tok[..end].to_string(), e)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn name(c: &Cty, call: &str) -> String {
        c.lookup(call).map(|c| c.name).unwrap_or_default()
    }

    #[test]
    fn countries() {
        let c = Cty::built_in();
        for (call, want) in [
            ("W1AW", "United States"),
            ("K1ABC", "United States"),
            ("VE3ABC", "Canada"),
            ("DL1ABC", "Fed. Rep. of Germany"),
            ("G4ABC", "England"),
            ("2E0ABC", "England"),
            ("GM4ABC", "Scotland"),
            ("JA1XYZ", "Japan"),
            ("KH6ABC", "Hawaii"),
            ("KL7ABC", "Alaska"),
            ("EA8ABC", "Canary Islands"),
            ("UA1ABC", "European Russia"),
            ("UA9ABC", "Asiatic Russia"),
            ("UA9ABC/1", "European Russia"),
            ("DL/W1ABC", "Fed. Rep. of Germany"),
            ("W1ABC/KH6", "Hawaii"),
            ("DL/W1ABC/P", "Fed. Rep. of Germany"),
            ("JA1XYZ/P", "Japan"),
            ("W6ABC/4", "United States"),
            ("w1aw", "United States"),
        ] {
            assert_eq!(name(&c, call), want, "{call}");
        }
        assert!(c.lookup("W1ABC/MM").is_none());
    }

    #[test]
    fn fields_and_overrides() {
        let c = Cty::built_in();
        let us = c.lookup("W1AW").unwrap();
        assert_eq!((us.cont.as_str(), us.prefix.as_str(), us.dxcc), ("NA", "K", true));
        assert!(us.lon < -60.0 && us.utc_offset == -5.0, "{us:?}");
        let de = c.lookup("DL1ABC").unwrap();
        assert_eq!((de.cq, de.itu, de.cont.as_str()), (14, 28, "EU"));
        assert!(de.lon > 0.0 && de.utc_offset == 1.0);
        // "=AA0BY(3)[6]": an individual call with its own zones.
        let x = c.lookup("AA0BY").unwrap();
        assert_eq!((x.name.as_str(), x.cq, x.itu), ("United States", 3, 6));
        // WAE-only entities are marked.
        assert!(!c.lookup("IT9ABC").unwrap().dxcc);
    }

    #[test]
    fn rejects_truncated_file() {
        assert!(Cty::parse(&BUILT_IN[..2000]).is_err());
    }
}
