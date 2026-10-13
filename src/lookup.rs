//! Callsign information lookup (name, location, photo) from free sources.
//!
//! - HamDB (api.hamdb.org): US and Canadian licence data, no account needed.
//! - QRZ.com XML: a free QRZ login returns name and country for any call;
//!   with a QRZ XML subscription it also returns address, grid and photo.
//! - HamQTH.com XML: free account, worldwide, often includes a photo.
//!
//! Lookups run on background threads; results are cached per call.

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Read;
use std::sync::Arc;
use std::time::Duration;

const AGENT: &str = "CwSkimmer";

#[derive(Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Credentials {
    pub qrz_user: String,
    pub qrz_pass: String,
    pub hamqth_user: String,
    pub hamqth_pass: String,
}

#[derive(Clone, Default, Debug)]
pub struct CallInfo {
    pub call: String,
    pub name: String,
    /// Address lines, most specific first.
    pub address: Vec<String>,
    pub country: String,
    pub grid: String,
    pub lat_lon: Option<(f64, f64)>,
    pub class: String,
    pub email: String,
    pub web: String,
    pub qsl: String,
    pub lotw: bool,
    pub image_url: String,
    pub image: Option<Arc<[u8]>>,
    /// Sources that answered, e.g. "HamDB, QRZ".
    pub sources: Vec<&'static str>,
}

impl CallInfo {
    fn merge(&mut self, o: CallInfo) {
        fn fill(a: &mut String, b: String) {
            if a.is_empty() && !b.trim().is_empty() {
                *a = b.trim().to_string();
            }
        }
        fill(&mut self.name, o.name);
        if self.address.is_empty() {
            self.address = o.address.into_iter().filter(|l| !l.trim().is_empty()).collect();
        }
        fill(&mut self.country, o.country);
        fill(&mut self.grid, o.grid);
        self.lat_lon = self.lat_lon.or(o.lat_lon);
        fill(&mut self.class, o.class);
        fill(&mut self.email, o.email);
        fill(&mut self.web, o.web);
        fill(&mut self.qsl, o.qsl);
        self.lotw |= o.lotw;
        fill(&mut self.image_url, o.image_url);
        self.sources.extend(o.sources);
    }
}

#[derive(Clone)]
pub enum LookupState {
    Pending,
    Done(CallInfo),
    /// Nothing found; message lists what was tried.
    Failed(String),
}

#[derive(Default)]
struct Sessions {
    qrz: Option<String>,
    hamqth: Option<String>,
}

#[derive(Clone, Default)]
pub struct Lookup {
    cache: Arc<Mutex<HashMap<String, LookupState>>>,
    sessions: Arc<Mutex<Sessions>>,
}

impl Lookup {
    /// Current state for `call`, starting a lookup if there is none.
    pub fn get(&self, call: &str, creds: &Credentials) -> LookupState {
        let call = call.to_ascii_uppercase();
        let mut cache = self.cache.lock();
        if let Some(s) = cache.get(&call) {
            return s.clone();
        }
        cache.insert(call.clone(), LookupState::Pending);
        let (me, creds) = (self.clone(), creds.clone());
        std::thread::spawn(move || {
            let r = me.fetch(&call, &creds);
            me.cache.lock().insert(call, r);
        });
        LookupState::Pending
    }

    /// Forget cached results (e.g. after the logins change).
    pub fn clear(&self) {
        self.cache.lock().clear();
        *self.sessions.lock() = Sessions::default();
    }

    fn fetch(&self, call: &str, creds: &Credentials) -> LookupState {
        let base = base_call(call);
        let mut info = CallInfo { call: call.to_string(), ..Default::default() };
        let mut errors = Vec::new();
        if !creds.qrz_user.is_empty() {
            match self.qrz(&base, creds) {
                Ok(i) => info.merge(i),
                Err(e) => errors.push(format!("QRZ: {e}")),
            }
        }
        if !creds.hamqth_user.is_empty() {
            match self.hamqth(&base, creds) {
                Ok(i) => info.merge(i),
                Err(e) => errors.push(format!("HamQTH: {e}")),
            }
        }
        match hamdb(&base) {
            Ok(i) => info.merge(i),
            Err(e) => errors.push(format!("HamDB: {e}")),
        }
        if info.sources.is_empty() {
            return LookupState::Failed(errors.join("\n"));
        }
        if !info.image_url.is_empty() {
            info.image = get_bytes(&info.image_url).ok().map(Arc::from);
        }
        LookupState::Done(info)
    }

    fn qrz(&self, call: &str, c: &Credentials) -> anyhow::Result<CallInfo> {
        for attempt in 0..2 {
            let key = match self.sessions.lock().qrz.clone() {
                Some(k) => k,
                None => {
                    let x = get_text(&format!(
                        "https://xmldata.qrz.com/xml/current/?username={};password={};agent={AGENT}",
                        enc(&c.qrz_user),
                        enc(&c.qrz_pass)
                    ))?;
                    let key = tag(&x, "Key").ok_or_else(|| anyhow::anyhow!(tag(&x, "Error").unwrap_or_else(|| "login failed".into())))?;
                    self.sessions.lock().qrz = Some(key.clone());
                    key
                }
            };
            let x = get_text(&format!("https://xmldata.qrz.com/xml/current/?s={key};callsign={}", enc(call)))?;
            let Some(cs) = section(&x, "Callsign") else {
                let err = tag(&x, "Error").unwrap_or_else(|| "no data".into());
                if attempt == 0 && err.contains("ession") {
                    self.sessions.lock().qrz = None;
                    continue;
                }
                anyhow::bail!(err);
            };
            let t = |n: &str| tag(cs, n).unwrap_or_default();
            let name = join(&[&t("fname"), &t("name")]);
            let city = join(&[&t("addr2"), &t("state"), &t("zip")]);
            return Ok(CallInfo {
                name: if t("nickname").is_empty() { name } else { format!("{name} \"{}\"", t("nickname")) },
                address: vec![t("attn"), t("addr1"), city],
                country: t("country"),
                grid: t("grid"),
                lat_lon: t("lat").parse().ok().zip(t("lon").parse().ok()),
                class: t("class"),
                email: t("email"),
                web: t("url"),
                qsl: t("qslmgr"),
                lotw: t("lotw") == "1",
                image_url: t("image"),
                sources: vec!["QRZ"],
                ..Default::default()
            });
        }
        anyhow::bail!("session expired")
    }

    fn hamqth(&self, call: &str, c: &Credentials) -> anyhow::Result<CallInfo> {
        for attempt in 0..2 {
            let id = match self.sessions.lock().hamqth.clone() {
                Some(k) => k,
                None => {
                    let x = get_text(&format!("https://www.hamqth.com/xml.php?u={}&p={}", enc(&c.hamqth_user), enc(&c.hamqth_pass)))?;
                    let id = tag(&x, "session_id").ok_or_else(|| anyhow::anyhow!(tag(&x, "error").unwrap_or_else(|| "login failed".into())))?;
                    self.sessions.lock().hamqth = Some(id.clone());
                    id
                }
            };
            let x = get_text(&format!("https://www.hamqth.com/xml.php?id={id}&callsign={}&prg={AGENT}", enc(call)))?;
            let Some(s) = section(&x, "search") else {
                let err = tag(&x, "error").unwrap_or_else(|| "no data".into());
                if attempt == 0 && err.contains("ession") {
                    self.sessions.lock().hamqth = None;
                    continue;
                }
                anyhow::bail!(err);
            };
            let t = |n: &str| tag(s, n).unwrap_or_default();
            let name = if t("adr_name").is_empty() { t("nick") } else { t("adr_name") };
            return Ok(CallInfo {
                name,
                address: vec![t("adr_street1"), join(&[&t("adr_city"), &t("us_state"), &t("adr_zip")]), t("qth")],
                country: t("country"),
                grid: t("grid"),
                lat_lon: t("latitude").parse().ok().zip(t("longitude").parse().ok()),
                email: t("email"),
                web: t("web"),
                qsl: t("qsl_via"),
                lotw: t("lotw") == "Y",
                image_url: t("picture"),
                sources: vec!["HamQTH"],
                ..Default::default()
            });
        }
        anyhow::bail!("session expired")
    }
}

/// HamDB: free US/Canada licence data.
fn hamdb(call: &str) -> anyhow::Result<CallInfo> {
    let v: serde_json::Value = serde_json::from_str(&get_text(&format!("https://api.hamdb.org/v1/{}/json/{AGENT}", enc(call)))?)?;
    let c = &v["hamdb"]["callsign"];
    let status = v["hamdb"]["messages"]["status"].as_str().unwrap_or("");
    if status != "OK" {
        anyhow::bail!("{}", if status.is_empty() { "no data" } else { status });
    }
    let t = |n: &str| c[n].as_str().unwrap_or("").trim().to_string();
    let name = join(&[&t("fname"), &t("mi"), &t("name"), &t("suffix")]);
    let class = match t("class").as_str() {
        "E" => "Extra",
        "A" => "Advanced",
        "G" => "General",
        "T" => "Technician",
        "N" => "Novice",
        _ => "",
    };
    Ok(CallInfo {
        name,
        address: vec![t("addr1"), join(&[&t("addr2"), &t("state"), &t("zip")])],
        country: t("country"),
        grid: t("grid"),
        lat_lon: t("lat").parse().ok().zip(t("lon").parse().ok()),
        class: class.into(),
        sources: vec!["HamDB"],
        ..Default::default()
    })
}

/// The home callsign inside a portable call: "DL/W1ABC/P" → "W1ABC".
pub fn base_call(call: &str) -> String {
    call.split('/').max_by_key(|p| (p.chars().any(|c| c.is_ascii_digit()) && p.len() >= 3, p.len())).unwrap_or(call).to_string()
}

pub fn qrz_url(call: &str) -> String {
    format!("https://www.qrz.com/db/{}", base_call(call))
}

pub fn hamqth_url(call: &str) -> String {
    format!("https://www.hamqth.com/{}", base_call(call))
}

/// Maidenhead locator of a position (6 characters).
pub fn grid_of(lat: f64, lon: f64) -> String {
    let (x, y) = (lon + 180.0, lat + 90.0);
    let c = |base: u8, v: f64| (base + v as u8) as char;
    [
        c(b'A', x / 20.0),
        c(b'A', y / 10.0),
        c(b'0', (x % 20.0) / 2.0),
        c(b'0', y % 10.0),
        c(b'a', (x % 2.0) * 12.0),
        c(b'a', (y % 1.0) * 24.0),
    ]
    .iter()
    .collect()
}

/// Centre of a Maidenhead locator (4 or 6 characters).
pub fn grid_pos(grid: &str) -> Option<(f64, f64)> {
    let g: Vec<u8> = grid.trim().to_ascii_uppercase().bytes().collect();
    if g.len() < 4 || !(b'A'..=b'R').contains(&g[0]) || !(b'A'..=b'R').contains(&g[1]) || !g[2].is_ascii_digit() || !g[3].is_ascii_digit() {
        return None;
    }
    let mut lon = (g[0] - b'A') as f64 * 20.0 + (g[2] - b'0') as f64 * 2.0 - 180.0;
    let mut lat = (g[1] - b'A') as f64 * 10.0 + (g[3] - b'0') as f64 - 90.0;
    if g.len() >= 6 && (b'A'..=b'X').contains(&g[4]) && (b'A'..=b'X').contains(&g[5]) {
        lon += (g[4] - b'A') as f64 / 12.0 + 1.0 / 24.0;
        lat += (g[5] - b'A') as f64 / 24.0 + 1.0 / 48.0;
    } else {
        lon += 1.0;
        lat += 0.5;
    }
    Some((lat, lon))
}

/// Great-circle distance (km) and initial bearing (degrees) from `a` to `b`.
pub fn distance_bearing(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    let (la1, lo1, la2, lo2) = (a.0.to_radians(), a.1.to_radians(), b.0.to_radians(), b.1.to_radians());
    let dlo = lo2 - lo1;
    let h = ((la2 - la1) / 2.0).sin().powi(2) + la1.cos() * la2.cos() * (dlo / 2.0).sin().powi(2);
    let d = 2.0 * 6371.0 * h.sqrt().asin();
    let y = dlo.sin() * la2.cos();
    let x = la1.cos() * la2.sin() - la1.sin() * la2.cos() * dlo.cos();
    (d, (y.atan2(x).to_degrees() + 360.0) % 360.0)
}

pub(crate) fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new().timeout(Duration::from_secs(10)).user_agent(concat!("CwSkimmer/", env!("CARGO_PKG_VERSION"))).build()
}

fn get_text(url: &str) -> anyhow::Result<String> {
    Ok(agent().get(url).call()?.into_string()?)
}

fn get_bytes(url: &str) -> anyhow::Result<Vec<u8>> {
    let mut v = Vec::new();
    agent().get(url).call()?.into_reader().take(8 << 20).read_to_end(&mut v)?;
    Ok(v)
}

fn enc(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

fn join(parts: &[&str]) -> String {
    parts.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).collect::<Vec<_>>().join(" ")
}

/// Inner text of the first `<name>…</name>` element in `xml`.
fn section<'a>(xml: &'a str, name: &str) -> Option<&'a str> {
    let open = format!("<{name}>");
    let close = format!("</{name}>");
    let a = xml.find(&open)? + open.len();
    let b = a + xml[a..].find(&close)?;
    Some(&xml[a..b])
}

/// Text of the first `<name>` element, with XML entities decoded.
fn tag(xml: &str, name: &str) -> Option<String> {
    let s = section(xml, name)?.trim();
    let s = s.strip_prefix("<![CDATA[").and_then(|s| s.strip_suffix("]]>")).unwrap_or(s);
    let s = s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&apos;", "'").replace("&amp;", "&");
    (!s.is_empty()).then_some(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base() {
        assert_eq!(base_call("W1AW"), "W1AW");
        assert_eq!(base_call("DL/W1ABC/P"), "W1ABC");
        assert_eq!(base_call("W1ABC/3"), "W1ABC");
    }

    #[test]
    fn grids() {
        let (lat, lon) = grid_pos("FN31pr").unwrap();
        assert!((lat - 41.729).abs() < 0.05 && (lon + 72.708).abs() < 0.05, "{lat} {lon}");
        assert_eq!(grid_of(41.7146, -72.7284), "FN31pr");
        let (d, b) = distance_bearing((41.71, -72.73), (51.5, -0.12));
        assert!((d - 5400.0).abs() < 150.0 && (b - 50.0).abs() < 5.0, "{d} {b}");
    }

    #[test]
    #[ignore = "network"]
    fn live_hamdb() {
        let l = Lookup::default();
        let LookupState::Done(i) = l.fetch("W1AW/P", &Credentials::default()) else { panic!("no result") };
        assert_eq!(i.grid, "FN31pr");
        assert!(i.name.contains("ARRL"), "{i:?}");
    }

    #[test]
    fn xml() {
        let x = "<QRZDatabase><Callsign><call>W1AW</call><name>A &amp; B</name></Callsign></QRZDatabase>";
        let cs = section(x, "Callsign").unwrap();
        assert_eq!(tag(cs, "name").unwrap(), "A & B");
        assert!(tag(cs, "grid").is_none());
    }
}
