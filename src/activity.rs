//! Activity flags: recognises on-air programs and contests (POTA, SKCC, CWT…)
//! from what a station sends, e.g. "CQ POTA" or "CQ SST".

use serde::{Deserialize, Serialize};

/// One flag. `patterns` is a comma-separated list; each pattern is one or more
/// words that must be sent in that order ("CQ POTA"). Matching ignores case,
/// and `*` in a word matches any characters ("K-*" for a park reference).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Activity {
    pub name: String,
    pub patterns: String,
    pub color: [u8; 3],
    pub enabled: bool,
}

impl Default for Activity {
    fn default() -> Self {
        Self { name: String::new(), patterns: String::new(), color: [200, 200, 200], enabled: true }
    }
}

impl Activity {
    fn new(name: &str, patterns: &str, color: [u8; 3]) -> Self {
        Self { name: name.into(), patterns: patterns.into(), color, enabled: true }
    }

    /// The pattern found in `words` (upper-case), if any.
    pub fn find(&self, words: &[String]) -> Option<String> {
        self.patterns.split(',').map(str::trim).filter(|p| !p.is_empty()).find_map(|p| {
            let pw: Vec<String> = p.split_whitespace().map(str::to_ascii_uppercase).collect();
            let hit = words.windows(pw.len()).any(|w| w.iter().zip(&pw).all(|(a, b)| glob(b, a)));
            hit.then(|| pw.join(" "))
        })
    }
}

/// Programs and contests commonly heard on CW.
pub fn defaults() -> Vec<Activity> {
    vec![
        Activity::new("POTA", "POTA", [80, 200, 90]),
        Activity::new("SOTA", "SOTA", [170, 120, 60]),
        Activity::new("WWFF", "WWFF, CQ FF", [60, 170, 130]),
        Activity::new("IOTA", "IOTA", [60, 160, 220]),
        Activity::new("SKCC", "SKCC, CQ SKS, CQ WES", [230, 190, 60]),
        Activity::new("NAQCC", "NAQCC", [200, 160, 120]),
        Activity::new("FISTS", "FISTS", [180, 140, 200]),
        Activity::new("CWT", "CWT, CQ CWT", [240, 130, 50]),
        Activity::new("SST", "SST, CQ SST", [230, 100, 160]),
        Activity::new("MST", "MST, CQ MST", [160, 110, 230]),
        Activity::new("NAQP", "NAQP, CQ NA", [100, 140, 240]),
        Activity::new("SS", "CQ SS", [220, 80, 80]),
        Activity::new("FD", "CQ FD, CQ WFD, WFD", [120, 200, 200]),
        Activity::new("QP", "CQ QP, QSO PARTY", [150, 200, 70]),
        Activity::new("TEST", "CQ TEST, CQ TST, CQ CONTEST, TEST", [235, 70, 70]),
    ]
}

/// Upper-case words of decoded text, prosigns and all.
pub fn words(text: &str) -> Vec<String> {
    text.split_whitespace().map(str::to_ascii_uppercase).collect()
}

/// Names of the enabled activities found in `words`, each with the pattern heard.
pub fn detect(acts: &[Activity], words: &[String]) -> Vec<(String, String)> {
    acts.iter().filter(|a| a.enabled && !a.name.trim().is_empty()).filter_map(|a| a.find(words).map(|p| (a.name.clone(), p))).collect()
}

/// `*` matches any run of characters; everything else literally.
fn glob(pat: &str, s: &str) -> bool {
    match pat.split_once('*') {
        None => pat == s,
        Some((head, rest)) => {
            let Some(s) = s.strip_prefix(head) else { return false };
            (0..=s.len()).filter(|&i| s.is_char_boundary(i)).any(|i| glob(rest, &s[i..]))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(text: &str) -> Vec<String> {
        detect(&defaults(), &words(text)).into_iter().map(|(n, _)| n).collect()
    }

    #[test]
    fn recognises_common_calls() {
        assert_eq!(names("CQ POTA CQ POTA DE NR5P K"), ["POTA"]);
        assert_eq!(names("cq sst nr5p"), ["SST"]);
        assert_eq!(names("TU NR5P TEST"), ["TEST"]);
        assert_eq!(names("CQ NA NR5P"), ["NAQP"]);
        assert_eq!(names("NR5P 5NN TX SKCC 1234T"), ["SKCC"]);
        // Words have to be whole and in order.
        assert!(names("CQ CQ DE NR5P TESTING NA SS").is_empty());
    }

    #[test]
    fn wildcards_and_custom() {
        let a = vec![Activity::new("Park", "K-*", [0, 0, 0]), Activity { enabled: false, ..Activity::new("Off", "CQ", [0; 3]) }];
        let hit = |t: &str| detect(&a, &words(t));
        assert_eq!(hit("CQ DE NR5P K-1234"), [("Park".to_string(), "K-*".to_string())]);
        assert!(hit("CQ DE NR5P KK-1").is_empty());
        assert!(glob("A*B*C", "AXXBYC") && !glob("A*B", "AXBX"));
    }
}
