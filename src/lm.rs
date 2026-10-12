//! Character trigram language model for amateur-radio CW traffic.
//!
//! The decoder combines element-timing likelihoods with this prior so that an
//! ambiguous mark/space is resolved toward text that looks like real CW
//! traffic (QSOs, contest exchanges, callsigns, plain English).
//!
//! The model is trained at start-up from a procedurally generated corpus so no
//! data files need to ship with the program.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

/// Symbols the model knows. Index 0 is the word space.
pub const ALPHABET: &[u8] = b" ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/?.,=+(%&-':@";

pub fn sym_index(c: u8) -> usize {
    // Small table; linear search is fine at build time, runtime uses SYM_LUT.
    ALPHABET.iter().position(|&a| a == c).unwrap_or(0)
}

pub struct LanguageModel {
    n: usize,
    /// Average -log P per character on the training text. Added back for
    /// every emitted character so the decoder is not biased toward
    /// hypotheses that emit fewer characters.
    pub char_bonus: f32,
    /// log P(c | a, b), indexed [a][b][c]
    tri: Vec<f32>,
    lut: [u8; 256],
}

impl Default for LanguageModel {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguageModel {
    /// Builds the model from the built-in corpus.
    pub fn new() -> Self {
        let corpus = generate_corpus(400_000, 0xC0FFEE);
        Self::train(&corpus)
    }

    /// A model that assigns uniform probabilities (disables prediction).
    pub fn uniform() -> Self {
        let n = ALPHABET.len();
        let v = -(n as f32).ln();
        Self { n, char_bonus: -v, tri: vec![v; n * n * n], lut: make_lut() }
    }

    pub fn train(corpus: &str) -> Self {
        let n = ALPHABET.len();
        let lut = make_lut();
        let mut uni = vec![1.0f64; n];
        let mut bi = vec![0.0f64; n * n];
        let mut tri = vec![0.0f64; n * n * n];
        let bytes: Vec<usize> = corpus
            .bytes()
            .map(|c| lut[c.to_ascii_uppercase() as usize] as usize)
            .collect();
        for i in 0..bytes.len() {
            let c = bytes[i];
            uni[c] += 1.0;
            if i >= 1 {
                bi[bytes[i - 1] * n + c] += 1.0;
            }
            if i >= 2 {
                tri[(bytes[i - 2] * n + bytes[i - 1]) * n + c] += 1.0;
            }
        }
        let uni_tot: f64 = uni.iter().sum();
        let mut out = vec![0.0f32; n * n * n];
        // Interpolated smoothing; weights favour context but never zero out
        // anything since callsigns are close to random strings.
        let (l3, l2, l1, l0) = (0.55, 0.25, 0.15, 0.05);
        for a in 0..n {
            for b in 0..n {
                let bi_tot: f64 = bi[b * n..(b + 1) * n].iter().sum();
                let tri_tot: f64 = tri[(a * n + b) * n..(a * n + b + 1) * n].iter().sum();
                for c in 0..n {
                    let p1 = uni[c] / uni_tot;
                    let p2 = if bi_tot > 0.0 { bi[b * n + c] / bi_tot } else { p1 };
                    let p3 = if tri_tot > 0.0 { tri[(a * n + b) * n + c] / tri_tot } else { p2 };
                    let p = l3 * p3 + l2 * p2 + l1 * p1 + l0 / n as f64;
                    out[(a * n + b) * n + c] = p.ln() as f32;
                }
            }
        }
        let mut ce = 0.0f64;
        for i in 2..bytes.len() {
            ce -= out[(bytes[i - 2] * n + bytes[i - 1]) * n + bytes[i]] as f64;
        }
        let char_bonus = (ce / (bytes.len().max(3) - 2) as f64) as f32;
        Self { n, char_bonus, tri: out, lut }
    }

    /// log P(c | a b) where a, b, c are ASCII characters.
    #[inline]
    pub fn logp(&self, a: u8, b: u8, c: u8) -> f32 {
        let ia = self.lut[a as usize] as usize;
        let ib = self.lut[b as usize] as usize;
        let ic = self.lut[c as usize] as usize;
        self.tri[(ia * self.n + ib) * self.n + ic]
    }

    /// Average log-probability per character of a string (for diagnostics).
    pub fn score(&self, text: &str) -> f32 {
        let b = text.as_bytes();
        let mut tot = 0.0;
        let (mut a, mut p) = (b' ', b' ');
        for &c in b {
            tot += self.logp(a, p, c);
            a = p;
            p = c;
        }
        tot / b.len().max(1) as f32
    }
}

fn make_lut() -> [u8; 256] {
    let mut lut = [0u8; 256];
    for (i, &c) in ALPHABET.iter().enumerate() {
        lut[c as usize] = i as u8;
        lut[c.to_ascii_lowercase() as usize] = i as u8;
    }
    lut
}

const NAMES: &[&str] = &[
    "BOB", "JOHN", "JIM", "BILL", "TOM", "MIKE", "DAVE", "DAN", "ED", "AL", "JOE", "RON", "DON",
    "KEN", "STEVE", "GARY", "LARRY", "PAUL", "PETE", "FRANK", "GEORGE", "HANS", "KLAUS", "PIERRE",
    "YURI", "TARO", "ANN", "SUE", "MARY", "LIZ", "CHUCK", "RICH", "RAY", "WALT", "HANK", "ART",
    "FRED", "JACK", "SAM", "TONY", "VIC", "LEE", "GENE", "JERRY", "BRUCE", "MARK", "RICK", "ROB",
];
const QTHS: &[&str] = &[
    "BOSTON", "NEW YORK", "CHICAGO", "DENVER", "DALLAS", "SEATTLE", "PORTLAND", "ATLANTA", "MIAMI",
    "PHOENIX", "TEXAS", "OHIO", "MAINE", "IOWA", "UTAH", "IDAHO", "NEAR LA", "BERLIN", "MUNICH",
    "LONDON", "PARIS", "ROME", "MADRID", "TOKYO", "SYDNEY", "TORONTO", "OTTAWA", "MOSCOW",
    "PRAGUE", "VIENNA", "OSLO", "HELSINKI", "STOCKHOLM", "DUBLIN", "SAN DIEGO", "NEWINGTON",
];
const STATES: &[&str] = &[
    "MA", "CT", "NH", "VT", "ME", "RI", "NY", "NJ", "PA", "DE", "MD", "VA", "NC", "SC", "GA", "FL",
    "AL", "TN", "KY", "OH", "MI", "IN", "IL", "WI", "MN", "IA", "MO", "AR", "LA", "TX", "OK", "KS",
    "NE", "SD", "ND", "MT", "WY", "CO", "NM", "AZ", "UT", "NV", "ID", "WA", "OR", "CA", "AK", "HI",
    "ON", "QC", "BC", "AB", "DX",
];
const RIGS: &[&str] = &[
    "FTDX3000", "FTDX101", "FT991", "FT891", "FT817", "IC7300", "IC7610", "IC705", "K3", "KX3",
    "TS590", "TS890", "K4", "FLEX", "HOMEBREW", "QRP",
];
const ANTS: &[&str] = &[
    "DIPOLE", "VERTICAL", "YAGI", "BEAM", "LOOP", "EFHW", "G5RV", "WIRE", "HEX BEAM", "INV V",
    "3 EL YAGI", "LONG WIRE",
];
const WX: &[&str] = &["SUNNY", "CLOUDY", "RAIN", "SNOW", "WARM", "COLD", "HOT", "FOGGY", "WINDY", "NICE", "CLR"];

/// Common English words (W1AW practice text, ragchews).
const ENGLISH: &[&str] = &[
    "THE", "OF", "AND", "TO", "A", "IN", "IS", "THAT", "FOR", "IT", "AS", "WAS", "WITH", "BE", "BY",
    "ON", "NOT", "HE", "THIS", "ARE", "OR", "HIS", "FROM", "AT", "WHICH", "BUT", "HAVE", "AN",
    "HAD", "THEY", "YOU", "WERE", "THEIR", "ONE", "ALL", "WE", "CAN", "HER", "HAS", "THERE",
    "BEEN", "IF", "MORE", "WHEN", "WILL", "WOULD", "WHO", "SO", "NO", "OUT", "UP", "INTO", "THAN",
    "THEM", "ABOUT", "ITS", "TIME", "ONLY", "NEW", "SOME", "COULD", "THESE", "TWO", "MAY", "FIRST",
    "THEN", "DO", "ANY", "LIKE", "MY", "NOW", "OVER", "SUCH", "OUR", "MAN", "ME", "EVEN", "MOST",
    "MADE", "AFTER", "ALSO", "DID", "MANY", "OFF", "BEFORE", "MUST", "WELL", "BACK", "THROUGH",
    "YEARS", "WHERE", "MUCH", "YOUR", "WAY", "DOWN", "SHOULD", "BECAUSE", "EACH", "JUST", "THOSE",
    "PEOPLE", "HOW", "TOO", "LITTLE", "STATE", "GOOD", "VERY", "MAKE", "WORLD", "STILL", "SEE",
    "OWN", "MEN", "WORK", "LONG", "HERE", "GET", "BOTH", "BETWEEN", "LIFE", "BEING", "UNDER",
    "NEVER", "DAY", "SAME", "ANOTHER", "KNOW", "WHILE", "LAST", "MIGHT", "US", "GREAT", "OLD",
    "YEAR", "COME", "SINCE", "AGAINST", "GO", "CAME", "RIGHT", "USED", "TAKE", "THREE", "RADIO",
    "AMATEUR", "STATION", "OPERATORS", "FREQUENCY", "BAND", "SIGNAL", "ANTENNA", "POWER",
    "LICENSE", "CODE", "MORSE", "PRACTICE", "LEAGUE", "RELAY", "AMERICAN", "NATIONAL", "SERVICE",
    "EMERGENCY", "COMMUNICATIONS", "NETWORK", "SYSTEM", "TRAFFIC", "MESSAGE", "CONTEST", "FIELD",
    "EQUIPMENT", "TRANSMITTER", "RECEIVER", "PROGRAM", "PUBLIC", "INFORMATION", "MEMBERS",
    "SPECTRUM", "COMMISSION", "FEDERAL", "RULES", "SATELLITE", "SPACE", "SOLAR", "WEATHER",
    "WHAT", "SAID", "PART", "PLACE", "HIGH", "EVERY", "NEAR", "ADD", "FOOD", "BELOW", "COUNTRY",
    "PLANT", "SCHOOL", "FATHER", "KEEP", "TREE", "NEVER", "START", "CITY", "EARTH", "EYES", "LIGHT",
    "THOUGHT", "HEAD", "STORY", "SAW", "FAR", "SEA", "DRAW", "LEFT", "LATE", "RUN", "WHILE",
    "PRESS", "CLOSE", "NIGHT", "REAL", "FEW", "NORTH", "OPEN", "SEEM", "TOGETHER", "NEXT", "WHITE",
    "CHILDREN", "BEGIN", "GOT", "WALK", "EXAMPLE", "EASE", "PAPER", "GROUP", "ALWAYS", "MUSIC",
    "MARK", "OFTEN", "LETTER", "UNTIL", "MILE", "RIVER", "CAR", "FEET", "CARE", "SECOND", "BOOK",
    "CARRY", "TOOK", "SCIENCE", "EAT", "ROOM", "FRIEND", "BEGAN", "IDEA", "FISH", "MOUNTAIN",
    "STOP", "ONCE", "BASE", "HEAR", "HORSE", "CUT", "SURE", "WATCH", "COLOR", "FACE", "WOOD",
    "MAIN", "ENOUGH", "PLAIN", "GIRL", "USUAL", "YOUNG", "READY", "ABOVE", "EVER", "RED", "LIST",
    "THOUGH", "FEEL", "TALK", "BIRD", "SOON", "BODY", "DOG", "FAMILY", "DIRECT", "POSE", "LEAVE",
    "SONG", "MEASURE", "DOOR", "PRODUCT", "BLACK", "SHORT", "NUMERAL", "CLASS", "WIND", "QUESTION",
    "HAPPEN", "COMPLETE", "SHIP", "AREA", "HALF", "ROCK", "ORDER", "FIRE", "SOUTH", "PROBLEM",
    "PIECE", "TOLD", "KNEW", "PASS", "FARM", "TOP", "WHOLE", "KING", "SIZE", "HEARD", "BEST",
    "HOUR", "BETTER", "TRUE", "DURING", "HUNDRED", "FIVE", "REMEMBER", "STEP", "EARLY", "HOLD",
    "WEST", "GROUND", "INTEREST", "REACH", "FAST", "VERB", "SING", "LISTEN", "SIX", "TABLE",
    "TRAVEL", "LESS", "MORNING", "TEN", "SIMPLE", "SEVERAL", "VOWEL", "TOWARD", "WAR", "LAY",
    "AGAINST", "PATTERN", "SLOW", "CENTER", "LOVE", "PERSON", "MONEY", "SERVE", "APPEAR", "ROAD",
    "MAP", "RAIN", "RULE", "GOVERN", "PULL", "COLD", "NOTICE", "VOICE", "UNIT", "POWER", "TOWN",
    "FINE", "CERTAIN", "FLY", "FALL", "LEAD", "CRY", "DARK", "MACHINE", "NOTE", "WAIT", "PLAN",
    "FIGURE", "STAR", "BOX", "NOUN", "REST", "CORRECT", "ABLE", "POUND", "DONE", "BEAUTY", "DRIVE",
    "STOOD", "CONTAIN", "FRONT", "TEACH", "WEEK", "FINAL", "GAVE", "GREEN", "OH", "QUICK",
    "DEVELOP", "OCEAN", "WARM", "FREE", "MINUTE", "STRONG", "SPECIAL", "MIND", "BEHIND", "CLEAR",
    "TAIL", "PRODUCE", "FACT", "STREET", "INCH", "MULTIPLY", "NOTHING", "COURSE", "STAY", "WHEEL",
    "FULL", "FORCE", "BLUE", "OBJECT", "DECIDE", "SURFACE", "DEEP", "MOON", "ISLAND", "FOOT",
    "BUSY", "TEST", "RECORD", "BOAT", "COMMON", "GOLD", "POSSIBLE", "PLANE", "STEAD", "DRY",
    "WONDER", "LAUGH", "THOUSAND", "AGO", "RAN", "CHECK", "GAME", "SHAPE", "EQUATE", "MISS",
    "BROUGHT", "HEAT", "SNOW", "TIRE", "BRING", "YES", "DISTANT", "FILL", "EAST", "PAINT",
    "LANGUAGE", "AMONG", "QUIET", "QUITE", "QUALITY", "EQUIP", "JUDGE", "JOIN", "JUMP", "ZERO",
    "ZONE", "SIZE", "EXTRA", "EXACT", "BOX", "VALVE", "VERSION", "WAVE", "WAVES", "OXYGEN",
];

const ABBREVS: &[&str] = &[
    "TNX", "TKS", "TU", "FB", "OM", "YL", "XYL", "ES", "HR", "UR", "RST", "NAME", "QTH", "WX",
    "ANT", "RIG", "PWR", "HW", "CPY", "AGN", "PSE", "BK", "KN", "SK", "CUL", "CUAGN", "GL", "GM",
    "GA", "GE", "GN", "DR", "VY", "73", "88", "QSL", "QRL", "QRZ", "QSB", "QRM", "QRN", "QRS",
    "QRQ", "QSY", "QRP", "QRO", "QTH", "QSO", "RPT", "SIG", "SIGS", "FER", "WID", "ABT", "NR",
    "HPE", "HV", "OP", "SRI", "TEMP", "WKD", "WRK", "BCNU", "DX", "TEST", "CQ", "DE", "R", "K",
    "5NN", "599", "579", "559", "339", "449", "569", "589", "ENN", "TT", "CFM", "MNI", "INFO",
];

pub fn random_call(rng: &mut impl Rng) -> String {
    const PFX: &[&str] = &[
        "K", "W", "N", "K", "W", "N", "AA", "AB", "AC", "AD", "AE", "AF", "AG", "AI", "AJ", "AK",
        "KA", "KB", "KC", "KD", "KE", "KF", "KG", "KI", "KJ", "KK", "KN", "KO", "KW", "WA", "WB",
        "WD", "N", "NA", "NO", "VE", "VA", "G", "M", "M0", "2E", "GM", "GW", "EI", "DL", "DK",
        "DJ", "DF", "F", "I", "IK", "IZ", "EA", "EC", "CT", "ON", "PA", "PD", "OH", "SM", "SA",
        "LA", "OZ", "OK", "OM", "SP", "SQ", "HA", "YO", "LZ", "9A", "S5", "YU", "UA", "RA", "RW",
        "UR", "UT", "LY", "ES", "YL", "JA", "JH", "JR", "BG", "BY", "HL", "VK", "ZL", "ZS", "PY",
        "LU", "CE", "XE", "CO", "KP", "KH", "KL", "VU", "4X", "SV", "TA", "HB", "OE", "LX", "EA8",
    ];
    let mut s = PFX.choose(rng).unwrap().to_string();
    if !s.chars().last().unwrap().is_ascii_digit() {
        s.push((b'0' + rng.gen_range(0..10)) as char);
    }
    let n = match rng.gen_range(0..10) {
        0..=1 => 1,
        2..=5 => 2,
        _ => 3,
    };
    for _ in 0..n {
        s.push((b'A' + rng.gen_range(0..26)) as char);
    }
    if rng.gen_ratio(1, 25) {
        s.push_str(["/P", "/M", "/QRP", "/1", "/2", "/3"].choose(rng).unwrap());
    }
    s
}

fn rst(rng: &mut impl Rng) -> String {
    ["599", "579", "589", "559", "569", "449", "479", "339", "5NN", "599", "57N", "55N"]
        .choose(rng)
        .unwrap()
        .to_string()
}

fn english_sentence(rng: &mut impl Rng) -> String {
    let n = rng.gen_range(4..16);
    let mut w: Vec<&str> = Vec::with_capacity(n);
    for _ in 0..n {
        w.push(ENGLISH.choose(rng).unwrap());
    }
    let mut s = w.join(" ");
    s.push_str([" .", " ", " ", ",", " ?"].choose(rng).unwrap());
    s
}

/// Generates a pseudo-random corpus of typical CW traffic.
pub fn generate_corpus(target_len: usize, seed: u64) -> String {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = String::with_capacity(target_len + 256);
    while out.len() < target_len {
        let c1 = random_call(&mut rng);
        let c2 = random_call(&mut rng);
        let name = NAMES.choose(&mut rng).unwrap();
        let qth = QTHS.choose(&mut rng).unwrap();
        let st = STATES.choose(&mut rng).unwrap();
        let line = match rng.gen_range(0..20) {
            0 | 1 => format!("CQ CQ CQ DE {c1} {c1} {c1} K"),
            2 => format!("CQ CQ DE {c1} {c1} K"),
            3 => format!("CQ TEST {c1} {c1}"),
            4 => format!("{c1} DE {c2} {c2} K"),
            5 => format!(
                "{c1} DE {c2} GM ES TNX FER CALL UR RST {r} {r} NAME {name} {name} QTH {qth} {qth} HW CPY? {c1} DE {c2} KN",
                r = rst(&mut rng)
            ),
            6 => format!(
                "R R TNX {name} UR RST {r} ES FB SIG HR RIG {rig} PWR {p}W ANT {ant} WX {wx} TEMP {t}F",
                r = rst(&mut rng),
                rig = RIGS.choose(&mut rng).unwrap(),
                p = [5, 10, 50, 100, 100, 200, 500][rng.gen_range(0..7)],
                ant = ANTS.choose(&mut rng).unwrap(),
                wx = WX.choose(&mut rng).unwrap(),
                t = rng.gen_range(10..99)
            ),
            7 => format!("{c1} 5NN {st}"),
            8 => format!("TU {c1} 5NN {}", rng.gen_range(1..2000)),
            9 => format!("{c2} TU 5NN {st} {c1}"),
            10 => format!("73 {name} TU {c1} DE {c2} %  EE"),
            11 => format!("{} {}", ABBREVS.choose(&mut rng).unwrap(), ABBREVS.choose(&mut rng).unwrap()),
            12 => format!("QRZ? {c1}"),
            13 => format!("AGN? {c2} {c2}"),
            14 => format!("FB {name} = HPE CUAGN = 73 ES GL {c2} DE {c1} % TU"),
            15..=19 => english_sentence(&mut rng),
            _ => unreachable!(),
        };
        out.push_str(&line);
        out.push(' ');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lm_prefers_plausible_text() {
        let lm = LanguageModel::new();
        assert!(lm.score("CQ CQ DE W1AW") > lm.score("XQ JQ ZE W1AW"));
        assert!(lm.score("THE ANTENNA") > lm.score("TIE ANTENNA"));
    }
}
