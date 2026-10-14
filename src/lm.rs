//! Character trigram language model for amateur-radio CW traffic.
//!
//! The decoder combines element-timing likelihoods with this prior so that an
//! ambiguous mark/space is resolved toward text that looks like real CW
//! traffic (QSOs, contest exchanges, callsigns, plain English).
//!
//! The model is trained at start-up from text compiled into the program:
//! on-air QSO traffic and plain English (`data/corpus/`), plus a
//! procedurally generated corpus for callsigns and exchanges. A dense
//! character trigram is extended to 5-grams (interpolated Witten-Bell
//! smoothing) for the contexts seen in training. A vocabulary of common
//! words is available for judging decoded text.

use rand::rngs::StdRng;
use std::collections::{HashMap, HashSet};
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
    /// Higher orders, keyed by `key(context, c)`: log P(c | 3 or 4
    /// preceding symbols) for continuations seen in training, and per
    /// context the log weight given to the lower order for unseen ones.
    hi: HashMap<u32, f32>,
    backoff: HashMap<u32, f32>,
    /// Known words (English, abbreviations, Q-codes, names).
    vocab: HashSet<Vec<u8>>,
    /// Longer known words with their Morse renderings, for `correct`.
    vocab_morse: Vec<(Vec<u8>, Vec<u8>)>,
    /// A large English dictionary: words in it are never corrected.
    dict: HashSet<&'static [u8]>,
}

/// Packs up to 4 context symbols (oldest first) and the predicted symbol.
/// Symbols are < 64; the order is in the top bits so orders cannot collide.
#[inline]
fn key(ctx: &[usize], c: usize) -> u32 {
    let mut k = ctx.len() as u32;
    for &s in ctx {
        k = (k << 6) | s as u32;
    }
    (k << 6) | c as u32
}

const QSO_TEXT: &str = include_str!("../data/corpus/qso.txt");
const ENGLISH_TEXT: &str = include_str!("../data/corpus/english.txt");
const WORDS: &str = include_str!("../data/corpus/words.txt");
/// English words of 5+ letters (SCOWL, see data/corpus/LICENSE-SCOWL.txt):
/// real words that `correct` must never change.
const DICT: &str = include_str!("../data/corpus/dict.txt");

/// Training text from the built-in corpora, with punctuation that is rarely
/// keyed removed: periods become the break sign in QSOs and disappear from
/// prose, as do commas and apostrophes.
fn corpus_lines() -> Vec<String> {
    let mut v = Vec::new();
    for l in QSO_TEXT.lines() {
        v.push(l.replace(". ", " = ").replace('.', " ").replace('\'', ""));
    }
    for l in ENGLISH_TEXT.lines() {
        v.push(l.replace(['.', ',', '\''], " "));
    }
    v.into_iter()
        .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|l| !l.is_empty())
        .collect()
}

impl Default for LanguageModel {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguageModel {
    /// Builds the model from the built-in corpus.
    pub fn new() -> Self {
        let mut train = generate_corpus(300_000, 0xC0FFEE);
        let mut held = generate_corpus(20_000, 0xBEEF);
        for (i, l) in corpus_lines().iter().enumerate() {
            // Every 20th line is held out to measure the per-character cost.
            let t = if i % 20 == 7 { &mut held } else { &mut train };
            t.push(' ');
            t.push_str(l);
        }
        let mut m = Self::train(&train);
        m.char_bonus = m.cross_entropy(&held);
        m.vocab = WORDS.lines().map(|w| w.trim().as_bytes().to_vec()).filter(|w| !w.is_empty()).collect();
        m.dict = DICT.lines().map(str::as_bytes).collect();
        // Correction targets: the common vocabulary only. (Taking them
        // from the whole dictionary fixed as many words as it broke on live
        // recordings: WEEKLY, AURORA, but also DURNING -> BURNING,
        // ETHAT -> ETHAN.)
        m.vocab_morse = m.vocab.iter().filter(|w| w.len() >= 4 && w.iter().all(u8::is_ascii_uppercase)).map(|w| (w.clone(), morse_render(w))).collect();
        m
    }

    /// A model that assigns uniform probabilities (disables prediction).
    pub fn uniform() -> Self {
        let n = ALPHABET.len();
        let v = -(n as f32).ln();
        Self {
            n,
            char_bonus: -v,
            tri: vec![v; n * n * n],
            lut: make_lut(),
            hi: HashMap::new(),
            backoff: HashMap::new(),
            vocab: HashSet::new(),
            vocab_morse: Vec::new(),
            dict: HashSet::new(),
        }
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
        let mut m = Self { n, char_bonus: 0.0, tri: out, lut, hi: HashMap::new(), backoff: HashMap::new(), vocab: HashSet::new(), vocab_morse: Vec::new(), dict: HashSet::new() };
        m.train_high(&bytes);
        m.char_bonus = m.cross_entropy(corpus);
        m
    }

    /// Adds orders 4 and 5 by interpolated Witten-Bell smoothing over the
    /// trigram: P(c|h) = (C(h c) + T(h) P(c|h')) / (C(h) + T(h)), where
    /// T(h) is the number of distinct symbols seen after h.
    fn train_high(&mut self, bytes: &[usize]) {
        for order in 3..=4usize {
            let mut counts: HashMap<u32, u32> = HashMap::new();
            let mut ctx_tot: HashMap<u32, (u32, u32)> = HashMap::new();
            for i in order..bytes.len() {
                let k = key(&bytes[i - order..i], bytes[i]);
                let e = counts.entry(k).or_insert(0);
                *e += 1;
                let t = ctx_tot.entry(key(&bytes[i - order..i], 0)).or_insert((0, 0));
                t.0 += 1;
                if *e == 1 {
                    t.1 += 1;
                }
            }
            let mut probs = Vec::with_capacity(counts.len());
            for (&k, &c) in &counts {
                let sym = (k & 63) as usize;
                let ctx = unpack(k >> 6, order);
                let (tot, types) = ctx_tot[&key(&ctx, 0)];
                let lower = self.logp_ctx(&ctx[1..], sym).exp() as f64;
                let p = (c as f64 + types as f64 * lower) / (tot + types) as f64;
                probs.push((k, p.ln() as f32));
            }
            for (k, lp) in probs {
                self.hi.insert(k, lp);
            }
            for (k, (tot, types)) in ctx_tot {
                self.backoff.insert(k, (types as f64 / (tot + types) as f64).ln() as f32);
            }
        }
    }

    /// log P(c | ctx) for up to 4 context symbols (oldest first).
    fn logp_ctx(&self, ctx: &[usize], c: usize) -> f32 {
        if ctx.len() <= 2 {
            let (a, b) = match ctx {
                [a, b] => (*a, *b),
                [b] => (0, *b),
                _ => (0, 0),
            };
            return self.tri[(a * self.n + b) * self.n + c];
        }
        if let Some(&lp) = self.hi.get(&key(ctx, c)) {
            return lp;
        }
        let bo = self.backoff.get(&key(ctx, 0)).copied().unwrap_or(0.0);
        bo + self.logp_ctx(&ctx[1..], c)
    }

    /// log P(c | the four preceding characters, oldest first). The 5-gram
    /// is mixed half and half with the trigram: the corpus is small and
    /// repetitive, so on its own it is far too sure that text it has not
    /// seen (an unusual name, a callsign) is wrong.
    #[inline]
    pub fn logp4(&self, ctx: &[u8; 4], c: u8) -> f32 {
        let l = |x: u8| self.lut[x as usize] as usize;
        let (a, b) = (l(ctx[2]), l(ctx[3]));
        let p3 = self.tri[(a * self.n + b) * self.n + l(c)];
        if self.hi.is_empty() {
            return p3;
        }
        let p5 = self.logp_ctx(&[l(ctx[0]), l(ctx[1]), a, b], l(c));
        let (hi, lo) = if p5 > p3 { (p5, p3) } else { (p3, p5) };
        hi + (0.5 + 0.5 * (lo - hi).exp()).ln()
    }

    /// Average -log P per character of a text under the full model.
    fn cross_entropy(&self, text: &str) -> f32 {
        let b = text.as_bytes();
        let mut ctx = [b' '; 4];
        let mut ce = 0.0f64;
        for &c in b {
            let c = c.to_ascii_uppercase();
            ce -= self.logp4(&ctx, c) as f64;
            ctx = [ctx[1], ctx[2], ctx[3], c];
        }
        (ce / b.len().max(1) as f64) as f32
    }

    /// True if the word is in the vocabulary.
    pub fn is_word(&self, w: &[u8]) -> bool {
        self.vocab.contains(w)
    }

    /// True if a model with a vocabulary was loaded.
    pub fn has_vocab(&self) -> bool {
        !self.vocab.is_empty()
    }

    /// The known word a garbled one most likely was, if exactly one known
    /// word is the garbled one plus a single lost dit: a dit missing inside
    /// a letter, or where two letters join into one (TM + dit = Y). That is
    /// the commonest error in fades and with clipped keying; other single
    /// edits (an extra dit, a dash for a dot) proved ambiguous on real
    /// signals. Only plain words of 5+ letters that are not known
    /// themselves are considered, so callsigns, numbers, abbreviations and
    /// short words are left alone.
    pub fn correct(&self, w: &[u8]) -> Option<&[u8]> {
        if w.len() < 5 || !w.iter().all(u8::is_ascii_uppercase) || self.vocab.contains(w) || self.dict.contains(w) {
            return None;
        }
        // Two known words run together (a squeezed word gap, "U EVER") is
        // as likely as a lost dit: leave those alone.
        let known = |p: &[u8]| self.vocab.contains(p) || matches!(p, b"A" | b"I" | b"U" | b"R");
        if (1..w.len()).any(|i| known(&w[..i]) && known(&w[i..])) {
            return None;
        }
        let m = morse_render(w);
        let mut found: Option<&[u8]> = None;
        for (word, wm) in &self.vocab_morse {
            if wm.len().abs_diff(m.len()) <= 1 && (lost_dit(&m, wm) || split_letter(&m, wm)) {
                if found.is_some() {
                    return None;
                }
                found = Some(word);
            }
        }
        found
    }

    /// log P(c | a b) where a, b, c are ASCII characters.
    #[inline]
    pub fn logp(&self, a: u8, b: u8, c: u8) -> f32 {
        let ia = self.lut[a as usize] as usize;
        let ib = self.lut[b as usize] as usize;
        let ic = self.lut[c as usize] as usize;
        self.tri[(ia * self.n + ib) * self.n + ic]
    }

    /// Average log-probability per character of a string.
    pub fn score(&self, text: &str) -> f32 {
        -self.cross_entropy(text)
    }
}

/// Morse elements of a word, characters separated by spaces.
fn morse_render(w: &[u8]) -> Vec<u8> {
    let mut v = Vec::new();
    for &c in w {
        if !v.is_empty() {
            v.push(b' ');
        }
        v.extend_from_slice(crate::morse::pattern_for(c).unwrap_or("#").as_bytes());
    }
    v
}

/// True if Morse rendering `word` is `got` with one dit restored: a '.'
/// inserted, or a character break (' ') that was really a '.'.
fn lost_dit(got: &[u8], word: &[u8]) -> bool {
    let pre = got.iter().zip(word).take_while(|(x, y)| x == y).count();
    if pre >= word.len() || word[pre] != b'.' {
        return false;
    }
    if word.len() == got.len() + 1 {
        got[pre..] == word[pre + 1..]
    } else if word.len() == got.len() {
        got[pre] == b' ' && got[pre + 1..] == word[pre + 1..]
    } else {
        false
    }
}

/// True if Morse rendering `word` is `got` with one character break
/// removed: a letter split in two because a fade stretched the gap inside
/// it (WEEKLY copied as WEEKEDY: L = E + D).
fn split_letter(got: &[u8], word: &[u8]) -> bool {
    if got.len() != word.len() + 1 {
        return false;
    }
    let pre = got.iter().zip(word).take_while(|(x, y)| x == y).count();
    got[pre] == b' ' && got[pre + 1..] == word[pre..]
}

fn unpack(mut k: u32, order: usize) -> Vec<usize> {
    let mut v = vec![0usize; order];
    for i in (0..order).rev() {
        v[i] = (k & 63) as usize;
        k >>= 6;
    }
    v
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
    fn corrects_lost_dits_only() {
        let lm = LanguageModel::new();
        assert_eq!(lm.correct(b"TMESTERDAY"), Some(&b"YESTERDAY"[..]));
        // Other single edits are not corrected, nor known words.
        assert_eq!(lm.correct(b"UEVER"), None);
        assert_eq!(lm.correct(b"ETHAT"), None);
        assert_eq!(lm.correct(b"WEATHER"), None);
        // A real word outside the common vocabulary is left alone.
        assert_eq!(lm.correct(b"STACK"), None);
        // A letter split in two by a fade (LITTLE: L = E + D).
        assert_eq!(lm.correct(b"EDITTLE"), Some(&b"LITTLE"[..]));
    }

    #[test]
    fn lm_prefers_plausible_text() {
        let lm = LanguageModel::new();
        assert!(lm.score("CQ CQ DE W1AW") > lm.score("XQ JQ ZE W1AW"));
        assert!(lm.score("THE ANTENNA") > lm.score("TIE ANTENNA"));
    }
}
