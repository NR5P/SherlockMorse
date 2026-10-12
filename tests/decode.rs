//! End-to-end decoding tests on synthetic impaired CW.

use cwskimmer::lm::LanguageModel;
use cwskimmer::skimmer::{Skimmer, SkimmerConfig};
use cwskimmer::synth::{cer, render, Scene, SignalSpec};
use std::sync::Arc;

fn decode(scene: &Scene) -> Vec<(f32, String)> {
    let lm = Arc::new(LanguageModel::new());
    let mut sk = Skimmer::new(SkimmerConfig::default(), lm);
    let audio = render(scene);
    for c in audio.chunks(800) {
        sk.process(c);
    }
    sk.finish();
    let mut per: std::collections::BTreeMap<u64, (f32, String)> = Default::default();
    for e in sk.events.drain(..) {
        let ent = per.entry(e.channel).or_insert((e.freq, String::new()));
        ent.1.push_str(&e.text);
    }
    per.into_values().collect()
}

fn best_near(out: &[(f32, String)], f: f32) -> String {
    out.iter().filter(|(ff, _)| (ff - f).abs() < 25.0).map(|(_, t)| t.clone()).max_by_key(|t| t.len()).unwrap_or_default()
}

#[test]
fn noisy_hand_sent_signal() {
    let text = "CQ CQ DE W1AW W1AW K";
    let scene = Scene {
        signals: vec![SignalSpec { text: text.into(), wpm: 18.0, snr_db: 3.0, jitter: 0.12, dash_ratio: 3.3, ..Default::default() }],
        seed: 7,
        ..Default::default()
    };
    let out = decode(&scene);
    let got = best_near(&out, 700.0);
    assert!(cer(text, &got) < 0.1, "got {got:?}");
}

#[test]
fn pileup_decodes_every_station() {
    let texts = ["CQ TEST K1ABC K1ABC", "DL5XYZ DE G4ABC 599 TU", "QRZ? DE JA1XYZ K"];
    let freqs = [500.0, 640.0, 900.0];
    let scene = Scene {
        signals: (0..3)
            .map(|i| SignalSpec { text: texts[i].into(), freq: freqs[i], wpm: [20.0, 26.0, 16.0][i], snr_db: 10.0, start: 0.3 * i as f32, ..Default::default() })
            .collect(),
        seed: 3,
        ..Default::default()
    };
    let out = decode(&scene);
    for i in 0..3 {
        let got = best_near(&out, freqs[i]);
        assert!(cer(texts[i], &got) < 0.1, "{} Hz: got {got:?}", freqs[i]);
    }
}

#[test]
fn noise_only_produces_no_text() {
    let scene = Scene { signals: vec![SignalSpec { text: "".into(), snr_db: 0.0, ..Default::default() }], tail: 20.0, seed: 11, ..Default::default() };
    let out = decode(&scene);
    let total: usize = out.iter().map(|(_, t)| t.trim().len()).sum();
    assert!(total < 10, "decoded noise: {out:?}");
}
