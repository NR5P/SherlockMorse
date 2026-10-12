//! Reads audio files (WAV, MP3, FLAC, OGG, M4A/AAC) into mono f32 samples.

use anyhow::{anyhow, Context, Result};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Returns (mono samples, sample rate).
pub fn read_audio(path: &std::path::Path) -> Result<(Vec<f32>, u32)> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .context("unsupported audio format")?;
    let mut format = probed.format;
    let track = format.default_track().ok_or_else(|| anyhow!("no audio track"))?;
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
    let mut rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut out = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(_)) => break,
            Err(symphonia::core::errors::Error::ResetRequired) => break,
            Err(e) => return Err(e.into()),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(e.into()),
        };
        let spec = *decoded.spec();
        rate = spec.rate;
        let ch = spec.channels.count().max(1);
        let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        buf.copy_interleaved_ref(decoded);
        for frame in buf.samples().chunks(ch) {
            out.push(frame.iter().sum::<f32>() / ch as f32);
        }
    }
    if rate == 0 {
        return Err(anyhow!("unknown sample rate"));
    }
    Ok((out, rate))
}

pub fn write_wav(path: &std::path::Path, samples: &[f32], rate: u32) -> Result<()> {
    let spec = hound::WavSpec { channels: 1, sample_rate: rate, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut w = hound::WavWriter::create(path, spec)?;
    let peak = samples.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-9);
    let g = if peak > 1.0 { 0.95 / peak } else { 1.0 };
    for &s in samples {
        w.write_sample((s * g * 32767.0) as i16)?;
    }
    w.finalize()?;
    Ok(())
}
