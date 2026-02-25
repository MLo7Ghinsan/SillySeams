use std::env;
use std::path::Path;
use std::f32::consts::PI;

use hound::{WavReader, WavWriter, WavSpec, SampleFormat};

// read shi
fn load_audio(path: &str) -> Option<(Vec<f32>, u32)> {
    let path_obj = Path::new(path);
    if !path_obj.exists() { return None; }

    let mut reader = WavReader::open(path).ok()?;
    let sr = reader.spec().sample_rate;
    
    fn average_channel_samples(samples: Vec<f32>, channels: usize) -> Vec<f32> {
        samples.chunks_exact(channels).map(|c| c.iter().sum::<f32>() / channels as f32).collect()
    }

    let audio: Vec<f32> = match reader.spec().sample_format {
        SampleFormat::Int => {
            let max_val = 2f32.powi(reader.spec().bits_per_sample as i32 - 1);
            let raw: Vec<f32> = reader.samples::<i32>().map(|s| s.unwrap_or(0) as f32 / max_val).collect();
            let averaged_raw = if reader.spec().channels > 1 {
                average_channel_samples(raw, reader.spec().channels as usize)
            } else { raw };
            averaged_raw
        }
        SampleFormat::Float => {
            let raw: Vec<f32> = reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect();
            let averaged_raw = if reader.spec().channels > 1 {
                average_channel_samples(raw, reader.spec().channels as usize)
            } else { raw };
            averaged_raw
        }
    };
    Some((audio, sr))
}

fn save_audio(path: &str, audio: &[f32], sr: u32) {
    let spec = WavSpec {
        channels: 1,
        sample_rate: sr,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec).unwrap();
    for &s in audio {
        let x = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        writer.write_sample(x).unwrap();
    }
}

fn ms_to_samples(ms: f32, sr: u32) -> usize {
    ((ms.max(0.0) / 1000.0) * sr as f32).round() as usize
}

fn calculate_rms(x: &[f32]) -> f32 {
    if x.is_empty() { return 0.0; }
    let s: f32 = x.iter().map(|&v| v * v).sum();
    (s / x.len() as f32).sqrt().max(1e-6)
}

fn str2float_length(s: &str) -> f32 {
    if let Ok(v) = s.parse::<f32>() { return v; }
    let parts: Vec<&str> = s.split('@').collect();
    if parts.len() != 2 { return 0.0; }
    let tick = parts[0].parse::<f32>().unwrap_or(0.0);
    let mut tempo_part = parts[1];
    let mut delta = 0.0;
    if let Some(idx) = tempo_part.find('+') {
        let (a, b) = tempo_part.split_at(idx);
        tempo_part = a;
        delta = b[1..].parse::<f32>().unwrap_or(0.0);
    } else if let Some(idx) = tempo_part.find('-') {
        let (a, b) = tempo_part.split_at(idx);
        tempo_part = a;
        delta = -b[1..].parse::<f32>().unwrap_or(0.0);
    }
    (60000.0 / tempo_part.parse().unwrap_or(120.0) / 480.0) * tick + delta
}

fn parse_envelope(envelope: &[f32], length_ms: f32) -> (Vec<f32>, Vec<f32>, f32) {
    if envelope.len() < 7 { return (vec![], vec![], 0.0); }
    let e = envelope;
    let (p1, p2, p3) = (e[0], e[1], e[2]);
    let (v1, v2, v3, v4) = (e[3], e[4], e[5], e[6]);
    let overlap = if e.len() >= 8 { e[7] } else { 0.0 };
    let p4 = if e.len() >= 9 { e[8] } else { 0.0 };

    let len = length_ms.max(1.0);
    let p_list = vec![0.0, p1, p1 + p2, len - p4 - p3, len - p4, len];
    let v_gain = vec![0.0, v1/100.0, v2/100.0, v3/100.0, v4/100.0, 0.0];
    (p_list, v_gain, overlap.max(0.0))
}

fn apply_envelope_waveform(audio: &mut [f32], sr: u32, p_list_ms: &[f32], v_gain: &[f32]) {
    if p_list_ms.len() < 2 || v_gain.len() < 2 { return; }
    let n = audio.len();

    let mut xp: Vec<usize> = p_list_ms.iter().map(|&ms| ms_to_samples(ms, sr).min(n)).collect();
    for i in 1..xp.len() { if xp[i] < xp[i-1] { xp[i] = xp[i-1]; } }

    for seg in 0..xp.len().saturating_sub(1) {
        let (a, b) = (xp[seg], xp[seg + 1]);
        if b <= a { continue; }
        let (ga, gb) = (v_gain[seg], v_gain[seg+1]);
        let span = (b - a) as f32;
        for i in a..b {
            let t = (i - a) as f32 / span;
            audio[i] *= ga + (gb - ga) * t;
        }
    }
}

// seam equalizer
fn equalize_seam_dynamics(existing: &mut [f32], seam_start: usize, seam_len: usize, sr: u32) {
    if seam_len == 0 || seam_start == 0 || seam_start + seam_len >= existing.len() { return; }

    let win_size = ms_to_samples(10.0, sr).max(32);
    
    let pre_rms = if seam_start > win_size {
        calculate_rms(&existing[seam_start - win_size .. seam_start]).max(1e-4)
    } else {
        calculate_rms(&existing[seam_start .. seam_start + win_size.min(seam_len)]).max(1e-4)
    };
    
    let post_start = seam_start + seam_len;
    let post_rms = if post_start + win_size <= existing.len() {
        calculate_rms(&existing[post_start .. post_start + win_size]).max(1e-4)
    } else {
        calculate_rms(&existing[post_start.saturating_sub(win_size) .. post_start]).max(1e-4)
    };

    let mut gain_curve = vec![1.0f32; seam_len];
    let half_win = win_size / 2;

    for i in 0..seam_len {
        let s = (seam_start + i).saturating_sub(half_win);
        let e = (seam_start + i + half_win).min(existing.len());
        let local_rms = calculate_rms(&existing[s..e]).max(1e-4);

        let t = i as f32 / seam_len.max(1) as f32;
        let target_rms = pre_rms * (1.0 - t) + post_rms * t;
        
        gain_curve[i] = (target_rms / local_rms).clamp(0.5, 3.5);
    }

    let mut smoothed = vec![1.0f32; seam_len];
    let smooth_win = ms_to_samples(15.0, sr).max(64);
    for i in 0..seam_len {
        let s = i.saturating_sub(smooth_win / 2);
        let e = (i + smooth_win / 2).min(seam_len);
        let sum: f32 = gain_curve[s..e].iter().sum();
        smoothed[i] = sum / (e - s).max(1) as f32;
    }

    // prevents boundary clicks
    let margin = ms_to_samples(5.0, sr).clamp(16, 256).min(seam_len / 4);
    for i in 0..seam_len {
        let mut final_gain = smoothed[i];
        
        if i < margin {
            let t = i as f32 / margin as f32;
            let curve = (PI * (t - 0.5)).sin() * 0.5 + 0.5;
            final_gain = 1.0 * (1.0 - curve) + final_gain * curve;
        } else if i >= seam_len - margin {
            let dist = seam_len - 1 - i;
            let t = dist as f32 / margin as f32;
            let curve = (PI * (t - 0.5)).sin() * 0.5 + 0.5;
            final_gain = 1.0 * (1.0 - curve) + final_gain * curve;
        }
        
        existing[seam_start + i] *= final_gain;
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 5 { return; }

    let outfile = &args[1];
    let infile = &args[2];
    let stp_ms: f32 = args[3].parse().unwrap_or(0.0);
    let length_ms: f32 = str2float_length(&args[4]);

    let mut env_args = Vec::new();
    for s in args.iter().skip(5) {
        if let Ok(v) = s.parse::<f32>() { env_args.push(v); }
    }

    let (raw_note, sr) = match load_audio(infile) {
        Some(x) => x,
        None => return,
    };

    let base_start = ms_to_samples(stp_ms, sr).min(raw_note.len());
    let want = ms_to_samples(length_ms, sr);
    
    // use overlap
    let effective_length_ms = if length_ms > 0.0 { length_ms } else { (raw_note.len() as f32) * 1000.0 / sr as f32 };
    let (p_list, v_gain, ove_ms) = parse_envelope(&env_args, effective_length_ms);
    let ovr = ms_to_samples(ove_ms, sr);

    let mut existing = load_audio(outfile).map(|x| x.0).unwrap_or_else(Vec::new);
    let mut best_start = base_start;

    // shift the start point
    // locate the perfect phase alignment before extracting the note length
    // technically bad because we shift time but well it will work
    if ovr > 32 && existing.len() >= ovr {
        let tail_a = &existing[existing.len() - ovr..];
        
        // search a 15ms window around start point- should not be a noticeable shift
        let search_radius = ms_to_samples(15.0, sr);
        let min_start = base_start.saturating_sub(search_radius);
        let max_start = (base_start + search_radius).min(raw_note.len().saturating_sub(want.max(ovr)));

        if max_start > min_start {
            let mut max_corr = f32::MIN;
            
            // cross correlate to find the exact peak match
            for s in min_start..=max_start {
                let mut corr = 0.0;
                let head_b = &raw_note[s .. s + ovr];
                for i in 0..ovr {
                    corr += tail_a[i] * head_b[i];
                }
                if corr > max_corr {
                    max_corr = corr;
                    best_start = s; // we gotta lock in yass
                }
            }
        }
    }

    // extract the note using the aligned start point.
    let end = if want > 0 { (best_start + want).min(raw_note.len()) } else { raw_note.len() };
    let mut note = raw_note[best_start..end].to_vec();
    if want > 0 && note.len() < want { note.resize(want, 0.0); }

    // standard UTAU dynamics envelope
    apply_envelope_waveform(&mut note, sr, &p_list, &v_gain);

    let final_ovr = ovr.min(existing.len()).min(note.len());

    // stitch notes together
    if final_ovr > 32 {
        let t_start = existing.len() - final_ovr;
        let tail_a = existing[t_start..].to_vec();

        // morph the shapes of the two phase-locked waves using an S curve
        for i in 0..final_ovr {
            let t = i as f32 / final_ovr as f32;
            let curve = (PI * (t - 0.5)).sin() * 0.5 + 0.5;
            existing[t_start + i] = tail_a[i] * (1.0 - curve) + note[i] * curve;
        }

        existing.extend_from_slice(&note[final_ovr..]);

        // fix the volume dip because envelope logic sucks
        equalize_seam_dynamics(&mut existing, t_start, final_ovr, sr);

    } else {
        // first note micro-fade logic for safety
        if existing.is_empty() {
            let f = ms_to_samples(2.0, sr).min(note.len());
            for i in 0..f { 
                let curve = (PI * ((i as f32 / f as f32) - 0.5)).sin() * 0.5 + 0.5;
                note[i] *= curve; 
            }
        }
        existing.extend_from_slice(&note);
    }
    save_audio(outfile, &existing, sr);
}
