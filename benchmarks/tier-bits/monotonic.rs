//! Every f32 in [0, 1], in order, through RowConverter: does a larger input ever give a smaller output?
use zenpixels::{ChannelLayout as L, ChannelType as T, PixelDescriptor as D, TransferFunction as Tf};
use zenpixels_convert::{ConvertOptions, RowConverter};

const CHUNK: usize = 1 << 20;
const TOP: u32 = 0x3f80_0000;

struct Found { count: u64, worst: f64, first: Option<(u32, f64, f64)>, last_input: u32, below: [u64; 4] }

fn scan<O: bytemuck::Pod + Copy + Into<f64>>(mut c: RowConverter) -> Found {
    let mut input = vec![0f32; CHUNK];
    let mut out = vec![<O as bytemuck::Zeroable>::zeroed(); CHUNK];
    let (mut prev, mut found) = (f64::NEG_INFINITY, Found { count: 0, worst: 0.0, first: None, last_input: 0, below: [0; 4] });
    let mut bits = 0u32;
    while bits <= TOP {
        let n = ((TOP - bits) as usize + 1).min(CHUNK);
        for (i, v) in input[..n].iter_mut().enumerate() { *v = f32::from_bits(bits + i as u32); }
        c.convert_row(bytemuck::cast_slice(&input[..n]), bytemuck::cast_slice_mut(&mut out[..n]), n as u32);
        for (i, o) in out[..n].iter().enumerate() {
            let v: f64 = (*o).into();
            if v < prev {
                let drop = prev - v;
                let b = bits + i as u32;
                found.count += 1;
                found.worst = found.worst.max(drop);
                found.first.get_or_insert((b, prev, v));
                found.last_input = b;
                let x = f32::from_bits(b);
                found.below[if x < 0.25 { 0 } else if x < 0.75 { 1 } else if x < 0.999 { 2 } else { 3 }] += 1;
            }
            prev = v;
        }
        bits += n as u32;
    }
    found
}

fn set_tier(t: &str) {
    if t == "baseline" {
        let _ = archmage::X64V3Token::dangerously_disable_token_process_wide(true);
    } else {
        let _ = archmage::X64V3Token::dangerously_disable_token_process_wide(false);
        let _ = archmage::X64V4Token::dangerously_disable_token_process_wide(true);
    }
}

fn main() {
    let opts = ConvertOptions::permissive();
    let d = |t, tf| D::new(t, L::Gray, None, tf);
    let tier = std::env::args().nth(1).unwrap_or("v3".into());
    set_tier(&tier);
    println!("tier {tier}: reversals among {} inputs; first = bits (previous -> output); by input range <0.25 / <0.75 / <0.999 / top", TOP as u64 + 1);
    for tf in [Tf::Pq, Tf::Hlg, Tf::Bt709, Tf::Srgb, Tf::Gamma22] {
        for (dir, from, to) in [("decode", tf, Tf::Linear), ("encode", Tf::Linear, tf)] {
            let plan = |x| RowConverter::new_explicit(d(T::F32, from), d(x, to), &opts).unwrap();
            let u: Found = scan::<u16>(plan(T::U16));
            let f: Found = scan::<f32>(plan(T::F32));
            let show = |x: &Found| match x.first {
                None => "none".to_string(),
                Some((b, p, v)) => format!("{} (worst {:.3e}, first {b:#010x}: {p} -> {v}, last {:#010x}, ranges {:?})", x.count, x.worst, x.last_input, x.below),
            };
            println!("{tf:?} {dir}\n  u16: {}\n  f32: {}", show(&u), show(&f));
        }
    }
}
