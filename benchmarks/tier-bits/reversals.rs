//! Classifies every reversal over every f32 in [0, 1], for each curve and direction.
use zenpixels::{ChannelLayout as L, ChannelType as T, PixelDescriptor as D, TransferFunction as Tf};
use zenpixels_convert::{ConvertOptions, RowConverter};
const CHUNK: usize = 1 << 20;
const TOP: u32 = 0x3f80_0000;
fn ulps(a: f32, b: f32) -> u32 { a.to_bits().abs_diff(b.to_bits()) }
fn main() {
    let opts = ConvertOptions::permissive();
    let d = |t, tf| D::new(t, L::Gray, None, tf);
    // Branch seams of the implementations, as input values, per curve and direction.
    let cases: [(&str, Tf, Tf, &[f32]); 8] = [
        ("PQ decode", Tf::Pq, Tf::Linear, &[0.02, 0.25]),
        ("PQ encode", Tf::Linear, Tf::Pq, &[1e-7, 1e-4]),
        ("HLG encode", Tf::Linear, Tf::Hlg, &[1.0 / 12.0]),
        ("HLG decode", Tf::Hlg, Tf::Linear, &[0.5]),
        ("BT.709 decode", Tf::Bt709, Tf::Linear, &[0.081242]),
        ("BT.709 encode", Tf::Linear, Tf::Bt709, &[0.018053968]),
        ("sRGB decode", Tf::Srgb, Tf::Linear, &[0.04045]),
        ("sRGB encode", Tf::Linear, Tf::Srgb, &[0.0031308]),
    ];
    for (name, from, to, seams) in cases {
        let mut cf = RowConverter::new_explicit(d(T::F32, from), d(T::F32, to), &opts).unwrap();
        let mut cu = RowConverter::new_explicit(d(T::F32, from), d(T::U16, to), &opts).unwrap();
        let mut input = vec![0f32; CHUNK];
        let (mut of, mut ou) = (vec![0f32; CHUNK], vec![0u16; CHUNK]);
        let (mut pf, mut pu) = (f32::NEG_INFINITY, 0u16);
        let (mut rf, mut ru, mut both, mut at_seam) = (0u64, 0u64, 0u64, 0u64);
        let mut hist = [0u64; 6];
        let (mut worst, mut worst_at) = (0u32, 0f32);
        let (mut steps, mut flat_or_rev) = ([0u64; 2], 0u64);
        let mut binades = std::collections::BTreeMap::<i32, (u64, u64)>::new();
        let mut bits = 0u32;
        while bits <= TOP {
            let n = ((TOP - bits) as usize + 1).min(CHUNK);
            for (i, v) in input[..n].iter_mut().enumerate() { *v = f32::from_bits(bits + i as u32); }
            cf.convert_row(bytemuck::cast_slice(&input[..n]), bytemuck::cast_slice_mut(&mut of[..n]), n as u32);
            cu.convert_row(bytemuck::cast_slice(&input[..n]), bytemuck::cast_slice_mut(&mut ou[..n]), n as u32);
            for i in 0..n {
                let (f, u, x) = (of[i], ou[i], input[i]);
                let rev_f = f < pf;
                let rev_u = u < pu;
                if f <= pf { flat_or_rev += 1; }
                steps[0] += 1;
                if rev_f {
                    rf += 1;
                    let drop = ulps(pf, f);
                    hist[match drop { 1 => 0, 2 => 1, 3..=4 => 2, 5..=16 => 3, 17..=1024 => 4, _ => 5 }] += 1;
                    if drop > worst { worst = drop; worst_at = x; }
                    if seams.iter().any(|s| x.to_bits().abs_diff(s.to_bits()) <= 64) { at_seam += 1; }
                }
                let e = ((x.to_bits() >> 23) as i32) - 127;
                let b = binades.entry(e).or_insert((0, 0));
                if rev_f { b.0 += 1; }
                if rev_u { b.1 += 1; ru += 1; if rev_f { both += 1; } }
                pf = f; pu = u;
            }
            bits += n as u32;
        }
        println!("{name}: f32 reversals {rf}; by drop in output ULP 1:{} 2:{} 3-4:{} 5-16:{} 17-1024:{} >1024:{}; worst {worst} ULP at input {worst_at:e}; within 64 input ULP of a branch seam: {at_seam}", hist[0], hist[1], hist[2], hist[3], hist[4], hist[5]);
        println!("  u16 reversals {ru}, of which also a reversal in the f32 path {both}; flat-or-reversed input steps {flat_or_rev} of {}", steps[0]);
        let top: Vec<String> = binades.iter().rev().filter(|(_, v)| v.0 + v.1 > 0).take(7).map(|(e, v)| format!("2^{e}:{}/{}", v.0, v.1)).collect();
        println!("  by input binade, f32/u16: {}", top.join(" "));
    }
}
