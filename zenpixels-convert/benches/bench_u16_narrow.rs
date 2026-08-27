//! u16 → u8 narrowing kernel shootout (imazen/zenpixels#72).
//!
//! garb 0.2.8's `convert_u16_to_u8` computes `(v * 255 + 32768) >> 16` — a
//! 65536-divisor approximation that floors 127 of the 65536 inputs by 1 LSB.
//! The replacement must be exact (`round(v / 257)`) with no throughput
//! loss. Arms:
//!
//!   - `garb (inexact)`  — the previous kernel, for reference.
//!   - `shift u32`       — `z = (v+128)·255; (z + (z>>16) + 1) >> 16`.
//!   - `byte lanes`      — `hi + [lo−hi ≥ 129] − [hi−lo ≥ 129]` on the two
//!                         bytes of `v` (saturating subs + compares; the
//!                         form LLVM vectorises widest).
//!   - `shipped`         — whatever `ConvertStep::U16ToU8` currently runs,
//!                         via the public `RowConverter`.
//!
//! All arms are exhaustively cross-checked against `(v + 128) / 257` before
//! timing so a fast-but-wrong arm cannot post a number.

use zenbench::prelude::*;
use zenpixels::{ChannelLayout, ChannelType, PixelDescriptor, TransferFunction};
use zenpixels_convert::RowConverter;

const SIZES: &[(&str, usize)] = &[
    ("  256px", 256),
    (" 4096px", 4096),
    ("1080p  ", 1920 * 1080),
];

fn garb_inexact(src: &[u8], dst: &mut [u8]) {
    garb::bytes::convert_u16_to_u8(src, dst).expect("sizes");
}

fn shift_u32(src: &[u8], dst: &mut [u8]) {
    let (pairs, _) = src.as_chunks::<2>();
    for (s, d) in pairs.iter().zip(dst.iter_mut()) {
        let v = u32::from(u16::from_ne_bytes(*s));
        let z = (v + 128) * 255;
        *d = ((z + (z >> 16) + 1) >> 16) as u8;
    }
}

fn byte_lanes(src: &[u8], dst: &mut [u8]) {
    let (pairs, _) = src.as_chunks::<2>();
    for (s, d) in pairs.iter().zip(dst.iter_mut()) {
        let [lo, hi] = u16::from_ne_bytes(*s).to_le_bytes();
        let up = u8::from(lo.saturating_sub(hi) > 128);
        let down = u8::from(hi.saturating_sub(lo) > 128);
        *d = hi.wrapping_add(up).wrapping_sub(down);
    }
}

fn exact_oracle(v: u16) -> u8 {
    ((u32::from(v) + 128) / 257) as u8
}

fn check_exact(name: &str, f: fn(&[u8], &mut [u8])) -> bool {
    let src: Vec<u8> = (0..=u16::MAX).flat_map(u16::to_ne_bytes).collect();
    let mut dst = vec![0u8; 65536];
    f(&src, &mut dst);
    let bad = (0..=u16::MAX)
        .filter(|&v| dst[v as usize] != exact_oracle(v))
        .count();
    if bad != 0 {
        eprintln!("[{name}] NOT exact: {bad} of 65536 inputs differ from round(v/257)");
    }
    bad == 0
}

fn main() {
    let exact_shift = check_exact("shift u32", shift_u32);
    let exact_bytes = check_exact("byte lanes", byte_lanes);
    let exact_garb = check_exact("garb", garb_inexact);
    assert!(
        exact_shift && exact_bytes,
        "candidate kernels must be exact"
    );
    eprintln!("[garb] exact = {exact_garb} (expected false on 0.2.8)");

    zenbench::run(|suite| {
        for &(label, width) in SIZES {
            let count = width * 3;
            let src: Vec<u8> = (0..count * 2).map(|i| (i * 31 % 251) as u8).collect();
            let bytes = (count * 3) as u64;

            let desc16 = PixelDescriptor::new(
                ChannelType::U16,
                ChannelLayout::Rgb,
                None,
                TransferFunction::Linear,
            );
            let desc8 = PixelDescriptor::new(
                ChannelType::U8,
                ChannelLayout::Rgb,
                None,
                TransferFunction::Linear,
            );
            let mut conv = RowConverter::new(desc16, desc8).unwrap();

            let s1 = src.clone();
            let s2 = src.clone();
            let s3 = src.clone();
            let s4 = src;
            suite.group(format!("u16→u8 {label}"), move |g| {
                g.throughput(Throughput::Bytes(bytes));
                let mut d = vec![0u8; count];
                g.bench("garb (inexact)", move |b| {
                    b.iter(|| garb_inexact(&s1, &mut d))
                });
                let mut d = vec![0u8; count];
                g.bench("shift u32", move |b| b.iter(|| shift_u32(&s2, &mut d)));
                let mut d = vec![0u8; count];
                g.bench("byte lanes", move |b| b.iter(|| byte_lanes(&s3, &mut d)));
                let mut d = vec![0u8; count];
                g.bench("shipped", move |b| {
                    b.iter(|| {
                        conv.convert_row(&s4, &mut d, width as u32);
                        black_box(());
                    })
                });
            });
        }
    });
}
