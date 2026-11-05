//! Changing primaries while changing layout, measured against independent math.
//!
//! Gamut kernels read RGB or RGBA, so a plan that removes channels first would feed
//! them the wrong rows: P3 `[200, 100, 50]` to gray once gave `117, 117, 117, 128, 0`.

use zenpixels::{
    AlphaMode, ChannelLayout as L, ChannelType as C, ColorPrimaries as P, PixelDescriptor as D,
    TransferFunction as T,
};
use zenpixels_convert::RowConverter;

type M = [[f64; 3]; 3];

/// The BT.709 curve constants that meet exactly, from BT.2020's published precision.
const ALPHA: f64 = 1.09929682680944;
const BETA: f64 = 0.018053968510807;

/// RGB to XYZ, from the published chromaticities and a D65 white.
fn to_xyz(p: P) -> M {
    let xy = match p {
        P::Bt709 => [[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]],
        P::DisplayP3 => [[0.68, 0.32], [0.265, 0.69], [0.15, 0.06]],
        P::Bt2020 => [[0.708, 0.292], [0.17, 0.797], [0.131, 0.046]],
        _ => unreachable!(),
    };
    let m: M = core::array::from_fn(|row| {
        core::array::from_fn(|c| match row {
            0 => xy[c][0] / xy[c][1],
            1 => 1.0,
            _ => (1.0 - xy[c][0] - xy[c][1]) / xy[c][1],
        })
    });
    let white = [0.3127 / 0.3290, 1.0, (1.0 - 0.3127 - 0.3290) / 0.3290];
    let s = apply(&invert(&m), white);
    core::array::from_fn(|r| core::array::from_fn(|c| m[r][c] * s[c]))
}

fn apply(m: &M, v: [f64; 3]) -> [f64; 3] {
    core::array::from_fn(|r| m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2])
}

fn invert(m: &M) -> M {
    let c = |a: usize, b: usize| {
        let (r0, r1) = ((a + 1) % 3, (a + 2) % 3);
        let (c0, c1) = ((b + 1) % 3, (b + 2) % 3);
        m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]
    };
    let det = m[0][0] * c(0, 0) + m[0][1] * c(0, 1) + m[0][2] * c(0, 2);
    core::array::from_fn(|r| core::array::from_fn(|k| c(k, r) / det))
}

/// From the given primaries to BT.709.
fn matrix(from: P) -> M {
    let (a, b) = (to_xyz(from), invert(&to_xyz(P::Bt709)));
    core::array::from_fn(|r| {
        core::array::from_fn(|k| b[r][0] * a[0][k] + b[r][1] * a[1][k] + b[r][2] * a[2][k])
    })
}

fn linearize(t: T, v: f64) -> f64 {
    match t {
        T::Bt709 if v < 4.5 * BETA => v / 4.5,
        T::Bt709 => ((v + ALPHA - 1.0) / ALPHA).powf(1.0 / 0.45),
        T::Gamma22 => v.powf(563.0 / 256.0),
        T::Pq => {
            let (m1, m2, c1, c2, c3) = (
                2610.0 / 16384.0,
                2523.0 / 4096.0 * 128.0,
                3424.0 / 4096.0,
                2413.0 / 4096.0 * 32.0,
                2392.0 / 4096.0 * 32.0,
            );
            let p = v.max(0.0).powf(1.0 / m2);
            (((p - c1).max(0.0)) / (c2 - c3 * p)).powf(1.0 / m1)
        }
        T::Hlg => {
            let (a, b, c) = (
                0.17883277,
                1.0 - 4.0 * 0.17883277,
                0.5 - 0.17883277 * (4.0f64 * 0.17883277).ln(),
            );
            if v <= 0.5 {
                v * v / 3.0
            } else {
                (((v - c) / a).exp() + b) / 12.0
            }
        }
        T::Srgb if v <= 0.04045 => v / 12.92,
        T::Srgb => ((v + 0.055) / 1.055).powf(2.4),
        T::Linear => v,
        _ => unreachable!(),
    }
}

/// By default, `f32` output keeps colors outside the target's gamut, except as sRGB.
/// The BT.709 curve continues its straight segment below zero.
fn encode(t: T, c: C, v: f64) -> f64 {
    let keeps = c == C::F32 && matches!(t, T::Linear | T::Bt709);
    let v = if keeps { v } else { v.clamp(0.0, 1.0) };
    match t {
        T::Bt709 if v < BETA => v * 4.5,
        T::Bt709 => ALPHA * v.powf(0.45) - (ALPHA - 1.0),
        T::Gamma22 => v.powf(256.0 / 563.0),
        T::Pq => {
            let (m1, m2, c1, c2, c3) = (
                2610.0 / 16384.0,
                2523.0 / 4096.0 * 128.0,
                3424.0 / 4096.0,
                2413.0 / 4096.0 * 32.0,
                2392.0 / 4096.0 * 32.0,
            );
            let p = v.powf(m1);
            ((c1 + c2 * p) / (1.0 + c3 * p)).powf(m2)
        }
        T::Hlg => {
            let (a, b, c) = (
                0.17883277,
                1.0 - 4.0 * 0.17883277,
                0.5 - 0.17883277 * (4.0f64 * 0.17883277).ln(),
            );
            if v <= 1.0 / 12.0 {
                (3.0 * v).sqrt()
            } else {
                a * (12.0 * v - b).ln() + c
            }
        }
        T::Srgb if v <= 0.0031308 => v * 12.92,
        T::Srgb => 1.055 * v.powf(1.0 / 2.4) - 0.055,
        T::Linear => v,
        _ => unreachable!(),
    }
}

/// Reads one channel as a fraction of full scale.
fn read(c: C, bytes: &[u8]) -> f64 {
    match c {
        C::U8 => f64::from(bytes[0]) / 255.0,
        C::U16 => f64::from(u16::from_ne_bytes([bytes[0], bytes[1]])) / 65535.0,
        C::F32 => f64::from(f32::from_ne_bytes(bytes.try_into().unwrap())),
        _ => unreachable!(),
    }
}

fn write(c: C, v: f64, out: &mut Vec<u8>) {
    match c {
        C::U8 => out.push((v * 255.0).round() as u8),
        C::U16 => out.extend(((v * 65535.0).round() as u16).to_ne_bytes()),
        C::F32 => out.extend((v as f32).to_ne_bytes()),
        _ => unreachable!(),
    }
}

/// Largest error one LSB of rounding could explain in a kernel, as a fraction.
/// F32 sits between U8 and U16 here because 8-bit sources quantize first in some routes.
fn tolerance(c: C) -> f64 {
    match c {
        C::U8 => 1.01 / 255.0,
        C::U16 => 1.5 / 65535.0,
        _ => 2e-4,
    }
}

const COLORS: [[f64; 3]; 8] = [
    [0.7843, 0.3922, 0.1961],
    [0.0392, 0.0784, 0.1176],
    [1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.5],
    [0.25, 0.75, 0.5],
    [0.9, 0.1, 0.1],
    [0.1, 0.2, 0.9],
];

/// A row of `COLORS` long enough to reach the SIMD paths, with varying alpha.
fn source_row(d: D) -> Vec<u8> {
    let mut row = Vec::new();
    for i in 0..41 {
        let color = COLORS[i % COLORS.len()];
        match d.layout() {
            L::Rgb | L::Rgba => {
                for v in color {
                    write(d.channel_type(), v, &mut row);
                }
            }
            L::Bgra => {
                for v in [color[2], color[1], color[0]] {
                    write(d.channel_type(), v, &mut row);
                }
            }
            L::Gray | L::GrayAlpha => write(d.channel_type(), color[0], &mut row),
            _ => unreachable!(),
        }
        if d.layout().has_alpha() {
            write(d.channel_type(), (i % 5) as f64 / 4.0, &mut row);
        }
    }
    row
}

fn descriptor(c: C, l: L, t: T, p: P) -> D {
    let alpha = l.has_alpha().then_some(AlphaMode::Straight);
    D::new(c, l, alpha, t).with_primaries(p)
}

const DEPTHS: [C; 3] = [C::U8, C::U16, C::F32];

fn eight_bit_only(l: L) -> bool {
    matches!(l, L::Bgra)
}

/// RGB order of a pixel's color channels, as they sit in the layout.
fn rgb_of(l: L, c: C, px: &[u8]) -> [f64; 3] {
    let n = c.byte_size();
    let at = |i: usize| read(c, &px[i * n..(i + 1) * n]);
    match l {
        L::Rgb | L::Rgba => [at(0), at(1), at(2)],
        L::Bgra => [at(2), at(1), at(0)],
        _ => unreachable!(),
    }
}

/// Every source layout into every color layout, at every depth and across transfers.
#[test]
fn color_to_color_matches_float_math() {
    let routes = [
        (T::Srgb, T::Srgb),
        (T::Srgb, T::Linear),
        (T::Linear, T::Srgb),
        (T::Bt709, T::Bt709),
        (T::Bt709, T::Srgb),
        (T::Srgb, T::Bt709),
        (T::Gamma22, T::Gamma22),
        (T::Gamma22, T::Srgb),
        (T::Srgb, T::Gamma22),
        (T::Pq, T::Pq),
        (T::Hlg, T::Hlg),
        (T::Pq, T::Linear),
        (T::Hlg, T::Linear),
    ];
    let mut failures = Vec::new();
    let mut checked = 0;
    for from_p in [P::DisplayP3, P::Bt2020] {
        let matrix = matrix(from_p);
        for from_l in [L::Rgb, L::Rgba, L::Bgra] {
            for to_l in [L::Rgb, L::Rgba, L::Bgra] {
                for from_c in DEPTHS {
                    for to_c in DEPTHS {
                        for (from_t, to_t) in routes {
                            let eight_bit_layout = (eight_bit_only(from_l) && from_c != C::U8)
                                || (eight_bit_only(to_l) && to_c != C::U8);
                            // 8 bits can't carry linear light or HDR through any conversion.
                            let too_coarse = (from_c == C::U8
                                && matches!(from_t, T::Linear | T::Pq | T::Hlg))
                                || (to_c == C::U8 && to_t == T::Linear);
                            if eight_bit_layout || too_coarse {
                                continue;
                            }
                            let from = descriptor(from_c, from_l, from_t, from_p);
                            let to = descriptor(to_c, to_l, to_t, P::Bt709);
                            let src = source_row(from);
                            let width = (src.len() / from.bytes_per_pixel()) as u32;
                            let mut out = vec![0u8; width as usize * to.bytes_per_pixel()];
                            RowConverter::new(from, to)
                                .unwrap_or_else(|e| panic!("{from:?} to {to:?}: {e:?}"))
                                .convert_row(&src, &mut out, width);
                            let limit = tolerance(to_c).max(tolerance(from_c) * 2.0);
                            let mut worst = 0.0f64;
                            for (s, d) in src
                                .chunks(from.bytes_per_pixel())
                                .zip(out.chunks(to.bytes_per_pixel()))
                            {
                                let input = rgb_of(from_l, from_c, s).map(|v| linearize(from_t, v));
                                let want = apply(&matrix, input).map(|v| encode(to_t, to_c, v));
                                let got = rgb_of(to_l, to_c, d);
                                for i in 0..3 {
                                    worst = worst.max((got[i] - want[i]).abs());
                                }
                                checked += 1;
                            }
                            if worst > limit {
                                failures.push(format!(
                                    "{:?} {from_t:?} {from_p:?} to {:?} {to_t:?}: {worst:.5}",
                                    from.format, to.format
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(checked > 10_000, "{checked}");
}

/// Wide RGB into gray is the luma of the colors once they are in the gray's primaries.
#[test]
fn color_to_gray_goes_through_the_targets_primaries() {
    for from_p in [P::DisplayP3, P::Bt2020] {
        for from_l in [L::Rgb, L::Rgba, L::Bgra] {
            for from_c in DEPTHS {
                for to_l in [L::Gray, L::GrayAlpha] {
                    for to_c in DEPTHS {
                        if eight_bit_only(from_l) && from_c != C::U8 {
                            continue;
                        }
                        let from = descriptor(from_c, from_l, T::Srgb, from_p);
                        let to = descriptor(to_c, to_l, T::Srgb, P::Bt709);
                        let src = source_row(from);
                        let width = (src.len() / from.bytes_per_pixel()) as u32;
                        let mut direct = vec![0u8; width as usize * to.bytes_per_pixel()];
                        RowConverter::new(from, to)
                            .unwrap_or_else(|e| panic!("{from:?} to {to:?}: {e:?}"))
                            .convert_row(&src, &mut direct, width);

                        let n = to_c.byte_size();
                        let limit = tolerance(to_c).max(tolerance(from_c) * 2.0);
                        for (s, d) in src
                            .chunks(from.bytes_per_pixel())
                            .zip(direct.chunks(to.bytes_per_pixel()))
                        {
                            let input = rgb_of(from_l, from_c, s).map(|v| linearize(T::Srgb, v));
                            let [r, g, b] =
                                apply(&matrix(from_p), input).map(|v| encode(T::Srgb, to_c, v));
                            let want = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                            let got = read(to_c, &d[..n]);
                            assert!(
                                (got - want).abs() <= limit,
                                "{from:?} to {to:?}: {got} wanted {want}"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Gray has no gamut, so primaries never change its samples.
#[test]
fn gray_sources_skip_the_matrix() {
    for from_l in [L::Gray, L::GrayAlpha] {
        for to_l in [L::Gray, L::GrayAlpha, L::Rgb, L::Rgba, L::Bgra] {
            for c in [C::U8, C::U16, C::F32] {
                if eight_bit_only(to_l) && c != C::U8 {
                    continue;
                }
                let from = descriptor(c, from_l, T::Srgb, P::DisplayP3);
                let to = descriptor(c, to_l, T::Srgb, P::Bt709);
                let same = descriptor(c, to_l, T::Srgb, P::DisplayP3);
                let src = source_row(from);
                let width = (src.len() / from.bytes_per_pixel()) as u32;
                let bytes = width as usize * to.bytes_per_pixel();
                let (mut crossed, mut kept) = (vec![0u8; bytes], vec![0u8; bytes]);
                RowConverter::new(from, to)
                    .unwrap_or_else(|e| panic!("{from:?} to {to:?}: {e:?}"))
                    .convert_row(&src, &mut crossed, width);
                RowConverter::new(from, same)
                    .unwrap()
                    .convert_row(&src, &mut kept, width);
                assert_eq!(crossed, kept, "{from:?} to {to:?}");
            }
        }
    }
}

/// `RGBA8` P3 once lost its alpha to an RGBA kernel running on RGB rows.
#[test]
fn rgba_to_rgb_keeps_colors_and_length() {
    let from = D::RGBA8_SRGB.with_primaries(P::DisplayP3);
    let rows = [200u8, 100, 50, 128, 200, 100, 50, 0, 200, 100, 50, 255];
    let mut out = [0u8; 9];
    RowConverter::new_explicit(
        from,
        D::RGB8_SRGB,
        &zenpixels_convert::policy::ConvertOptions::permissive()
            .with_alpha_policy(zenpixels_convert::policy::AlphaPolicy::DiscardUnchecked),
    )
    .unwrap()
    .convert_row(&rows, &mut out, 3);
    let mut rgb = [0u8; 3];
    RowConverter::new(D::RGB8_SRGB.with_primaries(P::DisplayP3), D::RGB8_SRGB)
        .unwrap()
        .convert_row(&[200, 100, 50], &mut rgb, 1);
    assert_eq!(
        out,
        [
            rgb[0], rgb[1], rgb[2], rgb[0], rgb[1], rgb[2], rgb[0], rgb[1], rgb[2]
        ]
    );
}

/// Half of the scratch buffer once fell 2 bytes off a `u32` boundary at this width, and F32
/// rows panicked. Every width near it must convert.
#[test]
fn odd_widths_keep_scratch_aligned() {
    let from = D::RGB16_SRGB;
    let to = descriptor(C::U8, L::Gray, T::Gamma22, P::Bt709);
    for width in 125..=141usize {
        let src = vec![0x55u8; width * from.bytes_per_pixel()];
        let mut out = vec![0u8; width];
        RowConverter::new(from, to)
            .unwrap()
            .convert_row(&src, &mut out, width as u32);
    }
}
