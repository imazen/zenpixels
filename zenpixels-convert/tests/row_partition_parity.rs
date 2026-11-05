//! A row converts to the same bytes however a caller divides it, across every route.
//! `row_partition.rs` covers one long row per route. This also crosses primaries and depths,
//! and uses a width that once put half of `ConvertScratch` off a `u32` boundary.

use zenpixels::{
    AlphaMode, ChannelLayout as L, ChannelType as C, ColorPrimaries as P, PixelDescriptor as D,
    TransferFunction as T,
};
use zenpixels_convert::RowConverter;

fn descriptor(c: C, l: L, t: T, p: P) -> D {
    let alpha = l.has_alpha().then_some(AlphaMode::Straight);
    D::new(c, l, alpha, t).with_primaries(p)
}

fn source_row(d: D, pixels: usize) -> Vec<u8> {
    let channels = d.layout().channels();
    let mut row = Vec::new();
    for i in 0..pixels * channels {
        let v = (i * 37 % 256) as f32 / 255.0;
        match d.channel_type() {
            C::U8 => row.push((v * 255.0) as u8),
            C::U16 => row.extend(((v * 65535.0) as u16).to_ne_bytes()),
            _ => row.extend(v.to_ne_bytes()),
        }
    }
    row
}

#[test]
fn splits_change_nothing() {
    let layouts = [L::Rgb, L::Rgba, L::Bgr, L::Bgra, L::Gray, L::GrayAlpha];
    let transfers = [T::Srgb, T::Linear, T::Bt709, T::Gamma22, T::Pq, T::Hlg];
    let mut failures = Vec::new();
    let mut checked = 0;
    for from_p in [P::Bt709, P::DisplayP3] {
        for from_l in layouts {
            for to_l in layouts {
                for from_c in [C::U8, C::U16, C::F32] {
                    for to_c in [C::U8, C::U16, C::F32] {
                        for from_t in transfers {
                            for to_t in transfers {
                                let fixed = |l, c| matches!(l, L::Bgr | L::Bgra) && c != C::U8;
                                if fixed(from_l, from_c) || fixed(to_l, to_c) {
                                    continue;
                                }
                                let from = descriptor(from_c, from_l, from_t, from_p);
                                let to = descriptor(to_c, to_l, to_t, P::Bt709);
                                let Ok(mut c) = RowConverter::new(from, to) else {
                                    continue;
                                };
                                let pixels = 133;
                                let src = source_row(from, pixels);
                                let (s, d) = (from.bytes_per_pixel(), to.bytes_per_pixel());
                                let mut whole = vec![0u8; pixels * d];
                                c.convert_row(&src, &mut whole, pixels as u32);
                                for n in [1, 3, 7, 8, 9, 16, 17, 64] {
                                    let mut parts = vec![0u8; pixels * d];
                                    for (a, b) in src.chunks(n * s).zip(parts.chunks_mut(n * d)) {
                                        c.convert_row(a, b, (a.len() / s) as u32);
                                    }
                                    checked += 1;
                                    if parts != whole {
                                        failures.push(format!(
                                            "{:?} {from_t:?} {from_p:?} to {:?} {to_t:?} in {n}",
                                            from.format, to.format
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {checked}:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(checked > 50_000, "{checked}");
}
