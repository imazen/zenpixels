//! Sample content scanning for [`SampleFacts`] (`FloatFacts`, opacity, grayscale, U16 8-bit exactness).

use zenpixels::{
    AlphaMode, ChannelOrder, ExactDepth, FloatFacts, PixelView, SampleFacts, SampleType,
};

/// Scan all active pixels of `view` and return exact [`SampleFacts`].
#[must_use]
pub fn scan_facts(view: &PixelView<'_>) -> SampleFacts {
    let layout = view.spec().layout;
    let alpha_mode = view.spec().color.alpha();
    let float = if layout.sample == SampleType::F32 {
        view.measure_float_facts()
    } else if layout.sample == SampleType::F16 && layout.packing.is_full() {
        measure_f16_float_facts(view)
    } else {
        FloatFacts::NONE
    };

    let mut all_opaque = matches!(
        alpha_mode,
        AlphaMode::None | AlphaMode::Opaque | AlphaMode::Undefined
    );
    let mut all_grayscale = matches!(layout.order, ChannelOrder::Gray | ChannelOrder::GrayAlpha);
    let mut exact_8bit = layout.sample == SampleType::U8 || layout.sample == SampleType::U16;

    if !layout.packing.is_full() {
        return SampleFacts {
            float,
            exact: ExactDepth {
                effective_bits: 0,
                all_opaque,
                all_grayscale,
            },
        };
    }

    let check_alpha = !all_opaque && layout.order.has_alpha_slot();
    let check_gray = !all_grayscale && layout.order.is_rgb_family();
    if check_alpha {
        all_opaque = true;
    }
    if check_gray {
        all_grayscale = true;
    }

    let ch = layout.channels();
    for y in 0..view.rows() {
        let row = view.row(y);
        match layout.sample {
            SampleType::U8 => {
                for px in row.chunks_exact(ch) {
                    if check_alpha && px[ch - 1] != 255 {
                        all_opaque = false;
                    }
                    if check_gray && !(px[0] == px[1] && px[1] == px[2]) {
                        all_grayscale = false;
                    }
                }
            }
            SampleType::U16 => {
                for px in row.chunks_exact(ch * 2) {
                    let s0 = u16::from_ne_bytes([px[0], px[1]]);
                    if exact_8bit {
                        for c in 0..ch {
                            let v = u16::from_ne_bytes([px[c * 2], px[c * 2 + 1]]);
                            if v != (v >> 8) * 257 {
                                exact_8bit = false;
                                break;
                            }
                        }
                    }
                    if check_alpha {
                        let a = u16::from_ne_bytes([px[(ch - 1) * 2], px[(ch - 1) * 2 + 1]]);
                        if a != 65535 {
                            all_opaque = false;
                        }
                    }
                    if check_gray {
                        let s1 = u16::from_ne_bytes([px[2], px[3]]);
                        let s2 = u16::from_ne_bytes([px[4], px[5]]);
                        if !(s0 == s1 && s1 == s2) {
                            all_grayscale = false;
                        }
                    }
                }
            }
            SampleType::F16 => {
                exact_8bit = false;
                for px in row.chunks_exact(ch * 2) {
                    if check_alpha {
                        let a_bits = u16::from_ne_bytes([px[(ch - 1) * 2], px[(ch - 1) * 2 + 1]]);
                        if crate::f16_scalar::f16_bits_to_f32(a_bits) != 1.0 {
                            all_opaque = false;
                        }
                    }
                    if check_gray {
                        let s0 =
                            crate::f16_scalar::f16_bits_to_f32(u16::from_ne_bytes([px[0], px[1]]));
                        let s1 =
                            crate::f16_scalar::f16_bits_to_f32(u16::from_ne_bytes([px[2], px[3]]));
                        let s2 =
                            crate::f16_scalar::f16_bits_to_f32(u16::from_ne_bytes([px[4], px[5]]));
                        if !(s0 == s1 && s1 == s2) {
                            all_grayscale = false;
                        }
                    }
                }
            }
            SampleType::F32 => {
                exact_8bit = false;
                for px in row.chunks_exact(ch * 4) {
                    if check_alpha {
                        let a = f32::from_ne_bytes(px[(ch - 1) * 4..ch * 4].try_into().unwrap());
                        if a != 1.0 {
                            all_opaque = false;
                        }
                    }
                    if check_gray {
                        let s0 = f32::from_ne_bytes(px[0..4].try_into().unwrap());
                        let s1 = f32::from_ne_bytes(px[4..8].try_into().unwrap());
                        let s2 = f32::from_ne_bytes(px[8..12].try_into().unwrap());
                        if !(s0 == s1 && s1 == s2) {
                            all_grayscale = false;
                        }
                    }
                }
            }
            _ => {
                exact_8bit = false;
                if check_alpha {
                    all_opaque = false;
                }
                if check_gray {
                    all_grayscale = false;
                }
            }
        }
    }

    SampleFacts {
        float,
        exact: ExactDepth {
            effective_bits: if exact_8bit { 8 } else { 0 },
            all_opaque,
            all_grayscale,
        },
    }
}

fn measure_f16_float_facts(view: &PixelView<'_>) -> FloatFacts {
    let mut no_nan = true;
    let mut no_inf = true;
    let mut no_sub = true;
    let mut non_neg = true;
    let mut positive = true;
    let mut at_most_one = true;
    for y in 0..view.rows() {
        for &chunk in view.row(y).as_chunks::<2>().0 {
            let bits = u16::from_ne_bytes(chunk);
            let exp = (bits >> 10) & 0x1f;
            let mant = bits & 0x3ff;
            if exp == 0 && mant != 0 {
                no_sub = false;
            }
            let v = crate::f16_scalar::f16_bits_to_f32(bits);
            if v.is_nan() {
                no_nan = false;
                non_neg = false;
                positive = false;
                at_most_one = false;
                continue;
            }
            if v.is_infinite() {
                no_inf = false;
            }
            if v < 0.0 {
                non_neg = false;
                positive = false;
            } else if v == 0.0 {
                positive = false;
            }
            if v > 1.0 {
                at_most_one = false;
            }
        }
    }
    let mut out = FloatFacts::NONE;
    if no_nan {
        out = out | FloatFacts::NO_NAN;
    }
    if no_inf {
        out = out | FloatFacts::NO_INFINITE;
    }
    if no_sub {
        out = out | FloatFacts::NO_SUBNORMALS;
    }
    if non_neg {
        out = out | FloatFacts::NON_NEGATIVE;
    }
    if positive {
        out = out | FloatFacts::POSITIVE;
    }
    if at_most_one {
        out = out | FloatFacts::AT_MOST_ONE;
    }
    out
}
