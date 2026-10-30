use zenpixels::{AlphaMode, ChannelLayout, ChannelType, PixelDescriptor, TransferFunction};
use zenpixels_convert::RowConverter;

fn oracle(v: u16) -> u8 {
    ((u32::from(v) * 255 + 32767) / 65535) as u8
}

fn check(layout: ChannelLayout, alpha: Option<AlphaMode>, tf: TransferFunction) {
    let src = PixelDescriptor::new(ChannelType::U16, layout, alpha, tf);
    let dst = PixelDescriptor::new(ChannelType::U8, layout, alpha, tf);
    let channels = layout.channels();
    let width = 65536usize;
    let mut input = Vec::with_capacity(width * channels * 2);
    for v in 0..=u16::MAX {
        for _ in 0..channels {
            input.extend_from_slice(&v.to_ne_bytes());
        }
    }
    let mut output = vec![0u8; width * channels];
    let mut conv = RowConverter::new(src, dst).unwrap();
    conv.convert_row(&input, &mut output, width as u32);
    let mut bad = 0usize;
    let mut first = None;
    for v in 0..=u16::MAX {
        let got = output[v as usize * channels];
        let want = oracle(v);
        if got != want {
            bad += 1;
            first.get_or_insert((v, got, want));
        }
    }
    assert_eq!(
        bad, 0,
        "{layout:?} {alpha:?} {tf:?}: {bad} bad codes; first {first:?}"
    );

    // Two rows with padding on both sides exercise the public stride path.
    let src_stride = input.len() + 8;
    let dst_stride = output.len() + 5;
    let mut strided_src = vec![0x91; src_stride * 2];
    strided_src[..input.len()].copy_from_slice(&input);
    strided_src[src_stride..src_stride + input.len()].copy_from_slice(&input);
    let mut strided_dst = vec![0x59; dst_stride * 2];
    conv.convert_rows(
        &strided_src,
        src_stride,
        &mut strided_dst,
        dst_stride,
        width as u32,
        2,
    )
    .unwrap();
    assert_eq!(&strided_dst[..output.len()], &output);
    assert_eq!(&strided_dst[dst_stride..dst_stride + output.len()], &output);
    assert_eq!(&strided_dst[output.len()..dst_stride], &[0x59; 5]);

    let mid = PixelDescriptor::new(ChannelType::F32, layout, alpha, tf);
    let mut via_f32 = vec![0u8; width * channels * 4];
    RowConverter::new(src, mid)
        .unwrap()
        .convert_row(&input, &mut via_f32, width as u32);
    let mut f32_output = vec![0u8; width * channels];
    RowConverter::new(mid, dst)
        .unwrap()
        .convert_row(&via_f32, &mut f32_output, width as u32);
    assert_eq!(
        f32_output, output,
        "{layout:?} {alpha:?} {tf:?}: f32 narrowing mismatch"
    );
}

#[test]
fn exhaustive_same_transfer_u16_to_u8() {
    for tf in [
        TransferFunction::Srgb,
        TransferFunction::Bt709,
        TransferFunction::Linear,
    ] {
        for (layout, alpha) in [
            (ChannelLayout::Rgb, None),
            (ChannelLayout::Rgba, Some(AlphaMode::Straight)),
            (ChannelLayout::Rgba, Some(AlphaMode::Premultiplied)),
            (ChannelLayout::Gray, None),
            (ChannelLayout::GrayAlpha, Some(AlphaMode::Straight)),
            (ChannelLayout::GrayAlpha, Some(AlphaMode::Premultiplied)),
        ] {
            check(layout, alpha, tf);
        }
    }
}

#[test]
fn exhaustive_same_transfer_all_available_tiers() {
    use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

    let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        for tf in [
            TransferFunction::Srgb,
            TransferFunction::Bt709,
            TransferFunction::Linear,
        ] {
            for (layout, alpha) in [
                (ChannelLayout::Rgb, None),
                (ChannelLayout::Rgba, Some(AlphaMode::Straight)),
                (ChannelLayout::Rgba, Some(AlphaMode::Premultiplied)),
                (ChannelLayout::Gray, None),
                (ChannelLayout::GrayAlpha, Some(AlphaMode::Straight)),
                (ChannelLayout::GrayAlpha, Some(AlphaMode::Premultiplied)),
            ] {
                check(layout, alpha, tf);
            }
        }
        eprintln!("tier={} routes=18 errors=0", perm.label);
    });
    eprintln!("RGB16 exact narrowing tier report: {report}");
}

#[cfg(feature = "__trace_ops")]
#[test]
fn tagged_srgb_uses_integer_narrowing() {
    use zenpixels_convert::__trace_ops as trace;

    let src = PixelDescriptor::RGB16_SRGB;
    let dst = PixelDescriptor::RGB8_SRGB;
    let mut conv = RowConverter::new(src, dst).unwrap();
    let code = 33025u16;
    let mut input = Vec::new();
    for _ in 0..3 {
        input.extend_from_slice(&code.to_ne_bytes());
    }
    let mut output = [0u8; 3];
    trace::start_recording();
    conv.convert_row(&input, &mut output, 1);
    let steps = trace::stop_recording();
    assert!(steps.contains(&"U16ToU8"), "unexpected plan: {steps:?}");
    assert_eq!(output, [129; 3]);
}

#[cfg(feature = "__trace_ops")]
#[test]
fn changed_sdr_transfer_uses_precise_composite_step() {
    use zenpixels_convert::__trace_ops as trace;

    let from = PixelDescriptor::new(
        ChannelType::U16,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Bt709,
    );
    let to = PixelDescriptor::RGB8_SRGB;
    let mut conv = RowConverter::new(from, to).unwrap();
    let mut input = Vec::new();
    for _ in 0..3 {
        input.extend_from_slice(&1035u16.to_ne_bytes());
    }
    let mut output = [0u8; 3];
    trace::start_recording();
    conv.convert_row(&input, &mut output, 1);
    let steps = trace::stop_recording();
    assert!(steps.contains(&"SdrU16ToU8"), "unexpected plan: {steps:?}");
    assert_eq!(output, [12; 3]);
}

// Independently evaluated composite transfer oracle. The BT.709 constants
// are the continuous H.273 parameters used by linear-srgb's f64 reference.
fn decode_f64(tf: TransferFunction, x: f64) -> f64 {
    match tf {
        TransferFunction::Linear => x,
        TransferFunction::Srgb => {
            if x <= 0.04045 {
                x / 12.92
            } else {
                ((x + 0.055) / 1.055).powf(2.4)
            }
        }
        TransferFunction::Bt709 => {
            const BETA: f64 = 0.018053968510807;
            const ALPHA: f64 = 0.09929682680944;
            if x < 4.5 * BETA {
                x / 4.5
            } else {
                ((x + ALPHA) / (1.0 + ALPHA)).powf(1.0 / 0.45)
            }
        }
        _ => unreachable!(),
    }
}

fn encode_f64(tf: TransferFunction, x: f64) -> f64 {
    match tf {
        TransferFunction::Linear => x,
        TransferFunction::Srgb => {
            if x <= 0.0031308 {
                x * 12.92
            } else {
                1.055 * x.powf(1.0 / 2.4) - 0.055
            }
        }
        TransferFunction::Bt709 => {
            const BETA: f64 = 0.018053968510807;
            const ALPHA: f64 = 0.09929682680944;
            if x < BETA {
                x * 4.5
            } else {
                (1.0 + ALPHA) * x.powf(0.45) - ALPHA
            }
        }
        _ => unreachable!(),
    }
}

#[test]
fn audit_real_transfer_pair_quantisation() {
    let tfs = [
        TransferFunction::Linear,
        TransferFunction::Bt709,
        TransferFunction::Srgb,
    ];
    let width = 65536usize;
    let mut input = Vec::with_capacity(width * 6);
    for v in 0..=u16::MAX {
        for _ in 0..3 {
            input.extend_from_slice(&v.to_ne_bytes());
        }
    }
    for src_tf in tfs {
        for dst_tf in tfs {
            if src_tf == dst_tf {
                continue;
            }
            let src = PixelDescriptor::new(ChannelType::U16, ChannelLayout::Rgb, None, src_tf);
            let dst = PixelDescriptor::new(ChannelType::U8, ChannelLayout::Rgb, None, dst_tf);
            let mut output = vec![0u8; width * 3];
            RowConverter::new(src, dst)
                .unwrap()
                .convert_row(&input, &mut output, width as u32);
            let mut bad = 0usize;
            let mut first = None;
            for v in 0..=u16::MAX {
                let normalized = f64::from(v) / 65535.0;
                let exact = encode_f64(dst_tf, decode_f64(src_tf, normalized));
                let want = (exact.clamp(0.0, 1.0) * 255.0 + 0.5).floor() as u8;
                let got = output[v as usize * 3];
                if got != want {
                    bad += 1;
                    first.get_or_insert((v, got, want));
                }
            }
            eprintln!("transfer_pair={src_tf:?}->{dst_tf:?} errors={bad} first={first:?}");
            assert_eq!(bad, 0, "{src_tf:?}->{dst_tf:?}: first {first:?}");
        }
    }
}

#[test]
fn real_transfer_rgba_keeps_alpha_linear() {
    let tfs = [
        TransferFunction::Linear,
        TransferFunction::Bt709,
        TransferFunction::Srgb,
    ];
    let width = 65536usize;
    let mut input = Vec::with_capacity(width * 8);
    for v in 0..=u16::MAX {
        for _ in 0..4 {
            input.extend_from_slice(&v.to_ne_bytes());
        }
    }
    for alpha in [AlphaMode::Straight, AlphaMode::Premultiplied] {
        for from_tf in tfs {
            for to_tf in tfs {
                if from_tf == to_tf {
                    continue;
                }
                let from = PixelDescriptor::new(
                    ChannelType::U16,
                    ChannelLayout::Rgba,
                    Some(alpha),
                    from_tf,
                );
                let to =
                    PixelDescriptor::new(ChannelType::U8, ChannelLayout::Rgba, Some(alpha), to_tf);
                let mut output = vec![0u8; width * 4];
                RowConverter::new(from, to)
                    .unwrap()
                    .convert_row(&input, &mut output, width as u32);
                for v in 0..=u16::MAX {
                    let encoded = encode_f64(to_tf, decode_f64(from_tf, f64::from(v) / 65535.0));
                    let want_color = if alpha == AlphaMode::Premultiplied {
                        // RGB == alpha: unassociated RGB is white, in every TF.
                        oracle(v)
                    } else {
                        (encoded.clamp(0.0, 1.0) * 255.0 + 0.5).floor() as u8
                    };
                    let px = &output[v as usize * 4..v as usize * 4 + 4];
                    assert_eq!(
                        &px[..3],
                        &[want_color; 3],
                        "{alpha:?} {from_tf:?}->{to_tf:?}, code {v}"
                    );
                    assert_eq!(
                        px[3],
                        oracle(v),
                        "alpha {alpha:?} {from_tf:?}->{to_tf:?}, code {v}"
                    );
                }
            }
        }
    }
}
