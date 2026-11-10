use zenpixels::{
    AlphaMode, ChannelLayout as L, ChannelType as T, ColorPrimaries as P, ConvertOptions,
    PixelDescriptor as D, TransferFunction as F,
    hdr::{DiffuseWhite, HlgRendering, Nits},
};
use zenpixels_convert::RowConverter;
fn descriptor(tf: F, alpha: bool) -> D {
    D::new(
        T::F32,
        if alpha { L::Rgba } else { L::Rgb },
        alpha.then_some(AlphaMode::Straight),
        tf,
    )
    .with_primaries(P::Bt2020)
}
fn options() -> ConvertOptions {
    ConvertOptions::permissive()
        .with_sdr_white(DiffuseWhite::new(1000.0))
        .with_hlg_rendering(HlgRendering::REFERENCE_1000_NIT)
        .with_clip_out_of_gamut(false)
}
fn run(from: D, to: D, options: ConvertOptions, src: &[f32]) -> Vec<f32> {
    let mut c = RowConverter::new_explicit(from, to, &options).unwrap();
    let mut dst = vec![0.0; src.len()];
    c.try_convert_row(
        bytemuck::cast_slice(src),
        bytemuck::cast_slice_mut(&mut dst),
        (src.len() / from.channels()) as u32,
    )
    .unwrap();
    dst
}
#[test]
fn reference_gray_and_colored_luminance() {
    let h = descriptor(F::Hlg, false);
    let l = descriptor(F::Linear, false);
    // At E'=0.5 inverse OETF is exactly 1/12. The OOTF acts on
    // luminance, not independently on each channel.
    let output = run(
        h,
        l,
        options(),
        &[0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 1.0],
    );
    let gray = (1.0f64 / 12.0).powf(1.2);
    let red = (1.0f64 / 12.0) * (0.2627f64 / 12.0).powf(0.2);
    assert!((output[0] as f64 - gray).abs() < 2e-8);
    assert!((output[3] as f64 - red).abs() < 2e-8);
    assert_eq!(&output[4..6], &[0.0, 0.0]);
    assert!((output[6] - 1.0).abs() < 2e-6);
    assert!((output[0] - output[3]).abs() > 0.01);
}
#[test]
fn black_lift_and_inverse_and_partition() {
    let h = descriptor(F::Hlg, true);
    let l = descriptor(F::Linear, true);
    let opt =
        options().with_hlg_rendering(HlgRendering::new(Nits::new(1000.0), 0.005, 1.2).unwrap());
    let alpha = f32::from_bits(0x7fc12345);
    let input = [
        0.0, 0.0, 0.0, alpha, 0.5, 0.2, 0.8, -0.0, 1.2, 1.1, 1.3, 0.5,
    ];
    let out = run(h, l, opt, &input);
    assert!((out[0] * 1000.0 - 0.005).abs() < 1e-7);
    for i in [3, 7, 11] {
        assert_eq!(out[i].to_bits(), input[i].to_bits());
    }
    let back = run(l, h, opt, &out);
    for (i, (&a, &b)) in input.iter().zip(&back).enumerate() {
        if i % 4 != 3 {
            assert!((a - b).abs() < 2e-6, "{a} {b}");
        }
    }
    for (i, row) in input.as_chunks::<4>().0.iter().enumerate() {
        let one = run(h, l, opt, row);
        assert_eq!(
            bytemuck::cast_slice::<f32, u8>(&one),
            bytemuck::cast_slice::<f32, u8>(&out[i * 4..i * 4 + 4])
        );
    }
}
#[test]
fn explicit_policy_and_white_required_and_identity_does_not_render() {
    let h = descriptor(F::Hlg, false);
    let p = descriptor(F::Pq, false);
    assert!(RowConverter::new_explicit(h, p, &ConvertOptions::permissive()).is_err());
    assert!(
        RowConverter::new_explicit(
            h,
            p,
            &ConvertOptions::permissive().with_hlg_rendering(HlgRendering::REFERENCE_1000_NIT)
        )
        .is_err()
    );
    let src = [0.5, 0.1, 0.8];
    let out = run(h, h, options(), &src);
    assert_eq!(out, src);
    let pq = run(h, p, options(), &src);
    let back = run(p, h, options(), &pq);
    for (&a, &b) in src.iter().zip(&back) {
        assert!((a - b).abs() < 2e-5, "{a} {b}");
    }
    assert!(HlgRendering::new(Nits::new(1000.0), f32::NAN, 1.2).is_none());
    assert!(HlgRendering::new(Nits::new(1000.0), 0.0, 0.0).is_none());
}

#[test]
fn encoded_depth_and_premultiplied_boundaries() {
    let h = descriptor(F::Hlg, true);
    let l = descriptor(F::Linear, true);
    let premul = h.with_alpha(Some(AlphaMode::Premultiplied));
    let src = [0.25, 0.1, 0.4, 0.5];
    let out = run(premul, l, options(), &src);
    let straight = run(h, l, options(), &[0.5, 0.2, 0.8, 0.5]);
    for (a, b) in out.iter().zip(straight) {
        assert!((a - b).abs() < 1e-6);
    }
    let back = run(l, premul, options(), &out);
    for (a, b) in src.iter().zip(back) {
        assert!((a - b).abs() < 1e-6);
    }
    let h16 = D::new(T::U16, L::Rgb, None, F::Hlg).with_primaries(P::Bt2020);
    let src = [32768u16, 13107, 52428];
    let mut out = [0f32; 3];
    RowConverter::new_explicit(h16, descriptor(F::Linear, false), &options())
        .unwrap()
        .try_convert_row(
            bytemuck::cast_slice(&src),
            bytemuck::cast_slice_mut(&mut out),
            1,
        )
        .unwrap();
    let reference = run(
        descriptor(F::Hlg, false),
        descriptor(F::Linear, false),
        options(),
        &src.map(|v| v as f32 / 65535.0),
    );
    for (a, b) in out.iter().zip(reference) {
        assert!((a - b).abs() < 1e-6, "{a} {b}");
    }
}

#[test]
fn hlg_compositing_requires_rendering_and_uses_display_white() {
    let h = descriptor(F::Hlg, true);
    let rgb = descriptor(F::Hlg, false);
    let base = ConvertOptions::permissive()
        .with_alpha_policy(zenpixels::AlphaPolicy::CompositeOnto {
            background: zenpixels::color::Color::srgb8(255, 255, 255),
            reference_white: Some(DiffuseWhite::new(203.0)),
        })
        .with_sdr_white(DiffuseWhite::new(203.0));
    assert!(RowConverter::new_explicit(h, rgb, &base).is_err());
    let opt = base.with_hlg_rendering(HlgRendering::REFERENCE_1000_NIT);
    let mut c = RowConverter::new_explicit(h, rgb, &opt).unwrap();
    let mut out = [0f32; 3];
    c.try_convert_row(
        bytemuck::cast_slice(&[0f32, 0.0, 0.0, 0.0]),
        bytemuck::cast_slice_mut(&mut out),
        1,
    )
    .unwrap();
    for v in out {
        assert!((v - 0.75).abs() < 0.001, "{v}");
    }
}

#[cfg(feature = "bt2446a")]
#[test]
fn rendered_hlg_to_sdr_uses_explicit_content_peak() {
    let h = descriptor(F::Hlg, false);
    let to = D::RGB8_SRGB;
    let opt = options()
        .with_hdr_peak(zenpixels::hdr::HdrPeak::Nits(Nits::new(1000.0)))
        .with_sdr_peak(Nits::new(203.0));
    let mut c = RowConverter::new_explicit(h, to, &opt).unwrap();
    let mut output = [0u8; 3];
    c.try_convert_row(bytemuck::cast_slice(&[0.75f32; 3]), &mut output, 1)
        .unwrap();
    assert_eq!(output[0], output[1]);
    assert_eq!(output[1], output[2]);
    assert!(output[0] > 0 && output[0] < 255);
}

#[test]
fn strided_buffer_matches_rows_and_preserves_padding() {
    use zenpixels_convert::PixelSliceConvertExt;
    let h = descriptor(F::Hlg, true);
    let l = descriptor(F::Linear, true);
    let src = [0.5f32, 0.2, 0.8, 0.5, -999.0, 0.1, 0.3, 0.2, 1.0];
    let view = zenpixels::PixelSlice::new(bytemuck::cast_slice(&src), 1, 2, 20, h).unwrap();
    let buffer = view.convert_to_with_options(l, &options()).unwrap();
    for y in 0..2 {
        let expected = run(h, l, options(), &src[y * 5..y * 5 + 4]);
        assert_eq!(
            buffer.as_slice().row(y as u32),
            bytemuck::cast_slice::<f32, u8>(&expected)
        );
    }
    assert_eq!(src[4], -999.0);
}

#[cfg(feature = "bt2446a")]
#[test]
fn buffer_measures_rendered_hlg_peak() {
    use zenpixels_convert::PixelSliceConvertExt;
    let h = descriptor(F::Hlg, false);
    let src = [0.75f32; 3];
    let view = zenpixels::PixelSlice::new(bytemuck::cast_slice(&src), 1, 1, 12, h).unwrap();
    let opt = options().with_sdr_peak(Nits::new(203.0));
    let rendered = view.convert_to_with_options(D::RGB8_SRGB, &opt).unwrap();
    assert_eq!(rendered.width(), 1);
    assert!(rendered.as_slice().row(0).iter().any(|&v| v > 0));
}

#[test]
fn internal_float_does_not_weaken_or_invent_depth_policy() {
    let opts = ConvertOptions::retain_channels()
        .with_sdr_white(DiffuseWhite::BT2408)
        .with_hlg_rendering(HlgRendering::REFERENCE_1000_NIT);
    assert!(RowConverter::new_explicit(D::RGB16_BT2100_HLG, D::RGB16_BT2100_PQ, &opts).is_ok());
    assert!(
        RowConverter::new_explicit(descriptor(F::Hlg, false), D::RGB16_BT2100_PQ, &opts).is_err()
    );
}

#[test]
fn extended_sign_clipping_and_nonfinite_rgb_are_explicit() {
    let h = descriptor(F::Hlg, true);
    let l = descriptor(F::Linear, true);
    let src = [-0.5, -0.5, -0.5, -0.0, 1.2, 1.2, 1.2, 0.5];
    let extended = run(h, l, options(), &src);
    assert!(extended[0] < 0.0 && extended[4] > 1.0);
    let back = run(l, h, options(), &extended);
    for (&a, &b) in src.iter().zip(&back) {
        assert!((a - b).abs() < 2e-6);
    }
    let clipped = run(h, l, options().with_clip_out_of_gamut(true), &src);
    assert_eq!(clipped[0], 0.0);
    assert!((clipped[4] - 1.0).abs() < 2e-6);
    let nan = run(h, l, options(), &[f32::INFINITY, 0.5, 0.5, -0.0]);
    assert!(nan[..3].iter().all(|v| v.is_nan()));
    assert_eq!(nan[3].to_bits(), (-0.0f32).to_bits());
}

#[cfg(any(feature = "cms-moxcms", feature = "cms-lcms"))]
#[test]
fn profile_backends_do_not_silently_render_hlg_without_conditions() {
    use zenpixels::{Cicp, ColorProfileSource, PixelFormat};
    use zenpixels_convert::cms::PluggableCms;
    let from = ColorProfileSource::Cicp(Cicp::new(9, 18, 0, true));
    let to = ColorProfileSource::Cicp(Cicp::new(9, 8, 0, true));
    let check = |cms: &dyn PluggableCms| {
        let built = cms.build_source_transform(
            from.clone(),
            to.clone(),
            PixelFormat::RgbF32,
            PixelFormat::RgbF32,
            &ConvertOptions::permissive(),
        );
        assert!(
            !matches!(built, Some(Ok(_))),
            "backend bypassed explicit HLG rendering"
        );
    };
    #[cfg(feature = "cms-moxcms")]
    check(&zenpixels_convert::MoxCms);
    #[cfg(feature = "cms-lcms")]
    check(&zenpixels_convert::Lcms);
}
