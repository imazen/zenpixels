//! Tests for `OutputPlan` / `EncodeReady` (atomic CICP/ICC + zero-copy identity),
//! `Negotiate::select`, `orient::apply_orientation` across all 8 orientations & widths,
//! and `MoxCms` ICC transforms.

use zenpixels_convert::{
    Cicp, ColorOrigin, ConvertPolicy, LossKind, Negotiate, Orientation, OutputPlan,
    OutputSignaling, PeakSource, PixelBuf, PixelCow, PixelFormat, Primaries, apply_orientation,
};

#[test]
fn output_plan_zero_copy_identity_and_atomic_signaling() {
    let mut buf = PixelBuf::new(8, 4, PixelFormat::RGB8_SRGB).unwrap();
    buf.as_view_mut().row_mut(0)[0..3].copy_from_slice(&[42, 84, 126]);
    let in_ptr = buf.as_view().row(0).as_ptr();

    // Identity on owned PixelBuf moves storage ZERO-COPY!
    let id_plan = OutputPlan::build(
        PixelFormat::RGB8_SRGB,
        PixelFormat::RGB8_SRGB,
        &ConvertPolicy::lossless(),
        PeakSource::Measured,
        OutputSignaling::CicpOnly,
    )
    .unwrap();
    assert!(id_plan.is_identity());
    let ready = id_plan.execute_buf(buf).unwrap();
    assert_eq!(ready.cicp, Some(Cicp::SRGB));
    assert!(ready.icc.is_none());
    assert_eq!(ready.pixels.as_view().row(0).as_ptr(), in_ptr);

    // Identity on borrowed PixelView returns PixelCow::Borrowed zero-copy!
    let owned = ready.pixels.into_owned().unwrap();
    let ready_view = id_plan
        .execute_view(owned.as_view(), &ColorOrigin::assumed())
        .unwrap();
    assert!(matches!(ready_view.pixels, PixelCow::Borrowed(_)));
    assert_eq!(
        ready_view.pixels.as_view().row(0).as_ptr(),
        owned.as_view().row(0).as_ptr()
    );

    // Non-identity: sRGB -> Display P3 Auto emits Display P3 CICP + bundled P3 ICC!
    let p3_spec = PixelFormat::RGB8_SRGB.with_primaries(Primaries::DisplayP3);
    let p3_plan = OutputPlan::build(
        PixelFormat::RGB8_SRGB,
        p3_spec,
        &ConvertPolicy::permissive(),
        PeakSource::Measured,
        OutputSignaling::Auto,
    )
    .unwrap();
    let p3_ready = p3_plan.execute_buf(owned).unwrap();
    assert_eq!(p3_ready.cicp, Some(Cicp::DISPLAY_P3));
    assert!(p3_ready.icc.is_some());
}

#[test]
fn negotiate_selects_lowest_loss_and_respects_policy() {
    let src = PixelFormat::RGBA8_SRGB;
    let supported = [
        PixelFormat::GRAY8_SRGB,
        PixelFormat::RGB8_SRGB,
        PixelFormat::BGRA8_SRGB,
        PixelFormat::RGBA16_SRGB,
    ];

    // Under lossless policy (forbids alpha drop & depth loss), BGRA8 beats RGBA16 (smaller bpp, lossless swizzle!)
    let best = Negotiate::select(&src, &supported, &ConvertPolicy::lossless()).unwrap();
    assert_eq!(best.format, PixelFormat::BGRA8_SRGB);
    assert_eq!(best.loss, LossKind::None);

    // If exact identity is in supported list, it wins immediately (effort = 0).
    let supported_with_id = [PixelFormat::BGRA8_SRGB, PixelFormat::RGBA8_SRGB];
    let best_id = Negotiate::select(&src, &supported_with_id, &ConvertPolicy::lossless()).unwrap();
    assert_eq!(best_id.format, PixelFormat::RGBA8_SRGB);
    assert_eq!(best_id.cost.effort, 0);
}

#[test]
fn orient_all_8_orientations_and_pixel_widths_match_forward_map_oracle() {
    // Prime non-multiple-of-tile dims (37x29) test BOTH AVX2/NEON register bodies AND edge tails!
    let (w, h) = (37u32, 29u32);
    let specs = [
        PixelFormat::GRAY8_SRGB,     // 1 B/px
        PixelFormat::GRAY16_SRGB,    // 2 B/px
        PixelFormat::RGB8_SRGB,      // 3 B/px
        PixelFormat::RGBA8_SRGB,     // 4 B/px
        PixelFormat::RGB16_SRGB,     // 6 B/px
        PixelFormat::RGBA16_SRGB,    // 8 B/px
        PixelFormat::RGBF32_LINEAR,  // 12 B/px
        PixelFormat::RGBAF32_LINEAR, // 16 B/px
    ];

    for spec in specs {
        let bpp = spec.layout.bytes_per_pixel();
        let mut src = PixelBuf::new(w, h, spec.clone()).unwrap();
        {
            let mut mv = src.as_view_mut();
            for y in 0..h {
                let row = mv.row_mut(y);
                for x in 0..w {
                    let off = (x as usize) * bpp;
                    for k in 0..bpp {
                        row[off + k] = ((x * 7 + y * 13 + k as u32 * 29) & 0xFF) as u8;
                    }
                }
            }
        }

        for orient in Orientation::ALL {
            let out = apply_orientation(&src.as_view(), orient).unwrap();
            let (ow, oh) = orient.output_dimensions(w, h);
            assert_eq!((out.width(), out.height()), (ow, oh));
            let sv = src.as_view();
            let ov = out.as_view();
            for sy in 0..h {
                let srow = sv.row(sy);
                for sx in 0..w {
                    let (dx, dy) = orient.forward_map(sx, sy, w, h);
                    let got = &ov.row(dy)[(dx as usize) * bpp..(dx as usize + 1) * bpp];
                    let want = &srow[(sx as usize) * bpp..(sx as usize + 1) * bpp];
                    assert_eq!(
                        got, want,
                        "orient={orient:?} bpp={bpp} src=({sx},{sy})->dst=({dx},{dy})"
                    );
                }
            }
        }
    }
}

#[test]
fn safe_icc_substitution_without_cms_matches_adobe_oracle() {
    extern crate alloc;
    use alloc::sync::Arc;
    use zenpixels_convert::{AlphaMode, ColorSpace, RenderingIntent, SampleLayout};

    // A08: Safe known ICC profile (Adobe RGB) substitutes directly without needing a CMS.
    // Test midtone [180, 90, 40], toe [12, 6, 3], and out-of-gamut green [0, 255, 0], pinning alpha.
    let adobe_icc = Arc::<[u8]>::from(zenpixels_convert::icc_profiles::ADOBE_RGB);
    let icc_spec = PixelFormat::new(
        SampleLayout::RGBA8,
        ColorSpace::Icc {
            profile: Arc::clone(&adobe_icc),
            channels: 3,
            alpha: AlphaMode::Straight,
            intent: RenderingIntent::RelativeColorimetric,
        },
    );
    let sub_plan = zenpixels_convert::RowPlan::build(
        icc_spec.clone(),
        PixelFormat::RGBA8_SRGB,
        &ConvertPolicy::permissive(),
        None,
    )
    .unwrap();
    let src_px: [u8; 16] = [
        180, 90, 40, 200, // midtone
        12, 6, 3, 77, // toe
        0, 255, 0, 255, // out-of-gamut Adobe green
        255, 255, 255, 128, // white
    ];
    let mut sub_out = [0u8; 16];
    sub_plan.convert_row(&src_px, &mut sub_out, 4, 0).unwrap();

    // Direct structured AdobeRgb -> sRGB oracle (no ICC path).
    let direct_spec = PixelFormat::new(
        SampleLayout::RGBA8,
        ColorSpace::Rgb {
            primaries: zenpixels_convert::Primaries::AdobeRgb,
            transfer: zenpixels_convert::Transfer::AdobeRgb,
            range: zenpixels_convert::SignalRange::Full,
            alpha: AlphaMode::Straight,
            white: None,
        },
    );
    let direct_plan = zenpixels_convert::RowPlan::build(
        direct_spec,
        PixelFormat::RGBA8_SRGB,
        &ConvertPolicy::permissive(),
        None,
    )
    .unwrap();
    let mut direct_out = [0u8; 16];
    direct_plan
        .convert_row(&src_px, &mut direct_out, 4, 0)
        .unwrap();
    assert_eq!(
        sub_out, direct_out,
        "ICC substitution must match structured Adobe RGB -> sRGB bit-exact"
    );
    // Pin alpha bytes untouched on all 4 pixels.
    assert_eq!(
        [sub_out[3], sub_out[7], sub_out[11], sub_out[15]],
        [200, 77, 255, 128]
    );
    // Out-of-gamut Adobe green [0,255,0] clips R<=0, G=255, B<=0 in sRGB!
    assert_eq!(&sub_out[8..11], &[0, 255, 0]);
    // White stays exact white.
    assert_eq!(&sub_out[12..15], &[255, 255, 255]);

    // AbsoluteColorimetric must NOT use ICC substitution without a CMS!
    let abs_spec = PixelFormat::new(
        SampleLayout::RGBA8,
        ColorSpace::Icc {
            profile: adobe_icc,
            channels: 3,
            alpha: AlphaMode::Straight,
            intent: RenderingIntent::AbsoluteColorimetric,
        },
    );
    assert!(
        zenpixels_convert::RowPlan::build(
            abs_spec,
            PixelFormat::RGBA8_SRGB,
            &ConvertPolicy::permissive(),
            None,
        )
        .is_err(),
        "AbsoluteColorimetric must refuse ICC substitution when no CMS is attached"
    );
}

#[cfg(feature = "cms-moxcms")]
#[test]
fn moxcms_icc_transform_and_safe_substitution() {
    extern crate alloc;
    use alloc::sync::Arc;
    use zenpixels_convert::{AlphaMode, ColorSpace, RenderingIntent, SampleLayout};

    // A08: Compare the SAME non-neutral Adobe RGB pixels through BOTH safe substitution AND
    // explicit MoxCms (with substitution disabled) on every channel!
    let adobe_icc = Arc::<[u8]>::from(zenpixels_convert::icc_profiles::ADOBE_RGB);
    let icc_spec = PixelFormat::new(
        SampleLayout::RGBA8,
        ColorSpace::Icc {
            profile: adobe_icc,
            channels: 3,
            alpha: AlphaMode::Straight,
            intent: RenderingIntent::RelativeColorimetric,
        },
    );
    let src_px: [u8; 16] = [
        180, 90, 40, 200, // midtone
        12, 6, 3, 77, // toe
        0, 255, 0, 255, // out-of-gamut Adobe green
        255, 255, 255, 128, // white
    ];
    let sub_plan = zenpixels_convert::RowPlan::build(
        icc_spec.clone(),
        PixelFormat::RGBA8_SRGB,
        &ConvertPolicy::permissive(),
        None,
    )
    .unwrap();
    let cms_policy = ConvertPolicy::permissive()
        .with_icc_substitution(false)
        .with_cms(Arc::new(zenpixels_convert::MoxCms));
    let cms_plan =
        zenpixels_convert::RowPlan::build(icc_spec, PixelFormat::RGBA8_SRGB, &cms_policy, None)
            .unwrap();

    let mut sub_out = [0u8; 16];
    let mut cms_out = [0u8; 16];
    sub_plan.convert_row(&src_px, &mut sub_out, 4, 0).unwrap();
    cms_plan.convert_row(&src_px, &mut cms_out, 4, 0).unwrap();

    for px in 0..4 {
        for c in 0..3 {
            let s = sub_out[px * 4 + c];
            let m = cms_out[px * 4 + c];
            assert!(
                (i32::from(s) - i32::from(m)).abs() <= 1,
                "Adobe->sRGB px {px} ch {c}: substitution={s} vs MoxCms={m} (sub={sub_out:?}, cms={cms_out:?})"
            );
        }
        assert_eq!(sub_out[px * 4 + 3], src_px[px * 4 + 3]);
        assert_eq!(cms_out[px * 4 + 3], src_px[px * 4 + 3]);
    }
    // Independent binary64 Adobe RGB (1998) -> sRGB oracle for [180, 90, 40]:
    // Decode Adobe RGB via x^(563/256), convert Adobe->sRGB linear via exact chromaticity matrix,
    // encode sRGB OETF, round to u8.
    let adobe_lin = [180.0_f64, 90.0, 40.0].map(|c| (c / 255.0).powf(563.0 / 256.0));
    // Standard Adobe RGB (1998) -> sRGB (D65) linear matrix (IEC 61966-2-1 / Adobe RGB 1998 spec):
    let srgb_lin = [
        1.3983576 * adobe_lin[0] - 0.3983576 * adobe_lin[1],
        adobe_lin[1],
        -0.0429383 * adobe_lin[1] + 1.0429383 * adobe_lin[2],
    ];
    let srgb_enc = |v: f64| -> u8 {
        let c = v.clamp(0.0, 1.0);
        let e = if c <= 0.0031308 {
            12.92 * c
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        };
        (e * 255.0).round() as u8
    };
    let expected_mid = srgb_lin.map(srgb_enc);
    for c in 0..3 {
        assert!(
            (i32::from(cms_out[c]) - i32::from(expected_mid[c])).abs() <= 1,
            "Adobe->sRGB midtone ch {c}: got={} expected_oracle={} (full={:?} vs {expected_mid:?})",
            cms_out[c],
            expected_mid[c],
            &cms_out[0..3]
        );
    }
}
