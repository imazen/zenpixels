//! Tests for `OutputPlan` / `EncodeReady` (atomic CICP/ICC + zero-copy identity),
//! `Negotiate::select`, `orient::apply_orientation` across all 8 orientations & widths,
//! and `MoxCms` ICC transforms.

use zenpixels_convert::{
    Cicp, ColorOrigin, ConvertPolicy, ImageSpec, LossKind, Negotiate, Orientation, OutputPlan,
    OutputProfile, PeakSource, PixelBuf, PixelCow, Primaries, apply_orientation,
};

#[test]
fn output_plan_zero_copy_identity_and_atomic_signaling() {
    let mut buf = PixelBuf::new(8, 4, ImageSpec::rgb8_srgb()).unwrap();
    buf.as_view_mut().row_mut(0)[0..3].copy_from_slice(&[42, 84, 126]);
    let in_ptr = buf.as_view().row(0).as_ptr();

    // Identity on owned PixelBuf moves storage ZERO-COPY!
    let id_plan = OutputPlan::build(
        ImageSpec::rgb8_srgb(),
        ImageSpec::rgb8_srgb(),
        ConvertPolicy::lossless(),
        PeakSource::Measured,
        OutputProfile::CicpOnly,
    )
    .unwrap();
    assert!(id_plan.is_identity());
    let ready = id_plan.execute_buf(buf).unwrap();
    assert_eq!(ready.cicp, Some(Cicp::SRGB));
    assert!(ready.icc.is_none());
    assert_eq!(ready.pixels.as_view().row(0).as_ptr(), in_ptr);

    // Identity on borrowed PixelView returns PixelCow::Borrowed zero-copy!
    let owned = ready.pixels.try_into_owned().unwrap();
    let ready_view = id_plan
        .execute_view(owned.as_view(), &ColorOrigin::assumed())
        .unwrap();
    assert!(matches!(ready_view.pixels, PixelCow::Borrowed(_)));
    assert_eq!(
        ready_view.pixels.as_view().row(0).as_ptr(),
        owned.as_view().row(0).as_ptr()
    );

    // Non-identity: sRGB -> Display P3 Auto emits Display P3 CICP + bundled P3 ICC!
    let p3_spec = ImageSpec::rgb8_srgb().with_primaries(Primaries::DisplayP3);
    let p3_plan = OutputPlan::build(
        ImageSpec::rgb8_srgb(),
        p3_spec,
        ConvertPolicy::permissive(),
        PeakSource::Measured,
        OutputProfile::Auto,
    )
    .unwrap();
    let p3_ready = p3_plan.execute_buf(owned).unwrap();
    assert_eq!(p3_ready.cicp, Some(Cicp::DISPLAY_P3));
    assert!(p3_ready.icc.is_some());
}

#[test]
fn negotiate_selects_lowest_loss_and_respects_policy() {
    let src = ImageSpec::rgba8_srgb();
    let supported = [
        ImageSpec::gray8_srgb(),
        ImageSpec::rgb8_srgb(),
        ImageSpec::bgra8_srgb(),
        ImageSpec::rgba16_srgb(),
    ];

    // Under lossless policy (forbids alpha drop & depth loss), BGRA8 beats RGBA16 (smaller bpp, lossless swizzle!)
    let best = Negotiate::select(&src, &supported, &ConvertPolicy::lossless()).unwrap();
    assert_eq!(best.spec, ImageSpec::bgra8_srgb());
    assert_eq!(best.loss, LossKind::None);

    // If exact identity is in supported list, it wins immediately (effort = 0).
    let supported_with_id = [ImageSpec::bgra8_srgb(), ImageSpec::rgba8_srgb()];
    let best_id = Negotiate::select(&src, &supported_with_id, &ConvertPolicy::lossless()).unwrap();
    assert_eq!(best_id.spec, ImageSpec::rgba8_srgb());
    assert_eq!(best_id.cost.effort, 0);
}

#[test]
fn orient_all_8_orientations_and_pixel_widths_match_forward_map_oracle() {
    // Prime non-multiple-of-tile dims (37x29) test BOTH AVX2/NEON register bodies AND edge tails!
    let (w, h) = (37u32, 29u32);
    let specs = [
        ImageSpec::gray8_srgb(),     // 1 B/px
        ImageSpec::gray16_srgb(),    // 2 B/px
        ImageSpec::rgb8_srgb(),      // 3 B/px
        ImageSpec::rgba8_srgb(),     // 4 B/px
        ImageSpec::rgb16_srgb(),     // 6 B/px
        ImageSpec::rgba16_srgb(),    // 8 B/px
        ImageSpec::rgbf32_linear(),  // 12 B/px
        ImageSpec::rgbaf32_linear(), // 16 B/px
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

#[cfg(feature = "cms-moxcms")]
#[test]
fn moxcms_icc_transform_and_safe_substitution() {
    extern crate alloc;
    use alloc::sync::Arc;
    use zenpixels_convert::{ColorSpace, SampleLayout, Transfer};

    // 1. Safe known ICC profile (Adobe RGB) substitutes directly without even needing a CMS!
    let adobe_icc = Arc::<[u8]>::from(zenpixels_convert::icc_profiles::ADOBE_RGB);
    let icc_spec = ImageSpec::new(
        SampleLayout::RGB8,
        ColorSpace::Icc {
            profile: Arc::clone(&adobe_icc),
            channels: 3,
            alpha: zenpixels_convert::AlphaMode::None,
            intent: zenpixels_convert::RenderingIntent::RelativeColorimetric,
        },
    );
    let sub_plan = zenpixels_convert::RowPlan::build(
        icc_spec.clone(),
        ImageSpec::rgb8_srgb(),
        &ConvertPolicy::permissive(),
        None,
    )
    .unwrap();
    let mut sub_out = [0u8; 3];
    sub_plan
        .convert_row(&[0, 255, 0], &mut sub_out, 1, 0)
        .unwrap();

    // 2. Explicit MoxCms transform on a custom Display P3 ICC profile agrees closely!
    let p3_icc = Arc::<[u8]>::from(zenpixels_convert::icc_profiles::DISPLAY_P3_V4);
    let p3_icc_spec = ImageSpec::new(
        SampleLayout::RGB8,
        ColorSpace::Icc {
            profile: p3_icc,
            channels: 3,
            alpha: zenpixels_convert::AlphaMode::None,
            intent: zenpixels_convert::RenderingIntent::RelativeColorimetric,
        },
    );
    let cms_policy = ConvertPolicy::permissive().with_cms(Arc::new(zenpixels_convert::MoxCms));
    let cms_plan =
        zenpixels_convert::RowPlan::build(p3_icc_spec, ImageSpec::rgb8_srgb(), &cms_policy, None)
            .unwrap();
    let mut white_out = [0u8; 3];
    cms_plan
        .convert_row(&[255, 255, 255], &mut white_out, 1, 0)
        .unwrap();
    assert_eq!(white_out, [255, 255, 255]);
    assert!(sub_out[1] >= 250);
    let _ = Transfer::Srgb;
}
