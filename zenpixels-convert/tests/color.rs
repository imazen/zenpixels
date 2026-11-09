//! One color, with what its samples mean: alignment, conversion and wide-gamut matte.
use zenpixels::{
    AlphaMode, AlphaPolicy, ChannelLayout as L, ChannelType as T, Color, ColorPrimaries as P,
    PixelDescriptor as D, TransferFunction as Tf,
};
use zenpixels_convert::{ColorConvertExt, ConvertError, ConvertOptions, RowConverter};

fn f32_color(descriptor: D, samples: &[f32]) -> Color {
    Color::try_new(descriptor, bytemuck::cast_slice(samples)).unwrap()
}

fn lossless() -> ConvertOptions {
    ConvertOptions::permissive().with_clip_out_of_gamut(false)
}

#[test]
fn samples_cast_without_copying_or_failing() {
    let wide = f32_color(D::RGBAF32_LINEAR, &[0.25, 0.5, 0.75, 1.0]);
    let color = core::hint::black_box(wide);
    assert_eq!(color.as_bytes().as_ptr() as usize % 4, 0);
    assert_eq!(
        bytemuck::cast_slice::<u8, f32>(color.as_bytes()),
        [0.25, 0.5, 0.75, 1.0]
    );

    let u16s = Color::try_new(D::RGB16_SRGB, bytemuck::cast_slice(&[1u16, 2, 3])).unwrap();
    assert_eq!(bytemuck::cast_slice::<u8, u16>(u16s.as_bytes()), [1, 2, 3]);
    assert_eq!(Color::srgb8(1, 2, 3).as_bytes(), [1, 2, 3]);
}

#[test]
fn a_pixel_of_the_wrong_size_is_refused() {
    assert!(Color::try_new(D::RGB8_SRGB, &[1, 2]).is_err());
    assert!(Color::try_new(D::RGB8_SRGB, &[1, 2, 3, 4]).is_err());
}

#[test]
fn equal_means_same_bits_and_same_meaning() {
    assert_eq!(Color::srgb8(1, 2, 3), Color::srgb8(1, 2, 3));
    assert_ne!(Color::srgb8(1, 2, 3), Color::srgb8(1, 2, 4));
    let srgb = Color::srgb8(1, 2, 3);
    let linear = Color::try_new(D::RGB8_SRGB.with_transfer(Tf::Linear), &[1, 2, 3]).unwrap();
    assert_ne!(srgb, linear);
    let (a, b) = (
        f32_color(D::RGBF32_LINEAR, &[0.0, 0.0, 0.0]),
        f32_color(D::RGBF32_LINEAR, &[-0.0, 0.0, 0.0]),
    );
    assert_ne!(a, b);
}

/// Display P3 red is out of sRGB's gamut, and that survives conversion.
#[test]
fn conversion_keeps_out_of_gamut_values() {
    let p3 = D::RGBF32_LINEAR.with_primaries(P::DisplayP3);
    let srgb = f32_color(p3, &[1.0, 0.0, 0.0])
        .try_convert_to(D::RGBF32_LINEAR, &lossless())
        .unwrap();
    let c = bytemuck::cast_slice::<u8, f32>(srgb.as_bytes());
    assert!(c[0] > 1.0 && c[1] < 0.0 && c[2] < 0.0, "{c:?}");
    assert_eq!(srgb.descriptor(), D::RGBF32_LINEAR);

    let back = srgb.try_convert_to(p3, &lossless()).unwrap();
    let c = bytemuck::cast_slice::<u8, f32>(back.as_bytes());
    assert!(
        (c[0] - 1.0).abs() < 1e-5 && c[1].abs() < 1e-5 && c[2].abs() < 1e-5,
        "{c:?}"
    );
}

#[test]
fn conversion_refuses_targets_wider_than_a_color() {
    let wide = D::new(T::F64, L::Rgba, Some(AlphaMode::Straight), Tf::Linear);
    assert!(
        Color::srgb8(1, 2, 3)
            .try_convert_to(wide, &lossless())
            .is_err()
    );
}

/// `background` under a fully transparent F32 RGBA pixel of `source`'s transfer and primaries.
fn background_of(source: D, background: Color) -> Result<[f32; 3], String> {
    let from = D::new(
        T::F32,
        L::Rgba,
        Some(AlphaMode::Straight),
        source.transfer(),
    )
    .with_primaries(source.primaries);
    let to = D::new(T::F32, L::Rgb, None, source.transfer()).with_primaries(source.primaries);
    let policy = AlphaPolicy::CompositeOnto {
        background,
        reference_white: None,
    };
    let options = ConvertOptions::permissive().with_alpha_policy(policy);
    let mut c = RowConverter::new_explicit(from, to, &options).map_err(|e| format!("{e}"))?;
    let mut out = [0f32; 3];
    c.convert_row(
        bytemuck::cast_slice(&[0f32; 4]),
        bytemuck::cast_slice_mut(&mut out),
        1,
    );
    Ok(out)
}

fn close(a: [f32; 3], b: [f32; 3], tolerance: f32) {
    for (x, y) in a.iter().zip(&b) {
        assert!((x - y).abs() <= tolerance, "{a:?} against {b:?}");
    }
}

/// sRGB red on BT.2020 linear: the matrix result, not red's numbers reinterpreted.
#[test]
fn background_converts_to_wide_gamut_sources() {
    let m = P::Bt709.gamut_matrix_to(P::Bt2020).unwrap();
    let out = background_of(
        D::RGBF32_LINEAR.with_primaries(P::Bt2020),
        Color::srgb8(255, 0, 0),
    )
    .unwrap();
    close(out, [m[0][0], m[1][0], m[2][0]], 2e-5);
    assert!(out[1] > 0.05, "{out:?}");
}

/// White is white in every D65 gamut.
#[test]
fn white_background_stays_white_across_primaries() {
    for primaries in [P::Bt709, P::DisplayP3, P::Bt2020] {
        let out = background_of(
            D::RGBF32_LINEAR.with_primaries(primaries),
            Color::srgb8(255, 255, 255),
        )
        .unwrap();
        close(out, [1.0, 1.0, 1.0], 2e-5);
    }
}

/// A wide-gamut background on an sRGB source keeps its out-of-gamut values through the blend.
#[test]
fn wide_gamut_background_on_narrow_source() {
    let p3_red = f32_color(
        D::RGBF32_LINEAR.with_primaries(P::DisplayP3),
        &[1.0, 0.0, 0.0],
    );
    let out = background_of(D::RGBF32_LINEAR, p3_red).unwrap();
    let want = p3_red
        .try_convert_to(D::RGBF32_LINEAR, &lossless())
        .unwrap();
    let want = bytemuck::cast_slice::<u8, f32>(want.as_bytes());
    close(out, [want[0], want[1], want[2]], 1e-6);
    assert!(out[0] > 1.0 && out[1] < 0.0, "{out:?}");
}

/// Unknown primaries get the background as-is: refusing would break ordinary ICC images.
#[test]
fn unknown_source_primaries_take_the_background_unconverted() {
    let out = background_of(
        D::RGBF32_LINEAR.with_primaries(P::Unknown),
        Color::srgb8(255, 0, 0),
    )
    .unwrap();
    close(out, [1.0, 0.0, 0.0], 2e-6);
}

/// A PQ-tagged background is absolute: no reference white applies.
#[test]
fn hdr_background_keeps_its_own_luminance() {
    let source = D::RGBF32_LINEAR
        .with_transfer(Tf::Pq)
        .with_primaries(P::Bt2020);
    let bright = f32_color(source, &[0.75, 0.75, 0.75]);
    let out = background_of(source, bright).unwrap();
    close(out, [0.75, 0.75, 0.75], 5e-5);
}

#[test]
fn a_background_with_alpha_is_refused() {
    let rgba = Color::try_new(D::RGBA8_SRGB, &[1, 2, 3, 128]).unwrap();
    assert!(background_of(D::RGBF32_LINEAR, rgba).is_err());
}

/// HLG and PQ linear aren't comparable, and an SDR source can't hold a PQ background without a peak.
#[test]
fn an_hdr_background_on_another_transfer_is_refused() {
    let pq = f32_color(D::RGBF32_LINEAR.with_transfer(Tf::Pq), &[0.5, 0.5, 0.5]);
    assert!(background_of(D::RGBF32_LINEAR.with_transfer(Tf::Hlg), pq).is_err());
    assert!(background_of(D::RGBF32_LINEAR, pq).is_err());
}

/// PQ and HLG don't say where SDR white belongs, so a color refuses rather than land at the peak.
#[test]
fn sdr_to_hdr_transfers_are_refused() {
    let pq = D::RGBF32_LINEAR.with_transfer(Tf::Pq);
    let hlg = D::RGBF32_LINEAR.with_transfer(Tf::Hlg);
    for sdr in [Tf::Srgb, Tf::Bt709, Tf::Gamma22] {
        let white = Color::try_new(D::RGB8_SRGB.with_transfer(sdr), &[255, 255, 255]).unwrap();
        for target in [pq, hlg] {
            let e = white.try_convert_to(target, &lossless()).unwrap_err();
            assert!(
                matches!(
                    e.error(),
                    ConvertError::SdrWhiteRequired | ConvertError::HlgRenderingRequired
                ),
                "{sdr:?} to {target:?}: {e:?}"
            );
        }
    }
}

/// Linear isn't SDR's question: PQ and HLG decode to it and come back.
#[test]
fn linear_reaches_pq_with_explicit_units_and_hlg_requires_ootf() {
    let options = lossless().with_sdr_white(zenpixels::hdr::DiffuseWhite::new(10000.));
    let hdr = D::RGBF32_LINEAR.with_transfer(Tf::Pq);
    let signal = 0.5807_f32;
    let linear = f32_color(hdr, &[signal; 3])
        .try_convert_to(D::RGBF32_LINEAR, &options)
        .unwrap();
    let back = linear.try_convert_to(hdr, &options).unwrap();
    let values = bytemuck::cast_slice::<u8, f32>(back.as_bytes());
    assert!((values[0] - signal).abs() < 1e-4);
    let hlg = D::RGBF32_LINEAR.with_transfer(Tf::Hlg);
    assert!(matches!(
        f32_color(hlg, &[0.7499; 3])
            .try_convert_to(D::RGBF32_LINEAR, &options)
            .unwrap_err()
            .error(),
        ConvertError::HlgRenderingRequired
    ));
}
