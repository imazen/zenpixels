//! `Color::map_to_gamut` against an `f64` port of the CSS Color 4 algorithm that agreed
//! with Color.js within 1.55e-14 on 2,048 samples (`benchmarks/gamut-survey`).
use zenpixels::{Color, ColorPrimaries as P, PixelDescriptor as D, TransferFunction as Tf};
use zenpixels_convert::{ColorConvertExt, ConvertError};

fn linear(p: P, rgb: [f32; 3]) -> Color {
    Color::try_new(D::RGBF32_LINEAR.with_primaries(p), bytemuck::bytes_of(&rgb)).unwrap()
}

fn srgb_f32() -> D {
    D::RGBF32_LINEAR.with_transfer(Tf::Srgb)
}

#[test]
fn matches_the_reference() {
    #[rustfmt::skip]
    let cases = [
        (P::DisplayP3, [1.0, 0.0, 0.0], [1.000000, 0.044570, 0.045932]),
        (P::DisplayP3, [0.0, 1.0, 0.0], [0.000000, 0.985764, 0.159742]),
        (P::DisplayP3, [0.0, 0.0, 1.0], [0.000000, 0.000000, 1.000000]),
        (P::DisplayP3, [0.9, 0.2, 0.3], [1.000000, 0.449895, 0.580341]),
        (P::Bt2020, [1.0, 0.0, 0.0], [1.000000, 0.288023, 0.308861]),
        (P::Bt2020, [0.0, 1.0, 0.0], [0.000000, 0.947344, 0.446281]),
        (P::Bt2020, [0.0, 0.0, 1.0], [0.000000, 0.317974, 0.576247]),
        (P::Bt2020, [0.2, 0.8, 0.6], [0.000000, 0.922375, 0.789058]),
    ];
    for (primaries, input, want) in cases {
        let mapped = linear(primaries, input).map_to_gamut(srgb_f32()).unwrap();
        let got: &[f32] = bytemuck::cast_slice(mapped.as_bytes());
        for c in 0..3 {
            assert!(
                (got[c] - want[c]).abs() < 2e-3,
                "{primaries:?} {input:?}: {got:?} vs {want:?}"
            );
        }
    }
}

#[test]
fn in_gamut_colors_are_a_plain_conversion() {
    for rgb in [
        [0.5f32, 0.25, 0.75],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.18; 3],
    ] {
        let color = linear(P::Bt709, rgb);
        let mapped = color.map_to_gamut(srgb_f32()).unwrap();
        let plain = color
            .try_convert_to(srgb_f32(), &zenpixels::ConvertOptions::permissive())
            .unwrap();
        let (a, b): (&[f32], &[f32]) = (
            bytemuck::cast_slice(mapped.as_bytes()),
            bytemuck::cast_slice(plain.as_bytes()),
        );
        for c in 0..3 {
            assert!((a[c] - b[c]).abs() < 1e-5, "{rgb:?}: {a:?} vs {b:?}");
        }
    }
}

#[test]
fn eight_bit_targets_and_alpha() {
    let p3 = Color::try_new(
        D::RGBA8_SRGB.with_primaries(P::DisplayP3),
        &[255, 0, 0, 128],
    )
    .unwrap();
    let mapped = p3.map_to_gamut(D::RGBA8_SRGB).unwrap();
    assert_eq!(mapped.descriptor(), D::RGBA8_SRGB);
    assert_eq!(mapped.as_bytes()[3], 128);
    assert_eq!(mapped.as_bytes()[0], 255);
    assert!(mapped.as_bytes()[1] > 0 && mapped.as_bytes()[2] > 0);
}

#[test]
fn refuses_what_it_cannot_map() {
    let c = Color::srgb8(1, 2, 3);
    let refused = [
        D::RGB16_SRGB
            .with_transfer(Tf::Pq)
            .with_primaries(P::Bt2020),
        D::GRAY8_SRGB,
        D::RGB8_SRGB.with_primaries(P::AdobeRgb),
        D::RGB8_SRGB.with_primaries(P::Unknown),
    ];
    for target in refused {
        let e = c.map_to_gamut(target).unwrap_err();
        assert!(
            matches!(*e.error(), ConvertError::NoPath { .. }),
            "{target:?}: {e:?}"
        );
    }
}

mod same_color {
    use zenpixels::{
        ChannelLayout as L, ChannelType as C, Color, ColorPrimaries as P, PixelDescriptor as D,
        TransferFunction as T,
    };
    use zenpixels_convert::ColorConvertExt;

    fn from_f32(d: D, samples: &[f32]) -> Color {
        let bytes: Vec<u8> = samples.iter().flat_map(|v| v.to_ne_bytes()).collect();
        Color::try_new(d, &bytes).unwrap()
    }

    fn from_u16(d: D, samples: &[u16]) -> Color {
        let bytes: Vec<u8> = samples.iter().flat_map(|v| v.to_ne_bytes()).collect();
        Color::try_new(d, &bytes).unwrap()
    }

    #[test]
    fn formats_of_one_color_match() {
        let red = Color::srgb8(255, 0, 0);
        assert!(red.same_color(&Color::srgba8(255, 0, 0, 255)));
        let white = Color::srgb8(255, 255, 255);
        assert!(white.same_color(&from_f32(D::RGBF32_LINEAR, &[1.0, 1.0, 1.0])));
        assert!(white.same_color(&from_u16(D::RGB16_SRGB, &[65535, 65535, 65535])));
        for v in [1u8, 7, 18, 128, 200, 254] {
            let eight = Color::srgb8(v, v, v);
            let sixteen = u16::from(v) * 257;
            assert!(
                eight.same_color(&from_u16(D::RGB16_SRGB, &[sixteen, sixteen, sixteen])),
                "{v}"
            );
            let fraction = f32::from(v) / 255.0;
            let float = D::new(C::F32, L::Rgb, None, T::Srgb);
            assert!(
                eight.same_color(&from_f32(float, &[fraction, fraction, fraction])),
                "{v}"
            );
        }
    }

    #[test]
    fn neutrals_match_across_primaries() {
        for v in [0.0f32, 0.1, 0.5, 1.0] {
            let wide = from_f32(D::RGBF32_LINEAR.with_primaries(P::DisplayP3), &[v, v, v]);
            let narrow = from_f32(D::RGBF32_LINEAR, &[v, v, v]);
            assert!(wide.same_color(&narrow), "{v}");
        }
        let wide = from_f32(
            D::RGBF32_LINEAR.with_primaries(P::DisplayP3),
            &[1.0, 0.0, 0.0],
        );
        assert!(!wide.same_color(&from_f32(D::RGBF32_LINEAR, &[1.0, 0.0, 0.0])));
    }

    #[test]
    fn different_colors_differ() {
        let red = Color::srgb8(255, 0, 0);
        assert!(!red.same_color(&Color::srgb8(254, 0, 0)));
        assert!(!red.same_color(&Color::srgba8(255, 0, 0, 254)));
        // One 16-bit step, in the darks where linear light moves least.
        let dark = from_u16(D::RGB16_SRGB, &[1, 1, 1]);
        assert!(!dark.same_color(&from_u16(D::RGB16_SRGB, &[2, 2, 2])));
    }

    #[test]
    fn transparent_colors_match() {
        assert!(Color::srgba8(255, 0, 0, 0).same_color(&Color::TRANSPARENT));
        assert!(!Color::srgba8(255, 0, 0, 1).same_color(&Color::srgba8(0, 0, 0, 1)));
    }

    #[test]
    fn scales_that_differ_never_match() {
        let pq = from_f32(D::new(C::F32, L::Rgb, None, T::Pq), &[0.5, 0.5, 0.5]);
        let hlg = from_f32(D::new(C::F32, L::Rgb, None, T::Hlg), &[0.5, 0.5, 0.5]);
        assert!(!pq.same_color(&hlg));
        assert!(!pq.same_color(&Color::srgb8(128, 128, 128)));
        assert!(pq.same_color(&pq));
    }

    #[test]
    fn colors_with_no_linear_light_match_only_themselves() {
        let half = [0.5f32, 0.5, 0.5];
        let unknown_primaries = from_f32(D::RGBF32_LINEAR.with_primaries(P::Unknown), &half);
        assert!(unknown_primaries.same_color(&unknown_primaries));
        assert!(!unknown_primaries.same_color(&from_f32(D::RGBF32_LINEAR, &half)));
        assert!(
            !unknown_primaries
                .same_color(&from_f32(D::RGBF32_LINEAR.with_primaries(P::Bt2020), &half))
        );
        let unknown_transfer = from_f32(D::new(C::F32, L::Rgb, None, T::Unknown), &half);
        assert!(
            !unknown_transfer.same_color(&from_f32(D::new(C::F32, L::Rgb, None, T::Srgb), &half))
        );

        let cmyk = D::new(C::U8, L::Cmyk, None, T::Unknown);
        let ink = Color::try_new(cmyk, &[10, 20, 30, 40]).unwrap();
        assert!(ink.same_color(&ink));
        assert!(!ink.same_color(&Color::try_new(cmyk, &[10, 20, 30, 41]).unwrap()));
        assert!(!ink.same_color(&Color::srgb8(10, 20, 30)));
    }

    #[test]
    fn layouts_of_one_color_match() {
        let gray = Color::try_new(D::GRAY8_SRGB, &[128]).unwrap();
        assert!(gray.same_color(&Color::srgb8(128, 128, 128)));
        assert!(!gray.same_color(&Color::srgb8(128, 128, 129)));
        let bgr = Color::try_new(D::BGR8.with_transfer(T::Srgb), &[30, 20, 10]).unwrap();
        assert!(bgr.same_color(&Color::srgb8(10, 20, 30)));
        assert!(!bgr.same_color(&Color::srgb8(30, 20, 10)));
        let bgra = Color::try_new(D::BGRA8_SRGB, &[30, 20, 10, 255]).unwrap();
        assert!(bgra.same_color(&Color::srgb8(10, 20, 30)));
    }
}
