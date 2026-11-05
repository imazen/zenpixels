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
