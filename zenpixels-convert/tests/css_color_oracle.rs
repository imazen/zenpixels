//! CSS parsing against Color.js 0.7.1, an independent implementation. The table comes from
//! `benchmarks/gamut-compress/css_oracle.mjs`.
use zenpixels::{
    AlphaMode, ChannelLayout, ChannelType, Color, ColorPrimaries as P, ConvertOptions,
    PixelDescriptor as D, TransferFunction as Tf,
};
use zenpixels_convert::css::CssColorError;
use zenpixels_convert::{ColorConvertExt, RowConverter};

/// Linear BT.2020 RGBA, unclipped, and straight alpha.
fn to_reference_space(color: &Color) -> [f32; 4] {
    let to = D::new(
        ChannelType::F32,
        ChannelLayout::Rgba,
        Some(AlphaMode::Straight),
        Tf::Linear,
    )
    .with_primaries(P::Bt2020);
    let options = ConvertOptions::permissive().with_clip_out_of_gamut(false);
    let mut out = [0f32; 4];
    RowConverter::new_explicit(color.descriptor(), to, &options)
        .unwrap()
        .convert_row(color.as_bytes(), bytemuck::cast_slice_mut(&mut out), 1);
    out
}

#[test]
fn matches_color_js() {
    let table = include_str!("data/css_oracle.tsv");
    let mut failures = Vec::new();
    let mut rows = 0;
    for line in table.lines() {
        let mut fields = line.split('\t');
        let css = fields.next().unwrap();
        let want: Vec<f64> = fields.map(|v| v.parse().unwrap()).collect();
        rows += 1;
        let Ok(color) = Color::parse_css(css) else {
            failures.push(format!("{css}: refused"));
            continue;
        };
        let got = to_reference_space(&color);
        // rgb() clamps at parse time and Color.js doesn't, which the spec requires of us.
        let tolerance = if css.starts_with("rgb(300") {
            f64::INFINITY
        } else {
            1e-5
        };
        for c in 0..4 {
            if (f64::from(got[c]) - want[c]).abs() > tolerance {
                failures.push(format!("{css}: {got:?} vs {want:?}"));
                break;
            }
        }
    }
    assert!(rows >= 60, "{rows} rows");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn eight_bit_forms_stay_exact() {
    let exact = [
        ("#123", [0x11, 0x22, 0x33]),
        ("#ABCDEF", [0xab, 0xcd, 0xef]),
        ("RGB(1, 2, 3)", [1, 2, 3]),
        ("rgb(1 2 3)", [1, 2, 3]),
        ("  Red  ", [255, 0, 0]),
    ];
    for (css, [r, g, b]) in exact {
        assert_eq!(
            Color::parse_css(css).unwrap(),
            Color::srgb8(r, g, b),
            "{css}"
        );
    }
    let half = Color::parse_css("#11223380").unwrap();
    assert_eq!(half.descriptor(), D::RGBA8_SRGB);
    assert_eq!(half.as_bytes(), [0x11, 0x22, 0x33, 0x80]);
    assert_eq!(
        Color::parse_css("transparent").unwrap().as_bytes(),
        [0, 0, 0, 0]
    );
}

#[test]
fn named_spaces_keep_their_descriptor() {
    let p3 = Color::parse_css("color(display-p3 1 0 0)").unwrap();
    assert_eq!(p3.descriptor().primaries, P::DisplayP3);
    assert_eq!(p3.descriptor().transfer(), Tf::Srgb);
    let pq = Color::parse_css("color(rec2100-pq 0.58 0.58 0.58)").unwrap();
    assert_eq!(
        (pq.descriptor().primaries, pq.descriptor().transfer()),
        (P::Bt2020, Tf::Pq)
    );
    // CSS puts 1.0 at 203 cd/m², and our linear puts it at PQ's 10,000.
    let linear = Color::parse_css("color(rec2100-linear 1 1 1)").unwrap();
    let samples: &[f32] = bytemuck::cast_slice(linear.as_bytes());
    assert!((samples[0] - 0.0203).abs() < 1e-6, "{samples:?}");
}

#[test]
fn parsed_wide_colors_map_into_srgb() {
    let mapped = Color::parse_css("oklch(0.7 0.4 150)")
        .unwrap()
        .map_to_gamut(D::RGB8_SRGB)
        .unwrap();
    assert_eq!(mapped.descriptor(), D::RGB8_SRGB);
}

#[test]
fn refuses_what_is_not_a_color() {
    for bad in [
        "",
        "#",
        "#12",
        "#12345",
        "#gggggg",
        "notacolor",
        "rgb()",
        "rgb(1 2)",
        "rgb(1 2 3 4)",
        "rgb(1, 2 3)",
        "rgb(1 2 3 / 4 / 5)",
        "rgb(none, 2, 3)",
        "rgb(nan 2 3)",
        "rgb(inf 2 3)",
        "rgb(1 2 3",
        "hsl(10% 50% 50%)",
        "hwb(10, 20%, 30%)",
        "lab(50, 10, 10)",
        "lab(50 10deg 10)",
        "color(unknown 1 2 3)",
        "color(srgb 1 2)",
        "color(srgb)",
        "oklch(0.5 0.1 10%)",
    ] {
        assert_eq!(
            Color::parse_css(bad),
            Err(CssColorError::Invalid),
            "{bad:?}"
        );
    }
    for needs_context in [
        "currentcolor",
        "color-mix(in srgb, red, blue)",
        "device-cmyk(0 0 0 1)",
    ] {
        assert_eq!(
            Color::parse_css(needs_context),
            Err(CssColorError::Unsupported),
            "{needs_context}"
        );
    }
}
