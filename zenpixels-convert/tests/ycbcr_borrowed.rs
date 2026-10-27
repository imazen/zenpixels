use zenpixels::{
    ChannelType, Cicp, PixelDescriptor, PixelSliceMut, TransferFunction,
    sample::{ChromaSiting as Site, ChromaSubsampling as Sub, CodePlane, SampleEncoding},
};
use zenpixels_convert::ycbcr::{YcbcrError, YcbcrToRgbF32};

fn descriptor() -> PixelDescriptor {
    PixelDescriptor::RGBF32_LINEAR.with_transfer(TransferFunction::Srgb)
}
fn run(plan: &YcbcrToRgbF32<'_>, x: usize, y: usize, width: u32, rows: u32) -> Vec<[f32; 3]> {
    let stride = width as usize + 2;
    let mut output = vec![[f32::from_bits(0x7fc12345); 3]; stride * rows as usize];
    let dst = PixelSliceMut::new(
        bytemuck::cast_slice_mut(&mut output),
        width,
        rows,
        stride * 12,
        plan.output_descriptor(),
    )
    .unwrap();
    let result = plan.convert_rows(x, y, dst).unwrap();
    assert_eq!(result.color_context().unwrap().cicp, Some(Cicp::SRGB));
    output
        .chunks_exact(stride)
        .flat_map(|r| {
            assert!(
                r[width as usize..]
                    .iter()
                    .flatten()
                    .all(|v| v.to_bits() == 0x7fc12345)
            );
            r[..width as usize].iter().copied()
        })
        .collect()
}

#[test]
fn code_placement_and_row_partition_do_not_change_rgb() {
    for bits in [8, 10, 12] {
        for sub in [Sub::Yuv444, Sub::Yuv422, Sub::Yuv420] {
            for full in [false, true] {
                let (w, h) = (5usize, 3usize);
                let (cw, ch) = match sub {
                    Sub::Yuv444 => (w, h),
                    Sub::Yuv422 => (3, h),
                    _ => (3, 2),
                };
                let y: Vec<u16> = (0..w * h)
                    .map(|i| (((i * 17) % 256) as u16) << (bits - 8))
                    .collect();
                let u: Vec<u16> = (0..cw * ch)
                    .map(|i| (((i * 41) % 256) as u16) << (bits - 8))
                    .collect();
                let v: Vec<u16> = (0..cw * ch)
                    .map(|i| ((255 - (i * 31) % 256) as u16) << (bits - 8))
                    .collect();
                let enc = SampleEncoding::new(ChannelType::U16, bits, 0).unwrap();
                let planes = [
                    CodePlane::from_u16(&y, w, h, w * 2, enc).unwrap(),
                    CodePlane::from_u16(&u, cw, ch, cw * 2, enc).unwrap(),
                    CodePlane::from_u16(&v, cw, ch, cw * 2, enc).unwrap(),
                ];
                let color = Cicp::new(1, 13, 1, full);
                let plan = YcbcrToRgbF32::new(planes, sub, Site::Center, color).unwrap();
                let expected = run(&plan, 0, 0, w as u32, h as u32);
                let crop = run(&plan, 1, 1, 3, 2);
                assert_eq!(crop, [&expected[6..9], &expected[11..14]].concat());
                for row in 0..h {
                    assert_eq!(run(&plan, 0, row, w as u32, 1), expected[row * w..][..w]);
                }
                for shift in [0, 16 - bits] {
                    // Both unused low and high bits are padding, not signal.
                    let mask = (!(((1u32 << bits) - 1) << shift)) as u16;
                    let encode = |input: &[u16], width: usize| {
                        input
                            .chunks_exact(width)
                            .flat_map(|r| r.iter().map(|v| (v << shift) | mask).chain([0xdead]))
                            .collect::<Vec<_>>()
                    };
                    let (ys, us, vs) = (encode(&y, w), encode(&u, cw), encode(&v, cw));
                    let enc = SampleEncoding::new(ChannelType::U16, bits, shift).unwrap();
                    let planes = [
                        CodePlane::from_u16(&ys, w, h, (w + 1) * 2, enc).unwrap(),
                        CodePlane::from_u16(&us, cw, ch, (cw + 1) * 2, enc).unwrap(),
                        CodePlane::from_u16(&vs, cw, ch, (cw + 1) * 2, enc).unwrap(),
                    ];
                    let plan = YcbcrToRgbF32::new(planes, sub, Site::Center, color).unwrap();
                    assert_eq!(run(&plan, 0, 0, w as u32, h as u32), expected);
                }
            }
        }
    }
}

#[test]
fn limited_neutral_codes_preserve_excursions_and_endpoints() {
    let y = [0u8, 16, 235, 255];
    let uv = [128u8; 4];
    let enc = SampleEncoding::new(ChannelType::U8, 8, 0).unwrap();
    let planes = [
        CodePlane::from_u8(&y, 4, 1, 4, enc).unwrap(),
        CodePlane::from_u8(&uv, 4, 1, 4, enc).unwrap(),
        CodePlane::from_u8(&uv, 4, 1, 4, enc).unwrap(),
    ];
    let plan = YcbcrToRgbF32::new(
        planes,
        Sub::Yuv444,
        Site::Unknown,
        Cicp::new(1, 13, 1, false),
    )
    .unwrap();
    let out = run(&plan, 0, 0, 4, 1);
    assert!(out[0].iter().all(|v| *v < 0.0));
    assert_eq!(out[1], [0.0; 3]);
    assert!(out[2].iter().all(|v| (*v - 1.0).abs() <= f32::EPSILON));
    assert!(out[3].iter().all(|v| *v > 1.0));
}

#[test]
fn bad_interpretation_and_output_fail_before_writes() {
    let samples = [128u8; 4];
    let enc = SampleEncoding::new(ChannelType::U8, 8, 0).unwrap();
    let p = CodePlane::from_u8(&samples, 2, 2, 2, enc).unwrap();
    let c = CodePlane::from_u8(&samples, 1, 1, 1, enc).unwrap();
    let color = Cicp::new(1, 13, 1, true);
    for site in [Site::Unknown, Site::Left, Site::TopLeft] {
        assert_eq!(
            *YcbcrToRgbF32::new([p, c, c], Sub::Yuv420, site, color)
                .err()
                .unwrap()
                .error(),
            YcbcrError::UnsupportedChroma {
                subsampling: Sub::Yuv420,
                siting: site
            }
        );
    }
    assert_eq!(
        *YcbcrToRgbF32::new([p, p, c], Sub::Yuv420, Site::Center, color)
            .err()
            .unwrap()
            .error(),
        YcbcrError::SourceGeometry {
            plane: 1,
            actual: (2, 2),
            expected: Some((1, 1))
        }
    );
    assert_eq!(
        *YcbcrToRgbF32::new(
            [p, c, c],
            Sub::Yuv420,
            Site::Center,
            Cicp::new(9, 16, 9, true)
        )
        .err()
        .unwrap()
        .error(),
        YcbcrError::UnsupportedColor {
            actual: Cicp::new(9, 16, 9, true)
        }
    );
    let bad_encoding = SampleEncoding::new(ChannelType::U16, 16, 0).unwrap();
    let words = [0u16; 4];
    let wide = CodePlane::from_u16(&words, 2, 2, 4, bad_encoding).unwrap();
    assert_eq!(
        *YcbcrToRgbF32::new([wide, wide, wide], Sub::Yuv444, Site::Unknown, color)
            .err()
            .unwrap()
            .error(),
        YcbcrError::UnsupportedEncoding {
            plane: 0,
            actual: bad_encoding
        }
    );
    let other_encoding = SampleEncoding::new(ChannelType::U16, 8, 0).unwrap();
    let other = CodePlane::from_u16(&words, 1, 1, 2, other_encoding).unwrap();
    assert_eq!(
        *YcbcrToRgbF32::new([p, c, other], Sub::Yuv420, Site::Center, color)
            .err()
            .unwrap()
            .error(),
        YcbcrError::UnsupportedEncoding {
            plane: 2,
            actual: other_encoding
        }
    );
    let empty = CodePlane::from_u8(&[], 0, 0, 0, enc).unwrap();
    assert_eq!(
        *YcbcrToRgbF32::new([empty, empty, empty], Sub::Yuv444, Site::Unknown, color)
            .err()
            .unwrap()
            .error(),
        YcbcrError::SourceGeometry {
            plane: 0,
            actual: (0, 0),
            expected: None
        }
    );
    let plan = YcbcrToRgbF32::new([p, c, c], Sub::Yuv420, Site::Center, color).unwrap();
    for (x, y, desc, error) in [
        (
            usize::MAX,
            0,
            descriptor(),
            YcbcrError::OutputGeometry {
                origin: (usize::MAX, 0),
                extent: (2, 2),
                source: (2, 2),
            },
        ),
        (
            0,
            2,
            descriptor(),
            YcbcrError::OutputGeometry {
                origin: (0, 2),
                extent: (2, 2),
                source: (2, 2),
            },
        ),
        (
            0,
            0,
            PixelDescriptor::RGBF32_LINEAR,
            YcbcrError::OutputDescriptor {
                actual: PixelDescriptor::RGBF32_LINEAR,
            },
        ),
    ] {
        let mut output = [[3.25f32; 3]; 4];
        let dst =
            PixelSliceMut::new(bytemuck::cast_slice_mut(&mut output), 2, 2, 24, desc).unwrap();
        assert_eq!(*plan.convert_rows(x, y, dst).err().unwrap().error(), error);
        assert_eq!(output, [[3.25; 3]; 4]);
    }
}
