//! Unclipped requests must not collapse negative or above-white color values.
use zenpixels_convert::{ConvertOptions, PixelDescriptor, RowConverter, TransferFunction};

#[test]
fn explicit_unclipped_sdr_transfers_preserve_extended_values() {
    let options = ConvertOptions::retain_channels().with_clip_out_of_gamut(false);
    for transfer in [
        TransferFunction::Srgb,
        TransferFunction::Bt709,
        TransferFunction::Gamma22,
    ] {
        let encoded = PixelDescriptor::RGBF32_LINEAR.with_transfer(transfer);
        let linear = PixelDescriptor::RGBF32_LINEAR;
        for (from, to) in [(encoded, linear), (linear, encoded)] {
            let mut converter = RowConverter::new_explicit(from, to, &options).unwrap();
            let source = [-0.25f32, 0.5, 1.5];
            let mut output = [0.0f32; 3];
            converter
                .try_convert_row(
                    bytemuck::cast_slice(&source),
                    bytemuck::cast_slice_mut(&mut output),
                    1,
                )
                .unwrap();
            assert!(
                output[0].is_finite()
                    && output[0] < 0.0
                    && output[2].is_finite()
                    && output[2] > 1.0,
                "unclipped {from:?} -> {to:?} collapsed extended samples: {output:?}"
            );
        }
    }
}

#[test]
fn extended_gamma_preserves_in_range_results_alpha_padding_and_row_partitions() {
    let width = 5usize;
    let rows = 3usize;
    let source_stride = 7 * 4;
    let target_stride = 8 * 4;
    let sentinel = f32::from_bits(0x7fc1_2345);
    let values = [
        -8.0f32, -1.5, -0.25, -0.0, 0.0, 0.2, 0.8, 1.0, 1.5, 2.0, 8.0,
    ];
    let mut source = vec![sentinel; source_stride * rows];
    for y in 0..rows {
        for x in 0..width {
            for c in 0..3 {
                source[y * source_stride + x * 4 + c] =
                    values[(y * width * 3 + x * 3 + c) % values.len()];
            }
            source[y * source_stride + x * 4 + 3] = if x % 2 == 0 { sentinel } else { -0.0 };
        }
    }
    let linear = PixelDescriptor::RGBAF32_LINEAR;
    let encoded = linear.with_transfer(TransferFunction::Gamma22);
    for (from, to, exponent) in [
        (encoded, linear, 563.0f64 / 256.0),
        (linear, encoded, 256.0 / 563.0),
    ] {
        let unclipped = ConvertOptions::retain_channels().with_clip_out_of_gamut(false);
        let mut all = RowConverter::new_explicit(from, to, &unclipped).unwrap();
        all.prepare(width as u32).unwrap();
        let mut output = vec![sentinel; target_stride * rows];
        all.convert_rows(
            bytemuck::cast_slice(&source),
            source_stride * 4,
            bytemuck::cast_slice_mut(&mut output),
            target_stride * 4,
            width as u32,
            rows as u32,
        )
        .unwrap();
        let mut split = vec![sentinel; target_stride * rows];
        let mut individual = RowConverter::new_explicit(from, to, &unclipped).unwrap();
        let mut default =
            RowConverter::new_explicit(from, to, &ConvertOptions::retain_channels()).unwrap();
        for y in 0..rows {
            let input = &source[y * source_stride..y * source_stride + width * 4];
            let destination = &mut split[y * target_stride..y * target_stride + width * 4];
            individual
                .try_convert_row(
                    bytemuck::cast_slice(input),
                    bytemuck::cast_slice_mut(destination),
                    width as u32,
                )
                .unwrap();
            let mut historical = vec![0f32; width * 4];
            default
                .try_convert_row(
                    bytemuck::cast_slice(input),
                    bytemuck::cast_slice_mut(&mut historical),
                    width as u32,
                )
                .unwrap();
            for x in 0..width {
                for c in 0..3 {
                    let s = input[x * 4 + c];
                    let actual = destination[x * 4 + c];
                    let expected = f64::from(s).abs().powf(exponent).copysign(f64::from(s));
                    assert!(
                        (f64::from(actual) - expected).abs() <= 2e-5 * expected.abs().max(1.0),
                        "{s} -> {actual}, expected {expected}"
                    );
                    assert_eq!(actual.is_sign_negative(), s.is_sign_negative());
                    if !s.is_sign_negative() && s <= 1.0 {
                        assert_eq!(actual.to_bits(), historical[x * 4 + c].to_bits());
                    }
                }
                assert_eq!(destination[x * 4 + 3].to_bits(), input[x * 4 + 3].to_bits());
            }
            assert!(
                output[y * target_stride + width * 4..(y + 1) * target_stride]
                    .iter()
                    .all(|v| v.to_bits() == sentinel.to_bits())
            );
        }
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&output),
            bytemuck::cast_slice::<_, u8>(&split)
        );
    }
}

// Retired estimation is checked by the downstream extended_gamma_estimate
// compile-failure probe; pixel and stride assertions above remain unchanged.
