//! Validate native CMYK row layouts against a separately compiled C LCMS oracle.
#![forbid(unsafe_code)]
use std::{path::PathBuf, process::Command};
use zenpixels::{ColorProfileSource as Source, PixelFormat as F};
use zenpixels_convert::{ConvertOptions, Lcms, MoxCms, PluggableCms};

type Error = Box<dyn std::error::Error>;
fn decode(bytes: &[u8], size: usize) -> f32 {
    match size {
        1 => f32::from(bytes[0]) / 255.0,
        2 => f32::from(u16::from_ne_bytes(bytes.try_into().unwrap())) / 65535.0,
        _ => f32::from_ne_bytes(bytes.try_into().unwrap()),
    }
}
fn encode(value: f32, size: usize) -> Vec<u8> {
    match size {
        1 => vec![(f64::from(value.clamp(0.0, 1.0)) * 255.0).round() as u8],
        2 => ((f64::from(value.clamp(0.0, 1.0)) * 65535.0).round() as u16)
            .to_ne_bytes()
            .to_vec(),
        _ => value.to_ne_bytes().to_vec(),
    }
}
fn main() -> Result<(), Error> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 4 {
        return Err("expected SOURCE.icc DEST.icc C_ORACLE FRESH_OUTPUT".into());
    }
    let source = std::fs::read(&args[0])?;
    let dest = std::fs::read(&args[1])?;
    let output = PathBuf::from(&args[3]);
    std::fs::create_dir(&output)?;
    const WIDTH: usize = 257;
    const ROWS: usize = 2;
    let options = ConvertOptions::permissive().with_clip_out_of_gamut(false);
    for (name, backend) in [("lcms", &Lcms as &dyn PluggableCms), ("mox", &MoxCms)] {
        for (from, rgb, rgba, size, kind, inverted) in [
            (F::CMYK8, F::Rgb8, F::Rgba8, 1, "u8", false),
            (F::CMYK16, F::Rgb16, F::Rgba16, 2, "u16", false),
            (F::CMYKF32, F::RgbF32, F::RgbaF32, 4, "f32", false),
            (F::CMYKF32_INVERTED, F::RgbF32, F::RgbaF32, 4, "f32", true),
        ] {
            let stride = (WIDTH + 3) * 4 * size;
            let prefix = size;
            let mut input_words = vec![0xa5a5a5a5u32; (prefix + ROWS * stride + 7).div_ceil(4)];
            let input = bytemuck::cast_slice_mut::<u32, u8>(&mut input_words);
            let mut canonical = Vec::new();
            for y in 0..ROWS {
                for x in 0..WIDTH {
                    for c in 0..4 {
                        let code = if x < 16 {
                            if x & (1 << c) == 0 { 0 } else { 65535 }
                        } else {
                            ((x * 257 + y * 101 + c * 3253) % 65536) as u16
                        };
                        let ink = f32::from(code) / 65535.0;
                        let value = if inverted { 1.0 - ink } else { ink };
                        let bytes = encode(value, size);
                        let at = prefix + y * stride + (x * 4 + c) * size;
                        input[at..at + size].copy_from_slice(&bytes);
                        let actual = decode(&bytes, size);
                        let actual = if inverted { 1.0 - actual } else { actual };
                        canonical.extend_from_slice(&actual.to_ne_bytes());
                    }
                }
            }
            let input_before = input.to_vec();
            let input_file = output.join(format!("{name}-{from}.f32"));
            std::fs::write(&input_file, canonical)?;
            let reference = Command::new(&args[2])
                .args([
                    &args[0],
                    &args[1],
                    input_file.as_os_str(),
                    std::ffi::OsStr::new("-"),
                    std::ffi::OsStr::new("1:bpc"),
                    std::ffi::OsStr::new(kind),
                ])
                .output()?;
            if !reference.status.success() {
                return Err(format!(
                    "C oracle failed: {}",
                    String::from_utf8_lossy(&reference.stderr)
                )
                .into());
            }
            let reference: Vec<f32> = reference
                .stdout
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_ne_bytes(*v))
                .collect();
            assert_eq!(reference.len(), WIDTH * ROWS * 3);
            for to in [rgb, rgba] {
                let Some(built) = backend.build_source_transform(
                    Source::Icc(&source),
                    Source::Icc(&dest),
                    from,
                    to,
                    &options,
                ) else {
                    assert_eq!(
                        (name, from),
                        ("mox", F::CMYK16),
                        "unexpected capability decline"
                    );
                    println!("{name}|{from:?}|{to:?}|declined");
                    continue;
                };
                let mut transform = built.map_err(|e| format!("{e:?}"))?;
                transform
                    .prepare(WIDTH as u32)
                    .map_err(|e| format!("{e:?}"))?;
                let channels = if to == rgba { 4 } else { 3 };
                let row_bytes = WIDTH * channels * size;
                let out_stride = row_bytes + 3 * size;
                let mut output_words =
                    vec![0xa5a5a5a5u32; (prefix + ROWS * out_stride + 5).div_ceil(4)];
                let result = bytemuck::cast_slice_mut::<u32, u8>(&mut output_words);
                let mut worst = 0.0f32;
                let mut clipped_worst = 0.0f32;
                for y in 0..ROWS {
                    let start = prefix + y * out_stride;
                    transform
                        .try_transform_row(
                            &input[prefix + y * stride..prefix + (y + 1) * stride],
                            &mut result[start..start + out_stride],
                            WIDTH as u32,
                        )
                        .map_err(|e| format!("{e:?}"))?;
                    assert!(
                        result[start + row_bytes..start + out_stride]
                            .iter()
                            .all(|&v| v == 0xa5)
                    );
                    for x in 0..WIDTH {
                        for c in 0..3 {
                            let at = start + (x * channels + c) * size;
                            let actual = decode(&result[at..at + size], size);
                            let expected = reference[(y * WIDTH + x) * 3 + c];
                            assert!(actual.is_finite() && expected.is_finite());
                            worst = worst.max((actual - expected).abs());
                            clipped_worst = clipped_worst
                                .max((actual.clamp(0.0, 1.0) - expected.clamp(0.0, 1.0)).abs());
                            if name == "lcms" {
                                assert_eq!(
                                    actual.to_bits(),
                                    expected.to_bits(),
                                    "C/port mismatch: {from:?} {to:?} pixel {x},{y} channel {c}"
                                );
                            }
                        }
                        if channels == 4 {
                            let at = start + (x * channels + 3) * size;
                            assert_eq!(
                                decode(&result[at..at + size], size).to_bits(),
                                1.0f32.to_bits()
                            );
                        }
                    }
                }
                assert_eq!(result[0], 0xa5);
                assert!(
                    result[prefix + ROWS * out_stride..]
                        .iter()
                        .all(|&v| v == 0xa5)
                );
                assert_eq!(&*input, input_before.as_slice());
                println!(
                    "{name}|{from:?}|{to:?}|ok|{worst:.9}|{clipped_worst:.9}|alpha_exact|padding_exact|input_unchanged"
                );
            }
        }
    }
    Ok(())
}
