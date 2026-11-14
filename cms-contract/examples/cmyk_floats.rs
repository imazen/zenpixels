//! Existing JXL CMYK regression samples through the pinned Rust LCMS adapter.
use cms_contract::LcmsPort;
use zenpixels::{Cicp, ColorProfileSource, PixelFormat};
use zenpixels_convert::{cms::PluggableCms, policy::ConvertOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let path = args.next().ok_or("expected CMYK ICC profile")?;
    let oracle = args.next().map(std::fs::read_to_string).transpose()?;
    if args.next().is_some() {
        return Err("expected profile and optional C oracle TSV".into());
    }
    let expected: Option<Vec<[f32; 3]>> = oracle.map(|text| {
        text.lines()
            .map(|line| {
                let fields: Vec<_> = line.split('\t').collect();
                assert_eq!(fields.len(), 4);
                std::array::from_fn(|c| fields[c + 1].parse().expect("C float output"))
            })
            .collect()
    });
    if let Some(expected) = &expected {
        assert_eq!(expected.len(), 4);
    }
    let icc = std::fs::read(path)?;
    let mut transform = LcmsPort
        .build_source_transform(
            ColorProfileSource::Icc(&icc),
            ColorProfileSource::Cicp(Cicp::new(1, 13, 0, true)),
            PixelFormat::CMYKF32_INVERTED,
            PixelFormat::RgbF32,
            &ConvertOptions::permissive().with_clip_out_of_gamut(false),
        )
        .ok_or("backend declined")?
        .map_err(|e| format!("{e:?}"))?;
    for (index, sample) in [
        [0.2863f32, 1.0, 0.6706, 1.0],
        [0.2347, 0.9601, 0.5918, 0.9813],
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ]
    .into_iter()
    .enumerate()
    {
        let mut rgb = [0.0f32; 3];
        transform
            .try_transform_row(
                bytemuck::cast_slice(&sample),
                bytemuck::cast_slice_mut(&mut rgb),
                1,
            )
            .map_err(|e| format!("{e:?}"))?;
        println!("rust-lcms\t{sample:?}\t{rgb:?}");
        if let Some(expected) = &expected {
            assert_eq!(
                rgb.map(f32::to_bits),
                expected[index].map(f32::to_bits),
                "C/Rust float mismatch at sample {sample:?}"
            );
        }
    }
    Ok(())
}
