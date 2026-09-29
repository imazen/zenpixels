//! Every accepted built-in format/depth route must be executable after setup.
use zenpixels_convert::{ConvertPlan, PixelFormat, RowConverter, TransferFunction};

#[test]
fn accepted_format_and_transfer_pairs_prepare_and_execute() {
    use PixelFormat::*;
    let formats = [
        Rgb8, Rgba8, Rgb16, Rgba16, RgbF32, RgbaF32, Gray8, Gray16, GrayF32, GrayA8, GrayA16,
        GrayAF32, Bgra8, Rgbx8, Bgrx8, OklabF32, OklabaF32, Cmyk8, RgbF16, RgbaF16, GrayF16,
        GrayAF16,
    ];
    for source in formats {
        for target in formats {
            for from_tf in [
                TransferFunction::Linear,
                TransferFunction::Srgb,
                TransferFunction::Bt709,
                TransferFunction::Gamma22,
            ] {
                for to_tf in [
                    TransferFunction::Linear,
                    TransferFunction::Srgb,
                    TransferFunction::Bt709,
                    TransferFunction::Gamma22,
                ] {
                    let from = source.descriptor().with_transfer(from_tf);
                    let to = target.descriptor().with_transfer(to_tf);
                    let Ok(plan) = ConvertPlan::new(from, to) else {
                        continue;
                    };
                    let result = std::panic::catch_unwind(|| {
                        let mut worker = RowConverter::from_plan(plan);
                        worker.prepare(17).unwrap();
                        // u32 backing ensures every sample depth is aligned. Zeros
                        // include alpha=0, which must also be a valid kernel input.
                        let src = [0u32; 17 * 4];
                        let mut dst = [0u32; 17 * 4];
                        worker
                            .try_convert_row(
                                bytemuck::cast_slice(&src),
                                bytemuck::cast_slice_mut(&mut dst),
                                17,
                            )
                            .unwrap();
                    });
                    assert!(
                        result.is_ok(),
                        "accepted but unexecutable: {from:?} -> {to:?}"
                    );
                }
            }
        }
    }
}
