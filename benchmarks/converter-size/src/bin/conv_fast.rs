use std::hint::black_box as bb;
use zenpixels::ImageSpec;
use zenpixels_convert::{AlphaPolicy, ConvertPolicy, RowPlan, accel};
fn main() {
    accel::enable_all();
    let policy = ConvertPolicy::lossless().with_alpha(AlphaPolicy::DiscardUnchecked);
    let plan = RowPlan::build(
        bb(ImageSpec::rgba8_srgb()),
        bb(ImageSpec::rgb8_srgb()),
        &policy,
        None,
    )
    .unwrap();
    let src = [1u8, 2, 3, 255, 4, 5, 6, 255];
    let mut dst = [0u8; 6];
    plan.convert_row(bb(&src), &mut dst, 2, 0).unwrap();
    println!("{dst:?}");
}
