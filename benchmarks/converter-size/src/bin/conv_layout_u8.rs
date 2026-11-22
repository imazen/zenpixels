use std::hint::black_box as bb;
use zenpixels::PixelDescriptor as D;
use zenpixels_convert::{AlphaPolicy, ConversionSpec, ConvertOptions, ConverterProfile, caps};
fn main() {
    let opts = ConvertOptions::retain_channels().with_alpha_policy(AlphaPolicy::DiscardUnchecked);
    let mut c = ConverterProfile::<{ caps::LAYOUT_U8 }>::build(
        D::RGBA8_SRGB.into(),
        D::RGB8_SRGB.into(),
        &ConversionSpec::new(opts),
    )
    .unwrap()
    .prepare(2)
    .unwrap();
    let src = [1u8, 2, 3, 255, 4, 5, 6, 255];
    let mut dst = [0u8; 6];
    c.try_convert_row(bb(&src), &mut dst, 2).unwrap();
    println!("{dst:?}");
}
