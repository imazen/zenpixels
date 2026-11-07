use std::hint::black_box as bb;
use zenpixels::PixelDescriptor as D;
use zenpixels_convert::RowConverter;
fn main() {
    let mut c = RowConverter::new(D::RGBA8_SRGB, D::RGB8_SRGB).unwrap();
    let src = [1u8, 2, 3, 255, 4, 5, 6, 255];
    let mut dst = [0u8; 6];
    c.convert_row(bb(&src), &mut dst, 2);
    println!("{dst:?}");
}
