use zenpixels::{ImageSpec, PixelBuf};
fn main() {
    let b = PixelBuf::new(std::hint::black_box(4), 1, ImageSpec::rgb8_srgb()).unwrap();
    println!("{}", b.as_view().row(0)[0]);
}
