use zenpixels::{PixelBuffer, PixelDescriptor as D};
fn main() {
    let b = PixelBuffer::new(std::hint::black_box(4), 1, D::RGB8_SRGB);
    println!("{}", b.as_slice().row(0)[0]);
}
