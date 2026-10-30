// Structural layout of zenpixels views at the review head. Not heap or RSS.
use std::mem::{align_of, size_of};
use std::sync::Arc;
macro_rules! row { ($t:ty) => { println!("{}\t{}\t{}", stringify!($t), size_of::<$t>(), align_of::<$t>()) } }
fn main() {
    println!("type\tsize\talign");
    row!(zenpixels::PixelSlice<'static>);
    row!(zenpixels::PixelSliceMut<'static>);
    row!(zenpixels::PixelBuffer);
    row!(zenpixels::sample::CodePlane<'static>);
    row!(zenpixels::PixelDescriptor);
    row!(Option<Arc<zenpixels::ColorContext>>);
    row!(zenpixels::sample::ExactDepth);
    row!(Option<zenpixels::sample::ExactDepth>);
}
