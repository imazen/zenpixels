//! Which row pitches can each zenpixels type describe without copying?
use zenpixels::plane::PlaneRef;
use zenpixels::{PixelDescriptor, PixelSlice};
fn main() {
    const W: u32 = 37;
    const H: u32 = 4;
    let buf = vec![0u8; 8192];
    for (name, d, bpp) in [
        ("RGB8", PixelDescriptor::RGB8_SRGB, 3usize),
        ("RGBA8", PixelDescriptor::RGBA8, 4),
        ("RGB16", PixelDescriptor::RGB16_SRGB, 6),
    ] {
        let min = W as usize * bpp;
        for (label, stride) in [
            ("tight", min),
            ("4-byte rows", min.next_multiple_of(4)),
            ("16-byte rows", min.next_multiple_of(16)),
            ("64-byte rows", min.next_multiple_of(64)),
        ] {
            let aligned = buf.as_ptr().align_offset(2);
            let r = PixelSlice::new(&buf[aligned..], W, H, stride, d);
            println!(
                "{name:<6} {label:<13} stride={stride:<4} PixelSlice: {}",
                match r {
                    Ok(_) => "ok".to_string(),
                    Err(e) => format!("REFUSED {:?}", e.decompose().0),
                }
            );
        }
    }
    let p = PlaneRef::new(&buf[..], W as usize, H as usize, 64);
    println!("PlaneRef<u8> width 37 stride 64: ok={}", p.is_ok());
}
