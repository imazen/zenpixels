//! Fixed-work profiler driver. No timers; use perf/callgrind externally.
use std::hint::black_box;
use zenpixels::plane::{
    AnyPlane, PlanarLayout, PlanarSlice, PlaneEncoding, PlaneRef, PlaneRole, PlaneSpec,
};
use zenpixels::sample::{ChromaSiting, ChromaSubsampling, CodePlane, SampleEncoding};
use zenpixels::{ChannelType, ColorContext, HashInclusions, PixelDescriptor, PixelSlice};

fn main() {
    let mode = std::env::args()
        .nth(1)
        .expect("mode: codes | component | hash | packed");
    let iterations: usize = std::env::args()
        .nth(2)
        .expect("iterations")
        .parse()
        .unwrap();
    let (width, height) = (3840usize, 2160usize);
    let input: Vec<u16> = (0..width * height)
        .map(|i| (((i as u16) & 1023) << 6) | 63)
        .collect();
    let chroma = &input[..width * height / 2];
    let encoding = SampleEncoding::new(ChannelType::U16, 10, 6).unwrap();
    let planes = [
        AnyPlane::U16(PlaneRef::new(&input[..], width, height, width).unwrap()),
        AnyPlane::U16(PlaneRef::new(chroma, width, height / 2, width).unwrap()),
    ];
    let layout = PlanarLayout::new(
        vec![
            PlaneSpec::new(PlaneRole::Luma, PlaneEncoding::Code(encoding)),
            PlaneSpec::new(PlaneRole::CbCr, PlaneEncoding::Code(encoding)),
        ],
        ChromaSubsampling::Yuv420,
        ChromaSiting::Left,
    )
    .unwrap();
    let color = ColorContext::default();
    let view =
        PlanarSlice::new(&planes, &layout, width as u32, height as u32, &color, None).unwrap();
    let codes = if mode == "component" {
        CodePlane::component(planes[0], 2, 0, encoding).unwrap()
    } else {
        CodePlane::new(planes[0], encoding).unwrap()
    };
    let packed = PixelSlice::<()>::new(
        bytemuck::cast_slice(&input),
        width as u32,
        height as u32,
        width * 2,
        PixelDescriptor::GRAY16,
    )
    .unwrap();
    let mut checksum = 0u64;
    for _ in 0..iterations {
        match mode.as_str() {
            "codes" | "component" => {
                let mut buffer = [0u16; 128];
                for y in 0..codes.height() {
                    for x in (0..codes.width()).step_by(buffer.len()) {
                        let n = buffer.len().min(codes.width() - x);
                        black_box(codes).read_codes(x, y, &mut buffer[..n]).unwrap();
                        checksum = checksum.wrapping_add(u64::from(buffer[n - 1]));
                        black_box(&buffer[..n]);
                    }
                }
            }
            "hash" => {
                checksum = checksum.wrapping_add(black_box(&view).hash_with(HashInclusions::All))
            }
            "packed" => {
                checksum = checksum.wrapping_add(black_box(&packed).hash_with(HashInclusions::All))
            }
            _ => panic!("mode: codes | component | hash | packed"),
        }
    }
    println!("{mode} iterations={iterations} checksum={checksum:016x}");
}
