//! Allocation gate: native code conversion, the exact-depth scan, encoder-side
//! narrowing and evidence carriage perform zero heap allocations on success.
//!
//! Links the `allocation-counter` global allocator, so this file is its own
//! test binary. Error paths are excluded: a `whereat` trace allocates.

use zenpixels::{
    ChannelType, PixelBuffer, PixelDescriptor, PixelSlice, PixelSliceMut, SignalRange,
    sample::{CodePlane, SampleEncoding},
};
use zenpixels_convert::codes::{CodeMap, CodeRole, measure_exact_depth, to_canonical};

#[test]
fn conversion_scan_and_evidence_allocate_nothing() {
    let (w, h) = (301usize, 7usize); // wider than one 256-code batch
    let twelve = SampleEncoding::new(ChannelType::U16, 12, 0).unwrap();
    let canonical = SampleEncoding::new(ChannelType::U16, 16, 0).unwrap();
    let ten = SampleEncoding::new(ChannelType::U16, 10, 0).unwrap();
    let src: Vec<u16> = (0..w * h * 3)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u16 & 0x0fff)
        .collect();
    let mut dst = vec![0u16; w * h * 3];
    let mut codes = vec![0u16; w * h * 3];
    let narrow = CodeMap::new(canonical, ten, CodeRole::Intensity, SignalRange::Full).unwrap();
    let mut buffer = PixelBuffer::new(w as u32, h as u32, PixelDescriptor::RGB16);

    let info = allocation_counter::measure(|| {
        let planes = [0usize, 1, 2]
            .map(|c| CodePlane::from_u16_interleaved(&src[c..], w, h, w * 6, 3, twelve).unwrap());
        let out = PixelSliceMut::new(
            bytemuck::cast_slice_mut(&mut dst),
            w as u32,
            h as u32,
            w * 6,
            PixelDescriptor::RGB16,
        )
        .unwrap();
        let (_, exact) = to_canonical(&planes, SignalRange::Full, 0, 0, out).unwrap();
        let slice = PixelSlice::new(
            bytemuck::cast_slice(&dst),
            w as u32,
            h as u32,
            w * 6,
            PixelDescriptor::RGB16,
        )
        .unwrap()
        .with_exact_depth(exact)
        .unwrap();
        let measured = measure_exact_depth(&slice).unwrap();
        assert_eq!(measured, exact);
        narrow.map_words(&dst, &mut codes).unwrap();
        // Carriage on an owned buffer: state, read, clear by mutable access.
        let (_, exact) =
            to_canonical(&planes, SignalRange::Full, 0, 0, buffer.as_slice_mut()).unwrap();
        buffer.set_exact_depth(Some(exact)).unwrap();
        assert_eq!(
            buffer.as_slice().crop_view(1, 1, 5, 3).exact_depth(),
            Some(exact)
        );
        let _ = buffer.as_slice_mut();
        assert_eq!(buffer.exact_depth(), None);
        std::hint::black_box((measured, &codes));
    });
    assert_eq!(info.count_total, 0, "{info:?}");
}
