use zenpixels::banded::{BandLayout, BandedPixelBuffer, DEFAULT_MAX_BAND_BYTES};
use zenpixels::{BufferError, Orientation, PixelBuffer, PixelDescriptor};

#[test]
fn gigapixel_streaming_geometry_and_contiguous_limits() {
    let desc = PixelDescriptor::RGB8_SRGB;
    let layout = BandLayout::new(40_000, 30_001, desc, DEFAULT_MAX_BAND_BYTES).unwrap();
    assert_eq!(layout.pixel_count(), 1_200_040_000);
    assert!(
        layout
            .bands()
            .all(|rows| u64::from(rows.end - rows.start) * 120_000 <= DEFAULT_MAX_BAND_BYTES)
    );
    assert_eq!(layout.bands().last().unwrap().end, 30_001);

    let layout = BandLayout::new(40_000, 30_001, desc, i32::MAX as u64).unwrap();
    assert!(layout.band_count() > 1);
    assert!(
        layout
            .bands()
            .all(|rows| u64::from(rows.end - rows.start) * 120_000 <= i32::MAX as u64)
    );
    assert_eq!(
        BandLayout::new(40_000, 30_001, desc, 119_999),
        Err(BufferError::InvalidDimensions)
    );
    let error = PixelBuffer::try_new(u32::MAX, u32::MAX, desc).unwrap_err();
    assert!(matches!(error.error(), BufferError::InvalidDimensions));
    #[cfg(target_pointer_width = "32")]
    assert!(matches!(
        PixelBuffer::try_new(40_000, 30_001, desc)
            .unwrap_err()
            .error(),
        BufferError::InvalidDimensions
    ));
}

#[test]
fn packed_rows_orientation_and_hash_independent_of_band_height() {
    let desc = PixelDescriptor::RGB8_SRGB;
    let mut input = BandedPixelBuffer::try_new(7, 5, desc, 42).unwrap();
    for y in 0..5 {
        for (i, v) in input.row_mut(y).unwrap().iter_mut().enumerate() {
            *v = (y * 21 + i as u32) as u8;
        }
    }
    assert_eq!(
        input.sub_rows(1..4).unwrap().map(|p| p.rows()).sum::<u32>(),
        3
    );
    for orientation in Orientation::ALL {
        let (w, h) = orientation.output_dimensions(7, 5);
        let mut output = BandedPixelBuffer::try_new(w, h, desc, u64::from(w) * 3).unwrap();
        input.oriented_into(orientation, &mut output).unwrap();
        for y in 0..5 {
            for x in 0..7 {
                let (dx, dy) = orientation.forward_map(x, y, 7, 5);
                let i = x as usize * 3;
                let j = dx as usize * 3;
                assert_eq!(
                    &input.row(y).unwrap()[i..i + 3],
                    &output.row(dy).unwrap()[j..j + 3]
                );
            }
        }
        #[cfg(feature = "hash")]
        {
            let bytes: Vec<u8> = (0..h)
                .flat_map(|y| output.row(y).unwrap().iter().copied())
                .collect();
            let contiguous = PixelBuffer::from_vec(bytes, w, h, desc).unwrap();
            for inclusion in [
                zenpixels::HashInclusions::SamplesOnly,
                zenpixels::HashInclusions::Pixels,
                zenpixels::HashInclusions::All,
            ] {
                assert_eq!(output.hash_with(inclusion), contiguous.hash_with(inclusion));
            }
        }
        if !orientation.swaps_axes() {
            let mut in_place = BandedPixelBuffer::try_new(7, 5, desc, 63).unwrap();
            for y in 0..5 {
                in_place
                    .row_mut(y)
                    .unwrap()
                    .copy_from_slice(input.row(y).unwrap());
            }
            in_place.orient_in_place(orientation).unwrap();
            for y in 0..5 {
                assert_eq!(in_place.row(y), output.row(y));
            }
        }
    }
}

#[cfg(feature = "planar")]
#[test]
fn planar_bands_keep_chroma_phase_at_boundaries() {
    use zenpixels::plane::{BandedPlanarBuffer, PlanarLayout, PlaneEncoding, PlaneRole, PlaneSpec};
    use zenpixels::sample::{ChromaSiting, ChromaSubsampling, SampleEncoding};
    let encoding =
        PlaneEncoding::Code(SampleEncoding::new(zenpixels::ChannelType::U8, 8, 0).unwrap());
    let layout = PlanarLayout::new(
        [PlaneRole::Luma, PlaneRole::Cb, PlaneRole::Cr]
            .map(|role| PlaneSpec::new(role, encoding))
            .into(),
        ChromaSubsampling::Yuv420,
        ChromaSiting::Center,
    )
    .unwrap();
    let huge = BandedPlanarBuffer::band_layout(&layout, 40_001, 40_001, i32::MAX as u64).unwrap();
    assert_eq!(huge.pixel_count(), 1_600_080_001);
    assert!(huge.band_count() > 1);
    assert_eq!(huge.band_rows() % 2, 0);
    let frame =
        BandedPlanarBuffer::try_new(&layout, 5, 5, zenpixels::ColorContext::default(), None, 16)
            .unwrap();
    assert_eq!(
        frame.bands().map(|band| band.height()).collect::<Vec<_>>(),
        [2, 2, 1]
    );
    assert_eq!(
        frame
            .bands()
            .map(|band| band.plane(1).unwrap().height())
            .sum::<usize>(),
        3
    );
}
