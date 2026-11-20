use zenpixels::{
    BandLayout, BandedPixelBuffer, BufferError, DEFAULT_MAX_BAND_BYTES, Orientation, PixelBuffer,
    PixelDescriptor,
};

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
        if orientation.swaps_axes() {
            let mut probe = BandedPixelBuffer::try_new(7, 5, desc, 63).unwrap();
            assert_eq!(
                probe.orient_in_place(orientation),
                Err(BufferError::InvalidDimensions)
            );
            continue;
        }
        let mut in_place = BandedPixelBuffer::try_new(7, 5, desc, 63).unwrap();
        for y in 0..5 {
            in_place
                .row_mut(y)
                .unwrap()
                .copy_from_slice(input.row(y).unwrap());
        }
        in_place.orient_in_place(orientation).unwrap();
        for y in 0..5 {
            for x in 0..7 {
                let (dx, dy) = orientation.forward_map(x, y, 7, 5);
                let i = x as usize * 3;
                let j = dx as usize * 3;
                assert_eq!(
                    &input.row(y).unwrap()[i..i + 3],
                    &in_place.row(dy).unwrap()[j..j + 3]
                );
            }
        }
        #[cfg(feature = "hash")]
        {
            let bytes: Vec<u8> = (0..5)
                .flat_map(|y| in_place.row(y).unwrap().iter().copied())
                .collect();
            let contiguous = PixelBuffer::from_vec(bytes, 7, 5, desc).unwrap();
            for inclusion in [
                zenpixels::HashInclusions::SamplesOnly,
                zenpixels::HashInclusions::Pixels,
                zenpixels::HashInclusions::All,
            ] {
                assert_eq!(
                    in_place.hash_with(inclusion),
                    contiguous.hash_with(inclusion)
                );
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
    let mut frame =
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
    for y in 0..3 {
        frame.plane_row_mut(1, y).unwrap().fill(10 + y as u8);
    }
    assert_eq!(frame.plane_row(1, 2), Some(&[12, 12, 12][..]));
    assert!(frame.plane_row(1, 3).is_none());
    assert!(frame.plane_row(3, 0).is_none());
    #[cfg(feature = "hash")]
    {
        use zenpixels::plane::{AnyPlane, PlanarSlice, PlaneRef};
        let y = [0u8; 25];
        let cb = [10u8, 10, 10, 11, 11, 11, 12, 12, 12];
        let cr = [0u8; 9];
        let planes = [
            AnyPlane::U8(PlaneRef::new(&y[..], 5, 5, 5).unwrap()),
            AnyPlane::U8(PlaneRef::new(&cb[..], 3, 3, 3).unwrap()),
            AnyPlane::U8(PlaneRef::new(&cr[..], 3, 3, 3).unwrap()),
        ];
        let color = zenpixels::ColorContext::default();
        let contiguous = PlanarSlice::new(&planes, &layout, 5, 5, &color, None).unwrap();
        for mode in [
            zenpixels::HashInclusions::SamplesOnly,
            zenpixels::HashInclusions::Pixels,
            zenpixels::HashInclusions::All,
        ] {
            assert_eq!(frame.hash_with(mode), contiguous.hash_with(mode));
        }
    }
}

#[cfg(feature = "planar")]
#[test]
fn typed_plane_orientation_matches_contiguous_samples_and_hash() {
    use zenpixels::plane::{BandedPlane, PlaneRef};
    let geometry = BandedPlane::<u16>::band_layout(40_000, 30_001, i32::MAX as u64).unwrap();
    assert_eq!(geometry.pixel_count(), 1_200_040_000);
    assert!(geometry.band_count() > 1);

    let mut source = BandedPlane::<u16>::try_new(32, 5, 128).unwrap();
    for y in 0..5 {
        for (x, sample) in source.row_mut(y).unwrap().iter_mut().enumerate() {
            *sample = (y * 32 + x as u32) as u16 * 7;
        }
    }
    let expected: Vec<u16> = (0..5)
        .flat_map(|y| source.row(y).unwrap().iter().copied())
        .collect();
    #[cfg(feature = "hash")]
    {
        let contiguous = PlaneRef::new(expected.as_slice(), 32, 5, 32).unwrap();
        assert_eq!(
            source.hash_with(zenpixels::HashInclusions::All),
            contiguous.hash_with(zenpixels::HashInclusions::All)
        );
    }
    #[cfg(not(feature = "hash"))]
    let _ = (expected, core::mem::size_of::<PlaneRef<'_, u16>>());

    for orientation in Orientation::ALL {
        let (w, h) = orientation.output_dimensions(32, 5);
        let mut dst = BandedPlane::try_new(w, h, 64).unwrap();
        source.oriented_into(orientation, &mut dst).unwrap();
        for y in 0..5 {
            for x in 0..32 {
                let (dx, dy) = orientation.forward_map(x, y, 32, 5);
                assert_eq!(
                    dst.row(dy).unwrap()[dx as usize],
                    source.row(y).unwrap()[x as usize]
                );
            }
        }
        if !orientation.swaps_axes() {
            let mut in_place = BandedPlane::try_new(32, 5, 192).unwrap();
            for y in 0..5 {
                in_place
                    .row_mut(y)
                    .unwrap()
                    .copy_from_slice(source.row(y).unwrap());
            }
            in_place.orient_in_place(orientation).unwrap();
            for y in 0..5 {
                assert_eq!(in_place.row(y), dst.row(y));
            }
        }
    }
}

#[cfg(feature = "planar")]
#[test]
fn planar_orientation_carries_chroma_lattice_across_band_boundaries() {
    use zenpixels::plane::{
        BandedPlanarBuffer, PlanarBandError, PlanarLayout, PlaneEncoding, PlaneRole, PlaneSpec,
    };
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
    let new = |w, h, cap| {
        BandedPlanarBuffer::try_new(&layout, w, h, zenpixels::ColorContext::default(), None, cap)
            .unwrap()
    };
    let mut source = new(6, 8, 20);
    for index in 0..3 {
        let (_, h) = layout.plane_size(index, 6, 8).unwrap();
        for y in 0..h {
            for (x, sample) in source
                .plane_row_mut(index, y)
                .unwrap()
                .iter_mut()
                .enumerate()
            {
                *sample = (y * 19 + x as u32 * 3) as u8;
            }
        }
    }
    for orientation in Orientation::ALL {
        let (w, h) = orientation.output_dimensions(6, 8);
        let mut output = new(w, h, 48);
        source.oriented_into(orientation, &mut output).unwrap();
        for index in 0..3 {
            let (pw, ph) = layout.plane_size(index, 6, 8).unwrap();
            for y in 0..ph {
                for x in 0..pw {
                    let (dx, dy) = orientation.forward_map(x, y, pw, ph);
                    assert_eq!(
                        output.plane_row(index, dy).unwrap()[dx as usize],
                        source.plane_row(index, y).unwrap()[x as usize],
                    );
                }
            }
        }
        if !orientation.swaps_axes() {
            let mut own = new(6, 8, 36);
            source
                .oriented_into(Orientation::Identity, &mut own)
                .unwrap();
            own.orient_in_place(orientation).unwrap();
            for index in 0..3 {
                for y in 0..layout.plane_size(index, 6, 8).unwrap().1 {
                    assert_eq!(own.plane_row(index, y), output.plane_row(index, y));
                }
            }
        }
    }
    let mut odd = new(5, 5, 16);
    odd.plane_row_mut(0, 0).unwrap().fill(99);
    assert!(matches!(
        odd.orient_in_place(Orientation::FlipH),
        Err(PlanarBandError::OrientationNeedsResampling)
    ));
    assert_eq!(odd.plane_row(0, 0), Some(&[99; 5][..]));
}

#[test]
fn adopted_buffer_context_authority_aligned_stride_and_exact_depth_preservation() {
    use alloc::sync::Arc;
    use zenpixels::sample::ExactDepth;
    use zenpixels::{ColorContext, RowAlign, SignalRange};
    extern crate alloc;

    let desc = PixelDescriptor::RGB16_SRGB;
    let ctx = Arc::new(ColorContext::from_icc(&b"authoritative-icc"[..]));
    let mut contig = PixelBuffer::try_new(5, 4, desc)
        .unwrap()
        .with_color_context(ctx);
    let exact8 = ExactDepth::new(desc.channel_type(), SignalRange::Full, 8, 1).unwrap();
    contig.set_exact_depth(Some(exact8)).unwrap();

    // Zero-copy adoption strips inner band context so `set_color_context(None)` clears borrowed views!
    let mut banded = BandedPixelBuffer::from_contiguous(contig).unwrap();
    assert_eq!(banded.exact_depth(), Some(exact8));
    assert_eq!(
        banded
            .bands()
            .next()
            .unwrap()
            .color_context()
            .and_then(|c| c.icc()),
        Some(&b"authoritative-icc"[..])
    );
    banded.set_color_context(None);
    assert!(banded.bands().next().unwrap().color_context().is_none());

    // ExactDepth survives geometric permutations (`orient_in_place`) and round-trips through contiguous!
    banded.orient_in_place(Orientation::Rotate180).unwrap();
    assert_eq!(banded.exact_depth(), Some(exact8));

    let dst_aligned =
        BandedPixelBuffer::try_new_with_align(4, 5, desc, 256, RowAlign::Cache64).unwrap();
    assert!(
        dst_aligned
            .bands()
            .next()
            .unwrap()
            .stride()
            .is_multiple_of(64)
    );

    #[cfg(feature = "hash")]
    {
        let rebuilt = banded.try_into_contiguous().unwrap();
        assert_eq!(rebuilt.exact_depth(), Some(exact8));
    }

    // Mutating a row clears ExactDepth evidence.
    let mut mutable = BandedPixelBuffer::try_new(4, 5, desc, 256).unwrap();
    mutable.set_exact_depth(Some(exact8)).unwrap();
    let _ = mutable.row_mut(0).unwrap();
    assert_eq!(mutable.exact_depth(), None);
}
