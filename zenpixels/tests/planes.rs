use std::sync::Arc;
use zenpixels::ChannelType;
use zenpixels::plane::*;
use zenpixels::sample::{ChromaSiting, ChromaSubsampling, SampleEncoding};

fn codes(storage: ChannelType, bits: u8, shift: u8) -> PlaneEncoding {
    PlaneEncoding::Code(SampleEncoding::new(storage, bits, shift).unwrap())
}

#[test]
fn construction_checks_geometry_and_allows_a_short_final_row() {
    // 3 rows of 4 samples, stride 6, no padding after the last row: 6+6+4.
    let samples = [0u8; 16];
    assert!(PlaneRef::new(&samples[..], 4, 3, 6).is_ok());
    assert_eq!(
        PlaneRef::new(&samples[..15], 4, 3, 6).unwrap_err(),
        PlaneError::ShortBuffer
    );
    assert_eq!(
        PlaneRef::new(&samples[..], 4, 3, 3).unwrap_err(),
        PlaneError::ShortStride
    );
    assert_eq!(
        PlaneRef::new(&samples[..], 4, usize::MAX, usize::MAX / 2).unwrap_err(),
        PlaneError::GeometryOverflow
    );
    // Empty planes need no storage, whatever the stride.
    let empty: [u16; 0] = [];
    let p = PlaneRef::new(&empty[..], 0, 5, 100).unwrap();
    assert_eq!(p.rows().count(), 5);
    assert!(p.rows().all(|r| r.is_empty()));
    assert_eq!(p.row(4), Some(&[][..]));
    assert_eq!(p.row(5), None);
    assert!(p.strided_samples().is_empty());
    assert!(PlaneRef::new(&empty[..], 7, 0, 7).is_ok());
}

#[test]
fn rows_exclude_padding_in_both_directions() {
    // Rows of 3 samples, stride 5, final row unpadded: 5 + 5 + 3 = 13 samples.
    let samples: Vec<u16> = (0..13).collect();
    let p = PlaneRef::new(&samples[..], 3, 3, 5).unwrap();
    let want: [&[u16]; 3] = [&[0, 1, 2], &[5, 6, 7], &[10, 11, 12]];
    assert_eq!(p.rows().collect::<Vec<_>>(), want);
    assert_eq!(
        p.rows().rev().collect::<Vec<_>>(),
        [want[2], want[1], want[0]]
    );
    assert_eq!(p.rows().len(), 3);
    assert_eq!(p.row(1), Some(want[1]));
    assert_eq!(p.strided_samples(), &samples[..]);
    assert_eq!((p.width(), p.height(), p.stride()), (3, 3, 5));
}

#[test]
fn sub_views_read_and_write_through() {
    let mut samples: Vec<f32> = (0..20).map(|v| v as f32).collect();
    let mut p = PlaneMut::new(&mut samples[..], 4, 5, 4).unwrap();
    let s = p.sub(1, 2, 2, 2).unwrap();
    assert_eq!(
        s.rows().collect::<Vec<_>>(),
        [&[9.0, 10.0][..], &[13.0, 14.0]]
    );
    assert_eq!(s.strided_samples(), &[9.0, 10.0, 11.0, 12.0, 13.0, 14.0]);
    assert_eq!(p.sub(3, 0, 2, 1).unwrap_err(), PlaneError::OutsidePlane);
    assert_eq!(p.sub(0, 5, 1, 1).unwrap_err(), PlaneError::OutsidePlane);
    assert_eq!(
        p.sub(usize::MAX, 0, 2, 1).unwrap_err(),
        PlaneError::OutsidePlane
    );
    assert_eq!(p.sub(4, 5, 0, 0).unwrap().rows().count(), 0);
    for row in p.sub_mut(1, 2, 2, 2).unwrap().rows_mut() {
        row.fill(-1.0);
    }
    assert_eq!(
        &samples[8..16],
        &[8.0, -1.0, -1.0, 11.0, 12.0, -1.0, -1.0, 15.0]
    );
}

#[test]
fn rows_mut_skip_padding_and_meet_in_the_middle() {
    let mut samples = [0u8; 13];
    {
        let mut p = PlaneMut::new(&mut samples[..], 3, 3, 5).unwrap();
        let mut it = p.rows_mut();
        assert_eq!(it.len(), 3);
        it.next().unwrap().fill(1);
        it.next_back().unwrap().fill(3);
        it.next().unwrap().fill(2);
        assert!(it.next().is_none() && it.next_back().is_none());
    }
    assert_eq!(samples, [1, 1, 1, 0, 0, 2, 2, 2, 0, 0, 3, 3, 3]);

    let mut none: [u8; 0] = [];
    let mut empty = PlaneMut::new(&mut none[..], 0, 4, 9).unwrap();
    assert_eq!(empty.rows_mut().count(), 4);
    assert_eq!(empty.rows_mut().rev().count(), 4);
    assert_eq!(empty.row_mut(3), Some(&mut [][..]));
}

#[test]
fn split_at_row_gives_disjoint_strips() {
    let mut samples = [0u16; 13];
    let p = PlaneMut::new(&mut samples[..], 3, 3, 5).unwrap();
    let (mut top, mut bottom) = p.split_at_row(1).unwrap();
    assert_eq!((top.height(), bottom.height()), (1, 2));
    top.row_mut(0).unwrap().fill(7);
    for row in bottom.rows_mut() {
        row.fill(9);
    }
    assert_eq!(samples, [7, 7, 7, 0, 0, 9, 9, 9, 0, 0, 9, 9, 9]);

    for y in 0..=3 {
        let p = PlaneMut::new(&mut samples[..], 3, 3, 5).unwrap();
        let (top, bottom) = p.split_at_row(y).unwrap();
        assert_eq!((top.height(), bottom.height()), (y, 3 - y));
        assert_eq!(top.rows().chain(bottom.rows()).count(), 3);
    }
    let p = PlaneMut::new(&mut samples[..], 3, 3, 5).unwrap();
    assert_eq!(p.split_at_row(4).unwrap_err(), PlaneError::OutsidePlane);
}

fn assert_aligned<T: PlaneSample>(width: usize, height: usize) {
    let mut p = PlaneBuf::<T>::alloc(width, height).unwrap();
    let size = std::mem::size_of::<T>();
    let stride_bytes = p.stride() * size;
    assert_eq!(stride_bytes % 64, 0, "{width}");
    assert!(stride_bytes >= width * size);
    assert!(stride_bytes < width * size + 128);
    if width > 0 {
        assert_ne!(stride_bytes % 4096, 0, "{width}: stride on a 4096 multiple");
    }
    assert!(p.rows().all(|r| r.iter().all(|v| *v == T::default())));
    for row in p.rows_mut() {
        assert_eq!(row.len(), width);
        if width > 0 {
            assert_eq!(row.as_ptr() as usize % 64, 0, "{width}");
        }
    }
}

#[test]
fn alloc_starts_rows_on_64_bytes_and_avoids_4096_strides() {
    for width in (0..=130).chain([1024, 2048, 4096, 4097]) {
        assert_aligned::<u8>(width, 3);
        assert_aligned::<u16>(width, 3);
        assert_aligned::<f32>(width, 3);
    }
    // 4096-byte rows get 64 bytes more; others are only rounded.
    assert_eq!(PlaneBuf::<u8>::alloc(4096, 1).unwrap().stride(), 4160);
    assert_eq!(PlaneBuf::<u16>::alloc(2048, 1).unwrap().stride(), 2080);
    assert_eq!(PlaneBuf::<f32>::alloc(1024, 1).unwrap().stride(), 1040);
    assert_eq!(PlaneBuf::<u8>::alloc(2048, 1).unwrap().stride(), 2048);
    assert_eq!(PlaneBuf::<u8>::alloc(100, 1).unwrap().stride(), 128);
    assert_eq!(PlaneBuf::<u8>::alloc(0, 0).unwrap().rows().count(), 0);
    // More than isize::MAX bytes without overflowing usize, on any sample size.
    let half = usize::BITS / 2;
    assert_eq!(
        PlaneBuf::<f32>::alloc(usize::MAX / 2, 1).unwrap_err(),
        PlaneError::GeometryOverflow
    );
    assert_eq!(
        PlaneBuf::<u8>::alloc(1 << half, 1 << (half - 1)).unwrap_err(),
        PlaneError::AllocationFailed
    );
}

/// Accounted or shared storage outside this crate implements `PlaneStorage`.
struct Shared(std::sync::Arc<[u16]>);
impl PlaneStorage for Shared {
    type Sample = u16;
    fn samples(&self) -> &[u16] {
        &self.0
    }
}

#[test]
fn any_owner_is_a_plane() {
    let v: Vec<u16> = (0..8).collect();
    let shared = Shared(Arc::from(v.clone()));
    let a = PlaneBuf::new(v, 2, 2, 4).unwrap();
    let c = Plane::new(shared, 2, 2, 4).unwrap();
    for p in [a.view(), c.view()] {
        assert_eq!(p.rows().collect::<Vec<_>>(), [&[0, 1][..], &[4, 5]]);
    }
    let any = AnyPlane::U16(c.view());
    assert_eq!(
        (any.width(), any.height(), any.channel_type()),
        (2, 2, ChannelType::U16)
    );
}

#[test]
fn float_encoding_is_finite_nonzero_and_compares_bitwise() {
    assert_eq!(
        FloatEncoding::new(0.0, 1.0).unwrap_err(),
        FloatEncodingError::ZeroScale
    );
    assert_eq!(
        FloatEncoding::new(f32::NAN, 0.0).unwrap_err(),
        FloatEncodingError::NotFinite
    );
    assert_eq!(
        FloatEncoding::new(1.0, f32::INFINITY).unwrap_err(),
        FloatEncodingError::NotFinite
    );
    assert_eq!(
        FloatEncoding::new(1.0, -0.0).unwrap(),
        FloatEncoding::NOMINAL
    );
    let jpeg_y = FloatEncoding::new(255.0, -128.0).unwrap();
    assert_eq!((jpeg_y.scale(), jpeg_y.offset()), (255.0, -128.0));
    assert_ne!(jpeg_y, FloatEncoding::NOMINAL);
    let set: std::collections::HashSet<_> = [jpeg_y, FloatEncoding::new(255.0, -128.0).unwrap()]
        .into_iter()
        .collect();
    assert_eq!(set.len(), 1);
    assert_eq!(
        PlaneEncoding::Float(jpeg_y).channel_type(),
        ChannelType::F32
    );
}

#[test]
fn layouts_accept_planar_semi_planar_rgb_and_mixed_depth() {
    use ChromaSubsampling::*;
    use PlaneRole::*;
    let u8c = codes(ChannelType::U8, 8, 0);
    let p010 = codes(ChannelType::U16, 10, 6);
    let ok = |planes: Vec<PlaneSpec>, s| {
        PlanarLayout::new(planes, s, ChromaSiting::Unknown).unwrap();
    };
    let spec = PlaneSpec::new;
    ok(vec![spec(Luma, u8c), spec(Cb, u8c), spec(Cr, u8c)], Yuv420);
    ok(vec![spec(Luma, u8c), spec(Cr, u8c), spec(Cb, u8c)], Yuv422);
    ok(vec![spec(Luma, u8c), spec(CbCr, u8c)], Yuv420);
    ok(vec![spec(Luma, u8c), spec(CrCb, u8c)], Yuv420);
    ok(
        vec![spec(Luma, p010), spec(CbCr, p010), spec(Alpha, u8c)],
        Yuv420,
    );
    ok(vec![spec(Luma, u8c)], Mono);
    ok(vec![spec(Luma, p010), spec(Alpha, u8c)], Mono);
    let f = PlaneEncoding::Float(FloatEncoding::NOMINAL);
    ok(
        vec![spec(Green, f), spec(Blue, f), spec(Red, f), spec(Alpha, f)],
        Yuv444,
    );
    assert_eq!(Luma.components(), 1);
    assert_eq!(CbCr.components(), 2);
}

#[test]
fn layouts_refuse_inconsistent_planes() {
    use ChromaSubsampling::*;
    use PlaneRole::*;
    let e = codes(ChannelType::U8, 8, 0);
    let err = |roles: &[PlaneRole], s| {
        let planes = roles.iter().map(|&r| PlaneSpec::new(r, e)).collect();
        PlanarLayout::new(planes, s, ChromaSiting::Center).unwrap_err()
    };
    assert_eq!(err(&[], Mono), LayoutError::Empty);
    assert_eq!(err(&[Luma, Luma], Mono), LayoutError::DuplicateRole(Luma));
    assert_eq!(
        err(&[Luma, Cb, CbCr], Yuv420),
        LayoutError::DuplicateRole(CbCr)
    );
    assert_eq!(
        err(&[Luma, CbCr, CrCb], Yuv420),
        LayoutError::DuplicateRole(CrCb)
    );
    assert_eq!(
        err(&[Luma, Red, Green, Blue], Yuv444),
        LayoutError::MixedFamilies
    );
    assert_eq!(err(&[Cb, Cr], Yuv420), LayoutError::MissingRole(Luma));
    assert_eq!(err(&[Alpha], Mono), LayoutError::MissingRole(Luma));
    assert_eq!(err(&[Luma, Cb], Yuv420), LayoutError::MissingRole(Cr));
    assert_eq!(err(&[Red, Green], Yuv444), LayoutError::MissingRole(Blue));
    assert_eq!(err(&[Luma, Cb, Cr], Mono), LayoutError::Subsampling);
    assert_eq!(err(&[Luma], Yuv420), LayoutError::Subsampling);
    assert_eq!(err(&[Red, Green, Blue], Yuv420), LayoutError::Subsampling);
}

#[test]
fn layout_sizes_planes_and_checks_supplied_planes() {
    use PlaneRole::*;
    let p010 = codes(ChannelType::U16, 10, 6);
    let u8c = codes(ChannelType::U8, 8, 0);
    let layout = PlanarLayout::new(
        vec![
            PlaneSpec::new(Luma, p010),
            PlaneSpec::new(CbCr, p010),
            PlaneSpec::new(Alpha, u8c),
        ],
        ChromaSubsampling::Yuv420,
        ChromaSiting::Left,
    )
    .unwrap();
    assert_eq!(layout.planes().len(), 3);
    assert_eq!(layout.siting(), ChromaSiting::Left);
    assert_eq!(layout.plane_size(0, 5, 3), Some((5, 3)));
    assert_eq!(layout.plane_size(1, 5, 3), Some((3, 2)));
    assert_eq!(layout.plane_size(2, 5, 3), Some((5, 3)));
    assert_eq!(layout.plane_size(3, 5, 3), None);

    let y = PlaneBuf::<u16>::alloc(5, 3).unwrap();
    let uv = PlaneBuf::<u16>::alloc(6, 2).unwrap();
    let a = PlaneBuf::<u8>::alloc(5, 3).unwrap();
    let a16 = PlaneBuf::<u16>::alloc(5, 3).unwrap();
    let uv_narrow = PlaneBuf::<u16>::alloc(3, 2).unwrap();
    let good = [
        AnyPlane::U16(y.view()),
        AnyPlane::U16(uv.view()),
        AnyPlane::U8(a.view()),
    ];
    layout.check(&good, 5, 3).unwrap();
    assert_eq!(
        layout.check(&good[..2], 5, 3).unwrap_err(),
        LayoutError::PlaneCount {
            expected: 3,
            actual: 2
        }
    );
    assert_eq!(
        layout
            .check(&[good[0], good[1], AnyPlane::U16(a16.view())], 5, 3)
            .unwrap_err(),
        LayoutError::SampleType { index: 2 }
    );
    assert_eq!(
        layout
            .check(&[good[0], AnyPlane::U16(uv_narrow.view()), good[2]], 5, 3)
            .unwrap_err(),
        LayoutError::Dimensions {
            index: 1,
            expected: (6, 2),
            actual: (3, 2)
        }
    );
    assert_eq!(
        layout.check(&good, 6, 3).unwrap_err(),
        LayoutError::Dimensions {
            index: 0,
            expected: (6, 3),
            actual: (5, 3)
        }
    );
}
