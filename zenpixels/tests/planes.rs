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
    let mut p = AlignedPlane::<T>::alloc(width, height).unwrap();
    assert_eq!(p.strided_samples().as_ptr() as usize % 64, 0, "{width}");
    let cloned = p.clone();
    assert_eq!(
        cloned.strided_samples().as_ptr() as usize % 64,
        0,
        "{width}"
    );
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
    assert_eq!(AlignedPlane::<u8>::alloc(4096, 1).unwrap().stride(), 4160);
    assert_eq!(AlignedPlane::<u16>::alloc(2048, 1).unwrap().stride(), 2080);
    assert_eq!(AlignedPlane::<f32>::alloc(1024, 1).unwrap().stride(), 1040);
    assert_eq!(AlignedPlane::<u8>::alloc(2048, 1).unwrap().stride(), 2048);
    assert_eq!(AlignedPlane::<u8>::alloc(100, 1).unwrap().stride(), 128);
    assert_eq!(AlignedPlane::<u8>::alloc(0, 0).unwrap().rows().count(), 0);
    // More than isize::MAX bytes without overflowing usize, on any sample size.
    let half = usize::BITS / 2;
    assert_eq!(
        AlignedPlane::<f32>::alloc(usize::MAX / 2, 1).unwrap_err(),
        PlaneError::GeometryOverflow
    );
    assert_eq!(
        AlignedPlane::<u8>::alloc(1 << half, 1 << (half - 1)).unwrap_err(),
        PlaneError::AllocationFailed
    );
}

/// Accounted or shared storage outside this crate implements `SampleStorage`.
struct Shared(std::sync::Arc<[u16]>);
impl SampleStorage for Shared {
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
    assert_eq!(err(&[Alpha], Yuv420), LayoutError::Subsampling);
    assert_eq!(err(&[Alpha, Cb], Mono), LayoutError::MissingRole(Luma));
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

    let y = AlignedPlane::<u16>::alloc(5, 3).unwrap();
    let uv = AlignedPlane::<u16>::alloc(6, 2).unwrap();
    let a = AlignedPlane::<u8>::alloc(5, 3).unwrap();
    let a16 = AlignedPlane::<u16>::alloc(5, 3).unwrap();
    let uv_narrow = AlignedPlane::<u16>::alloc(3, 2).unwrap();
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

#[test]
fn strided_samples_keep_the_borrowed_lifetime() {
    let samples: Vec<u16> = (0..13).collect();
    let span: &[u16] = {
        let plane = PlaneRef::new(&samples[..], 3, 3, 5).unwrap();
        plane.into_strided_samples()
    };
    assert_eq!(span.as_ptr(), samples.as_ptr());
    assert_eq!(span.len(), 13);
    let mut out = [0u8; 13];
    let span = PlaneMut::new(&mut out[..], 3, 3, 5)
        .unwrap()
        .into_strided_samples_mut();
    span[12] = 9;
    assert_eq!(out[12], 9);
}

#[cfg(feature = "imgref")]
mod imgref_interop {
    use super::*;
    use imgref::{Img, ImgRef, ImgRefMut, ImgVec};

    #[test]
    fn borrowed_images_and_planes_convert_without_copying() {
        let buf: Vec<u16> = (0..20).collect();
        let img = Img::new_stride(&buf[..], 3, 4, 5);
        let plane = PlaneRef::try_from(img).unwrap();
        assert_eq!((plane.width(), plane.height(), plane.stride()), (3, 4, 5));
        assert_eq!(plane.row(1).unwrap().as_ptr(), buf[5..].as_ptr());
        // A sub-image ends at its last pixel, as a plane may.
        let sub = img.sub_image(1, 1, 2, 3);
        let plane = PlaneRef::try_from(sub).unwrap();
        assert_eq!(
            plane.rows().collect::<Vec<_>>(),
            [&[6, 7][..], &[11, 12], &[16, 17]]
        );
        let back: ImgRef<'_, u16> = plane.try_into().unwrap();
        assert_eq!(back.buf().as_ptr(), buf[6..].as_ptr());
        assert_eq!((back.width(), back.height(), back.stride()), (2, 3, 5));

        // An aligned allocation converts without moving.
        let mut aligned = AlignedPlane::<f32>::alloc(7, 2).unwrap();
        aligned.row_mut(1).unwrap()[6] = 2.5;
        let first = aligned.row(0).unwrap().as_ptr();
        let img: ImgRef<'_, f32> = aligned.view().try_into().unwrap();
        assert_eq!(img.buf().as_ptr(), first);
        assert_eq!(img[(6usize, 1usize)], 2.5);

        let short = Img::new_stride(&buf[..3], 4, 2, 4);
        assert_eq!(
            PlaneRef::try_from(short).unwrap_err(),
            PlaneError::ShortBuffer
        );
        let empty: [u8; 0] = [];
        let img: ImgRef<'_, u8> = PlaneRef::new(&empty[..], 0, 3, 0)
            .unwrap()
            .try_into()
            .unwrap();
        assert_eq!((img.width(), img.height(), img.stride()), (0, 3, 1));
    }

    #[test]
    fn mutable_images_and_planes_write_through() {
        let mut buf = vec![0u8; 12];
        {
            let img = ImgRefMut::new_stride(&mut buf[..], 2, 3, 4);
            let mut plane = PlaneMut::try_from(img).unwrap();
            for row in plane.rows_mut() {
                row.fill(1);
            }
            let mut img: ImgRefMut<'_, u8> = plane.try_into().unwrap();
            img[(1usize, 2usize)] = 7;
        }
        assert_eq!(buf, [1, 1, 0, 0, 1, 1, 0, 0, 1, 7, 0, 0]);
    }

    #[test]
    fn planes_starting_at_row_zero_become_owned_images() {
        let plane = PlaneBuf::new((0..12u16).collect(), 3, 3, 4).unwrap();
        let ptr = plane.row(0).unwrap().as_ptr();
        let img = ImgVec::try_from(plane).unwrap();
        assert_eq!((img.buf().as_ptr(), img.width(), img.stride()), (ptr, 3, 4));
    }

    #[test]
    fn owned_images_become_planes_without_copying() {
        let img = ImgVec::new((0..12u16).collect(), 4, 3);
        let ptr = img.buf().as_ptr();
        let plane = PlaneBuf::try_from(img).unwrap();
        assert_eq!(plane.row(0).unwrap().as_ptr(), ptr);
        assert_eq!(plane.row(2).unwrap(), &[8, 9, 10, 11]);
    }
}

#[test]
fn into_vec_hands_an_adopted_vec_back() {
    let v: Vec<u16> = (0..12).collect();
    let ptr = v.as_ptr();
    let adopted = PlaneBuf::new(v, 3, 3, 4).unwrap();
    let back = adopted.try_into_vec().unwrap();
    assert_eq!(back.as_ptr(), ptr);
}

#[test]
fn aligned_samples_are_aligned_zeroed_and_exact_length() {
    for len in [0usize, 1, 31, 32, 33, 64, 65, 4096, 1 << 20] {
        let a = AlignedSamples::<u16>::try_zeroed(len).unwrap();
        assert_eq!(a.len(), len);
        assert!(a.iter().all(|&s| s == 0));
        if len > 0 {
            assert_eq!(a.as_ptr() as usize % 64, 0);
            assert_eq!(a.clone().as_ptr() as usize % 64, 0);
        }
    }
    assert_eq!(
        AlignedSamples::<f32>::try_zeroed(usize::MAX).unwrap_err(),
        PlaneError::GeometryOverflow
    );
}

#[test]
fn byte_methods_alias_the_samples() {
    let mut v: Vec<u16> = vec![0x0102, 0x0304];
    let ptr = v.as_ptr();
    assert_eq!(zenpixels::ByteStorage::as_bytes(&v).as_ptr().cast(), ptr);
    assert_eq!(zenpixels::ByteStorage::as_bytes(&v).len(), 4);
    zenpixels::ByteStorageMut::as_bytes_mut(&mut v)[..2].copy_from_slice(&0xBEEFu16.to_ne_bytes());
    assert_eq!(v[0], 0xBEEF);

    // Trait objects keep them, so storage can stay erased down to the reference.
    let erased: Box<dyn SampleStorage<Sample = u16>> = Box::new(v);
    assert_eq!(zenpixels::ByteStorage::as_bytes(&erased).len(), 4);
    assert_eq!(
        zenpixels::ByteStorage::as_bytes(&erased).as_ptr().cast(),
        ptr
    );
}

/// Storage from somewhere else (accounted storage, a decoder pool).
struct Pool {
    samples: Vec<u8>,
}
impl SampleStorage for Pool {
    type Sample = u8;
    fn samples(&self) -> &[u8] {
        &self.samples
    }
}
impl SampleStorageMut for Pool {
    fn samples_mut(&mut self) -> &mut [u8] {
        &mut self.samples
    }
}

type DynPlane = Plane<Box<dyn SampleStorageMut<Sample = u8> + Send + Sync>>;

#[test]
fn any_owner_works_behind_a_boxed_trait_object() {
    // Not a use case for one picture (one allocator per image); this covers
    // the blanket impls and dyn compatibility with several owner types.
    let owners: Vec<Box<dyn SampleStorageMut<Sample = u8> + Send + Sync>> = vec![
        Box::new((0..12u8).collect::<Vec<u8>>()),
        Box::new(AlignedSamples::<u8>::try_zeroed(12).unwrap()),
        Box::new(vec![1u8; 12].into_boxed_slice()),
        Box::new(Pool {
            samples: vec![2; 12],
        }),
    ];
    let mut planes: Vec<DynPlane> = owners
        .into_iter()
        .map(|o| Plane::new(o, 4, 3, 4).unwrap())
        .collect();
    for p in &mut planes {
        p.row_mut(2).unwrap()[3] = 99;
    }
    // Borrowed views erase the owner with no boxing.
    let views: Vec<AnyPlane<'_>> = planes.iter().map(|p| AnyPlane::U8(p.view())).collect();
    assert_eq!(views.len(), 4);
    assert!(planes.iter().all(|p| p.row(2).unwrap()[3] == 99));
    assert_eq!(planes[0].row(1).unwrap(), &[4, 5, 6, 7]);
    assert_eq!(planes[3].row(0).unwrap(), &[2, 2, 2, 2]);

    // References to any owner, and to trait objects, are owners too.
    let v: Vec<u16> = (0..6).collect();
    let by_ref = Plane::new(&v, 3, 2, 3).unwrap();
    let dyn_ref: &dyn SampleStorage<Sample = u16> = &v;
    let by_dyn = Plane::new(dyn_ref, 3, 2, 3).unwrap();
    assert_eq!(by_ref.row(1), by_dyn.row(1));
}

#[test]
fn storage_traits_stay_dyn_compatible() {
    fn read(_: &dyn SampleStorage<Sample = f32>) {}
    fn write(_: &mut dyn SampleStorageMut<Sample = u16>) {}
    read(&vec![0.0f32; 2]);
    write(&mut vec![0u16; 2]);
    let mut boxed: Box<dyn SampleStorageMut<Sample = u16>> = Box::new(vec![0u16; 4]);
    write(&mut boxed);
    assert_eq!(boxed.samples().len(), 4);
}

#[test]
fn a_lone_alpha_plane_is_a_layout() {
    use zenpixels::sample::{ChromaSiting, ChromaSubsampling, SampleEncoding};
    let code = PlaneEncoding::Code(SampleEncoding::new(ChannelType::U8, 8, 0).unwrap());
    let layout = PlanarLayout::new(
        vec![PlaneSpec::new(PlaneRole::Alpha, code)],
        ChromaSubsampling::Mono,
        ChromaSiting::Unknown,
    )
    .unwrap();
    assert_eq!(layout.plane_size(0, 5, 3), Some((5, 3)));
    let plane = AlignedPlane::<u8>::alloc(5, 3).unwrap();
    assert_eq!(plane.row(2).unwrap().len(), 5);
}

#[test]
fn planar_alpha_role_requires_a_defined_interpretation() {
    use zenpixels::{AlphaMode, ColorContext};
    let samples = [255u8];
    let planes = [AnyPlane::U8(PlaneRef::new(&samples[..], 1, 1, 1).unwrap())];
    let color = ColorContext::default();
    for role in [PlaneRole::Alpha, PlaneRole::Luma] {
        let layout = PlanarLayout::new(
            vec![PlaneSpec::new(role, codes(ChannelType::U8, 8, 0))],
            ChromaSubsampling::Mono,
            ChromaSiting::Unknown,
        )
        .unwrap();
        for alpha in [
            None,
            Some(AlphaMode::Undefined),
            Some(AlphaMode::Straight),
            Some(AlphaMode::Premultiplied),
            Some(AlphaMode::Opaque),
        ] {
            let result = PlanarSlice::new(&planes, &layout, 1, 1, &color, alpha);
            let valid = if role == PlaneRole::Alpha {
                alpha.is_some() && alpha != Some(AlphaMode::Undefined)
            } else {
                alpha.is_none()
            };
            if valid {
                assert_eq!(result.unwrap().alpha_mode(), alpha);
            } else {
                assert_eq!(result.unwrap_err(), LayoutError::AlphaInterpretation);
            }
        }
    }
}

#[test]
fn planar_picture_reuses_native_code_components_and_preserves_interpretation() {
    use zenpixels::sample::CodePlane;
    use zenpixels::{AlphaMode, Cicp, ColorContext, ColorProfileSource};
    let layout = PlanarLayout::new(
        vec![
            PlaneSpec::new(PlaneRole::Luma, codes(ChannelType::U16, 10, 6)),
            PlaneSpec::new(PlaneRole::CrCb, codes(ChannelType::U16, 10, 6)),
            PlaneSpec::new(PlaneRole::Alpha, codes(ChannelType::U8, 1, 0)),
        ],
        ChromaSubsampling::Yuv420,
        ChromaSiting::Left,
    )
    .unwrap();
    let y: Vec<u16> = (0..19).map(|i| ((i + 64) << 6) | 63).collect();
    let uv: Vec<u16> = (0..15).map(|i| ((i + 300) << 6) | 31).collect();
    let alpha: Vec<u8> = (0..17).map(|i| 0xfe | (i & 1)).collect();
    let planes = [
        AnyPlane::U16(PlaneRef::new(&y[..], 5, 3, 7).unwrap()),
        AnyPlane::U16(PlaneRef::new(&uv[..], 6, 2, 9).unwrap()),
        AnyPlane::U8(PlaneRef::new(&alpha[..], 5, 3, 6).unwrap()),
    ];
    // HLG signaling is not display rendering, and no white/peak is inferred.
    let context = ColorContext::from_cicp(Cicp::new(9, 18, 9, false));
    let image =
        PlanarSlice::new(&planes, &layout, 5, 3, &context, Some(AlphaMode::Straight)).unwrap();
    assert!(core::ptr::eq(image.color_context(), &context));
    assert!(image.color_context().diffuse_white().is_none());
    assert!(image.color_context().hdr_peak().is_none());
    assert_eq!(image.layout().siting(), ChromaSiting::Left);
    assert_eq!(image.alpha_mode(), Some(AlphaMode::Straight));
    let cr = CodePlane::component(
        image.plane(1).unwrap(),
        2,
        0,
        SampleEncoding::new(ChannelType::U16, 10, 6).unwrap(),
    )
    .unwrap();
    let cb = CodePlane::component(image.plane(1).unwrap(), 2, 1, cr.encoding()).unwrap();
    assert_eq!(cr.code(2, 1), Some(313));
    assert_eq!(cb.code(2, 1), Some(314));
    let a = CodePlane::new(
        image.plane(2).unwrap(),
        SampleEncoding::new(ChannelType::U8, 1, 0).unwrap(),
    )
    .unwrap();
    assert_eq!(a.code(4, 2), Some(0));
    let AnyPlane::U16(luma) = image.plane(0).unwrap() else {
        panic!("U16")
    };
    assert_eq!(luma.strided_samples().as_ptr(), y.as_ptr());
    assert_eq!(luma.row(2).unwrap(), &y[14..19]);
    assert_eq!(y[5], (69 << 6) | 63); // Padding and low payload padding are untouched.

    // Raw unknown signaling and opaque ICC bytes remain transportable;
    // ColorContext holds a single active profile authority at a time.
    let raw_cicp = ColorContext::from_cicp(Cicp::new(200, 201, 202, false));
    let image =
        PlanarSlice::new(&planes, &layout, 5, 3, &raw_cicp, Some(AlphaMode::Straight)).unwrap();
    assert_eq!(
        image.color_context().cicp(),
        Some(Cicp::new(200, 201, 202, false))
    );

    let icc: Arc<[u8]> = Arc::from([1, 2, 3, 4]);
    let declared = raw_cicp.with_icc(icc.clone());
    let image =
        PlanarSlice::new(&planes, &layout, 5, 3, &declared, Some(AlphaMode::Straight)).unwrap();
    assert_eq!(image.color_context().cicp(), None);
    assert!(Arc::ptr_eq(image.color_context().icc_arc().unwrap(), &icc));
    assert_eq!(
        image.color_context().as_profile_source(),
        Some(ColorProfileSource::Icc(icc.as_ref()))
    );
}

#[test]
fn signed_and_wide_encoding_descriptions_do_not_relabel_plane_storage() {
    use zenpixels::ColorContext;
    let data = [0xffffu16];
    let planes = [AnyPlane::U16(PlaneRef::new(&data[..], 1, 1, 1).unwrap())];
    let context = ColorContext::default();
    for storage in [ChannelType::I16, ChannelType::U32, ChannelType::I64] {
        let encoding = SampleEncoding::new(storage, 10, 0).unwrap();
        let layout = PlanarLayout::new(
            vec![PlaneSpec::new(
                PlaneRole::Luma,
                PlaneEncoding::Code(encoding),
            )],
            ChromaSubsampling::Mono,
            ChromaSiting::Unknown,
        )
        .unwrap();
        assert_eq!(
            PlanarSlice::new(&planes, &layout, 1, 1, &context, None).unwrap_err(),
            LayoutError::SampleType { index: 0 }
        );
    }
    assert_eq!(data, [0xffff]);
}
