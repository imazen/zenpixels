//! One sample roster and owner-preserving packed/planar storage boundaries.
#[path = "common/fail_allocator.rs"]
mod fail_allocator;
use fail_allocator::{failing_above, lock};
use std::sync::Arc;
use zenpixels::plane::*;
use zenpixels::sample::{
    ChromaSiting, ChromaSubsampling, CodePlane, CodePlaneError, SampleEncoding,
};
use zenpixels::{ByteStorage, ChannelType, ColorContext, PixelStorage};

fn gray(encoding: PlaneEncoding) -> PlanarLayout {
    PlanarLayout::new(
        vec![PlaneSpec::new(PlaneRole::Luma, encoding)],
        ChromaSubsampling::Mono,
        ChromaSiting::Unknown,
    )
    .unwrap()
}
fn code(t: ChannelType, bits: u8) -> PlaneEncoding {
    PlaneEncoding::Code(SampleEncoding::new(t, bits, 0).unwrap())
}

#[test]
fn all_native_storage_types_have_typed_aligned_and_runtime_plane_views() {
    let _guard = lock();
    macro_rules! check {
        ($t:ty, $variant:ident, $values:expr, $encoding:expr) => {{
            let values: [$t; 3] = $values;
            let expected = bytemuck::cast_slice::<_, u8>(&values).to_vec();
            let l = gray($encoding);
            let mut storage = AlignedSamples::<$t>::try_zeroed(3).unwrap();
            storage.copy_from_slice(&values);
            let pointer = storage.as_ptr();
            assert_eq!((pointer as usize) % 64, 0);
            let planes = vec![Plane::new(storage, 3, 1, 3).unwrap()];
            let adopted_layout = l.clone();
            let mut picture = failing_above(0, || {
                PlanarBuffer::new(planes, adopted_layout, 3, 1, ColorContext::default(), None)
                    .unwrap()
            });
            failing_above(0, || {
                let view = picture.as_slice();
                let AnyPlane::$variant(p) = view.plane(0).unwrap() else {
                    panic!()
                };
                assert_eq!(p.strided_samples().as_ptr(), pointer);
                assert_eq!(bytemuck::cast_slice::<_, u8>(p.strided_samples()), expected);
                let mut writable = picture.as_slice_mut();
                let AnyPlaneMut::$variant(mut p) = writable.plane_mut(0).unwrap() else {
                    panic!()
                };
                p.row_mut(0).unwrap()[1] = values[2];
            });
            let plane = picture.into_planes().pop().unwrap();
            let storage = plane.into_storage();
            assert_eq!(storage.as_ptr(), pointer);
            assert_eq!(
                bytemuck::bytes_of(&storage[1]),
                bytemuck::bytes_of(&values[2])
            );
            // The default packed owner and default planar owner retain the
            // same native vector type and exact bits (including float payloads).
            let storage = PixelStorage::from(values.to_vec());
            assert_eq!(storage.as_bytes(), expected);
            let values = storage.try_into_vec::<$t>().unwrap();
            let pointer = values.as_ptr();
            let mut owned = AnyPlaneBuf::$variant(PlaneBuf::new(values, 3, 1, 3).unwrap());
            assert_eq!(owned.as_ref().as_strided_bytes(), expected);
            assert_eq!(owned.as_mut().as_ref().as_strided_bytes(), expected);
            let picture =
                PlanarBuffer::new(vec![owned], l, 3, 1, ColorContext::default(), None).unwrap();
            let AnyPlane::$variant(view) = picture.as_slice().plane(0).unwrap() else {
                panic!()
            };
            assert_eq!(view.strided_samples().as_ptr(), pointer);
            assert_eq!(
                bytemuck::cast_slice::<_, u8>(view.strided_samples()),
                expected
            );
        }};
    }
    check!(u8, U8, [0, 1, u8::MAX], code(ChannelType::U8, 8));
    check!(i8, I8, [i8::MIN, 0, i8::MAX], code(ChannelType::I8, 8));
    check!(u16, U16, [0, 1, u16::MAX], code(ChannelType::U16, 16));
    check!(
        i16,
        I16,
        [i16::MIN, 0, i16::MAX],
        code(ChannelType::I16, 16)
    );
    check!(u32, U32, [0, 1, u32::MAX], code(ChannelType::U32, 32));
    check!(
        i32,
        I32,
        [i32::MIN, 0, i32::MAX],
        code(ChannelType::I32, 32)
    );
    check!(u64, U64, [0, 1, u64::MAX], code(ChannelType::U64, 64));
    check!(
        i64,
        I64,
        [i64::MIN, 0, i64::MAX],
        code(ChannelType::I64, 64)
    );
    check!(
        f32,
        F32,
        [-0.0, f32::from_bits(0x7fc00021), 2.5],
        PlaneEncoding::Float(FloatEncoding::NOMINAL)
    );
    check!(
        f64,
        F64,
        [-0.0, f64::from_bits(0x7ff8000000000021), 1.0 + f64::EPSILON],
        PlaneEncoding::Float64(FloatEncoding::NOMINAL)
    );
}

#[derive(Debug)]
struct PoolSlot {
    samples: Vec<u16>,
    release: Arc<std::sync::atomic::AtomicUsize>,
}
impl SampleStorage for PoolSlot {
    type Sample = u16;
    fn samples(&self) -> &[u16] {
        &self.samples
    }
}
impl SampleStorageMut for PoolSlot {
    fn samples_mut(&mut self) -> &mut [u16] {
        &mut self.samples
    }
}
impl Drop for PoolSlot {
    fn drop(&mut self) {
        self.release
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

#[derive(Debug)]
enum MixedOwner {
    Pool(Plane<PoolSlot>),
    Aligned(AlignedPlane<f32>),
}
impl PlaneStorage for MixedOwner {
    fn as_plane(&self) -> AnyPlane<'_> {
        match self {
            Self::Pool(p) => p.as_plane(),
            Self::Aligned(p) => p.as_plane(),
        }
    }
}
impl PlaneStorageMut for MixedOwner {
    fn as_plane_mut(&mut self) -> AnyPlaneMut<'_> {
        match self {
            Self::Pool(p) => p.as_plane_mut(),
            Self::Aligned(p) => p.as_plane_mut(),
        }
    }
}

#[test]
fn heterogeneous_custom_owners_keep_pointers_padding_and_release_ownership() {
    let _guard = lock();
    let release = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let slot = PoolSlot {
        samples: vec![19; 7],
        release: release.clone(),
    };
    let pointer = slot.samples.as_ptr();
    let mut alpha = AlignedSamples::<f32>::try_zeroed(8).unwrap();
    alpha.fill(0.5);
    let alpha_pointer = alpha.as_ptr();
    let planes = vec![
        MixedOwner::Pool(Plane::new(slot, 3, 2, 4).unwrap()),
        MixedOwner::Aligned(Plane::new(alpha, 3, 2, 5).unwrap()),
    ];
    let layout = PlanarLayout::new(
        vec![
            PlaneSpec::new(PlaneRole::Luma, code(ChannelType::U16, 12)),
            PlaneSpec::new(
                PlaneRole::Alpha,
                PlaneEncoding::Float(FloatEncoding::NOMINAL),
            ),
        ],
        ChromaSubsampling::Mono,
        ChromaSiting::Unknown,
    )
    .unwrap();
    let mut image = failing_above(0, || {
        PlanarBuffer::new(
            planes,
            layout,
            3,
            2,
            ColorContext::default(),
            Some(zenpixels::AlphaMode::Straight),
        )
        .unwrap()
    });
    failing_above(0, || {
        let mut image = image.as_slice_mut();
        for mut p in image.planes_mut() {
            match p.as_mut() {
                AnyPlaneMut::U16(mut p) => p.row_mut(1).unwrap().fill(4095),
                AnyPlaneMut::F32(mut p) => p.row_mut(1).unwrap().fill(0.25),
                _ => panic!(),
            }
        }
    });
    assert_eq!(release.load(std::sync::atomic::Ordering::SeqCst), 0);
    let mut planes = image.into_planes();
    let MixedOwner::Aligned(p) = planes.pop().unwrap() else {
        panic!()
    };
    let a = p.into_storage();
    assert_eq!(a.as_ptr(), alpha_pointer);
    assert_eq!(&a[3..5], &[0.5, 0.5]);
    assert_eq!(&a[5..8], &[0.25; 3]);
    let MixedOwner::Pool(p) = planes.pop().unwrap() else {
        panic!()
    };
    let slot = p.into_storage();
    assert_eq!(slot.samples.as_ptr(), pointer);
    assert_eq!(&slot.samples, &[19, 19, 19, 19, 4095, 4095, 4095]);
    assert_eq!(release.load(std::sync::atomic::Ordering::SeqCst), 0);
    drop(slot);
    assert_eq!(release.load(std::sync::atomic::Ordering::SeqCst), 1);
}

struct Shared(Arc<[i16]>);
impl SampleStorage for Shared {
    type Sample = i16;
    fn samples(&self) -> &[i16] {
        &self.0
    }
}
#[test]
fn shared_read_only_owner_needs_no_mutable_implementation() {
    let data: Arc<[i16]> = Arc::from([-32768, 0, 32767]);
    let owner = Plane::new(Shared(data.clone()), 3, 1, 3).unwrap();
    let image = PlanarBuffer::new(
        vec![owner],
        gray(code(ChannelType::I16, 16)),
        3,
        1,
        ColorContext::default(),
        None,
    )
    .ok()
    .unwrap();
    let AnyPlane::I16(view) = image.as_slice().plane(0).unwrap() else {
        panic!()
    };
    assert_eq!(view.row(0).unwrap().as_ptr(), data.as_ptr());
    assert_eq!(view.row(0).unwrap(), &[-32768, 0, 32767]);
    assert_eq!(Arc::strong_count(&data), 2);
    let owner = image.into_planes().pop().unwrap().into_storage();
    assert!(Arc::ptr_eq(&owner.0, &data));
    drop(owner);
    assert_eq!(Arc::strong_count(&data), 1);
}

#[test]
fn carrying_signed_or_wide_samples_does_not_claim_code_reader_support() {
    let p = Plane::new(vec![-1i32], 1, 1, 1).unwrap();
    assert_eq!(
        CodePlane::new(
            p.as_plane(),
            SampleEncoding::new(ChannelType::I32, 24, 4).unwrap()
        )
        .unwrap_err(),
        CodePlaneError::StorageMismatch
    );
}

#[test]
fn rejected_custom_storage_and_color_can_be_recovered_and_retried() {
    let _guard = lock();
    let release = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let slot = PoolSlot {
        samples: vec![3, 4, 5],
        release: release.clone(),
    };
    let pointer = slot.samples.as_ptr();
    let plane = Plane::new(slot, 3, 1, 3).unwrap();
    let color = ColorContext::from_icc(vec![1, 2, 3, 4]);
    let expected_color = color.clone();
    let parts = PlanarBufferParts::new(
        vec![plane],
        gray(code(ChannelType::U16, 12)),
        4,
        1,
        color,
        None,
    );
    let mut error = failing_above(0, || PlanarBuffer::try_from_parts(parts).unwrap_err());
    assert_eq!(
        *error.error(),
        LayoutError::Dimensions {
            index: 0,
            expected: (4, 1),
            actual: (3, 1)
        }
    );
    assert_eq!(release.load(std::sync::atomic::Ordering::SeqCst), 0);
    let mut parts = error.take_parts().unwrap();
    assert!(error.take_parts().is_none());
    assert_eq!(parts.color, expected_color);
    assert_eq!(parts.planes[0].strided_samples().as_ptr(), pointer);
    parts.width = 3;
    let picture = failing_above(0, || PlanarBuffer::try_from_parts(parts).unwrap());
    assert_eq!(picture.as_slice().color_context(), &expected_color);
    let parts = failing_above(0, || picture.into_parts());
    assert_eq!(parts.color, expected_color);
    let slot = parts.planes.into_iter().next().unwrap().into_storage();
    assert_eq!(slot.samples.as_ptr(), pointer);
    assert_eq!(release.load(std::sync::atomic::Ordering::SeqCst), 0);
    drop(slot);
    assert_eq!(release.load(std::sync::atomic::Ordering::SeqCst), 1);
}
