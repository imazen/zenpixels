//! `Vec<T, A>` and `Box<T, A>` as storage, on Rust 1.100 or newer. On older
//! compilers the `allocator_api` feature has no effect and this file is empty;
//! CI's beta job runs it until 1.100 is stable.

#[rustversion::since(1.100)]
mod with_allocator {
    use std::alloc::{Global, System};
    use zenpixels::plane::{Plane, PlaneMut, PlaneRef};
    use zenpixels::{ByteStorage, PixelBuffer, PixelBufferParts, PixelDescriptor};

    fn in_system<T: Copy + Default>(len: usize) -> Vec<T, System> {
        let mut v = Vec::with_capacity_in(len, System);
        v.resize(len, T::default());
        v
    }

    #[test]
    fn plain_vec_still_works_because_global_is_an_allocator() {
        let v: Vec<u8, Global> = vec![1; 12];
        let plane = Plane::new(v, 3, 3, 4).unwrap();
        let back: Vec<u8> = plane.try_into_vec().unwrap();
        assert_eq!(back.len(), 12);
    }

    #[test]
    fn planes_own_vecs_from_any_allocator() {
        let mut v = in_system::<u16>(12);
        v[5] = 9;
        let ptr = v.as_ptr();
        let mut plane = Plane::new(v, 3, 3, 4).unwrap();
        assert_eq!(plane.row(1).unwrap()[1], 9);
        plane.row_mut(2).unwrap()[0] = 7;
        assert_eq!(plane.view().row(0).unwrap().as_ptr(), ptr);
        let back: Vec<u16, System> = plane.try_into_vec().unwrap();
        assert_eq!((back.as_ptr(), back[8]), (ptr, 7));
    }

    #[test]
    fn boxes_from_any_allocator_are_storage() {
        let boxed: Box<[f32], System> = in_system::<f32>(6).into_boxed_slice();
        let ptr = boxed.as_ptr();
        let plane = Plane::new(boxed, 3, 2, 3).unwrap();
        assert_eq!(plane.row(0).unwrap().as_ptr(), ptr);
    }

    #[test]
    fn packed_buffers_own_vecs_from_any_allocator() {
        let v = in_system::<u16>(37 * 11 * 3);
        let ptr = v.as_ptr();
        let parts = PixelBufferParts::new(v, 37, 11, 37 * 6, PixelDescriptor::RGB16_SRGB);
        let buffer: PixelBuffer<(), Vec<u16, System>> = PixelBuffer::try_from_parts(parts).unwrap();
        assert_eq!(buffer.as_slice().as_strided_bytes().as_ptr().cast(), ptr);
        let back: Vec<u16, System> = buffer.into_parts().data;
        assert_eq!(back.as_ptr(), ptr);
    }

    #[test]
    fn byte_methods_and_views_reach_through_the_allocator() {
        let mut v = in_system::<u16>(4);
        v[0] = 0x0102;
        assert_eq!(ByteStorage::as_bytes(&v).len(), 8);
        assert_eq!(
            PlaneRef::new(&v[..], 2, 2, 2).unwrap().row(0).unwrap()[0],
            0x0102
        );
        let mut m = PlaneMut::new(&mut v[..], 2, 2, 2).unwrap();
        m.row_mut(1).unwrap()[1] = 5;
        assert_eq!(v[3], 5);
    }
    #[test]
    fn mutable_plane_and_picture_borrows_preserve_the_allocator_owner() {
        use zenpixels::plane::{
            AnyPlane, PlanarLayout, PlanarSlice, PlaneEncoding, PlaneRole, PlaneSpec,
        };
        use zenpixels::sample::{ChromaSiting, ChromaSubsampling, SampleEncoding};
        use zenpixels::{Cicp, ColorContext};
        let mut data = in_system::<u16>(19); // 5x3, stride 7, short final row.
        data[5] = 0xbeef; // Row padding must remain unchanged.
        let ptr = data.as_ptr();
        let mut owner = Plane::new(data, 5, 3, 7).unwrap();
        {
            let mut writable: PlaneMut<'_, u16> = owner.view_mut();
            writable.row_mut(2).unwrap()[4] = 0x1234;
        }
        let layout = PlanarLayout::new(
            vec![PlaneSpec::new(
                PlaneRole::Luma,
                PlaneEncoding::Code(
                    SampleEncoding::new(zenpixels::ChannelType::U16, 10, 6).unwrap(),
                ),
            )],
            ChromaSubsampling::Mono,
            ChromaSiting::Unknown,
        )
        .unwrap();
        let context = ColorContext::from_cicp(Cicp::new(1, 1, 0, true));
        {
            let borrowed: PlaneRef<'_, u16> = owner.view();
            let planes = [AnyPlane::U16(borrowed)];
            let picture = PlanarSlice::new(&planes, &layout, 5, 3, &context, None).unwrap();
            let AnyPlane::U16(plane) = picture.plane(0).unwrap() else {
                panic!("u16")
            };
            assert_eq!(plane.strided_samples().as_ptr(), ptr);
            assert_eq!(plane.row(2).unwrap()[4], 0x1234);
            assert!(core::ptr::eq(picture.color_context(), &context));
        }
        // Both view layers borrow; neither consumes/replaces the allocator.
        let recovered: Vec<u16, System> = owner.try_into_vec().unwrap();
        assert_eq!(recovered.as_ptr(), ptr);
        let _: &System = recovered.allocator();
        assert_eq!((recovered[5], recovered[18]), (0xbeef, 0x1234));
    }
}
