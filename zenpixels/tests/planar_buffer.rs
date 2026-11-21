//! Owned and mutable planar boundaries, including allocation-free borrowing.
#[path = "common/fail_allocator.rs"]
mod fail_allocator;
use fail_allocator::{failing_above, lock};
use zenpixels::plane::*;
use zenpixels::sample::{ChromaSiting, ChromaSubsampling, SampleEncoding};
use zenpixels::{AlphaMode, ChannelType, Cicp, ColorContext};

fn code(t: ChannelType, bits: u8) -> PlaneEncoding {
    PlaneEncoding::Code(SampleEncoding::new(t, bits, 0).unwrap())
}
fn layout(roles: &[(PlaneRole, PlaneEncoding)], sub: ChromaSubsampling) -> PlanarLayout {
    PlanarLayout::new(
        roles.iter().map(|&(r, e)| PlaneSpec::new(r, e)).collect(),
        sub,
        ChromaSiting::Left,
    )
    .unwrap()
}

#[test]
fn adopts_jpeg_ycbcr_allocations_and_lends_without_allocating() {
    let _guard = lock();
    let e = code(ChannelType::U8, 8);
    let layout = layout(
        &[(PlaneRole::Luma, e), (PlaneRole::Cb, e), (PlaneRole::Cr, e)],
        ChromaSubsampling::Yuv420,
    );
    let y = vec![10u8; 19]; // 5x3, stride 7, short final row.
    let cb = vec![20u8; 8]; // 3x2, stride 5.
    let cr = vec![30u8; 7]; // 3x2, stride 4.
    let pointers = [y.as_ptr(), cb.as_ptr(), cr.as_ptr()];
    let planes = vec![
        AnyPlaneBuf::U8(PlaneBuf::new(y, 5, 3, 7).unwrap()),
        AnyPlaneBuf::U8(PlaneBuf::new(cb, 3, 2, 5).unwrap()),
        AnyPlaneBuf::U8(PlaneBuf::new(cr, 3, 2, 4).unwrap()),
    ];
    let color = ColorContext::from_cicp(Cicp::new(1, 13, 6, true));
    let expected_color = color.clone();
    let mut buffer = failing_above(0, || {
        PlanarBuffer::new(planes, layout, 5, 3, color, None).unwrap()
    });
    failing_above(0, || {
        let picture = buffer.as_slice();
        assert_eq!(picture.color_context(), &expected_color);
        assert_eq!(picture.planes().len(), 3);
        for (p, ptr) in picture.planes().zip(pointers) {
            assert_eq!(p.as_strided_bytes().as_ptr(), ptr);
        }
        assert!(picture.plane(3).is_none());
        let mut picture = buffer.as_slice_mut();
        assert_eq!(picture.as_slice().width(), 5);
        assert!(picture.plane_mut(3).is_none());
        // Three simultaneously live writable views, no staging allocation.
        let mut planes = picture.planes_mut();
        let AnyPlaneMut::U8(mut y) = planes.next().unwrap() else {
            panic!()
        };
        let AnyPlaneMut::U8(mut cb) = planes.next().unwrap() else {
            panic!()
        };
        let AnyPlaneMut::U8(mut cr) = planes.next().unwrap() else {
            panic!()
        };
        assert_eq!(planes.len(), 0);
        for row in y.rows_mut() {
            row.fill(11);
        }
        for row in cb.rows_mut() {
            row.fill(21);
        }
        for row in cr.rows_mut() {
            row.fill(31);
        }
    });
    assert_eq!(buffer.as_slice().color_context(), &expected_color);
    let planes = buffer.into_planes();
    for (i, p) in planes.iter().enumerate() {
        let p = p.as_ref();
        assert_eq!(p.as_strided_bytes().as_ptr(), pointers[i]);
        let AnyPlane::U8(p) = p else { panic!() };
        for row in p.rows() {
            assert!(row.iter().all(|&v| v == [11, 21, 31][i]));
        }
        assert_eq!(p.strided_samples()[p.width()], [10, 20, 30][i]);
    }
}

#[test]
fn borrowed_mutable_mixed_planes_preserve_encodings_and_padding() {
    let layout = layout(
        &[
            (PlaneRole::Luma, code(ChannelType::U16, 10)),
            (PlaneRole::CbCr, code(ChannelType::U16, 10)),
            (
                PlaneRole::Alpha,
                PlaneEncoding::Float(FloatEncoding::NOMINAL),
            ),
        ],
        ChromaSubsampling::Yuv420,
    );
    let color = ColorContext::from_cicp(Cicp::new(9, 16, 9, false));
    let mut y = [7u16; 19];
    let mut uv = [512u16; 14];
    let mut alpha = [0.5f32; 17];
    {
        let mut planes = [
            AnyPlaneMut::U16(PlaneMut::new(&mut y[..], 5, 3, 7).unwrap()),
            AnyPlaneMut::U16(PlaneMut::new(&mut uv[..], 6, 2, 8).unwrap()),
            AnyPlaneMut::F32(PlaneMut::new(&mut alpha[..], 5, 3, 6).unwrap()),
        ];
        let mut picture = PlanarSliceMut::new(
            &mut planes,
            &layout,
            5,
            3,
            &color,
            Some(AlphaMode::Straight),
        )
        .unwrap();
        assert_eq!(picture.as_slice().alpha_mode(), Some(AlphaMode::Straight));
        assert_eq!(picture.as_slice().layout(), &layout);
        assert_eq!(picture.as_slice().color_context(), &color);
        {
            let AnyPlaneMut::U16(mut p) = picture.plane_mut(0).unwrap() else {
                panic!()
            };
            p.row_mut(2).unwrap().fill(1023);
        }
        {
            let AnyPlaneMut::F32(mut p) = picture.plane_mut(2).unwrap() else {
                panic!()
            };
            p.row_mut(1).unwrap().fill(0.25);
        }
        let AnyPlane::U16(p) = picture.as_slice().plane(0).unwrap() else {
            panic!()
        };
        assert_eq!(p.row(2).unwrap(), &[1023; 5]);
    }
    assert_eq!(&y[14..], &[1023; 5]);
    assert_eq!(&y[5..7], &[7; 2]);
    assert_eq!(&alpha[6..11], &[0.25; 5]);
    assert_eq!(alpha[11], 0.5);
    assert_eq!(uv, [512; 14]);
}

#[test]
fn replacing_reborrowed_descriptor_cannot_change_owned_geometry() {
    let e = code(ChannelType::U16, 12);
    let l = layout(&[(PlaneRole::Luma, e)], ChromaSubsampling::Mono);
    let mut b = PlanarBuffer::new(
        vec![AnyPlaneBuf::U16(
            PlaneBuf::new(vec![0u16; 2], 2, 1, 2).unwrap(),
        )],
        l,
        2,
        1,
        ColorContext::default(),
        None,
    )
    .unwrap();
    let mut other = [42u16];
    {
        let mut picture = b.as_slice_mut();
        let mut borrowed = picture.plane_mut(0).unwrap();
        let original = core::mem::replace(
            &mut borrowed,
            AnyPlaneMut::U16(PlaneMut::new(&mut other[..], 1, 1, 1).unwrap()),
        );
        let AnyPlaneMut::U16(mut p) = original else {
            panic!()
        };
        p.row_mut(0).unwrap()[1] = 4095;
    }
    let AnyPlane::U16(p) = b.as_slice().plane(0).unwrap() else {
        panic!()
    };
    assert_eq!(p.width(), 2);
    assert_eq!(p.row(0).unwrap(), &[0, 4095]);
}

#[test]
fn owned_u16_and_f32_keep_native_values() {
    for float in [false, true] {
        let e = if float {
            PlaneEncoding::Float(FloatEncoding::new(2.0, -1.0).unwrap())
        } else {
            code(ChannelType::U16, 12)
        };
        let l = layout(&[(PlaneRole::Luma, e)], ChromaSubsampling::Mono);
        let plane = if float {
            AnyPlaneBuf::F32(PlaneBuf::new(vec![-0.25, 1.5], 2, 1, 2).unwrap())
        } else {
            AnyPlaneBuf::U16(PlaneBuf::new(vec![0, 4095], 2, 1, 2).unwrap())
        };
        let mut b = PlanarBuffer::new(vec![plane], l, 2, 1, ColorContext::default(), None).unwrap();
        let mut view = b.as_slice_mut();
        match view.plane_mut(0).unwrap() {
            AnyPlaneMut::F32(mut p) => {
                assert_eq!(p.row(0).unwrap(), &[-0.25, 1.5]);
                p.row_mut(0).unwrap()[1] = 2.5;
            }
            AnyPlaneMut::U16(mut p) => {
                assert_eq!(p.row(0).unwrap(), &[0, 4095]);
                p.row_mut(0).unwrap()[1] = 2048;
            }
            _ => panic!(),
        }
        assert_eq!(view.as_slice().layout().planes()[0].encoding(), e);
    }
}

#[test]
fn owned_and_mutable_constructors_share_validation() {
    let e = code(ChannelType::U8, 8);
    let color = ColorContext::default();
    for (role, alpha, expected) in [
        (PlaneRole::Luma, None, None),
        (
            PlaneRole::Luma,
            Some(AlphaMode::Straight),
            Some(LayoutError::AlphaInterpretation),
        ),
        (
            PlaneRole::Alpha,
            None,
            Some(LayoutError::AlphaInterpretation),
        ),
        (
            PlaneRole::Alpha,
            Some(AlphaMode::Undefined),
            Some(LayoutError::AlphaInterpretation),
        ),
        (PlaneRole::Alpha, Some(AlphaMode::Straight), None),
        (PlaneRole::Alpha, Some(AlphaMode::Premultiplied), None),
        (PlaneRole::Alpha, Some(AlphaMode::Opaque), None),
    ] {
        for (w, h) in [(0, 0), (1, 1)] {
            let l = layout(&[(role, e)], ChromaSubsampling::Mono);
            let mut data = vec![0u8; (w * h) as usize];
            let owned = vec![AnyPlaneBuf::U8(
                PlaneBuf::new(data.clone(), w as usize, h as usize, w as usize).unwrap(),
            )];
            assert_eq!(
                PlanarBuffer::new(owned, l.clone(), w, h, color.clone(), alpha).err(),
                expected
            );
            let mut planes = [AnyPlaneMut::U8(
                PlaneMut::new(&mut data[..], w as usize, h as usize, w as usize).unwrap(),
            )];
            assert_eq!(
                PlanarSliceMut::new(&mut planes, &l, w, h, &color, alpha).err(),
                expected
            );
        }
    }
    let l = layout(&[(PlaneRole::Luma, e)], ChromaSubsampling::Mono);
    let expected = LayoutError::PlaneCount {
        expected: 1,
        actual: 0,
    };
    assert_eq!(
        PlanarBuffer::<AnyPlaneBuf>::new(vec![], l.clone(), 1, 1, color.clone(), None).unwrap_err(),
        expected
    );
    assert_eq!(
        PlanarSliceMut::<AnyPlaneMut<'_>>::new(&mut [], &l, 1, 1, &color, None).unwrap_err(),
        expected
    );
    let mut wrong = [0u16];
    let mut planes = [AnyPlaneMut::U16(
        PlaneMut::new(&mut wrong[..], 1, 1, 1).unwrap(),
    )];
    assert_eq!(
        PlanarSliceMut::new(&mut planes, &l, 1, 1, &color, None).unwrap_err(),
        LayoutError::SampleType { index: 0 }
    );
    let owned = vec![AnyPlaneBuf::U16(
        PlaneBuf::new(vec![0u16], 1, 1, 1).unwrap(),
    )];
    assert_eq!(
        PlanarBuffer::new(owned, l.clone(), 1, 1, color.clone(), None).unwrap_err(),
        LayoutError::SampleType { index: 0 }
    );
    let expected = LayoutError::Dimensions {
        index: 0,
        expected: (2, 1),
        actual: (1, 1),
    };
    let mut data = [0u8];
    let mut planes = [AnyPlaneMut::U8(
        PlaneMut::new(&mut data[..], 1, 1, 1).unwrap(),
    )];
    assert_eq!(
        PlanarSliceMut::new(&mut planes, &l, 2, 1, &color, None).unwrap_err(),
        expected
    );
    let owned = vec![AnyPlaneBuf::U8(PlaneBuf::new(vec![0u8], 1, 1, 1).unwrap())];
    assert_eq!(
        PlanarBuffer::new(owned, l, 2, 1, color, None).unwrap_err(),
        expected
    );
}

#[test]
fn mutable_descriptor_list_can_be_reborrowed_for_sequential_operations() {
    let e = code(ChannelType::U8, 8);
    let layout = layout(&[(PlaneRole::Luma, e)], ChromaSubsampling::Mono);
    let color = ColorContext::default();
    let mut data = [0u8];
    let mut planes = [AnyPlaneMut::U8(
        PlaneMut::new(&mut data[..], 1, 1, 1).unwrap(),
    )];
    for value in [1, 2] {
        let mut picture = PlanarSliceMut::new(&mut planes, &layout, 1, 1, &color, None).unwrap();
        let AnyPlaneMut::U8(mut p) = picture.plane_mut(0).unwrap() else {
            panic!()
        };
        p.row_mut(0).unwrap()[0] = value;
    }
    assert_eq!(planes[0].as_ref().as_strided_bytes(), &[2]);
}
