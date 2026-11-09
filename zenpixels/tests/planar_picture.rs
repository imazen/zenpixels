use zenpixels::ChannelType;
use zenpixels::plane::*;
use zenpixels::sample::{ChromaSiting, ChromaSubsampling, SampleEncoding};

fn code(kind: ChannelType, bits: u8) -> PlaneEncoding {
    PlaneEncoding::Code(SampleEncoding::new(kind, bits, 0).unwrap())
}

#[test]
fn borrowed_mixed_depth_picture_keeps_odd_edges_and_row_padding() {
    let layout = PlanarLayout::new(
        vec![
            PlaneSpec::new(PlaneRole::Luma, code(ChannelType::U16, 10)),
            PlaneSpec::new(PlaneRole::CbCr, code(ChannelType::U16, 10)),
            PlaneSpec::new(PlaneRole::Alpha, code(ChannelType::U8, 8)),
        ],
        ChromaSubsampling::Yuv420,
        ChromaSiting::Left,
    )
    .unwrap();
    let cloned = layout.clone();
    assert_eq!(cloned, layout);
    let y = [15_u16; 19]; // width 5, height 3, stride 7, no final padding.
    let uv = [512_u16; 14]; // 3 chroma positions = 6 samples, two rows, stride 8.
    let alpha = [128_u8; 17]; // width 5, height 3, stride 6.
    let planes = [
        AnyPlane::U16(PlaneRef::new(&y[..], 5, 3, 7).unwrap()),
        AnyPlane::U16(PlaneRef::new(&uv[..], 6, 2, 8).unwrap()),
        AnyPlane::U8(PlaneRef::new(&alpha[..], 5, 3, 6).unwrap()),
    ];
    layout.check(&planes, 5, 3).unwrap();
    assert_eq!(
        layout.roles().collect::<Vec<_>>(),
        vec![PlaneRole::Luma, PlaneRole::CbCr, PlaneRole::Alpha]
    );
    assert_eq!(planes[0].stride_bytes().unwrap(), 14);
    assert_eq!(planes[1].stride_bytes().unwrap(), 16);
    assert_eq!(planes[0].as_strided_bytes(), bytemuck::cast_slice(&y));
    assert_eq!(planes[0].as_strided_bytes().as_ptr(), y.as_ptr().cast());
    assert_eq!(planes[2].as_strided_bytes().as_ptr(), alpha.as_ptr());
    assert!(matches!(
        layout.check_plane(3, &planes[0], 5, 3),
        Err(LayoutError::PlaneIndex { index: 3, count: 3 })
    ));
    assert!(matches!(
        layout.check_plane(1, &planes[2], 5, 3),
        Err(LayoutError::SampleType { index: 1 })
    ));
    assert!(layout.check(&planes, 7, 3).is_err());
}

#[test]
fn byte_stride_reports_overflow_without_rejecting_sample_views() {
    let sample = [1_u16];
    let plane = AnyPlane::U16(PlaneRef::new(&sample[..], 1, 1, usize::MAX).unwrap());
    assert_eq!(plane.stride_bytes(), Err(PlaneError::GeometryOverflow));
    assert_eq!(plane.as_strided_bytes(), bytemuck::cast_slice(&sample));
}
