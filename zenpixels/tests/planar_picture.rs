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

#[test]
fn extra_plane_subsampling_shift_and_inverted_cmyk() {
    use zenpixels::sample::{ChromaSiting, ChromaSubsampling, SampleEncoding};
    use zenpixels::{AlphaMode, ChannelType, ColorContext};

    let u8_enc = PlaneEncoding::Code(SampleEncoding::new(ChannelType::U8, 8, 0).unwrap());
    let f16_enc = PlaneEncoding::Float16(FloatEncoding::NOMINAL);

    // Subsampling shift 0..=3 works on extra-channel roles (Depth, Alpha, Cfa, Auxiliary, etc.)
    let half_depth = PlaneSpec::new(PlaneRole::Depth, f16_enc)
        .with_subsampling_shift(1, 1)
        .unwrap();
    let quarter_aux = PlaneSpec::new(PlaneRole::Auxiliary(0), u8_enc)
        .with_subsampling_shift(2, 1)
        .unwrap();
    let eighth_alpha = PlaneSpec::new(PlaneRole::Alpha, u8_enc)
        .with_subsampling_shift(3, 3)
        .unwrap();
    assert_eq!((half_depth.h_shift(), half_depth.v_shift()), (1, 1));
    assert_eq!((quarter_aux.h_shift(), quarter_aux.v_shift()), (2, 1));
    assert_eq!((eighth_alpha.h_shift(), eighth_alpha.v_shift()), (3, 3));

    // Shift > 3 or non-zero shift on primary color roles is rejected.
    assert_eq!(
        PlaneSpec::new(PlaneRole::Depth, u8_enc).with_subsampling_shift(4, 0),
        Err(LayoutError::Subsampling)
    );
    for color_role in [
        PlaneRole::Luma,
        PlaneRole::Cb,
        PlaneRole::CbCr,
        PlaneRole::Red,
        PlaneRole::Cyan,
    ] {
        assert_eq!(
            PlaneSpec::new(color_role, u8_enc).with_subsampling_shift(1, 0),
            Err(LayoutError::Subsampling)
        );
        // Identity (0, 0) is valid on any role.
        assert!(
            PlaneSpec::new(color_role, u8_enc)
                .with_subsampling_shift(0, 0)
                .is_ok()
        );
    }

    // Planar CMYK + shifted extra planes + inverted ink polarity.
    let cmyk_specs = vec![
        PlaneSpec::new(PlaneRole::Cyan, u8_enc),
        PlaneSpec::new(PlaneRole::Magenta, u8_enc),
        PlaneSpec::new(PlaneRole::Yellow, u8_enc),
        PlaneSpec::new(PlaneRole::Black, u8_enc),
        half_depth,
        quarter_aux,
        eighth_alpha,
    ];
    let layout = PlanarLayout::new(cmyk_specs, ChromaSubsampling::Yuv444, ChromaSiting::Center)
        .unwrap()
        .with_inverted_cmyk(true)
        .unwrap();
    assert!(layout.is_inverted_cmyk());
    let conventional = layout.clone().with_inverted_cmyk(false).unwrap();
    assert!(!conventional.is_inverted_cmyk());
    assert_ne!(layout, conventional);

    // Odd picture dimensions 9x5 round up per-plane via div_ceil(1 << shift).
    assert_eq!(layout.plane_size(0, 9, 5), Some((9, 5)));
    assert_eq!(layout.plane_size(4, 9, 5), Some((5, 3))); // (9/2 ceil, 5/2 ceil)
    assert_eq!(layout.plane_size(5, 9, 5), Some((3, 3))); // (9/4 ceil, 5/2 ceil)
    assert_eq!(layout.plane_size(6, 9, 5), Some((2, 1))); // (9/8 ceil, 5/8 ceil)

    let c = [0u8; 45];
    let m = [0u8; 45];
    let y = [0u8; 45];
    let k = [255u8; 45];
    let d = [zenpixels::F16Sample::ZERO; 15];
    let aux = [1u8; 9];
    let a = [255u8; 2];
    let planes = [
        AnyPlane::U8(PlaneRef::new(&c[..], 9, 5, 9).unwrap()),
        AnyPlane::U8(PlaneRef::new(&m[..], 9, 5, 9).unwrap()),
        AnyPlane::U8(PlaneRef::new(&y[..], 9, 5, 9).unwrap()),
        AnyPlane::U8(PlaneRef::new(&k[..], 9, 5, 9).unwrap()),
        AnyPlane::F16(PlaneRef::new(&d[..], 5, 3, 5).unwrap()),
        AnyPlane::U8(PlaneRef::new(&aux[..], 3, 3, 3).unwrap()),
        AnyPlane::U8(PlaneRef::new(&a[..], 2, 1, 2).unwrap()),
    ];
    let ctx = ColorContext::default();
    let slice = PlanarSlice::new(&planes, &layout, 9, 5, &ctx, Some(AlphaMode::Straight)).unwrap();
    assert!(slice.layout().is_inverted_cmyk());

    // Non-CMYK layouts reject inverted_cmyk = true with LayoutError::InkPolarity.
    let rgb_layout = PlanarLayout::new(
        vec![
            PlaneSpec::new(PlaneRole::Red, u8_enc),
            PlaneSpec::new(PlaneRole::Green, u8_enc),
            PlaneSpec::new(PlaneRole::Blue, u8_enc),
        ],
        ChromaSubsampling::Yuv444,
        ChromaSiting::Center,
    )
    .unwrap();
    assert_eq!(
        rgb_layout.with_inverted_cmyk(true),
        Err(LayoutError::InkPolarity)
    );
}
