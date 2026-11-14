#![cfg(feature = "hash")]

use std::hash::{Hash, Hasher};
use std::sync::Arc;
use zenpixels::hdr::{DiffuseWhite, Nits};
use zenpixels::plane::*;
use zenpixels::sample::*;
use zenpixels::*;

// A caller's Hasher may distinguish both write content and call boundaries.
// finish panics deliberately: Hash must serialize, never finalize its caller.
#[derive(Default, Debug, PartialEq, Eq)]
struct Writes(Vec<Vec<u8>>);
impl Hasher for Writes {
    fn write(&mut self, bytes: &[u8]) {
        self.0.push(bytes.to_vec());
    }
    fn finish(&self) -> u64 {
        panic!("Hash must not finish the caller's state")
    }
}
fn writes(value: &impl Hash) -> Writes {
    let mut state = Writes::default();
    value.hash(&mut state);
    state
}
fn view(data: &[u8], w: u32, h: u32, stride: usize) -> PixelSlice<'_> {
    PixelSlice::new(data, w, h, stride, PixelDescriptor::RGBA8).unwrap()
}

#[test]
fn storage_stride_and_write_boundaries_do_not_change_hashes() {
    for width in [1, 2, 63, 64, 255, 256, 257, 1025] {
        let active = width * 4;
        let tight: Vec<u8> = (0..active * 3).map(|i| (i * 37 + 11) as u8).collect();
        let mut strided = vec![0x9b; (active + 19) * 3 + 31];
        for y in 0..3 {
            strided[7 + y * (active + 19)..7 + y * (active + 19) + active]
                .copy_from_slice(&tight[y * active..(y + 1) * active]);
        }
        let a = view(&tight, width as u32, 3, active);
        let b = view(&strided[7..], width as u32, 3, active + 19);
        assert_eq!(writes(&a), writes(&b), "width={width}");
        for mode in [
            HashInclusions::SamplesOnly,
            HashInclusions::Pixels,
            HashInclusions::All,
        ] {
            assert_eq!(a.hash_with(mode), b.hash_with(mode));
        }
        let owner =
            PixelBuffer::from_vec(tight.clone(), width as u32, 3, PixelDescriptor::RGBA8).unwrap();
        assert_eq!(writes(&owner), writes(&a));
        assert_eq!(writes(&PixelCow::Borrowed(a.clone())), writes(&a));
        let mut owner = owner;
        assert_eq!(writes(&owner.as_slice_mut()), writes(&a));
        let mut state = zenhash::Xxh3::new();
        a.hash(&mut state);
        assert_eq!(state.digest(), a.hash_with(HashInclusions::default()));
    }
}

#[test]
fn samples_metadata_geometry_and_exact_evidence_follow_inclusions() {
    let bytes = [1u8, 2, 3, 4, 5, 6, 7, 8];
    let base = view(&bytes, 2, 1, 8);
    let tagged = base
        .clone()
        .with_color_context(Arc::new(ColorContext::from_icc(vec![1, 2, 3])));
    assert_eq!(
        base.hash_with(HashInclusions::SamplesOnly),
        tagged.hash_with(HashInclusions::SamplesOnly)
    );
    assert_ne!(
        base.hash_with(HashInclusions::Pixels),
        tagged.hash_with(HashInclusions::Pixels)
    );
    let ctx = Arc::new(ColorContext::from_icc(vec![1, 2, 4]));
    assert_ne!(
        writes(&tagged),
        writes(&base.clone().with_color_context(ctx))
    );
    assert_ne!(
        writes(&base),
        writes(
            &base
                .clone()
                .with_color_context(Arc::new(ColorContext::default()))
        )
    );
    assert_ne!(
        writes(&base),
        writes(&base.clone().with_cicp(Cicp::new(1, 13, 9, false)))
    );
    assert_ne!(
        writes(&base),
        writes(&base.clone().with_diffuse_white(DiffuseWhite::BT2408))
    );
    assert_ne!(
        writes(&base),
        writes(&base.clone().with_color_context(Arc::new(
            ColorContext::default().with_hdr_peak(Nits::new(1000.0))
        )))
    );
    for other in [
        base.clone().with_transfer(TransferFunction::Linear),
        base.clone().with_primaries(ColorPrimaries::DisplayP3),
        base.clone().with_signal_range(SignalRange::Narrow),
        base.clone().with_alpha_mode(Some(AlphaMode::Premultiplied)),
    ] {
        assert_eq!(
            base.hash_with(HashInclusions::SamplesOnly),
            other.hash_with(HashInclusions::SamplesOnly)
        );
        assert_ne!(writes(&base), writes(&other));
    }
    let exact = ExactDepth::new(ChannelType::U8, SignalRange::Full, 7, 8).unwrap();
    let exact = base.clone().with_exact_depth(exact).unwrap();
    assert_eq!(
        base.hash_with(HashInclusions::Pixels),
        exact.hash_with(HashInclusions::Pixels)
    );
    assert_ne!(writes(&base), writes(&exact));
    assert_ne!(writes(&base), writes(&view(&bytes, 1, 2, 4)));
    let mut changed = bytes;
    changed[7] ^= 1;
    assert_ne!(writes(&base), writes(&view(&changed, 2, 1, 8)));
}

#[test]
fn undefined_alpha_and_float_payloads_are_handled_explicitly() {
    let a =
        PixelSlice::<()>::new(&[1, 2, 3, 0, 4, 5, 6, 9], 2, 1, 8, PixelDescriptor::RGBX8).unwrap();
    let b = PixelSlice::<()>::new(
        &[1, 2, 3, 255, 4, 5, 6, 10],
        2,
        1,
        8,
        PixelDescriptor::RGBX8,
    )
    .unwrap();
    assert_eq!(writes(&a), writes(&b));
    let rgb = PixelSlice::<()>::new(&[1, 2, 3, 4, 5, 6], 2, 1, 6, PixelDescriptor::RGB8).unwrap();
    assert_eq!(
        a.hash_with(HashInclusions::SamplesOnly),
        rgb.hash_with(HashInclusions::SamplesOnly)
    );
    for bits in [0u32, 0x8000_0000, 0x7fc0_0001, 0x7fc0_0002, 0x7f80_0000] {
        let words = [bits, 0, 0];
        let words2 = [bits ^ 1, 0, 0];
        let a = PixelSlice::<()>::new(
            bytemuck::cast_slice(&words),
            1,
            1,
            12,
            PixelDescriptor::RGBF32,
        )
        .unwrap();
        let b = PixelSlice::<()>::new(
            bytemuck::cast_slice(&words2),
            1,
            1,
            12,
            PixelDescriptor::RGBF32,
        )
        .unwrap();
        assert_ne!(writes(&a), writes(&b));
    }
}

#[test]
fn wide_native_sample_bits_are_not_narrowed() {
    for ty in [
        ChannelType::U8,
        ChannelType::I8,
        ChannelType::U16,
        ChannelType::I16,
        ChannelType::F16,
        ChannelType::U32,
        ChannelType::I32,
        ChannelType::F32,
        ChannelType::U64,
        ChannelType::I64,
        ChannelType::F64,
    ] {
        let desc =
            PixelDescriptor::try_new(ty, ChannelLayout::Gray, None, TransferFunction::Linear)
                .unwrap();
        let a = [0u64; 2];
        let mut b = a;
        // Change the most significant bit of the last active sample.
        let size = ty.byte_size();
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut b);
        bytes[if cfg!(target_endian = "little") {
            2 * size - 1
        } else {
            size
        }] ^= 128;
        let a = PixelSlice::<()>::new(bytemuck::cast_slice(&a), 2, 1, 2 * size, desc).unwrap();
        let b = PixelSlice::<()>::new(bytemuck::cast_slice(&b), 2, 1, 2 * size, desc).unwrap();
        assert_ne!(writes(&a), writes(&b), "{ty:?}");
    }
}

#[test]
fn planes_and_component_views_exclude_padding_and_other_components() {
    let a = [1u16, 2, 3, 4];
    let b = [1u16, 2, 999, 3, 4, 888];
    let a = PlaneRef::new(&a[..], 2, 2, 2).unwrap();
    let b = PlaneRef::new(&b[..], 2, 2, 3).unwrap();
    assert_eq!(writes(&a), writes(&b));
    assert_eq!(writes(&a), writes(&AnyPlane::U16(b)));
    let enc = SampleEncoding::new(ChannelType::U16, 10, 6).unwrap();
    let words = [64u16, 55, 128, 66];
    let other = [127u16, 999, 191, 111];
    let a = CodePlane::component(
        AnyPlane::U16(PlaneRef::new(&words[..], 4, 1, 4).unwrap()),
        2,
        0,
        enc,
    )
    .unwrap();
    let b = CodePlane::component(
        AnyPlane::U16(PlaneRef::new(&other[..], 4, 1, 4).unwrap()),
        2,
        0,
        enc,
    )
    .unwrap();
    assert_eq!(writes(&a), writes(&b));
    let other = [127u16, 999, 255, 111];
    let b = CodePlane::component(
        AnyPlane::U16(PlaneRef::new(&other[..], 4, 1, 4).unwrap()),
        2,
        0,
        enc,
    )
    .unwrap();
    assert_ne!(writes(&a), writes(&b));
}

#[test]
fn planar_native_padding_and_metadata_are_independent() {
    let e = PlaneEncoding::Code(SampleEncoding::new(ChannelType::U16, 10, 6).unwrap());
    let layout = PlanarLayout::new(
        vec![
            PlaneSpec::new(PlaneRole::Luma, e),
            PlaneSpec::new(PlaneRole::CrCb, e),
        ],
        ChromaSubsampling::Yuv420,
        ChromaSiting::Left,
    )
    .unwrap();
    let y = [64u16; 9];
    let uv = [128u16; 8];
    let yp = [127u16; 9];
    let uvp = [191u16; 8];
    let a = [
        AnyPlane::U16(PlaneRef::new(&y[..], 3, 3, 3).unwrap()),
        AnyPlane::U16(PlaneRef::new(&uv[..], 4, 2, 4).unwrap()),
    ];
    let b = [
        AnyPlane::U16(PlaneRef::new(&yp[..], 3, 3, 3).unwrap()),
        AnyPlane::U16(PlaneRef::new(&uvp[..], 4, 2, 4).unwrap()),
    ];
    let color = ColorContext::from_cicp(Cicp::new(9, 16, 9, false));
    let a = PlanarSlice::new(&a, &layout, 3, 3, &color, None).unwrap();
    let b = PlanarSlice::new(&b, &layout, 3, 3, &color, None).unwrap();
    assert_eq!(writes(&a), writes(&b));
    let different = PlanarLayout::new(
        vec![
            PlaneSpec::new(PlaneRole::Luma, e),
            PlaneSpec::new(PlaneRole::CbCr, e),
        ],
        ChromaSubsampling::Yuv420,
        ChromaSiting::Center,
    )
    .unwrap();
    let borrowed = b.planes().collect::<Vec<_>>();
    let c = PlanarSlice::new(&borrowed, &different, 3, 3, &color, None).unwrap();
    assert_eq!(
        a.hash_with(HashInclusions::SamplesOnly),
        c.hash_with(HashInclusions::SamplesOnly)
    );
    assert_ne!(writes(&a), writes(&c));
}

#[test]
fn zero_width_views_do_not_iterate_the_declared_empty_rows() {
    let a = PlaneRef::new(&[] as &[u8], 0, usize::MAX, usize::MAX).unwrap();
    let b = PlaneRef::new(&[] as &[u8], 0, usize::MAX - 1, usize::MAX).unwrap();
    assert_ne!(writes(&a), writes(&b));
    let a = view(&[], 0, u32::MAX, 0);
    let b = view(&[], 0, u32::MAX - 1, 0);
    assert_ne!(writes(&a), writes(&b));
}

#[test]
fn portable_v1_golden_hashes() {
    // Freeze the complete v1 serialization/backend output across targets.
    // Independent algorithm goldens are checked separately in icc::murmur3_tests
    // and zenhash's C-reference conformance matrix.
    let p = view(&[1, 2, 3, 4], 1, 1, 4);
    assert_eq!(p.hash_with(HashInclusions::All), 0xac0c_c9b2_0351_b75a);
    let samples = [0x1234u16, 0xabcd];
    let p = PlaneRef::new(&samples[..], 2, 1, 2).unwrap();
    assert_eq!(p.hash_with(HashInclusions::All), 0x959d_7fe7_6c2e_f808);

    let samples = [63u16, 127, 65535];
    let planes = [AnyPlane::U16(PlaneRef::new(&samples[..], 3, 1, 3).unwrap())];
    let e = PlaneEncoding::Code(SampleEncoding::new(ChannelType::U16, 10, 6).unwrap());
    let layout = PlanarLayout::new(
        vec![PlaneSpec::new(PlaneRole::Luma, e)],
        ChromaSubsampling::Mono,
        ChromaSiting::Center,
    )
    .unwrap();
    let color = ColorContext::default();
    let p = PlanarSlice::new(&planes, &layout, 3, 1, &color, None).unwrap();
    assert_eq!(p.hash_with(HashInclusions::All), 0x60ba_798c_2efa_a06e);

    let samples = [0x8000_0000_0000_0000u64, 0x7ff8_0000_0000_0001];
    let desc = PixelDescriptor::try_new(
        ChannelType::F64,
        ChannelLayout::Gray,
        None,
        TransferFunction::Linear,
    )
    .unwrap();
    let bytes: Vec<_> = samples.into_iter().flat_map(u64::to_ne_bytes).collect();
    let p = PixelBuffer::from_vec(bytes, 2, 1, desc).unwrap();
    assert_eq!(p.hash_with(HashInclusions::All), 0x6e78_65c3_8572_b77e);
}
#[test]
fn planar_u8_subbyte_codes_ignore_unused_bits() {
    let encoding = SampleEncoding::new(ChannelType::U8, 1, 3).unwrap();
    let layout = PlanarLayout::new(
        vec![PlaneSpec::new(
            PlaneRole::Luma,
            PlaneEncoding::Code(encoding),
        )],
        ChromaSubsampling::Mono,
        ChromaSiting::Unknown,
    )
    .unwrap();
    let color = ColorContext::default();
    let clean = [0x08u8, 0x00, 0x08];
    let padded = [0xffu8, 0xf7, 0xff];
    let changed = [0x08u8, 0x08, 0x08];
    let a = [AnyPlane::U8(PlaneRef::new(&clean[..], 3, 1, 3).unwrap())];
    let b = [AnyPlane::U8(PlaneRef::new(&padded[..], 3, 1, 3).unwrap())];
    let c = [AnyPlane::U8(PlaneRef::new(&changed[..], 3, 1, 3).unwrap())];
    let a = PlanarSlice::new(&a, &layout, 3, 1, &color, None).unwrap();
    let b = PlanarSlice::new(&b, &layout, 3, 1, &color, None).unwrap();
    let c = PlanarSlice::new(&c, &layout, 3, 1, &color, None).unwrap();
    for mode in [
        HashInclusions::SamplesOnly,
        HashInclusions::Pixels,
        HashInclusions::All,
    ] {
        assert_eq!(a.hash_with(mode), b.hash_with(mode));
        assert_ne!(a.hash_with(mode), c.hash_with(mode));
    }
}

#[test]
fn wide_and_signed_planar_hashes_ignore_unused_bits_and_row_padding() {
    macro_rules! check {
        ($t:ty, $variant:ident, $bits:expr, $shift:expr) => {{
            let e = SampleEncoding::new(ChannelType::$variant, $bits, $shift).unwrap();
            let mask = ((1u64 << $bits) - 1) << $shift;
            let low = [0u64, 1, (1u64 << ($bits - 1)), (1u64 << $bits) - 1];
            let clean: Vec<$t> = low.iter().map(|&v| (v << $shift) as $t).collect();
            let padded: Vec<$t> = vec![
                ((low[0] << $shift) | !mask) as $t,
                ((low[1] << $shift) | !mask) as $t,
                17 as $t, // padding between rows, not part of the picture
                ((low[2] << $shift) | !mask) as $t,
                ((low[3] << $shift) | !mask) as $t,
            ];
            let clean = Plane::new(clean, 2, 2, 2).unwrap();
            let padded = Plane::new(padded, 2, 2, 3).unwrap();
            let layout = PlanarLayout::new(
                vec![PlaneSpec::new(PlaneRole::Luma, PlaneEncoding::Code(e))],
                ChromaSubsampling::Mono,
                ChromaSiting::Unknown,
            )
            .unwrap();
            let color = ColorContext::default();
            let a = [clean.as_plane()];
            let b = [padded.as_plane()];
            let a = PlanarSlice::new(&a, &layout, 2, 2, &color, None).unwrap();
            let b = PlanarSlice::new(&b, &layout, 2, 2, &color, None).unwrap();
            assert_eq!(writes(&a), writes(&b));
            assert_eq!(
                a.hash_with(HashInclusions::SamplesOnly),
                b.hash_with(HashInclusions::SamplesOnly)
            );
            // The final payload bytes must be independently packed little-endian
            // codes of the storage width, not host-endian or sign-extended data.
            let mut expected = Vec::new();
            for v in low {
                expected.extend_from_slice(&v.to_le_bytes()[..core::mem::size_of::<$t>()]);
            }
            let serialized = writes(&a).0.concat();
            assert!(serialized.ends_with(&expected));
        }};
    }
    check!(i8, I8, 6, 1);
    check!(i16, I16, 10, 4);
    check!(u32, U32, 24, 4);
    check!(i32, I32, 24, 4);
    check!(u64, U64, 48, 8);
    check!(i64, I64, 48, 8);
}

#[test]
fn empty_wide_code_planes_do_not_iterate_declared_rows() {
    let plane = Plane::new(Vec::<i64>::new(), 0, u32::MAX as usize, usize::MAX).unwrap();
    let encoding = SampleEncoding::new(ChannelType::I64, 48, 8).unwrap();
    let layout = PlanarLayout::new(
        vec![PlaneSpec::new(
            PlaneRole::Luma,
            PlaneEncoding::Code(encoding),
        )],
        ChromaSubsampling::Mono,
        ChromaSiting::Unknown,
    )
    .unwrap();
    let image = PlanarBuffer::new(
        vec![plane],
        layout,
        0,
        u32::MAX,
        ColorContext::default(),
        None,
    )
    .unwrap();
    let view = image.as_slice();
    let _ = writes(&view); // Must finish without iterating u32::MAX empty rows.
}

#[test]
fn parametric_color_and_rendering_intent_participate_in_metadata_hashes() {
    use zenpixels::{
        CieXy, GammaCurve, ParametricColor, ParametricPrimaries, ParametricTransfer,
        RenderingIntent, WhitePoint,
    };

    let custom_a = ParametricColor::rgb(
        ParametricPrimaries::Custom {
            red: CieXy::new(0.7347, 0.2653),
            green: CieXy::new(0.0, 1.0),
            blue: CieXy::new(0.0001, -0.0770),
        },
        WhitePoint::Custom(CieXy::new(0.32168, 0.33767)),
        ParametricTransfer::Gamma(GammaCurve::from_encoding_exponent(0.45)),
    );
    let custom_b = custom_a.with_rendering_intent(RenderingIntent::Perceptual);

    let buf_a = PixelBuffer::from_vec(vec![10, 20, 30, 255], 1, 1, PixelDescriptor::RGBA8)
        .unwrap()
        .with_parametric(custom_a);
    let buf_b = PixelBuffer::from_vec(vec![10, 20, 30, 255], 1, 1, PixelDescriptor::RGBA8)
        .unwrap()
        .with_parametric(custom_b);
    let buf_cicp_rel = PixelBuffer::from_vec(vec![10, 20, 30, 255], 1, 1, PixelDescriptor::RGBA8)
        .unwrap()
        .with_cicp(Cicp::SRGB);
    let buf_cicp_abs = PixelBuffer::from_vec(vec![10, 20, 30, 255], 1, 1, PixelDescriptor::RGBA8)
        .unwrap()
        .with_cicp(Cicp::SRGB)
        .with_rendering_intent(RenderingIntent::AbsoluteColorimetric);

    assert_eq!(
        buf_a.hash_with(HashInclusions::SamplesOnly),
        buf_b.hash_with(HashInclusions::SamplesOnly)
    );
    assert_ne!(
        buf_a.hash_with(HashInclusions::Pixels),
        buf_b.hash_with(HashInclusions::Pixels)
    );
    assert_ne!(
        buf_a.hash_with(HashInclusions::All),
        buf_b.hash_with(HashInclusions::All)
    );
    assert_ne!(
        buf_cicp_rel.hash_with(HashInclusions::All),
        buf_cicp_abs.hash_with(HashInclusions::All)
    );

    // CodePlane over U32 masks unused bits and ignores row padding when hashed directly.
    let enc = SampleEncoding::new(ChannelType::U32, 24, 4).unwrap();
    let clean = [0x0123_4560u32, 0x0fed_cba0];
    let dirty = [0xf123_456fu32, 0x7fed_cbaf, 0xdead_beef];
    let cp_clean = CodePlane::new(
        AnyPlane::U32(PlaneRef::new(&clean[..], 2, 1, 2).unwrap()),
        enc,
    )
    .unwrap();
    let cp_dirty = CodePlane::new(
        AnyPlane::U32(PlaneRef::new(&dirty[..], 2, 1, 3).unwrap()),
        enc,
    )
    .unwrap();
    assert_eq!(writes(&cp_clean), writes(&cp_dirty));
}
