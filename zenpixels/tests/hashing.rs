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
    let tagged = base.clone().with_color_context(Arc::new(
        ColorContext::from_icc(vec![1, 2, 3]).with_cicp(Cicp::SRGB),
    ));
    assert_eq!(
        base.hash_with(HashInclusions::SamplesOnly),
        tagged.hash_with(HashInclusions::SamplesOnly)
    );
    assert_ne!(
        base.hash_with(HashInclusions::Pixels),
        tagged.hash_with(HashInclusions::Pixels)
    );
    let ctx = Arc::new(ColorContext::from_icc(vec![1, 2, 4]).with_cicp(Cicp::SRGB));
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
    let c = PlanarSlice::new(b.planes(), &different, 3, 3, &color, None).unwrap();
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
