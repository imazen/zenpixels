//! ZERO-TOLERANCE 0-BIT PARITY TESTS: `Accelerators::SCALAR` vs `Accelerators::ALL`.
//!
//! Every single SIMD accelerator in `zenpixels-convert 0.5` MUST produce
//! bit-for-bit identical output bytes to the pure scalar oracle across
//! non-aligned prime row widths (exercising SIMD bodies AND scalar tails).

use zenpixels_convert::{
    Accelerators, AlphaMode, AlphaPolicy, BitPacking, ChannelOrder, Color, ColorSpace,
    ConvertPolicy, DepthPolicy, DiffuseWhite, F16Sample, GamutPolicy, HlgMode, HlgRendering,
    ImageSpec, Nits, Primaries, RangePolicy, RowPlan, SampleLayout, SampleType, SignalRange,
    Transfer,
};

#[allow(clippy::too_many_arguments)]
fn assert_scalar_simd_bit_exact(
    label: &str,
    src_spec: ImageSpec,
    dst_spec: ImageSpec,
    base_policy: ConvertPolicy,
    src_bytes: &[u8],
    width: u32,
    y: u32,
    hdr_peak: Option<Nits>,
) {
    let p_scalar = base_policy.clone().with_accelerators(Accelerators::SCALAR);
    let p_simd = base_policy.with_accelerators(Accelerators::ALL);

    let plan_scalar = RowPlan::build(src_spec.clone(), dst_spec.clone(), &p_scalar, hdr_peak)
        .unwrap_or_else(|e| panic!("{label}: scalar build failed: {e}"));
    let plan_simd = RowPlan::build(src_spec, dst_spec.clone(), &p_simd, hdr_peak)
        .unwrap_or_else(|e| panic!("{label}: simd build failed: {e}"));

    let dst_len = (width as usize) * dst_spec.layout.bytes_per_pixel();
    let mut out_scalar = vec![0xA5u8; dst_len];
    let mut out_simd = vec![0x5Au8; dst_len];

    plan_scalar
        .convert_row(src_bytes, &mut out_scalar, width, y)
        .unwrap();
    plan_simd
        .convert_row(src_bytes, &mut out_simd, width, y)
        .unwrap();

    assert_eq!(
        out_scalar, out_simd,
        "0-bit SCALAR vs ALL parity mismatch on `{label}` (width={width}, y={y})"
    );
}

#[test]
fn parity_byte_layouts_and_u8_to_u16() {
    let width = 73u32; // crosses 64-px strip boundary + 9-px tail!
    let rgb8: Vec<u8> = (0..width as usize * 3)
        .map(|i| ((i * 37 + 11) % 256) as u8)
        .collect();
    let rgba8: Vec<u8> = (0..width as usize * 4)
        .map(|i| ((i * 53 + 7) % 256) as u8)
        .collect();
    let gray8: Vec<u8> = (0..width as usize)
        .map(|i| ((i * 19 + 3) % 256) as u8)
        .collect();

    let pol = ConvertPolicy::permissive();
    assert_scalar_simd_bit_exact(
        "RGB8->RGBA8",
        ImageSpec::rgb8_srgb(),
        ImageSpec::rgba8_srgb(),
        pol.clone(),
        &rgb8,
        width,
        0,
        None,
    );
    assert_scalar_simd_bit_exact(
        "RGB8->BGRA8",
        ImageSpec::rgb8_srgb(),
        ImageSpec::bgra8_srgb(),
        pol.clone(),
        &rgb8,
        width,
        0,
        None,
    );
    assert_scalar_simd_bit_exact(
        "RGB8->BGR8",
        ImageSpec::rgb8_srgb(),
        ImageSpec::bgr8_srgb(),
        pol.clone(),
        &rgb8,
        width,
        0,
        None,
    );
    assert_scalar_simd_bit_exact(
        "RGBA8->RGB8",
        ImageSpec::rgba8_srgb(),
        ImageSpec::rgb8_srgb(),
        pol.clone(),
        &rgba8,
        width,
        0,
        None,
    );
    assert_scalar_simd_bit_exact(
        "BGRA8->RGB8",
        ImageSpec::bgra8_srgb(),
        ImageSpec::rgb8_srgb(),
        pol.clone(),
        &rgba8,
        width,
        0,
        None,
    );
    assert_scalar_simd_bit_exact(
        "RGBA8->BGRA8",
        ImageSpec::rgba8_srgb(),
        ImageSpec::bgra8_srgb(),
        pol.clone(),
        &rgba8,
        width,
        0,
        None,
    );
    assert_scalar_simd_bit_exact(
        "RGBX8->RGBA8",
        ImageSpec::rgbx8_srgb(),
        ImageSpec::rgba8_srgb(),
        pol.clone(),
        &rgba8,
        width,
        0,
        None,
    );
    assert_scalar_simd_bit_exact(
        "RGBX8->BGRA8",
        ImageSpec::rgbx8_srgb(),
        ImageSpec::bgra8_srgb(),
        pol.clone(),
        &rgba8,
        width,
        0,
        None,
    );
    assert_scalar_simd_bit_exact(
        "GRAY8->RGB8",
        ImageSpec::gray8_srgb(),
        ImageSpec::rgb8_srgb(),
        pol.clone(),
        &gray8,
        width,
        0,
        None,
    );
    assert_scalar_simd_bit_exact(
        "GRAY8->RGBA8",
        ImageSpec::gray8_srgb(),
        ImageSpec::rgba8_srgb(),
        pol.clone(),
        &gray8,
        width,
        0,
        None,
    );
    assert_scalar_simd_bit_exact(
        "RGB8->RGB16",
        ImageSpec::rgb8_srgb(),
        ImageSpec::rgb16_srgb(),
        pol,
        &rgb8,
        width,
        0,
        None,
    );
}

#[test]
fn parity_srgb_gamut_f16_oklab_and_extended_floats() {
    let width = 71u32;
    // Extended float test row covering negatives, sub-0.04045 toe, midtones, >1.0 highlights!
    let mut f32_rgb = Vec::with_capacity(width as usize * 3);
    for i in 0..width as usize {
        let t = (i as f32) / (width as f32 - 1.0);
        f32_rgb.push(-0.25 + 1.75 * t);
        f32_rgb.push(0.01 + 0.98 * ((i * 7 % 71) as f32 / 70.0));
        f32_rgb.push(1.4 - 1.6 * t);
    }
    let f32_bytes: &[u8] = bytemuck::cast_slice(&f32_rgb);

    // sRGB <-> Linear (extended & clipped)
    assert_scalar_simd_bit_exact(
        "LinearF32->SrgbF32(extended)",
        ImageSpec::rgbf32_linear(),
        ImageSpec::new(SampleLayout::RGBF32, ColorSpace::SRGB),
        ConvertPolicy::working_linear().with_gamut(GamutPolicy::PreserveExtended),
        f32_bytes,
        width,
        0,
        None,
    );
    assert_scalar_simd_bit_exact(
        "SrgbF32->LinearF32(extended)",
        ImageSpec::new(SampleLayout::RGBF32, ColorSpace::SRGB),
        ImageSpec::rgbf32_linear(),
        ConvertPolicy::working_linear().with_gamut(GamutPolicy::PreserveExtended),
        f32_bytes,
        width,
        0,
        None,
    );

    // Cross-gamut 3x3 (DisplayP3 <-> Bt709 <-> Bt2020 <-> AdobeRgb)
    let p3_srgb = ImageSpec::rgb8_srgb().with_primaries(Primaries::DisplayP3);
    let bt2020_u16 = ImageSpec::rgb16_srgb().with_primaries(Primaries::Bt2020);
    let adobe_u16 = ImageSpec::rgb16_srgb()
        .with_primaries(Primaries::AdobeRgb)
        .with_transfer(Transfer::Gamma22);
    let rgb8: Vec<u8> = (0..width as usize * 3)
        .map(|i| ((i * 41 + 9) % 256) as u8)
        .collect();
    let rgb16: Vec<u16> = (0..width as usize * 3)
        .map(|i| ((i * 1973 + 511) % 65536) as u16)
        .collect();
    let rgb16_bytes: &[u8] = bytemuck::cast_slice(&rgb16);

    assert_scalar_simd_bit_exact(
        "P3U8->sRGBU8",
        p3_srgb.clone(),
        ImageSpec::rgb8_srgb(),
        ConvertPolicy::permissive(),
        &rgb8,
        width,
        1,
        None,
    );
    assert_scalar_simd_bit_exact(
        "Bt2020U16->sRGBU8(OrderedDither)",
        bt2020_u16.clone(),
        ImageSpec::rgb8_srgb(),
        ConvertPolicy::permissive().with_depth(DepthPolicy::OrderedDither),
        rgb16_bytes,
        width,
        3,
        None,
    );
    assert_scalar_simd_bit_exact(
        "AdobeRGB->DisplayP3(Truncate)",
        adobe_u16,
        p3_srgb.clone(),
        ConvertPolicy::permissive().with_depth(DepthPolicy::Truncate),
        rgb16_bytes,
        width,
        2,
        None,
    );

    // F16 <-> F32
    let f16s: Vec<F16Sample> = f32_rgb.iter().map(|&v| F16Sample::from_f32(v)).collect();
    let f16_bytes: &[u8] = bytemuck::cast_slice(&f16s);
    assert_scalar_simd_bit_exact(
        "RGBF16->RGBF32",
        ImageSpec::rgbf16_linear(),
        ImageSpec::rgbf32_linear(),
        ConvertPolicy::working_linear(),
        f16_bytes,
        width,
        0,
        None,
    );
    assert_scalar_simd_bit_exact(
        "RGBF32->RGBF16",
        ImageSpec::rgbf32_linear(),
        ImageSpec::rgbf16_linear(),
        ConvertPolicy::working_linear(),
        f32_bytes,
        width,
        0,
        None,
    );

    // Oklab SIMD vs scalar
    #[cfg(feature = "oklab")]
    {
        assert_scalar_simd_bit_exact(
            "P3U8->OKLABF32",
            p3_srgb.clone(),
            ImageSpec::oklab_f32(),
            ConvertPolicy::working_linear(),
            &rgb8,
            width,
            0,
            None,
        );
        assert_scalar_simd_bit_exact(
            "OKLABF32->P3U8",
            ImageSpec::oklab_f32(),
            p3_srgb,
            ConvertPolicy::permissive(),
            f32_bytes,
            width,
            0,
            None,
        );
    }
}

#[test]
fn parity_hlg_pq_narrow_matte_and_bitpacking() {
    let width = 69u32;
    let rgb16: Vec<u16> = (0..width as usize * 3)
        .map(|i| ((i * 2503 + 17) % 65536) as u16)
        .collect();
    let rgb16_bytes: &[u8] = bytemuck::cast_slice(&rgb16);

    // HLG BT.2100-3 display rendering (with non-zero black lift Lb = 0.05!)
    #[cfg(feature = "hdr-transfers")]
    {
        let hlg_lift = HlgRendering::new(Nits::new(1000.0), 0.05, 1.2).unwrap();
        let hlg_spec = ImageSpec::new(
            SampleLayout::RGB16,
            ColorSpace::BT2100_HLG_SCENE.with_transfer(Transfer::Hlg(HlgMode::Display(hlg_lift))),
        );
        let lin_2020 = ImageSpec::rgbf32_linear()
            .with_primaries(Primaries::Bt2020)
            .with_white(DiffuseWhite::BT2408);
        assert_scalar_simd_bit_exact(
            "HLG(lift)->LinearF32",
            hlg_spec.clone(),
            lin_2020.clone(),
            ConvertPolicy::working_linear(),
            rgb16_bytes,
            width,
            0,
            None,
        );

        // PQ U16 -> sRGB U8 via BT.2446A
        #[cfg(feature = "bt2446a")]
        assert_scalar_simd_bit_exact(
            "PQ16->sRGB8(BT2446A)",
            ImageSpec::rgb16_bt2100_pq(),
            ImageSpec::rgb8_srgb(),
            ConvertPolicy::display_sdr(DiffuseWhite::BT2408, Nits::new(100.0)),
            rgb16_bytes,
            width,
            0,
            Some(Nits::new(1000.0)),
        );
    }

    // Narrow <-> Full studio range + Matte composite + RGB565 / RGB10A2 / P010
    let rgba8: Vec<u8> = (0..width as usize * 4)
        .map(|i| ((i * 31 + 16) % 256) as u8)
        .collect();
    let narrow_spec = ImageSpec::rgb8_srgb().with_range(SignalRange::Narrow);
    assert_scalar_simd_bit_exact(
        "Narrow->Full",
        narrow_spec.clone(),
        ImageSpec::rgb8_srgb(),
        ConvertPolicy::permissive().with_range(RangePolicy::Convert),
        &rgba8[..width as usize * 3],
        width,
        0,
        None,
    );
    assert_scalar_simd_bit_exact(
        "RGBA8->RGB8(Matte)",
        ImageSpec::rgba8_srgb(),
        ImageSpec::rgb8_srgb(),
        ConvertPolicy::permissive().with_alpha(AlphaPolicy::CompositeOnto(Color::WHITE)),
        &rgba8,
        width,
        0,
        None,
    );
    let rgb565_spec = ImageSpec::new(SampleLayout::RGB565, ColorSpace::SRGB);
    let rgb10a2_spec = ImageSpec::new(SampleLayout::RGB10A2, ColorSpace::SRGB_ALPHA);
    let p010_rgb_spec = ImageSpec::new(
        SampleLayout::new(SampleType::U16, ChannelOrder::Rgb).with_packing(BitPacking::P010),
        ColorSpace::SRGB.with_alpha(AlphaMode::None),
    );
    assert_scalar_simd_bit_exact(
        "RGBA8->RGB10A2",
        ImageSpec::rgba8_srgb(),
        rgb10a2_spec,
        ConvertPolicy::permissive(),
        &rgba8,
        width,
        0,
        None,
    );
    assert_scalar_simd_bit_exact(
        "RGB8->RGB565",
        ImageSpec::rgb8_srgb(),
        rgb565_spec,
        ConvertPolicy::permissive(),
        &rgba8[..width as usize * 3],
        width,
        0,
        None,
    );
    assert_scalar_simd_bit_exact(
        "RGB16->P010RGB",
        ImageSpec::rgb16_srgb(),
        p010_rgb_spec,
        ConvertPolicy::permissive(),
        rgb16_bytes,
        width,
        0,
        None,
    );
}

#[test]
fn parity_fast_rows_wide_layouts_premul_and_all_transfers() {
    let width = 73u32; // crosses 64-px strip + SIMD widths + prime tail!
    let rgb8: Vec<u8> = (0..width as usize * 3)
        .map(|i| ((i * 43 + 5) % 256) as u8)
        .collect();
    let rgba8: Vec<u8> = (0..width as usize * 4)
        .map(|i| ((i * 59 + 13) % 256) as u8)
        .collect();
    let rgb16: Vec<u16> = (0..width as usize * 3)
        .map(|i| ((i * 1973 + 17) % 65536) as u16)
        .collect();
    let rgba16: Vec<u16> = (0..width as usize * 4)
        .map(|i| ((i * 2503 + 29) % 65536) as u16)
        .collect();
    let mut f32_rgb = Vec::with_capacity(width as usize * 3);
    for i in 0..width as usize {
        let t = (i as f32) / (width as f32 - 1.0);
        // Includes negative, dark toe (< 0.02), midtones, super-white (> 1.0), and 0.0
        f32_rgb.push(if i == 0 { 0.0 } else { -0.2 + 1.5 * t });
        f32_rgb.push((i as f32 * 0.003).min(1.0));
        f32_rgb.push(if i == 1 { f32::NAN } else { 1.2 - 1.3 * t });
    }
    let mut f32_rgba = Vec::with_capacity(width as usize * 4);
    for i in 0..width as usize {
        f32_rgba.push(f32_rgb[i * 3]);
        f32_rgba.push(f32_rgb[i * 3 + 1]);
        f32_rgba.push(f32_rgb[i * 3 + 2]);
        f32_rgba.push((i as f32) / (width as f32 - 1.0));
    }

    let rgb16_bytes: &[u8] = bytemuck::cast_slice(&rgb16);
    let rgba16_bytes: &[u8] = bytemuck::cast_slice(&rgba16);
    let f32_rgb_bytes: &[u8] = bytemuck::cast_slice(&f32_rgb);
    let f32_rgba_bytes: &[u8] = bytemuck::cast_slice(&f32_rgba);

    let pol = ConvertPolicy::permissive();
    let pol_trunc = ConvertPolicy::permissive().with_depth(DepthPolicy::Truncate);
    let pol_ext = ConvertPolicy::working_linear().with_gamut(GamutPolicy::PreserveExtended);

    // Wide layouts & integer depth
    assert_scalar_simd_bit_exact(
        "RGBA16->RGB16",
        ImageSpec::rgba16_srgb(),
        ImageSpec::rgb16_srgb(),
        pol.clone(),
        rgba16_bytes,
        width,
        0,
        None,
    );
    assert_scalar_simd_bit_exact(
        "RGB16->RGBA16",
        ImageSpec::rgb16_srgb(),
        ImageSpec::rgba16_srgb(),
        pol.clone(),
        rgb16_bytes,
        width,
        0,
        None,
    );
    assert_scalar_simd_bit_exact(
        "RGBAF32->RGBF32",
        ImageSpec::rgbaf32_linear(),
        ImageSpec::rgbf32_linear(),
        pol.clone(),
        f32_rgba_bytes,
        width,
        0,
        None,
    );
    assert_scalar_simd_bit_exact(
        "RGB16->RGB8(Round)",
        ImageSpec::rgb16_srgb(),
        ImageSpec::rgb8_srgb(),
        pol.clone(),
        rgb16_bytes,
        width,
        0,
        None,
    );
    assert_scalar_simd_bit_exact(
        "RGB16->RGB8(Trunc)",
        ImageSpec::rgb16_srgb(),
        ImageSpec::rgb8_srgb(),
        pol_trunc.clone(),
        rgb16_bytes,
        width,
        0,
        None,
    );

    // Premul U8 fast paths
    let rgba8_premul = ImageSpec::new(
        SampleLayout::RGBA8,
        ColorSpace::SRGB_ALPHA.with_alpha(AlphaMode::Premultiplied),
    );
    assert_scalar_simd_bit_exact(
        "RGBA8_Straight->Premul",
        ImageSpec::rgba8_srgb(),
        rgba8_premul.clone(),
        pol.clone(),
        &rgba8,
        width,
        0,
        None,
    );
    assert_scalar_simd_bit_exact(
        "RGBA8_Premul->Straight",
        rgba8_premul,
        ImageSpec::rgba8_srgb(),
        pol.clone(),
        &rgba8,
        width,
        0,
        None,
    );

    // Direct sRGB U8 <-> Linear F32 & F32 <-> U8/U16 depth
    let rgbf32_srgb = ImageSpec::new(SampleLayout::RGBF32, ColorSpace::SRGB);
    let rgbaf32_srgb = ImageSpec::new(SampleLayout::RGBAF32, ColorSpace::SRGB_ALPHA);
    assert_scalar_simd_bit_exact(
        "RGB8_sRGB->RGBF32_Lin",
        ImageSpec::rgb8_srgb(),
        ImageSpec::rgbf32_linear(),
        pol.clone(),
        &rgb8,
        width,
        0,
        None,
    );
    assert_scalar_simd_bit_exact(
        "RGBA8_sRGB->RGBAF32_Lin",
        ImageSpec::rgba8_srgb(),
        ImageSpec::rgbaf32_linear(),
        pol.clone(),
        &rgba8,
        width,
        0,
        None,
    );
    assert_scalar_simd_bit_exact(
        "RGBF32_Lin->RGB8_sRGB",
        ImageSpec::rgbf32_linear(),
        ImageSpec::rgb8_srgb(),
        pol.clone(),
        f32_rgb_bytes,
        width,
        0,
        None,
    );
    assert_scalar_simd_bit_exact(
        "RGBAF32_Lin->RGBA8_sRGB",
        ImageSpec::rgbaf32_linear(),
        ImageSpec::rgba8_srgb(),
        pol.clone(),
        f32_rgba_bytes,
        width,
        0,
        None,
    );
    assert_scalar_simd_bit_exact(
        "RGBF32_sRGB->RGB8_sRGB(Round)",
        rgbf32_srgb.clone(),
        ImageSpec::rgb8_srgb(),
        pol.clone(),
        f32_rgb_bytes,
        width,
        0,
        None,
    );
    assert_scalar_simd_bit_exact(
        "RGBF32_sRGB->RGB8_sRGB(Trunc)",
        rgbf32_srgb.clone(),
        ImageSpec::rgb8_srgb(),
        pol_trunc,
        f32_rgb_bytes,
        width,
        0,
        None,
    );
    assert_scalar_simd_bit_exact(
        "RGBAF32_sRGB->RGBAF32_Lin",
        rgbaf32_srgb,
        ImageSpec::rgbaf32_linear(),
        pol.clone(),
        f32_rgba_bytes,
        width,
        0,
        None,
    );

    // BT.709 & Gamma 2.2 (clamped & extended)
    let rgbf32_bt709 = rgbf32_srgb.clone().with_transfer(Transfer::Bt709);
    let rgbf32_g22 = rgbf32_srgb.clone().with_transfer(Transfer::Gamma22);
    assert_scalar_simd_bit_exact(
        "BT709_F32->Lin(Clamped)",
        rgbf32_bt709.clone(),
        ImageSpec::rgbf32_linear(),
        pol.clone(),
        f32_rgb_bytes,
        width,
        0,
        None,
    );
    assert_scalar_simd_bit_exact(
        "BT709_F32->Lin(Extended)",
        rgbf32_bt709.clone(),
        ImageSpec::rgbf32_linear(),
        pol_ext.clone(),
        f32_rgb_bytes,
        width,
        0,
        None,
    );
    assert_scalar_simd_bit_exact(
        "Lin->BT709_F32(Extended)",
        ImageSpec::rgbf32_linear(),
        rgbf32_bt709,
        pol_ext.clone(),
        f32_rgb_bytes,
        width,
        0,
        None,
    );
    assert_scalar_simd_bit_exact(
        "Gamma22_F32->Lin(Clamped)",
        rgbf32_g22.clone(),
        ImageSpec::rgbf32_linear(),
        pol.clone(),
        f32_rgb_bytes,
        width,
        0,
        None,
    );
    assert_scalar_simd_bit_exact(
        "Gamma22_F32->Lin(Extended)",
        rgbf32_g22.clone(),
        ImageSpec::rgbf32_linear(),
        pol_ext.clone(),
        f32_rgb_bytes,
        width,
        0,
        None,
    );
    assert_scalar_simd_bit_exact(
        "Lin->Gamma22_F32(Extended)",
        ImageSpec::rgbf32_linear(),
        rgbf32_g22,
        pol_ext.clone(),
        f32_rgb_bytes,
        width,
        0,
        None,
    );

    // Strip SIMD unpack/pack + gamut + gray
    let p3_f32_lin = ImageSpec::rgbf32_linear().with_primaries(Primaries::DisplayP3);
    assert_scalar_simd_bit_exact(
        "P3_LinF32->sRGB_LinF32",
        p3_f32_lin,
        ImageSpec::rgbf32_linear(),
        pol.clone(),
        f32_rgb_bytes,
        width,
        0,
        None,
    );
    assert_scalar_simd_bit_exact(
        "P3_U16->sRGB_U16",
        ImageSpec::rgb16_srgb().with_primaries(Primaries::DisplayP3),
        ImageSpec::rgb16_srgb(),
        pol.clone(),
        rgb16_bytes,
        width,
        0,
        None,
    );
    assert_scalar_simd_bit_exact(
        "RGB8_sRGB->Gray8_sRGB",
        ImageSpec::rgb8_srgb(),
        ImageSpec::gray8_srgb(),
        pol.clone(),
        &rgb8,
        width,
        0,
        None,
    );

    // PQ & HLG Scene SIMD parity
    #[cfg(feature = "hdr-transfers")]
    {
        let lin_2020 = ImageSpec::rgbf32_linear()
            .with_primaries(Primaries::Bt2020)
            .with_white(DiffuseWhite::BT2408);
        let pq_f32 = lin_2020.clone().with_transfer(Transfer::Pq);
        let hlg_scene_f32 = lin_2020
            .clone()
            .with_transfer(Transfer::Hlg(HlgMode::Scene));
        assert_scalar_simd_bit_exact(
            "PQ_U16->Lin2020_F32",
            ImageSpec::rgb16_bt2100_pq(),
            lin_2020.clone(),
            ConvertPolicy::working_linear(),
            rgb16_bytes,
            width,
            0,
            None,
        );
        assert_scalar_simd_bit_exact(
            "Lin2020_F32->PQ_U16",
            lin_2020.clone(),
            ImageSpec::rgb16_bt2100_pq(),
            ConvertPolicy::working_linear(),
            f32_rgb_bytes,
            width,
            0,
            None,
        );
        assert_scalar_simd_bit_exact(
            "PQ_F32->Lin2020_F32",
            pq_f32.clone(),
            lin_2020.clone(),
            ConvertPolicy::working_linear(),
            f32_rgb_bytes,
            width,
            0,
            None,
        );
        assert_scalar_simd_bit_exact(
            "Lin2020_F32->PQ_F32",
            lin_2020.clone(),
            pq_f32,
            ConvertPolicy::working_linear(),
            f32_rgb_bytes,
            width,
            0,
            None,
        );
        let pol_hlg = pol_ext
            .clone()
            .with_hlg_rendering(HlgRendering::REFERENCE_1000_NIT);
        assert_scalar_simd_bit_exact(
            "HLGScene_F32->Lin2020_F32",
            hlg_scene_f32.clone(),
            lin_2020.clone(),
            pol_hlg.clone(),
            f32_rgb_bytes,
            width,
            0,
            None,
        );
        assert_scalar_simd_bit_exact(
            "Lin2020_F32->HLGScene_F32",
            lin_2020,
            hlg_scene_f32,
            pol_hlg,
            f32_rgb_bytes,
            width,
            0,
            None,
        );
    }
}

#[cfg(feature = "gainmap")]
#[test]
fn parity_gainmap_applier_all_modes_zero_bit() {
    use core::num::NonZeroU32;
    use enough::Unstoppable;
    use zenpixels_convert::{
        DualLayerHdrView, GainMapApplyConfig, GainMapChannels, GainMapColorSpace, GainMapComponent,
        GainMapMetadata, GainMapResample, Orientation, PixelBuf, SignedFraction, UnsignedFraction,
        apply_gain_map_with,
    };

    let make_base = |w: u32, h: u32, spec: ImageSpec| -> PixelBuf {
        let mut buf = PixelBuf::new(w, h, spec.clone()).unwrap();
        let bpp = spec.layout.bytes_per_pixel();
        let mut v = buf.as_view_mut();
        for y in 0..h {
            let row = v.row_mut(y);
            for (i, b) in row.iter_mut().enumerate() {
                *b = (((y as usize * 73 + i * 37 + 19) * 13) % 256) as u8;
            }
            if bpp == 4 {
                // Include varied alpha including 0, 128, 255
                for x in 0..w as usize {
                    row[x * 4 + 3] = match x % 4 {
                        0 => 0,
                        1 => 128,
                        2 => 255,
                        _ => ((x * 29 + y as usize * 17) % 256) as u8,
                    };
                }
            }
        }
        buf
    };

    let make_gm_u8 = |w: u32, h: u32, ch: usize| -> PixelBuf {
        let spec = if ch == 3 {
            ImageSpec::new(
                SampleLayout::RGB8,
                ColorSpace::Untagged {
                    channels: 3,
                    alpha: AlphaMode::None,
                },
            )
        } else {
            ImageSpec::new(
                SampleLayout::GRAY8,
                ColorSpace::Untagged {
                    channels: 1,
                    alpha: AlphaMode::None,
                },
            )
        };
        let mut buf = PixelBuf::new(w, h, spec).unwrap();
        let mut v = buf.as_view_mut();
        for y in 0..h {
            for (i, b) in v.row_mut(y).iter_mut().enumerate() {
                *b = ((y as usize * 53 + i * 41 + 7) % 256) as u8;
            }
        }
        buf
    };

    let make_gm_u16 = |w: u32, h: u32| -> PixelBuf {
        let spec = ImageSpec::new(
            SampleLayout::GRAY16,
            ColorSpace::Untagged {
                channels: 1,
                alpha: AlphaMode::None,
            },
        );
        let mut buf = PixelBuf::new(w, h, spec).unwrap();
        let mut v = buf.as_view_mut();
        for y in 0..h {
            let row = v.row_mut(y);
            for x in 0..w as usize {
                let code = ((y as usize * 2503 + x * 1973 + 17) % 65536) as u16;
                row[x * 2..x * 2 + 2].copy_from_slice(&code.to_ne_bytes());
            }
        }
        buf
    };

    let comp_g1 = GainMapComponent {
        min_gain_stops: SignedFraction::new(-1, NonZeroU32::new(2).unwrap()),
        max_gain_stops: SignedFraction::new(7, NonZeroU32::new(2).unwrap()),
        gamma: UnsignedFraction::ONE,
        base_offset: SignedFraction::new(1, NonZeroU32::new(64).unwrap()),
        alternate_offset: SignedFraction::new(1, NonZeroU32::new(64).unwrap()),
    };
    let comp_g22 = GainMapComponent {
        gamma: UnsignedFraction::new(11, NonZeroU32::new(5).unwrap()),
        ..comp_g1
    };
    let comp_b = GainMapComponent {
        min_gain_stops: SignedFraction::new(0, NonZeroU32::new(1).unwrap()),
        max_gain_stops: SignedFraction::new(3, NonZeroU32::new(1).unwrap()),
        gamma: UnsignedFraction::new(3, NonZeroU32::new(2).unwrap()),
        base_offset: SignedFraction::new(1, NonZeroU32::new(32).unwrap()),
        alternate_offset: SignedFraction::new(1, NonZeroU32::new(64).unwrap()),
    };

    let meta_fwd_shared_g1 = GainMapMetadata::try_new(
        GainMapChannels::Shared(comp_g1),
        UnsignedFraction::ZERO,
        UnsignedFraction::new(3, NonZeroU32::new(1).unwrap()),
        GainMapColorSpace::Base,
    )
    .unwrap();
    let meta_fwd_shared_g22 = GainMapMetadata::try_new(
        GainMapChannels::Shared(comp_g22),
        UnsignedFraction::ZERO,
        UnsignedFraction::new(3, NonZeroU32::new(1).unwrap()),
        GainMapColorSpace::Base,
    )
    .unwrap();
    let meta_rev_perch = GainMapMetadata::try_new(
        GainMapChannels::PerChannel([comp_g1, comp_g22, comp_b]),
        UnsignedFraction::new(3, NonZeroU32::new(1).unwrap()),
        UnsignedFraction::ZERO,
        GainMapColorSpace::Base,
    )
    .unwrap();

    // Prime dimensions exercising AVX2 8-wide loops + scalar tails + edge taps + downscale!
    let base_rgb = make_base(37, 29, ImageSpec::rgb8_srgb());
    let base_rgba = make_base(37, 29, ImageSpec::rgba8_srgb());
    let base_small = make_base(19, 13, ImageSpec::rgb8_srgb());
    let gm_up_1 = make_gm_u8(11, 7, 1);
    let gm_up_3 = make_gm_u8(11, 7, 3);
    let gm_up_u16 = make_gm_u16(11, 7);
    let gm_1to1_1 = make_gm_u8(37, 29, 1);
    let gm_down_1 = make_gm_u8(41, 31, 1); // larger than 19x13 base!

    let cases = [
        (
            "up_1ch_shared_g1_rgb",
            &base_rgb,
            &gm_up_1,
            meta_fwd_shared_g1,
            2.2,
            None,
        ),
        (
            "up_1ch_shared_g22_rgba",
            &base_rgba,
            &gm_up_1,
            meta_fwd_shared_g22,
            2.2,
            None,
        ),
        (
            "up_1ch_perch_rev_rgba",
            &base_rgba,
            &gm_up_1,
            meta_rev_perch,
            0.8,
            None,
        ),
        (
            "up_3ch_perch_rev_rgb",
            &base_rgb,
            &gm_up_3,
            meta_rev_perch,
            1.2,
            None,
        ),
        (
            "up_3ch_shared_g1_rgba_f16",
            &base_rgba,
            &gm_up_3,
            meta_fwd_shared_g1,
            2.5,
            Some(ImageSpec::rgbaf16_linear()),
        ),
        (
            "up_u16_shared_g1_rgb",
            &base_rgb,
            &gm_up_u16,
            meta_fwd_shared_g1,
            1.75,
            None,
        ),
        (
            "1to1_lut_shared_g1_rgb",
            &base_rgb,
            &gm_1to1_1,
            meta_fwd_shared_g1,
            2.0,
            None,
        ),
        (
            "1to1_lut_perch_rev_rgba_srgb8",
            &base_rgba,
            &gm_1to1_1,
            meta_rev_perch,
            0.5,
            Some(ImageSpec::rgba8_srgb()),
        ),
        (
            "downscale_gm_larger_than_base",
            &base_small,
            &gm_down_1,
            meta_fwd_shared_g1,
            2.5,
            None,
        ),
    ];

    for (label, base, gm, meta, headroom, dst_spec) in cases {
        let dual = DualLayerHdrView::new(
            base.as_view(),
            gm.as_view(),
            meta,
            None,
            Orientation::Identity,
        )
        .unwrap();
        for resample in [
            GainMapResample::Nearest,
            GainMapResample::Bilinear,
            GainMapResample::Bicubic,
            GainMapResample::Mitchell,
        ] {
            let mut cfg_scalar = GainMapApplyConfig::new(headroom)
                .with_resample(resample)
                .with_policy(
                    ConvertPolicy::working_linear().with_accelerators(Accelerators::SCALAR),
                );
            let mut cfg_all = GainMapApplyConfig::new(headroom)
                .with_resample(resample)
                .with_policy(
                    ConvertPolicy::working_linear()
                        .with_accelerators(Accelerators::ALL.with_gainmap()),
                );
            if let Some(ds) = &dst_spec {
                cfg_scalar = cfg_scalar.with_dst_spec(ds.clone());
                cfg_all = cfg_all.with_dst_spec(ds.clone());
            }

            let out_s = apply_gain_map_with(&dual, &cfg_scalar, &Unstoppable).unwrap();
            let out_a = apply_gain_map_with(&dual, &cfg_all, &Unstoppable).unwrap();
            for y in 0..base.height() {
                assert_eq!(
                    out_s.as_view().row(y),
                    out_a.as_view().row(y),
                    "0-bit gainmap SCALAR vs ALL mismatch on `{label}` resample={resample:?} row={y}"
                );
            }
        }
    }
}
