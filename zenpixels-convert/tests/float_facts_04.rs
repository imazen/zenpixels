use half::f16;
use zenpixels::banded::BandedPixelBuffer;
use zenpixels::sample::FloatFacts;
use zenpixels::{
    AlphaMode, ChannelLayout, ChannelType, Cicp, Color, ColorPrimaries, Orientation, PixelBuffer,
    PixelDescriptor, PixelFormat, TransferFunction,
};
use zenpixels_convert::{
    AlphaPolicy, ConversionSpec, ConvertError, ConvertOptions, ConvertPlan, DepthPolicy,
    EncodeReady, GamutPolicy, OutputPlan, OutputProfile, OutputTarget, PixelBufferConvertExt,
    PixelSliceConvertExt,
};

const RGB8: PixelDescriptor = PixelDescriptor::RGB8_SRGB;
const RGBF32_LIN: PixelDescriptor = PixelDescriptor::RGBF32_LINEAR;
const RGBAF32_LIN: PixelDescriptor = PixelDescriptor::RGBAF32_LINEAR;

fn oklab_desc(has_alpha: bool, primaries: ColorPrimaries) -> PixelDescriptor {
    PixelDescriptor::new(
        ChannelType::F32,
        if has_alpha {
            ChannelLayout::OklabA
        } else {
            ChannelLayout::Oklab
        },
        has_alpha.then_some(AlphaMode::Straight),
        TransferFunction::Linear,
    )
    .with_primaries(primaries)
}

fn oklch_desc(primaries: ColorPrimaries) -> PixelDescriptor {
    PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Oklch,
        None,
        TransferFunction::Unknown,
    )
    .with_primaries(primaries)
}

fn assert_bits_eq(context: &str, a: &[u8], b: &[u8]) {
    assert_eq!(a.len(), b.len(), "{context}: byte length mismatch");
    let fa: &[f32] = bytemuck::cast_slice(a);
    let fb: &[f32] = bytemuck::cast_slice(b);
    for (idx, (&va, &vb)) in fa.iter().zip(fb.iter()).enumerate() {
        assert_eq!(
            va.to_bits(),
            vb.to_bits(),
            "{context}: bit mismatch at float index {idx}: untrusted={va:?} ({:#010x}) vs specialized={vb:?} ({:#010x})",
            va.to_bits(),
            vb.to_bits()
        );
    }
}

#[test]
fn automatic_float_facts_propagation_and_orthogonal_subnormals() {
    // 1. RGB8_SRGB -> RGBF32_LINEAR proves unit non-negative normal facts automatically.
    let mut src8 = PixelBuffer::try_new(17, 3, RGB8).unwrap();
    for (i, b) in src8.as_slice_mut().row_mut(0).iter_mut().enumerate() {
        *b = ((i * 37) & 0xFF) as u8;
    }
    let lin = src8.convert_to(RGBF32_LIN).unwrap();
    let expected_u8_lin = FloatFacts::NO_NAN
        | FloatFacts::NO_INFINITE
        | FloatFacts::NO_SUBNORMALS
        | FloatFacts::NON_NEGATIVE
        | FloatFacts::AT_MOST_ONE;
    assert_eq!(lin.float_facts(), expected_u8_lin);
    assert!(lin.float_facts().holds_for(&lin.as_slice()).unwrap());

    // 2. F16 signed finite values -> F32 proves NO_NAN | NO_INFINITE | NO_SUBNORMALS
    //    WITHOUT claiming NON_NEGATIVE (subnormals are orthogonal to positivity).
    let rgbf16 = PixelDescriptor::new(
        ChannelType::F16,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
    );
    let f16_bits = [
        f16::from_f32(-0.5).to_bits(),
        f16::from_f32(-0.0).to_bits(),
        f16::from_f32(1e-7).to_bits(), // F16 subnormal -> normal F32!
        f16::from_f32(0.25).to_bits(),
        f16::from_f32(1.5).to_bits(),
        f16::from_f32(-2.0).to_bits(),
    ];
    let mut src_f16 =
        PixelBuffer::from_vec(bytemuck::cast_slice(&f16_bits).to_vec(), 2, 1, rgbf16).unwrap();
    let f16_measured = FloatFacts::measure(&src_f16.as_slice()).unwrap();
    assert_eq!(f16_measured, FloatFacts::NO_NAN | FloatFacts::NO_INFINITE);
    assert!(!f16_measured.contains(FloatFacts::NO_SUBNORMALS));
    assert!(!f16_measured.contains(FloatFacts::NON_NEGATIVE));
    src_f16.set_float_facts(f16_measured).unwrap();

    let promoted = src_f16.convert_to(RGBF32_LIN).unwrap();
    assert_eq!(
        promoted.float_facts(),
        FloatFacts::NO_NAN | FloatFacts::NO_INFINITE | FloatFacts::NO_SUBNORMALS
    );
    assert!(!promoted.float_facts().contains(FloatFacts::NON_NEGATIVE));
    assert!(
        promoted
            .float_facts()
            .holds_for(&promoted.as_slice())
            .unwrap()
    );

    // 3. BandedPixelBuffer convert_bands and oriented_into preserve/stamp FloatFacts.
    let mut banded_src = BandedPixelBuffer::try_new(19, 11, RGB8, 19 * 3 * 4).unwrap();
    for y in 0..11 {
        let row = banded_src.row_mut(y).unwrap();
        for (i, b) in row.iter_mut().enumerate() {
            *b = ((i + y as usize * 13) & 0xFF) as u8;
        }
    }
    let mut banded_dst = BandedPixelBuffer::try_new(19, 11, RGBF32_LIN, 19 * 12 * 3).unwrap();
    let spec = ConversionSpec::new(ConvertOptions::retain_channels());
    let plan = ConvertPlan::build(RGB8.into(), RGBF32_LIN.into(), &spec).unwrap();
    assert_eq!(plan.output_float_facts(), expected_u8_lin);
    let mut worker = plan.prepare(19).unwrap();
    worker.convert_bands(&banded_src, &mut banded_dst).unwrap();
    assert_eq!(banded_dst.float_facts(), expected_u8_lin);
    assert_eq!(
        FloatFacts::measure_banded(&banded_dst).unwrap() & expected_u8_lin,
        expected_u8_lin
    );

    let mut rotated = BandedPixelBuffer::try_new(11, 19, RGBF32_LIN, 11 * 12 * 5).unwrap();
    zenpixels_convert::orient_into(&banded_dst, Orientation::Rotate90, &mut rotated).unwrap();
    assert_eq!(rotated.float_facts(), expected_u8_lin);

    // 4. Contiguous oriented and EncodeReady preserve/stamp FloatFacts.
    let oriented_contig = lin.as_slice().oriented(Orientation::Rotate270).unwrap();
    assert_eq!(oriented_contig.float_facts(), expected_u8_lin);

    let ready_plan = OutputPlan::build(
        RGB8.into(),
        OutputTarget::new(
            OutputProfile::Named(Cicp::new(1, 8, 0, true)),
            PixelFormat::RgbF32,
        ),
        &ConversionSpec::new(ConvertOptions::retain_channels()),
    )
    .unwrap();
    let ready = EncodeReady::from_owned(src8, ready_plan).unwrap();
    assert_eq!(ready.pixels().float_facts(), expected_u8_lin);
}

#[test]
fn float_facts_contract_enforcement_across_plan_worker_bands_and_output_plan() {
    let required = FloatFacts::NO_NAN | FloatFacts::NO_INFINITE | FloatFacts::NON_NEGATIVE;
    let opts = ConvertOptions::retain_channels().with_gamut_policy(GamutPolicy::PreserveExtended);
    let spec = ConversionSpec::new(opts).with_float_facts(required);

    // 1. Integer source rejects non-empty FloatFacts at compile time.
    let err_int = ConvertPlan::build(RGB8.into(), RGBF32_LIN.into(), &spec).unwrap_err();
    assert!(matches!(
        err_int.error(),
        ConvertError::FloatFactsUnsatisfied {
            required: r,
            actual: FloatFacts::NONE
        } if *r == required
    ));

    // 2. Float plan records input_float_facts and rejects untrusted PixelSlice in convert_rows.
    //    Unbounded F32 (without AT_MOST_ONE) soundly refrains from claiming finite Oklab
    //    output (f32::MAX through RGB->LMS could overflow), whereas bounded AT_MOST_ONE proves it.
    let oklab = oklab_desc(false, ColorPrimaries::Bt709);
    let unbounded_plan = ConvertPlan::build(RGBF32_LIN.into(), oklab.into(), &spec).unwrap();
    assert_eq!(unbounded_plan.output_float_facts(), FloatFacts::NONE);

    let bounded_required = required | FloatFacts::AT_MOST_ONE;
    let bounded_spec = ConversionSpec::new(opts).with_float_facts(bounded_required);
    let plan = ConvertPlan::build(RGBF32_LIN.into(), oklab.into(), &bounded_spec).unwrap();
    assert_eq!(plan.input_float_facts(), bounded_required);
    assert!(
        plan.output_float_facts()
            .contains(FloatFacts::NO_NAN | FloatFacts::NO_INFINITE)
    );

    let mut worker = plan.try_clone().unwrap().prepare(4).unwrap();
    assert_eq!(worker.input_float_facts(), bounded_required);

    let mut src_buf = PixelBuffer::from_vec(
        bytemuck::cast_slice(&[0.2f32; 12]).to_vec(),
        4,
        1,
        RGBF32_LIN,
    )
    .unwrap();
    let mut dst_buf = PixelBuffer::try_new(4, 1, oklab).unwrap();

    let err_rows = worker
        .convert_rows(src_buf.as_slice(), dst_buf.as_slice_mut())
        .unwrap_err();
    assert!(matches!(
        err_rows.error(),
        ConvertError::FloatFactsUnsatisfied {
            required: r,
            actual: FloatFacts::NONE
        } if *r == bounded_required
    ));

    // Measuring and stamping unlocks convert_rows.
    let measured = FloatFacts::measure(&src_buf.as_slice()).unwrap();
    src_buf.set_float_facts(measured).unwrap();
    worker
        .convert_rows(src_buf.as_slice(), dst_buf.as_slice_mut())
        .unwrap();

    // 3. BandedPixelBuffer convert_bands enforces input_float_facts too.
    let mut b_src = BandedPixelBuffer::try_new(4, 2, RGBF32_LIN, 4 * 12).unwrap();
    let mut b_dst = BandedPixelBuffer::try_new(4, 2, oklab, 4 * 12).unwrap();
    assert!(matches!(
        worker
            .convert_bands(&b_src, &mut b_dst)
            .unwrap_err()
            .error(),
        ConvertError::FloatFactsUnsatisfied { .. }
    ));
    let b_measured = FloatFacts::measure_banded(&b_src).unwrap();
    b_src.set_float_facts(b_measured).unwrap();
    worker.convert_bands(&b_src, &mut b_dst).unwrap();
    assert_eq!(b_dst.float_facts(), worker.output_float_facts());

    // 4. ConvertPlan::then enforces intermediate FloatFacts compatibility.
    let untrusted_spec = ConversionSpec::new(opts);
    let untrusted_id =
        ConvertPlan::build(RGBF32_LIN.into(), RGBF32_LIN.into(), &untrusted_spec).unwrap();
    assert!(matches!(
        untrusted_id
            .then(plan.try_clone().unwrap())
            .unwrap_err()
            .error(),
        ConvertError::FloatFactsUnsatisfied { .. }
    ));

    let u8_to_lin = ConvertPlan::build(RGB8.into(), RGBF32_LIN.into(), &untrusted_spec).unwrap();
    let chained = u8_to_lin.then(plan).unwrap();
    assert_eq!(chained.input_float_facts(), FloatFacts::NONE);
    assert!(
        chained
            .output_float_facts()
            .contains(FloatFacts::NO_NAN | FloatFacts::NO_INFINITE)
    );

    // 5. OutputPlan enforces input_float_facts on convert_rows and borrow_rows.
    let mut out_plan = OutputPlan::build(
        RGBF32_LIN.into(),
        OutputTarget::new(OutputProfile::Named(Cicp::SRGB), PixelFormat::Rgb8),
        &ConversionSpec::new(
            ConvertOptions::retain_channels().with_depth_policy(DepthPolicy::Round),
        )
        .with_float_facts(required),
    )
    .unwrap();
    assert_eq!(out_plan.input_float_facts(), required);
    out_plan.prepare(4).unwrap();

    let untrusted_f32 = PixelBuffer::from_vec(
        bytemuck::cast_slice(&[0.5f32; 12]).to_vec(),
        4,
        1,
        RGBF32_LIN,
    )
    .unwrap();
    let mut out_u8 = PixelBuffer::try_new(4, 1, RGB8).unwrap();
    assert!(matches!(
        out_plan
            .convert_rows(untrusted_f32.as_slice(), out_u8.as_slice_mut())
            .unwrap_err()
            .error(),
        ConvertError::FloatFactsUnsatisfied { .. }
    ));
    out_plan
        .convert_rows(src_buf.as_slice(), out_u8.as_slice_mut())
        .unwrap();

    let id_out_plan = OutputPlan::build(
        RGBF32_LIN.into(),
        OutputTarget::new(OutputProfile::SameAsOrigin, PixelFormat::RgbF32),
        &ConversionSpec::new(ConvertOptions::retain_channels()).with_float_facts(required),
    )
    .unwrap();
    assert!(matches!(
        id_out_plan
            .borrow_rows(untrusted_f32.as_slice())
            .unwrap_err()
            .error(),
        ConvertError::FloatFactsUnsatisfied { .. }
    ));
    let borrowed = id_out_plan
        .borrow_rows(src_buf.as_slice())
        .unwrap()
        .unwrap();
    assert!(borrowed.float_facts().contains(required));
}

#[test]
fn oklab_const_specializations_are_bit_identical_to_untrusted_across_widths_and_primaries() {
    let opts = ConvertOptions::retain_channels().with_gamut_policy(GamutPolicy::PreserveExtended);
    let widths = [1usize, 7, 16, 17, 63, 65];
    let primaries_list = [
        ColorPrimaries::Bt709,
        ColorPrimaries::DisplayP3,
        ColorPrimaries::AdobeRgb,
        ColorPrimaries::Bt2020,
    ];

    let all_positive_unit = FloatFacts::NO_NAN
        | FloatFacts::NO_INFINITE
        | FloatFacts::NO_SUBNORMALS
        | FloatFacts::POSITIVE
        | FloatFacts::AT_MOST_ONE;
    let nonneg_unit_nosub = FloatFacts::NO_NAN
        | FloatFacts::NO_INFINITE
        | FloatFacts::NO_SUBNORMALS
        | FloatFacts::NON_NEGATIVE
        | FloatFacts::AT_MOST_ONE;
    let nonneg_unit_with_sub = FloatFacts::NO_NAN
        | FloatFacts::NO_INFINITE
        | FloatFacts::NON_NEGATIVE
        | FloatFacts::AT_MOST_ONE;
    let signed_unit_nosub = FloatFacts::NO_NAN
        | FloatFacts::NO_INFINITE
        | FloatFacts::NO_SUBNORMALS
        | FloatFacts::AT_MOST_ONE;
    let finite_only = FloatFacts::NO_NAN | FloatFacts::NO_INFINITE;

    for primaries in primaries_list {
        for has_alpha in [false, true] {
            let ch = if has_alpha { 4 } else { 3 };
            let src_desc =
                if has_alpha { RGBAF32_LIN } else { RGBF32_LIN }.with_primaries(primaries);
            let dst_desc = oklab_desc(has_alpha, primaries);

            let untrusted_plan =
                ConvertPlan::build(src_desc.into(), dst_desc.into(), &ConversionSpec::new(opts))
                    .unwrap();

            for &w in &widths {
                let mut w_untrusted = untrusted_plan
                    .try_clone()
                    .unwrap()
                    .prepare(w as u32)
                    .unwrap();

                // Domain A: Strictly positive normal unit values -> tests FinitePosNoSub + subsets.
                let mut dom_pos = Vec::with_capacity(w * ch);
                for i in 0..(w * ch) {
                    let v = 0.001 + 0.998 * ((i as f32 * 0.6180339) % 1.0);
                    dom_pos.push(v);
                }
                let src_bytes: &[u8] = bytemuck::cast_slice(&dom_pos);
                let mut ref_out = vec![0u8; w * ch * 4];
                w_untrusted
                    .try_convert_row(src_bytes, &mut ref_out, w as u32)
                    .unwrap();

                for facts in [
                    finite_only,
                    signed_unit_nosub,
                    nonneg_unit_with_sub,
                    nonneg_unit_nosub,
                    all_positive_unit,
                ] {
                    let spec = ConversionSpec::new(opts).with_float_facts(facts);
                    let mut worker = ConvertPlan::build(src_desc.into(), dst_desc.into(), &spec)
                        .unwrap()
                        .prepare(w as u32)
                        .unwrap();
                    let mut spec_out = vec![0u8; w * ch * 4];
                    worker
                        .try_convert_row(src_bytes, &mut spec_out, w as u32)
                        .unwrap();
                    assert_bits_eq(
                        &format!("oklab pos {primaries:?} alpha={has_alpha} w={w} facts={facts:?}"),
                        &ref_out,
                        &spec_out,
                    );
                }

                // Domain B: Non-negative with +0.0 and positive subnormals -> tests FiniteNonNeg & Finite.
                let mut dom_sub = dom_pos.clone();
                dom_sub[0] = 0.0;
                if dom_sub.len() > ch {
                    dom_sub[1] = f32::from_bits(1); // smallest positive subnormal
                    dom_sub[2] = f32::from_bits(0x007f_ffff); // largest positive subnormal
                }
                let sub_bytes: &[u8] = bytemuck::cast_slice(&dom_sub);
                w_untrusted
                    .try_convert_row(sub_bytes, &mut ref_out, w as u32)
                    .unwrap();
                for facts in [finite_only, nonneg_unit_with_sub] {
                    let spec = ConversionSpec::new(opts).with_float_facts(facts);
                    let mut worker = ConvertPlan::build(src_desc.into(), dst_desc.into(), &spec)
                        .unwrap()
                        .prepare(w as u32)
                        .unwrap();
                    let mut spec_out = vec![0u8; w * ch * 4];
                    worker
                        .try_convert_row(sub_bytes, &mut spec_out, w as u32)
                        .unwrap();
                    assert_bits_eq(
                        &format!("oklab sub {primaries:?} alpha={has_alpha} w={w} facts={facts:?}"),
                        &ref_out,
                        &spec_out,
                    );
                }

                // Domain C: Signed out-of-gamut normals including -0.0 -> tests FiniteNoSub & Finite.
                let mut dom_signed = Vec::with_capacity(w * ch);
                for i in 0..(w * ch) {
                    let v = match i % 5 {
                        0 => -0.0f32,
                        1 => -0.35f32,
                        2 => 0.0f32,
                        3 => 0.42f32,
                        _ => -0.002f32,
                    };
                    dom_signed.push(v);
                }
                let signed_bytes: &[u8] = bytemuck::cast_slice(&dom_signed);
                w_untrusted
                    .try_convert_row(signed_bytes, &mut ref_out, w as u32)
                    .unwrap();
                for facts in [finite_only, signed_unit_nosub] {
                    let spec = ConversionSpec::new(opts).with_float_facts(facts);
                    let mut worker = ConvertPlan::build(src_desc.into(), dst_desc.into(), &spec)
                        .unwrap()
                        .prepare(w as u32)
                        .unwrap();
                    let mut spec_out = vec![0u8; w * ch * 4];
                    worker
                        .try_convert_row(signed_bytes, &mut spec_out, w as u32)
                        .unwrap();
                    assert_bits_eq(
                        &format!(
                            "oklab signed {primaries:?} alpha={has_alpha} w={w} facts={facts:?}"
                        ),
                        &ref_out,
                        &spec_out,
                    );
                }
            }
        }
    }
}

#[test]
fn extended_gamma22_pq_truncate_matte_perceptual_and_tonemap_bit_exact_parity() {
    let w = 35usize;
    let opts = ConvertOptions::retain_channels().with_gamut_policy(GamutPolicy::PreserveExtended);

    // 1. Extended Gamma 2.2 EOTF & OETF across unit [0, 1], HDR [0, 8], and signed [-2.5, 4.5].
    let g22 = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Gamma22,
    );
    for (from, to) in [(g22, RGBF32_LIN), (RGBF32_LIN, g22)] {
        let mut ref_w = ConvertPlan::build(from.into(), to.into(), &ConversionSpec::new(opts))
            .unwrap()
            .prepare(w as u32)
            .unwrap();

        // Unit non-negative [0, 1]
        let unit_vals: Vec<f32> = (0..(w * 3))
            .map(|i| {
                if i == 0 {
                    0.0
                } else {
                    (i as f32) / ((w * 3) as f32)
                }
            })
            .collect();
        let unit_facts = FloatFacts::NO_NAN
            | FloatFacts::NO_INFINITE
            | FloatFacts::NO_SUBNORMALS
            | FloatFacts::NON_NEGATIVE
            | FloatFacts::AT_MOST_ONE;
        let mut ref_out = vec![0u8; w * 12];
        let mut spec_out = vec![0u8; w * 12];
        ref_w
            .try_convert_row(bytemuck::cast_slice(&unit_vals), &mut ref_out, w as u32)
            .unwrap();
        ConvertPlan::build(
            from.into(),
            to.into(),
            &ConversionSpec::new(opts).with_float_facts(unit_facts),
        )
        .unwrap()
        .prepare(w as u32)
        .unwrap()
        .try_convert_row(bytemuck::cast_slice(&unit_vals), &mut spec_out, w as u32)
        .unwrap();
        assert_bits_eq(
            &format!("gamma22 unit {from:?}->{to:?}"),
            &ref_out,
            &spec_out,
        );

        // HDR non-negative [0, 8]
        let hdr_vals: Vec<f32> = (0..(w * 3)).map(|i| (i as f32) * 0.075).collect();
        let hdr_facts = FloatFacts::NO_NAN | FloatFacts::NO_INFINITE | FloatFacts::NON_NEGATIVE;
        ref_w
            .try_convert_row(bytemuck::cast_slice(&hdr_vals), &mut ref_out, w as u32)
            .unwrap();
        ConvertPlan::build(
            from.into(),
            to.into(),
            &ConversionSpec::new(opts).with_float_facts(hdr_facts),
        )
        .unwrap()
        .prepare(w as u32)
        .unwrap()
        .try_convert_row(bytemuck::cast_slice(&hdr_vals), &mut spec_out, w as u32)
        .unwrap();
        assert_bits_eq(
            &format!("gamma22 hdr {from:?}->{to:?}"),
            &ref_out,
            &spec_out,
        );

        // Signed finite [-2.5, 4.5] including -0.0
        let signed_vals: Vec<f32> = (0..(w * 3))
            .map(|i| {
                if i == 1 {
                    -0.0f32
                } else {
                    -2.5 + (i as f32) * 0.065
                }
            })
            .collect();
        let signed_facts = FloatFacts::NO_NAN | FloatFacts::NO_INFINITE;
        ref_w
            .try_convert_row(bytemuck::cast_slice(&signed_vals), &mut ref_out, w as u32)
            .unwrap();
        ConvertPlan::build(
            from.into(),
            to.into(),
            &ConversionSpec::new(opts).with_float_facts(signed_facts),
        )
        .unwrap()
        .prepare(w as u32)
        .unwrap()
        .try_convert_row(bytemuck::cast_slice(&signed_vals), &mut spec_out, w as u32)
        .unwrap();
        assert_bits_eq(
            &format!("gamma22 signed {from:?}->{to:?}"),
            &ref_out,
            &spec_out,
        );
    }

    // 2. PQ F32 <-> Linear F32 bit-exact parity.
    let pq_f32 = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Pq,
    )
    .with_primaries(ColorPrimaries::Bt2020);
    let lin_2020 = RGBF32_LIN.with_primaries(ColorPrimaries::Bt2020);
    let unit_vals: Vec<f32> = (0..(w * 3))
        .map(|i| (i as f32) / ((w * 3 - 1) as f32))
        .collect();
    let unit_facts = FloatFacts::NO_NAN
        | FloatFacts::NO_INFINITE
        | FloatFacts::NON_NEGATIVE
        | FloatFacts::AT_MOST_ONE;
    let pq_opts = opts.with_sdr_white(zenpixels::hdr::DiffuseWhite::BT2408);
    for (from, to) in [(pq_f32, lin_2020), (lin_2020, pq_f32)] {
        let mut ref_out = vec![0u8; w * 12];
        let mut spec_out = vec![0u8; w * 12];
        ConvertPlan::build(from.into(), to.into(), &ConversionSpec::new(pq_opts))
            .unwrap()
            .prepare(w as u32)
            .unwrap()
            .try_convert_row(bytemuck::cast_slice(&unit_vals), &mut ref_out, w as u32)
            .unwrap();
        ConvertPlan::build(
            from.into(),
            to.into(),
            &ConversionSpec::new(pq_opts).with_float_facts(unit_facts),
        )
        .unwrap()
        .prepare(w as u32)
        .unwrap()
        .try_convert_row(bytemuck::cast_slice(&unit_vals), &mut spec_out, w as u32)
        .unwrap();
        assert_bits_eq(&format!("pq {from:?}->{to:?}"), &ref_out, &spec_out);
    }

    // 3. DepthPolicy::Truncate (F32 -> U8 and F32 -> U16) byte-exact parity.
    let trunc_opts = opts.with_depth_policy(DepthPolicy::Truncate);
    for to in [
        PixelDescriptor::new(
            ChannelType::U8,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
        ),
        PixelDescriptor::new(
            ChannelType::U16,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
        ),
    ] {
        let bpp = to.bytes_per_pixel();
        let mut ref_out = vec![0u8; w * bpp];
        let mut spec_out = vec![0u8; w * bpp];
        ConvertPlan::build(
            RGBF32_LIN.into(),
            to.into(),
            &ConversionSpec::new(trunc_opts),
        )
        .unwrap()
        .prepare(w as u32)
        .unwrap()
        .try_convert_row(bytemuck::cast_slice(&unit_vals), &mut ref_out, w as u32)
        .unwrap();
        ConvertPlan::build(
            RGBF32_LIN.into(),
            to.into(),
            &ConversionSpec::new(trunc_opts).with_float_facts(unit_facts),
        )
        .unwrap()
        .prepare(w as u32)
        .unwrap()
        .try_convert_row(bytemuck::cast_slice(&unit_vals), &mut spec_out, w as u32)
        .unwrap();
        assert_eq!(ref_out, spec_out, "truncate mismatch for {to:?}");
    }

    // 4. MatteComposite on RGBAF32_LINEAR -> RGBF32_LINEAR bit-exact parity.
    let matte_opts = opts.with_alpha_policy(AlphaPolicy::CompositeOnto {
        background: Color::srgb8(240, 128, 16),
        reference_white: None,
    });
    let rgba_vals: Vec<f32> = (0..(w * 4))
        .map(|i| (i as f32) / ((w * 4 - 1) as f32))
        .collect();
    let mut ref_matte = vec![0u8; w * 12];
    let mut spec_matte = vec![0u8; w * 12];
    ConvertPlan::build(
        RGBAF32_LIN.into(),
        RGBF32_LIN.into(),
        &ConversionSpec::new(matte_opts),
    )
    .unwrap()
    .prepare(w as u32)
    .unwrap()
    .try_convert_row(bytemuck::cast_slice(&rgba_vals), &mut ref_matte, w as u32)
    .unwrap();
    ConvertPlan::build(
        RGBAF32_LIN.into(),
        RGBF32_LIN.into(),
        &ConversionSpec::new(matte_opts).with_float_facts(unit_facts),
    )
    .unwrap()
    .prepare(w as u32)
    .unwrap()
    .try_convert_row(bytemuck::cast_slice(&rgba_vals), &mut spec_matte, w as u32)
    .unwrap();
    assert_bits_eq("matte composite", &ref_matte, &spec_matte);

    // 5. Perceptual spaces: Hsl, Hsv, Lch, Oklch forward & inverse bit-exact parity.
    let srgb_f32 = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Srgb,
    );
    let hsl = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Hsl,
        None,
        TransferFunction::Srgb,
    );
    let hsv = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Hsv,
        None,
        TransferFunction::Srgb,
    );
    let lch = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Lch,
        None,
        TransferFunction::Unknown,
    );
    let oklch = oklch_desc(ColorPrimaries::Bt709);

    for (from, to) in [
        (srgb_f32, hsl),
        (srgb_f32, hsv),
        (RGBF32_LIN, lch),
        (RGBF32_LIN, oklch),
    ] {
        let mut ref_fwd = vec![0u8; w * 12];
        let mut spec_fwd = vec![0u8; w * 12];
        ConvertPlan::build(from.into(), to.into(), &ConversionSpec::new(opts))
            .unwrap()
            .prepare(w as u32)
            .unwrap()
            .try_convert_row(bytemuck::cast_slice(&unit_vals), &mut ref_fwd, w as u32)
            .unwrap();
        let fwd_plan = ConvertPlan::build(
            from.into(),
            to.into(),
            &ConversionSpec::new(opts).with_float_facts(unit_facts),
        )
        .unwrap();
        let inv_facts = fwd_plan.output_float_facts();
        assert!(inv_facts.contains(FloatFacts::NO_NAN | FloatFacts::NO_INFINITE));
        fwd_plan
            .prepare(w as u32)
            .unwrap()
            .try_convert_row(bytemuck::cast_slice(&unit_vals), &mut spec_fwd, w as u32)
            .unwrap();
        assert_bits_eq(
            &format!("perceptual fwd {from:?}->{to:?}"),
            &ref_fwd,
            &spec_fwd,
        );

        let mut ref_inv = vec![0u8; w * 12];
        let mut spec_inv = vec![0u8; w * 12];
        ConvertPlan::build(to.into(), from.into(), &ConversionSpec::new(opts))
            .unwrap()
            .prepare(w as u32)
            .unwrap()
            .try_convert_row(&ref_fwd, &mut ref_inv, w as u32)
            .unwrap();
        ConvertPlan::build(
            to.into(),
            from.into(),
            &ConversionSpec::new(opts).with_float_facts(inv_facts),
        )
        .unwrap()
        .prepare(w as u32)
        .unwrap()
        .try_convert_row(&spec_fwd, &mut spec_inv, w as u32)
        .unwrap();
        assert_bits_eq(
            &format!("perceptual inv {to:?}->{from:?}"),
            &ref_inv,
            &spec_inv,
        );
    }

    // 6. ToneMap (hdr-experimental) bit-exact parity on 3ch and 4ch linear HDR float.
    #[cfg(feature = "hdr-experimental")]
    {
        let hdr = zenpixels_convert::HdrConfig::for_source_peak(1000.0);
        let hdr_facts = FloatFacts::NO_NAN | FloatFacts::NO_INFINITE | FloatFacts::NON_NEGATIVE;
        let rgba_2020 = RGBAF32_LIN.with_primaries(ColorPrimaries::Bt2020);
        for (src_desc, dst_desc, ch) in
            [(lin_2020, lin_2020, 3usize), (rgba_2020, rgba_2020, 4usize)]
        {
            let hdr_in: Vec<f32> = (0..(w * ch))
                .map(|i| {
                    if ch == 4 && i % 4 == 3 {
                        0.8
                    } else {
                        (i as f32) * 0.05
                    }
                })
                .collect();
            let mut ref_tm = vec![0u8; w * ch * 4];
            let mut spec_tm = vec![0u8; w * ch * 4];
            ConvertPlan::build(
                src_desc.into(),
                dst_desc.into(),
                &ConversionSpec::new(opts).with_hdr(&hdr),
            )
            .unwrap()
            .prepare(w as u32)
            .unwrap()
            .try_convert_row(bytemuck::cast_slice(&hdr_in), &mut ref_tm, w as u32)
            .unwrap();
            let tm_plan = ConvertPlan::build(
                src_desc.into(),
                dst_desc.into(),
                &ConversionSpec::new(opts)
                    .with_hdr(&hdr)
                    .with_float_facts(hdr_facts),
            )
            .unwrap();
            if ch == 3 {
                assert!(tm_plan.output_float_facts().contains(
                    FloatFacts::NO_NAN | FloatFacts::NO_INFINITE | FloatFacts::AT_MOST_ONE
                ));
            } else {
                // On 4ch RGBA, ToneMap clamps RGB to <= 1.0 while alpha passes through
                // unchanged; since hdr_facts lacked AT_MOST_ONE, output soundly omits AT_MOST_ONE!
                assert_eq!(
                    tm_plan.output_float_facts(),
                    FloatFacts::NO_NAN | FloatFacts::NO_INFINITE
                );
            }
            tm_plan
                .prepare(w as u32)
                .unwrap()
                .try_convert_row(bytemuck::cast_slice(&hdr_in), &mut spec_tm, w as u32)
                .unwrap();
            assert_bits_eq(&format!("tonemap ch={ch}"), &ref_tm, &spec_tm);
        }
    }
}

#[test]
fn unbounded_hlg_srgb_extended_and_bt709_do_not_overclaim_output_float_facts() {
    let ext_opts =
        ConvertOptions::retain_channels().with_gamut_policy(GamutPolicy::PreserveExtended);
    let unbounded_finite_nonneg =
        FloatFacts::NO_NAN | FloatFacts::NO_INFINITE | FloatFacts::NON_NEGATIVE;
    let unbounded_no_nan = FloatFacts::NO_NAN;
    let unit_facts = FloatFacts::NO_NAN
        | FloatFacts::NO_INFINITE
        | FloatFacts::NON_NEGATIVE
        | FloatFacts::AT_MOST_ONE;

    // 1. HLG rendering (`HlgRender`) does NOT clamp above 1.0 (`2.0`, `17.0`) and soundly
    //    reports `FloatFacts::NONE` on unbounded inputs.
    let hlg_opts = ext_opts
        .with_sdr_white(zenpixels::hdr::DiffuseWhite::new(1000.0))
        .with_hlg_rendering(zenpixels::hdr::HlgRendering::REFERENCE_1000_NIT);
    let hlg_2020 = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Hlg,
    )
    .with_primaries(ColorPrimaries::Bt2020);
    let lin_2020 = RGBF32_LIN.with_primaries(ColorPrimaries::Bt2020);

    for (from, to) in [(hlg_2020, lin_2020), (lin_2020, hlg_2020)] {
        for facts in [unbounded_no_nan, unbounded_finite_nonneg] {
            let plan = ConvertPlan::build(
                from.into(),
                to.into(),
                &ConversionSpec::new(hlg_opts).with_float_facts(facts),
            )
            .unwrap();
            assert_eq!(
                plan.output_float_facts(),
                FloatFacts::NONE,
                "HLG {from:?}->{to:?} with {facts:?} must not claim bounded/finite output"
            );
        }
    }

    // Verify actual witness values (`2.0`, `17.0`) satisfy `plan.output_float_facts()` on `PixelSlice`.
    let mut hlg_src = PixelBuffer::from_vec(
        bytemuck::cast_slice(&[2.0f32, 17.0, 0.5]).to_vec(),
        1,
        1,
        hlg_2020,
    )
    .unwrap();
    hlg_src.set_float_facts(unbounded_finite_nonneg).unwrap();
    let hlg_plan = ConvertPlan::build(
        hlg_2020.into(),
        lin_2020.into(),
        &ConversionSpec::new(hlg_opts).with_float_facts(unbounded_finite_nonneg),
    )
    .unwrap();
    let hlg_out_facts = hlg_plan.output_float_facts();
    let mut hlg_dst = PixelBuffer::try_new(1, 1, lin_2020).unwrap();
    hlg_plan
        .prepare(1)
        .unwrap()
        .convert_rows(hlg_src.as_slice(), hlg_dst.as_slice_mut())
        .unwrap();
    hlg_dst.set_float_facts(hlg_out_facts).unwrap();
    assert!(
        hlg_dst
            .float_facts()
            .holds_for(&hlg_dst.as_slice())
            .unwrap()
    );

    // 2. sRGB extended & BT.709 rational polynomials overflow to `Inf/Inf = NaN` around `1e20` / `+Inf`.
    let srgb_f32 = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Srgb,
    );
    let bt709_f32 = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Bt709,
    );
    for (from, to) in [
        (srgb_f32, RGBF32_LIN),
        (RGBF32_LIN, srgb_f32),
        (bt709_f32, RGBF32_LIN),
        (RGBF32_LIN, bt709_f32),
    ] {
        for facts in [unbounded_no_nan, unbounded_finite_nonneg] {
            let plan = ConvertPlan::build(
                from.into(),
                to.into(),
                &ConversionSpec::new(ext_opts).with_float_facts(facts),
            )
            .unwrap();
            assert_eq!(
                plan.output_float_facts(),
                FloatFacts::NONE,
                "rational {from:?}->{to:?} with {facts:?} must not claim NO_NAN/NO_INFINITE"
            );
        }
        // Unit-capped inputs DO prove unit output.
        let unit_plan = ConvertPlan::build(
            from.into(),
            to.into(),
            &ConversionSpec::new(ext_opts).with_float_facts(unit_facts),
        )
        .unwrap();
        assert!(unit_plan.output_float_facts().contains(unit_facts));
    }

    // Witness `1e20` through sRGB extended OETF produces NaN in rational polynomial, so
    // `output_float_facts() == NONE` soundly holds on the resulting buffer!
    let mut srgb_src = PixelBuffer::from_vec(
        bytemuck::cast_slice(&[1.0e20f32, 0.5, 1.0]).to_vec(),
        1,
        1,
        RGBF32_LIN,
    )
    .unwrap();
    srgb_src.set_float_facts(unbounded_finite_nonneg).unwrap();
    let srgb_plan = ConvertPlan::build(
        RGBF32_LIN.into(),
        srgb_f32.into(),
        &ConversionSpec::new(ext_opts).with_float_facts(unbounded_finite_nonneg),
    )
    .unwrap();
    let srgb_out_facts = srgb_plan.output_float_facts();
    let mut srgb_dst = PixelBuffer::try_new(1, 1, srgb_f32).unwrap();
    srgb_plan
        .prepare(1)
        .unwrap()
        .convert_rows(srgb_src.as_slice(), srgb_dst.as_slice_mut())
        .unwrap();
    srgb_dst.set_float_facts(srgb_out_facts).unwrap();
    assert!(
        srgb_dst
            .float_facts()
            .holds_for(&srgb_dst.as_slice())
            .unwrap()
    );
}
