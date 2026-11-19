//! Tests for `zenpixels::sample::FloatFacts` across F16, F32, F64, padded views,
//! descriptor relabeling, mutation clearing, banded buffers, and hashing.

use zenpixels::{
    AlphaMode, BufferError, ChannelLayout, ChannelType, F16Sample, Orientation, PixelBuffer,
    PixelDescriptor, PixelSlice, TransferFunction,
    banded::BandedPixelBuffer,
    sample::{ExactDepth, FloatFacts, FloatFactsError},
};

#[test]
fn atomic_flags_and_implication_closure_keep_subnormals_independent_of_positivity() {
    // Subnormals are independent of sign and positivity:
    assert!(!FloatFacts::POSITIVE.contains(FloatFacts::NO_SUBNORMALS));
    assert!(!FloatFacts::POSITIVE.contains(FloatFacts::NO_INFINITE));
    assert!(FloatFacts::POSITIVE.contains(FloatFacts::NON_NEGATIVE));
    assert!(FloatFacts::POSITIVE.contains(FloatFacts::NO_NAN));

    assert!(!FloatFacts::NON_NEGATIVE.contains(FloatFacts::NO_SUBNORMALS));
    assert!(!FloatFacts::NON_NEGATIVE.contains(FloatFacts::POSITIVE));
    assert!(FloatFacts::NON_NEGATIVE.contains(FloatFacts::NO_NAN));

    assert!(!FloatFacts::NO_SUBNORMALS.contains(FloatFacts::NON_NEGATIVE));
    assert!(!FloatFacts::NO_SUBNORMALS.contains(FloatFacts::NO_NAN));
    assert!(!FloatFacts::NO_SUBNORMALS.contains(FloatFacts::NO_INFINITE));

    // Combining NON_NEGATIVE with AT_MOST_ONE implies NO_INFINITE and NO_NAN,
    // still without implying NO_SUBNORMALS or POSITIVE.
    let unit = FloatFacts::NON_NEGATIVE | FloatFacts::AT_MOST_ONE;
    assert!(unit.contains(FloatFacts::NO_NAN));
    assert!(unit.contains(FloatFacts::NO_INFINITE));
    assert!(!unit.contains(FloatFacts::NO_SUBNORMALS));
    assert!(!unit.contains(FloatFacts::POSITIVE));
}

#[test]
fn f32_special_values_signed_zero_subnormals_and_unit_bounds() {
    let check_one = |v: f32| -> FloatFacts {
        let samples = [v, v, v];
        let slice = PixelSlice::new(
            bytemuck::cast_slice(&samples),
            1,
            1,
            12,
            PixelDescriptor::RGBF32_LINEAR,
        )
        .unwrap();
        let measured = FloatFacts::measure(&slice).unwrap();
        assert!(measured.holds_for(&slice).unwrap());
        measured
    };

    // +0.0: NO_NAN | NO_INFINITE | NO_SUBNORMALS | NON_NEGATIVE | AT_MOST_ONE, NOT POSITIVE.
    let pos_zero = check_one(0.0);
    assert_eq!(
        pos_zero,
        FloatFacts::NO_NAN
            | FloatFacts::NO_INFINITE
            | FloatFacts::NO_SUBNORMALS
            | FloatFacts::NON_NEGATIVE
            | FloatFacts::AT_MOST_ONE
    );
    assert!(!pos_zero.contains(FloatFacts::POSITIVE));

    // -0.0: NO_NAN | NO_INFINITE | NO_SUBNORMALS | AT_MOST_ONE, NOT NON_NEGATIVE.
    let neg_zero = check_one(-0.0);
    assert_eq!(
        neg_zero,
        FloatFacts::NO_NAN
            | FloatFacts::NO_INFINITE
            | FloatFacts::NO_SUBNORMALS
            | FloatFacts::AT_MOST_ONE
    );
    assert!(!neg_zero.contains(FloatFacts::NON_NEGATIVE));

    // Positive subnormal (+1e-40): NO_NAN | NO_INFINITE | NON_NEGATIVE | POSITIVE | AT_MOST_ONE,
    // NOT NO_SUBNORMALS.
    let pos_sub = check_one(f32::from_bits(1));
    assert!(pos_sub.contains(
        FloatFacts::NO_NAN
            | FloatFacts::NO_INFINITE
            | FloatFacts::NON_NEGATIVE
            | FloatFacts::POSITIVE
            | FloatFacts::AT_MOST_ONE
    ));
    assert!(!pos_sub.contains(FloatFacts::NO_SUBNORMALS));

    // Negative normal (-0.5): NO_NAN | NO_INFINITE | NO_SUBNORMALS | AT_MOST_ONE.
    let neg_norm = check_one(-0.5);
    assert_eq!(
        neg_norm,
        FloatFacts::NO_NAN
            | FloatFacts::NO_INFINITE
            | FloatFacts::NO_SUBNORMALS
            | FloatFacts::AT_MOST_ONE
    );

    // HDR normal (+4.0): NO_NAN | NO_INFINITE | NO_SUBNORMALS | NON_NEGATIVE | POSITIVE,
    // NOT AT_MOST_ONE.
    let hdr = check_one(4.0);
    assert_eq!(
        hdr,
        FloatFacts::NO_NAN
            | FloatFacts::NO_INFINITE
            | FloatFacts::NO_SUBNORMALS
            | FloatFacts::NON_NEGATIVE
            | FloatFacts::POSITIVE
    );
    assert!(!hdr.contains(FloatFacts::AT_MOST_ONE));

    // +Inf: NO_NAN | NO_SUBNORMALS | NON_NEGATIVE | POSITIVE, NOT NO_INFINITE, NOT AT_MOST_ONE.
    let pos_inf = check_one(f32::INFINITY);
    assert_eq!(
        pos_inf,
        FloatFacts::NO_NAN
            | FloatFacts::NO_SUBNORMALS
            | FloatFacts::NON_NEGATIVE
            | FloatFacts::POSITIVE
    );

    // -Inf: NO_NAN | NO_SUBNORMALS | AT_MOST_ONE, NOT NO_INFINITE, NOT NON_NEGATIVE.
    let neg_inf = check_one(f32::NEG_INFINITY);
    assert_eq!(
        neg_inf,
        FloatFacts::NO_NAN | FloatFacts::NO_SUBNORMALS | FloatFacts::AT_MOST_ONE
    );

    // NaN: satisfies NO_INFINITE | NO_SUBNORMALS only.
    let nan = check_one(f32::NAN);
    assert_eq!(nan, FloatFacts::NO_INFINITE | FloatFacts::NO_SUBNORMALS);
}

#[test]
fn f16_and_f64_measure_and_holds_for() {
    // F16: normal 0.5 and subnormal 2^-24 (bits = 0x0001).
    let f16_norm = [F16Sample::from_f32(0.5); 3];
    let f16_desc = PixelDescriptor::new(
        ChannelType::F16,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Linear,
    );
    let norm_slice = PixelSlice::new(bytemuck::cast_slice(&f16_norm), 1, 1, 6, f16_desc).unwrap();
    let norm_facts = FloatFacts::measure(&norm_slice).unwrap();
    assert!(norm_facts.contains(
        FloatFacts::NO_NAN
            | FloatFacts::NO_INFINITE
            | FloatFacts::NO_SUBNORMALS
            | FloatFacts::POSITIVE
            | FloatFacts::AT_MOST_ONE
    ));

    let f16_sub = [
        F16Sample::from_f32(0.5),
        F16Sample::from_bits(0x0001),
        F16Sample::from_f32(1.0),
    ];
    let sub_slice = PixelSlice::new(bytemuck::cast_slice(&f16_sub), 1, 1, 6, f16_desc).unwrap();
    let sub_facts = FloatFacts::measure(&sub_slice).unwrap();
    assert!(sub_facts.contains(FloatFacts::POSITIVE | FloatFacts::AT_MOST_ONE));
    assert!(!sub_facts.contains(FloatFacts::NO_SUBNORMALS));
    assert!(!FloatFacts::NO_SUBNORMALS.holds_for(&sub_slice).unwrap());

    // F64: signed zero and normal 1.0.
    let f64_desc = PixelDescriptor::try_new(
        ChannelType::F64,
        ChannelLayout::Gray,
        None,
        TransferFunction::Linear,
    )
    .unwrap();
    let f64_vals = [0.0f64, 1.0f64];
    let f64_slice = PixelSlice::new(bytemuck::cast_slice(&f64_vals), 2, 1, 16, f64_desc).unwrap();
    let f64_facts = FloatFacts::measure(&f64_slice).unwrap();
    assert_eq!(
        f64_facts,
        FloatFacts::NO_NAN
            | FloatFacts::NO_INFINITE
            | FloatFacts::NO_SUBNORMALS
            | FloatFacts::NON_NEGATIVE
            | FloatFacts::AT_MOST_ONE
    );

    // Integer storage is rejected.
    let u8_slice = PixelSlice::new(&[0u8, 128, 255], 1, 1, 3, PixelDescriptor::RGB8_SRGB).unwrap();
    assert_eq!(
        FloatFacts::measure(&u8_slice),
        Err(FloatFactsError::UnsupportedStorage)
    );
    assert_eq!(
        FloatFacts::NO_NAN.holds_for(&u8_slice),
        Err(FloatFactsError::UnsupportedStorage)
    );
    assert_eq!(
        u8_slice
            .clone()
            .with_float_facts(FloatFacts::NO_NAN)
            .unwrap_err(),
        FloatFactsError::UnsupportedStorage
    );
    // Clearing with NONE is harmless on any storage.
    assert!(u8_slice.with_float_facts(FloatFacts::NONE).is_ok());
}

#[test]
fn row_padding_and_undefined_alpha_are_ignored_until_promoted() {
    // Stride-padded row containing NaN and -Inf in row padding, and Undefined alpha channel
    // containing NaN and subnormal in pixel[3].
    let rgbx_f32 = PixelDescriptor::new(
        ChannelType::F32,
        ChannelLayout::Rgba,
        Some(AlphaMode::Undefined),
        TransferFunction::Linear,
    );
    let raw: [f32; 12] = [
        // Row 0: pixel 0 (active RGB in (0, 1], pad = NaN), stride pad (2 floats = -Inf, subnormal)
        0.25,
        0.5,
        1.0,
        f32::NAN,
        f32::NEG_INFINITY,
        f32::from_bits(1),
        // Row 1: pixel 0 (active RGB in (0, 1], pad = -5.0), stride pad (2 floats = NaN, +Inf)
        0.125,
        0.75,
        0.5,
        -5.0,
        f32::NAN,
        f32::INFINITY,
    ];
    let slice = PixelSlice::new(bytemuck::cast_slice(&raw), 1, 2, 24, rgbx_f32).unwrap();
    let facts = FloatFacts::measure(&slice).unwrap();
    assert!(facts.contains(
        FloatFacts::NO_NAN
            | FloatFacts::NO_INFINITE
            | FloatFacts::NO_SUBNORMALS
            | FloatFacts::POSITIVE
            | FloatFacts::AT_MOST_ONE
    ));

    let annotated = slice.with_float_facts(facts).unwrap();
    assert_eq!(annotated.float_facts(), facts);

    // Promoting Undefined padding into Straight alpha MUST drop FloatFacts!
    let promoted = annotated.clone().with_alpha_mode(Some(AlphaMode::Straight));
    assert_eq!(promoted.float_facts(), FloatFacts::NONE);
    assert!(!facts.holds_for(&promoted).unwrap());

    // Same soundness guard for ExactDepth on RGBX8 -> RGBA8!
    let rgbx8_bytes = [0u8, 255, 0, 123]; // RGB exact at 1 bit, padding byte = 123 (8-bit only)
    let rgbx8 = PixelSlice::new(&rgbx8_bytes, 1, 1, 4, PixelDescriptor::RGBX8)
        .unwrap()
        .with_exact_depth(
            ExactDepth::measure(
                &PixelSlice::new(&rgbx8_bytes, 1, 1, 4, PixelDescriptor::RGBX8).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(rgbx8.exact_depth().unwrap().color_bits(), 1);
    let promoted_rgba8 = rgbx8.reinterpret(PixelDescriptor::RGBA8).unwrap();
    assert_eq!(promoted_rgba8.exact_depth(), None);
}

#[test]
fn buffer_and_banded_preserve_facts_across_views_and_clear_on_mutation()
-> Result<(), whereat::At<BufferError>> {
    let mut buf = PixelBuffer::try_new(4, 4, PixelDescriptor::RGBF32_LINEAR)?;
    let facts = FloatFacts::measure(&buf.as_slice()).unwrap();
    assert!(facts.contains(FloatFacts::NON_NEGATIVE | FloatFacts::NO_SUBNORMALS));
    buf.set_float_facts(facts).unwrap();
    assert_eq!(buf.float_facts(), facts);
    assert_eq!(buf.as_slice().float_facts(), facts);
    assert_eq!(buf.rows(1, 2).float_facts(), facts);
    assert_eq!(buf.crop_view(1, 1, 2, 2).float_facts(), facts);

    let copied = buf.crop_copy(0, 0, 2, 2)?;
    assert_eq!(copied.float_facts(), facts);

    // Mutable access clears facts immediately.
    let _ = buf.rows_mut(0, 1);
    assert_eq!(buf.float_facts(), FloatFacts::NONE);

    // BandedPixelBuffer preserves and clears symmetrically.
    buf.set_float_facts(facts).unwrap();
    let mut banded = BandedPixelBuffer::from_contiguous(buf)?;
    assert_eq!(banded.float_facts(), facts);
    for band in banded.bands() {
        assert_eq!(band.float_facts(), facts);
    }
    banded.orient_in_place(Orientation::FlipH).unwrap();
    assert_eq!(banded.float_facts(), facts);

    let mut dst = BandedPixelBuffer::try_new(4, 4, PixelDescriptor::RGBF32_LINEAR, 64)?;
    banded
        .oriented_into(Orientation::Rotate90, &mut dst)
        .unwrap();
    assert_eq!(dst.float_facts(), facts);
    assert_eq!(FloatFacts::measure_banded(&dst).unwrap(), facts);

    let rebuilt = dst.try_into_contiguous()?;
    assert_eq!(rebuilt.float_facts(), facts);

    let _ = banded.row_mut(0);
    assert_eq!(banded.float_facts(), FloatFacts::NONE);
    Ok(())
}

#[cfg(feature = "hash")]
#[test]
fn hashing_includes_float_facts_only_in_all_mode_and_preserves_unannotated_digest() {
    use zenpixels::HashInclusions;

    let samples = [0.25f32, 0.5, 0.75];
    let plain = PixelSlice::new(
        bytemuck::cast_slice(&samples),
        1,
        1,
        12,
        PixelDescriptor::RGBF32_LINEAR,
    )
    .unwrap();
    let none_explicit = plain.clone().with_float_facts(FloatFacts::NONE).unwrap();
    let annotated = plain
        .clone()
        .with_float_facts(FloatFacts::NO_NAN | FloatFacts::NO_SUBNORMALS)
        .unwrap();
    let annotated_more = plain
        .clone()
        .with_float_facts(FloatFacts::POSITIVE | FloatFacts::AT_MOST_ONE)
        .unwrap();

    assert_eq!(
        plain.hash_with(HashInclusions::All),
        none_explicit.hash_with(HashInclusions::All)
    );
    assert_eq!(
        plain.hash_with(HashInclusions::SamplesOnly),
        annotated.hash_with(HashInclusions::SamplesOnly)
    );
    assert_eq!(
        plain.hash_with(HashInclusions::Pixels),
        annotated.hash_with(HashInclusions::Pixels)
    );
    assert_ne!(
        plain.hash_with(HashInclusions::All),
        annotated.hash_with(HashInclusions::All)
    );
    assert_ne!(
        annotated.hash_with(HashInclusions::All),
        annotated_more.hash_with(HashInclusions::All)
    );
}
