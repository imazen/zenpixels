//! The rule tables of `docs/precision-policy-design.md`, row by row.

use core::fmt;

use whereat::At;

whereat::define_at_crate_info!();

use zenpixels::precision::{
    AlphaContent, ConversionDecision, ConversionEffect, ConversionFailure, ConversionOffer,
    ConversionPolicy, ConversionPreset, ConversionQuestion, ConversionRefusal, DecidedBy,
    OutputChoice, SourcePrecision,
};
use zenpixels::sample::ExactDepth;
use zenpixels::{
    AlphaMode, ChannelLayout, ChannelType, PixelDescriptor as D, PixelSlice, SignalRange,
    TransferFunction as T,
};

fn declared(bits: u8) -> SourcePrecision {
    SourcePrecision::declared(bits, T::Srgb, SignalRange::Full).unwrap()
}

fn real16() -> SourcePrecision {
    SourcePrecision::from_storage(D::RGB16_SRGB)
}

fn offer(output: D) -> ConversionOffer {
    ConversionOffer::new(output)
}

fn keep() -> ConversionPolicy {
    ConversionPolicy::new()
}

fn linear(channel: ChannelType, layout: ChannelLayout) -> D {
    let alpha = layout.has_alpha().then_some(AlphaMode::Straight);
    D::new(channel, layout, alpha, T::Linear)
}

// ---- The rule ----

#[test]
fn row_8bit_source_any_working_format_back_to_8bit_has_no_event() {
    // The significant bits don't care what the samples sit in now.
    for working in [
        linear(ChannelType::U16, ChannelLayout::Rgb),
        linear(ChannelType::I16, ChannelLayout::Rgb),
        linear(ChannelType::F32, ChannelLayout::Rgb),
        D::RGB16_SRGB,
    ] {
        let source = declared(8);
        let through = ConversionEffect::of(&source, &offer(working));
        assert!(through.is_lossless(), "{working:?}: {through:?}");
        let back = keep()
            .resolve_preset(&source, OutputChoice::Hidden, &[offer(D::RGB8_SRGB)])
            .unwrap();
        assert!(back.effect.is_lossless());
        assert_eq!(back.effect.bits_kept, 8);
        assert_eq!(back.decided_by, DecidedBy::Preset);
    }
}

#[test]
fn linear_u8_is_the_one_working_format_that_does_not_hold_8bit_srgb() {
    // The design said "any working format". Eight-bit linear crowds the darks together.
    let effect = ConversionEffect::of(
        &declared(8),
        &offer(linear(ChannelType::U8, ChannelLayout::Rgb)),
    );
    assert_eq!(effect.bits_kept, 4);
    assert_eq!(effect.bits_dropped, 4);
    assert!(
        keep()
            .resolve_preset(
                &declared(8),
                OutputChoice::Hidden,
                &[offer(linear(ChannelType::U8, ChannelLayout::Rgb))]
            )
            .is_err()
    );
}

#[test]
fn row_8bit_content_in_u16_storage_needs_to_be_known() {
    let eight_bit_content: Vec<u16> = (0..=255u16).map(|v| v * 257).collect();
    let bytes: &[u8] = bytemuck::cast_slice(&eight_bit_content);
    let pixels = PixelSlice::new(bytes, 256, 1, 512, D::GRAY16_SRGB).unwrap();

    // Before anyone looks, all 16 bits count.
    let unknown = SourcePrecision::from_storage(D::GRAY16_SRGB);
    let err = keep()
        .resolve_preset(&unknown, OutputChoice::Hidden, &[offer(D::GRAY8_SRGB)])
        .unwrap_err();
    assert!(matches!(
        err.error(),
        ConversionRefusal::DropsPrecision {
            significant_bits: 16,
            best_kept: 8,
            ..
        }
    ));

    // After a scan it's 8-bit content, and the same conversion raises nothing.
    let measured = ExactDepth::measure(&pixels).unwrap();
    assert_eq!(measured.color_bits(), 8);
    let known =
        SourcePrecision::declared(measured.color_bits(), T::Srgb, SignalRange::Full).unwrap();
    let record = keep()
        .resolve_preset(&known, OutputChoice::Hidden, &[offer(D::GRAY8_SRGB)])
        .unwrap();
    assert!(record.effect.is_lossless());
}

#[test]
fn a_container_claim_counts_only_once_a_scan_verifies_it() {
    // PNG sBIT says 8, and the samples disagree.
    let samples: Vec<u16> = vec![0x1234, 0x5678];
    let pixels = PixelSlice::new(bytemuck::cast_slice(&samples), 2, 1, 4, D::GRAY16_SRGB).unwrap();
    let claim = ExactDepth::new(ChannelType::U16, SignalRange::Full, 8, 16).unwrap();
    assert!(!claim.holds_for(&pixels).unwrap());
}

#[test]
fn row_real_16bit_output_that_can_hold_it_keeps_it() {
    let record = keep()
        .resolve_preset(&real16(), OutputChoice::Hidden, &[offer(D::RGB16_SRGB)])
        .unwrap();
    assert!(record.effect.is_lossless());
    assert_eq!(record.effect.bits_kept, 16);
}

#[test]
fn row_real_16bit_automatic_format_picks_one_that_holds_it() {
    let record = keep()
        .resolve_preset(
            &real16(),
            OutputChoice::Hidden,
            &[offer(D::RGB8_SRGB), offer(D::RGB16_SRGB)],
        )
        .unwrap();
    assert_eq!(record.index, 1);
    assert_eq!(record.output, D::RGB16_SRGB);
}

#[test]
fn row_real_16bit_unnamed_narrowing_refuses() {
    let err = keep()
        .resolve_preset(&real16(), OutputChoice::Hidden, &[offer(D::RGB8_SRGB)])
        .unwrap_err();
    assert_eq!(
        *err.error(),
        ConversionRefusal::DropsPrecision {
            significant_bits: 16,
            best_kept: 8,
            output: D::RGB8_SRGB,
        }
    );
}

#[test]
fn row_8bit_onto_u16_canvas_follows_the_canvas() {
    let record = keep()
        .resolve_preset(&declared(8), OutputChoice::Hidden, &[offer(D::RGB16_SRGB)])
        .unwrap();
    assert_eq!(record.output, D::RGB16_SRGB);
    assert!(record.effect.is_lossless());
}

#[test]
fn composing_onto_a_real_16bit_canvas_raises_the_floor() {
    let composite = declared(8).joined_with(real16());
    assert_eq!(composite.significant_bits(), 16);
    assert!(
        keep()
            .resolve_preset(&composite, OutputChoice::Hidden, &[offer(D::RGB8_SRGB)])
            .is_err()
    );
    // Either order.
    assert_eq!(real16().joined_with(declared(8)).significant_bits(), 16);
}

#[test]
fn row_real_16bit_onto_8bit_canvas_refuses_unless_allowed_by_name() {
    let canvas = [offer(D::RGB8_SRGB).with_peak_bytes(100)];
    assert!(
        keep()
            .resolve_preset(&real16(), OutputChoice::Hidden, &canvas)
            .is_err()
    );
    let named = keep().with_preset(ConversionPreset::DropPrecisionForMemory);
    let record = named
        .resolve_preset(&real16(), OutputChoice::Hidden, &canvas)
        .unwrap();
    assert_eq!(record.effect.bits_dropped, 8);
    assert!(!record.effect.is_lossless());
}

#[test]
fn row_translucent_source_no_alpha_no_matte_refuses() {
    let translucent = declared(8).with_alpha(AlphaContent::Translucent);
    let err = keep()
        .resolve_preset(&translucent, OutputChoice::Hidden, &[offer(D::RGB8_SRGB)])
        .unwrap_err();
    assert_eq!(
        *err.error(),
        ConversionRefusal::DropsAlpha {
            output: D::RGB8_SRGB
        }
    );

    // A matte makes it fine, as does keeping alpha.
    let record = keep()
        .resolve_preset(
            &translucent,
            OutputChoice::Hidden,
            &[offer(D::RGB8_SRGB).with_matte()],
        )
        .unwrap();
    assert!(record.effect.is_lossless());
    let record = keep()
        .resolve_preset(
            &translucent,
            OutputChoice::Hidden,
            &[offer(D::RGB8_SRGB), offer(D::RGBA8_SRGB)],
        )
        .unwrap();
    assert_eq!(record.index, 1);

    // Unknown counts as translucent; a scan that found it opaque clears it.
    let unknown = SourcePrecision::from_storage(D::RGBA8_SRGB);
    assert!(
        keep()
            .resolve_preset(&unknown, OutputChoice::Hidden, &[offer(D::RGB8_SRGB)])
            .is_err()
    );
    let opaque = unknown.with_alpha(AlphaContent::Opaque);
    assert!(
        keep()
            .resolve_preset(&opaque, OutputChoice::Hidden, &[offer(D::RGB8_SRGB)])
            .is_ok()
    );
}

#[test]
fn row_memory_over_limit_refuses_naming_both_costs_unless_a_resolver_chooses() {
    let offers = [
        offer(D::RGB8_SRGB).with_peak_bytes(50),
        offer(D::RGB16_SRGB).with_peak_bytes(101),
    ];
    let limited = keep().with_max_peak_bytes(60);
    let err = limited
        .resolve_preset(&real16(), OutputChoice::Hidden, &offers)
        .unwrap_err();
    assert_eq!(
        *err.error(),
        ConversionRefusal::MemoryOrPrecision {
            limit: 60,
            keep_bytes: 101,
            narrow_bytes: 50,
            narrow_drops: 8,
        }
    );
    let message = err.error().to_string();
    assert!(message.contains("101") && message.contains("50") && message.contains("60"));

    // A resolver can take the narrow one: that is what choosing means.
    let mut take_first = |q: ConversionQuestion<'_>| -> Result<ConversionDecision, At<Never>> {
        Ok(ConversionDecision::Choose(
            q.admitted().next().unwrap().index(),
        ))
    };
    let record = limited
        .resolve(&real16(), OutputChoice::Hidden, &offers, &mut take_first)
        .unwrap();
    assert_eq!(record.index, 0);
    assert_eq!(record.decided_by, DecidedBy::Resolver);
    assert_eq!(record.effect.bits_dropped, 8);

    // And a loud preset can too.
    let record = limited
        .with_preset(ConversionPreset::DropPrecisionForMemory)
        .resolve_preset(&real16(), OutputChoice::Hidden, &offers)
        .unwrap();
    assert_eq!(record.index, 0);
}

// ---- Which conversions are hidden ----

#[test]
fn named_format_that_holds_the_source_keeps_it() {
    let record = keep()
        .resolve_preset(
            &real16(),
            OutputChoice::CallerNamed,
            &[offer(D::RGB8_SRGB), offer(D::RGB16_SRGB)],
        )
        .unwrap();
    assert_eq!(record.output, D::RGB16_SRGB);
}

#[test]
fn named_format_that_cannot_hold_the_source_is_consent_and_is_recorded() {
    let record = keep()
        .resolve_preset(&real16(), OutputChoice::CallerNamed, &[offer(D::RGB8_SRGB)])
        .unwrap();
    assert_eq!(record.output, D::RGB8_SRGB);
    assert_eq!(record.effect.bits_kept, 8);
    assert_eq!(record.effect.bits_dropped, 8);
    assert!(!record.effect.is_lossless());
}

#[test]
fn named_format_does_not_consent_to_dropping_alpha() {
    // What was under the alpha shows, so this isn't a rendition of the same image.
    let translucent = declared(8).with_alpha(AlphaContent::Translucent);
    assert!(matches!(
        keep()
            .resolve_preset(
                &translucent,
                OutputChoice::CallerNamed,
                &[offer(D::RGB8_SRGB)]
            )
            .unwrap_err()
            .error(),
        ConversionRefusal::DropsAlpha { .. }
    ));
    // Choosing the offer in a resolver does.
    let mut first = |_: ConversionQuestion<'_>| -> Result<ConversionDecision, At<Never>> {
        Ok(ConversionDecision::Choose(0))
    };
    let record = keep()
        .resolve(
            &translucent,
            OutputChoice::CallerNamed,
            &[offer(D::RGB8_SRGB)],
            &mut first,
        )
        .unwrap();
    assert!(record.effect.alpha_dropped);
}

#[test]
fn twelve_bit_jpeg_to_jpeg_takes_the_path_that_keeps_it_or_refuses() {
    // "JPEG" names no depth. The same format in and out is not consent to narrow.
    let twelve = declared(12);
    let coefficient_path = offer(D::RGB16_SRGB);
    let pixel_path = offer(D::RGB8_SRGB);
    let record = keep()
        .resolve_preset(
            &twelve,
            OutputChoice::Hidden,
            &[pixel_path, coefficient_path],
        )
        .unwrap();
    assert_eq!(record.index, 1);
    assert_eq!(record.effect.bits_kept, 12);

    assert_eq!(
        *keep()
            .resolve_preset(&twelve, OutputChoice::Hidden, &[pixel_path])
            .unwrap_err()
            .error(),
        ConversionRefusal::DropsPrecision {
            significant_bits: 12,
            best_kept: 8,
            output: D::RGB8_SRGB,
        }
    );
}

#[test]
fn twelve_bit_jpeg_decoder_rounding_to_u8_is_a_narrowing() {
    // Today's zenjpeg decoder when U8 is preferred, reported only as bit_depth 12.
    let effect = ConversionEffect::of(&declared(12), &offer(D::RGB8_SRGB));
    assert_eq!((effect.bits_kept, effect.bits_dropped), (8, 4));
    // Its F32 exit holds them.
    assert!(ConversionEffect::of(&declared(12), &offer(D::RGBF32_LINEAR)).is_lossless());
}

#[test]
fn depth_is_counted_in_bits_not_channel_types() {
    // A depth that isn't a storage width, against one that has no integer width at all.
    let half = linear(ChannelType::F16, ChannelLayout::Rgb);
    assert_eq!(
        ConversionEffect::of(&declared(12), &offer(half)).bits_kept,
        11
    );
    assert_eq!(
        ConversionEffect::of(&declared(10), &offer(half)).bits_kept,
        10
    );
}

// ---- HDR ----

fn pq(bits: u8) -> SourcePrecision {
    SourcePrecision::declared(bits, T::Pq, SignalRange::Full).unwrap()
}

#[test]
fn hdr_to_an_sdr_output_drops_range_and_is_refused_when_hidden() {
    let sdr8 = D::RGB8_SRGB;
    let effect = ConversionEffect::of(&pq(10), &offer(sdr8));
    assert!(effect.range_dropped);
    assert!(matches!(
        keep()
            .resolve_preset(&pq(10), OutputChoice::Hidden, &[offer(D::RGB16_SRGB)])
            .unwrap_err()
            .error(),
        ConversionRefusal::DropsRange { .. }
    ));
    // A named SDR format is consent, and the record says range went: this is zencodecs' default
    // tone map for SDR-only formats.
    let record = keep()
        .resolve_preset(&pq(10), OutputChoice::CallerNamed, &[offer(sdr8)])
        .unwrap();
    assert!(record.effect.range_dropped);
    assert_eq!(record.effect.bits_dropped, 2);
}

#[test]
fn hdr_to_hdr_storage_compares_bits() {
    let pq8 = D::new(ChannelType::U8, ChannelLayout::Rgb, None, T::Pq);
    let pq16 = D::new(ChannelType::U16, ChannelLayout::Rgb, None, T::Pq);
    let hlg16 = D::new(ChannelType::U16, ChannelLayout::Rgb, None, T::Hlg);
    assert!(ConversionEffect::of(&pq(12), &offer(pq16)).is_lossless());
    assert!(ConversionEffect::of(&pq(10), &offer(hlg16)).is_lossless());
    let narrowed = ConversionEffect::of(&pq(10), &offer(pq8));
    assert_eq!(narrowed.bits_dropped, 2);
    assert!(!narrowed.range_dropped);
    assert!(
        keep()
            .resolve_preset(&pq(10), OutputChoice::Hidden, &[offer(pq8)])
            .is_err()
    );
}

#[test]
fn hdr_held_in_float_stays_hdr_and_linear_integer_does_not_hold_it() {
    assert!(ConversionEffect::of(&pq(12), &offer(D::RGBF32_LINEAR)).is_lossless());
    let effect = ConversionEffect::of(
        &pq(12),
        &offer(linear(ChannelType::U16, ChannelLayout::Rgb)),
    );
    assert!(effect.range_dropped);
}

#[test]
fn a_float_source_counts_as_extended_until_someone_says_otherwise() {
    let float = SourcePrecision::from_storage(D::RGBF32_LINEAR);
    assert_eq!(float.significant_bits(), 24);
    let effect = ConversionEffect::of(&float, &offer(D::RGB16_SRGB));
    assert!(effect.range_dropped);
    assert!(effect.bits_dropped > 0);
    // Half keeps range and loses mantissa.
    let half = ConversionEffect::of(&float, &offer(linear(ChannelType::F16, ChannelLayout::Rgb)));
    assert_eq!(
        (half.bits_kept, half.bits_dropped, half.range_dropped),
        (11, 13, false)
    );
    // A half source doesn't lose bits to U16, and still loses range.
    let half_source = SourcePrecision::from_storage(linear(ChannelType::F16, ChannelLayout::Rgb));
    let effect = ConversionEffect::of(&half_source, &offer(D::RGB16_SRGB));
    assert_eq!((effect.bits_dropped, effect.range_dropped), (0, true));
}

#[test]
fn drop_presets_never_drop_range_or_alpha() {
    let memory = keep().with_preset(ConversionPreset::DropPrecisionForMemory);
    assert!(matches!(
        memory
            .resolve_preset(
                &pq(10),
                OutputChoice::Hidden,
                &[offer(D::RGB8_SRGB).with_peak_bytes(1)]
            )
            .unwrap_err()
            .error(),
        ConversionRefusal::DropsRange { .. }
    ));
    let translucent = declared(16).with_alpha(AlphaContent::Translucent);
    assert!(matches!(
        memory
            .resolve_preset(
                &translucent,
                OutputChoice::Hidden,
                &[offer(D::RGB8_SRGB).with_peak_bytes(1)]
            )
            .unwrap_err()
            .error(),
        ConversionRefusal::DropsAlpha { .. }
    ));
}

// ---- Who decides ----

#[test]
fn requirements_remove_offers_before_any_resolver_is_asked() {
    let offers = [
        offer(D::RGB8_SRGB).with_peak_bytes(10),
        offer(D::RGB16_SRGB).with_peak_bytes(20),
    ];
    let mut seen = Vec::new();
    let mut spy = |q: ConversionQuestion<'_>| -> Result<ConversionDecision, At<Never>> {
        seen.extend(q.admitted().map(|c| c.index()));
        Ok(ConversionDecision::Defer)
    };
    let floored = keep().with_min_bits(12);
    let record = floored
        .resolve(&real16(), OutputChoice::Hidden, &offers, &mut spy)
        .unwrap();
    assert_eq!(seen, [1]);
    assert_eq!(record.index, 1);

    // A resolver can't bring a removed offer back.
    let mut cheat = |_: ConversionQuestion<'_>| -> Result<ConversionDecision, At<Never>> {
        Ok(ConversionDecision::Choose(0))
    };
    let err = floored
        .resolve(&real16(), OutputChoice::Hidden, &offers, &mut cheat)
        .unwrap_err();
    assert!(matches!(
        err.error(),
        ConversionFailure::Refused(ConversionRefusal::InvalidChoice { index: 0 })
    ));
}

#[test]
fn a_floor_binds_a_named_format_too() {
    let err = keep()
        .with_min_bits(12)
        .resolve_preset(&real16(), OutputChoice::CallerNamed, &[offer(D::RGB8_SRGB)])
        .unwrap_err();
    assert_eq!(
        *err.error(),
        ConversionRefusal::NoneAdmitted {
            min_bits: Some(12),
            max_peak_bytes: None,
        }
    );
}

#[test]
fn a_floor_above_the_source_does_not_refuse_what_loses_nothing() {
    let record = keep()
        .with_min_bits(12)
        .resolve_preset(&declared(8), OutputChoice::Hidden, &[offer(D::RGB8_SRGB)])
        .unwrap();
    assert!(record.effect.is_lossless());
}

#[test]
fn unknown_peak_is_not_zero() {
    let err = keep()
        .with_max_peak_bytes(1_000_000)
        .resolve_preset(&declared(8), OutputChoice::Hidden, &[offer(D::RGB8_SRGB)])
        .unwrap_err();
    assert!(matches!(
        err.error(),
        ConversionRefusal::NoneAdmitted { .. }
    ));
}

#[test]
fn the_memory_preset_picks_the_smallest_peak_and_refuses_when_none_is_stated() {
    let offers = [
        offer(D::RGB16_SRGB).with_peak_bytes(100),
        offer(D::RGB8_SRGB).with_peak_bytes(50),
    ];
    let memory = keep().with_preset(ConversionPreset::DropPrecisionForMemory);
    let source = real16();
    assert_eq!(
        memory
            .resolve_preset(&source, OutputChoice::Hidden, &offers)
            .unwrap()
            .index,
        1
    );
    assert_eq!(
        *memory
            .resolve_preset(&source, OutputChoice::Hidden, &[offer(D::RGB8_SRGB)])
            .unwrap_err()
            .error(),
        ConversionRefusal::CostUnknown
    );
}

#[test]
fn no_offers_is_its_own_refusal() {
    assert_eq!(
        *keep()
            .resolve_preset(&declared(8), OutputChoice::Hidden, &[])
            .unwrap_err()
            .error(),
        ConversionRefusal::NoOffers
    );
}

#[derive(Debug)]
struct Never;
impl fmt::Display for Never {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("never")
    }
}
impl core::error::Error for Never {}

#[derive(Debug, PartialEq, Eq)]
struct TenantRule(&'static str);
impl fmt::Display for TenantRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}
impl core::error::Error for TenantRule {}

#[test]
fn resolvers_compose_by_deferring() {
    let offers = [offer(D::RGB8_SRGB), offer(D::RGB16_SRGB)];
    let mut calls = 0;
    let mut outer = |q: ConversionQuestion<'_>| -> Result<ConversionDecision, At<Never>> {
        calls += 1;
        let mut inner = |q: ConversionQuestion<'_>| -> Result<ConversionDecision, At<Never>> {
            // Only cares about 8-bit sources.
            if q.source.significant_bits() == 8 {
                Ok(ConversionDecision::Choose(0))
            } else {
                Ok(ConversionDecision::Defer)
            }
        };
        zenpixels::precision::ConversionResolver::resolve(&mut inner, q)
    };
    let from_preset = keep()
        .resolve(&real16(), OutputChoice::Hidden, &offers, &mut outer)
        .unwrap();
    assert_eq!(from_preset.index, 1);
    assert_eq!(from_preset.decided_by, DecidedBy::Preset);
    let from_inner = keep()
        .resolve(&declared(8), OutputChoice::Hidden, &offers, &mut outer)
        .unwrap();
    assert_eq!(from_inner.index, 0);
    assert_eq!(from_inner.decided_by, DecidedBy::Resolver);
    assert_eq!(calls, 2);
}

#[test]
fn a_resolver_failure_keeps_its_own_error_and_trace() {
    let mut failing = |_: ConversionQuestion<'_>| -> Result<ConversionDecision, At<TenantRule>> {
        Err(whereat::at!(TenantRule("tenant forbids this")))
    };
    let err = keep()
        .resolve(
            &declared(8),
            OutputChoice::Hidden,
            &[offer(D::RGB8_SRGB)],
            &mut failing,
        )
        .unwrap_err();
    assert!(matches!(
        err.error(),
        ConversionFailure::Resolver(TenantRule("tenant forbids this"))
    ));
    // The closure's frame and the policy's both show.
    assert!(err.frames().count() >= 2, "{err:?}");
}

#[test]
fn a_refusal_carries_a_trace_to_its_caller() {
    let err = keep()
        .resolve_preset(&real16(), OutputChoice::Hidden, &[offer(D::RGB8_SRGB)])
        .unwrap_err();
    assert!(err.frames().count() >= 1, "{err:?}");
    assert!(err.to_string().contains("16 significant bits"));
}

#[test]
fn resolver_refusal_and_invalid_index_are_refusals() {
    let offers = [offer(D::RGB8_SRGB)];
    let mut refuse = |_: ConversionQuestion<'_>| -> Result<ConversionDecision, At<Never>> {
        Ok(ConversionDecision::Refuse)
    };
    assert!(matches!(
        keep()
            .resolve(&declared(8), OutputChoice::Hidden, &offers, &mut refuse)
            .unwrap_err()
            .error(),
        ConversionFailure::Refused(ConversionRefusal::ResolverRefused)
    ));
    let mut wild = |_: ConversionQuestion<'_>| -> Result<ConversionDecision, At<Never>> {
        Ok(ConversionDecision::Choose(7))
    };
    assert!(matches!(
        keep()
            .resolve(&declared(8), OutputChoice::Hidden, &offers, &mut wild)
            .unwrap_err()
            .error(),
        ConversionFailure::Refused(ConversionRefusal::InvalidChoice { index: 7 })
    ));
}

#[test]
fn invalid_depths_are_rejected() {
    assert_eq!(
        SourcePrecision::declared(0, T::Srgb, SignalRange::Full),
        None
    );
    assert_eq!(
        SourcePrecision::declared(65, T::Srgb, SignalRange::Full),
        None
    );
    assert!(SourcePrecision::declared(64, T::Srgb, SignalRange::Full).is_some());
}

// ---- The chain solver, against the old solver's validation table ----

const RGBA8: D = D::RGBA8_SRGB;

fn frame(descriptor: D) -> ConversionOffer {
    // 4096 x 3072.
    offer(descriptor).with_peak_bytes(4096 * 3072 * descriptor.bytes_per_pixel() as u64)
}

#[test]
fn chain_8bit_resize_back_to_8_is_allowed() {
    let source = declared(8).with_alpha(AlphaContent::Translucent);
    let working = [frame(RGBA8), frame(D::RGBAF32_LINEAR)];
    let output = [frame(RGBA8)];
    let plan = keep()
        .plan_chain_preset(&source, RGBA8, &[&working, &output], OutputChoice::Hidden)
        .unwrap();
    // The cheaper working format, and no conversion.
    assert_eq!(plan.steps[0].output, RGBA8);
    assert_eq!(plan.steps[1].output, RGBA8);
    assert!(plan.steps.iter().all(|s| s.effect.is_lossless()));
    assert_eq!(plan.peak_bytes, Some(4096 * 3072 * 4));
}

#[test]
fn chain_real_16bit_resize_to_8_refuses_at_the_working_format_or_output() {
    let source = SourcePrecision::from_storage(D::RGBA16_SRGB).with_alpha(AlphaContent::Opaque);
    let working = [frame(RGBA8), frame(D::RGBAF32_LINEAR)];
    let output = [frame(RGBA8)];
    let err = keep()
        .plan_chain_preset(
            &source,
            D::RGBA16_SRGB,
            &[&working, &output],
            OutputChoice::Hidden,
        )
        .unwrap_err();
    assert_eq!(err.error().node, 1);
    assert!(matches!(
        err.error().failure,
        ConversionFailure::Refused(ConversionRefusal::DropsPrecision {
            significant_bits: 16,
            best_kept: 8,
            ..
        })
    ));
}

#[test]
fn chain_real_16bit_no_operation_to_8_refuses() {
    let source = real16();
    let output = [frame(D::RGB8_SRGB)];
    let err = keep()
        .plan_chain_preset(&source, D::RGB16_SRGB, &[&output], OutputChoice::Hidden)
        .unwrap_err();
    assert_eq!(err.error().node, 0);
}

#[test]
fn chain_8bit_onto_16bit_canvas_is_allowed() {
    let source = declared(8);
    let canvas = [frame(D::RGB16_SRGB)];
    let plan = keep()
        .plan_chain_preset(&source, D::RGB8_SRGB, &[&canvas], OutputChoice::Hidden)
        .unwrap();
    assert!(plan.steps[0].effect.is_lossless());
}

#[test]
fn chain_real_16bit_resize_to_16_is_allowed_through_a_working_format_that_holds_it() {
    let source = SourcePrecision::from_storage(D::RGBA16_SRGB).with_alpha(AlphaContent::Opaque);
    let working = [frame(RGBA8), frame(D::RGBAF32_LINEAR)];
    let output = [frame(D::RGBA16_SRGB)];
    let plan = keep()
        .plan_chain_preset(
            &source,
            D::RGBA16_SRGB,
            &[&working, &output],
            OutputChoice::Hidden,
        )
        .unwrap();
    assert_eq!(plan.steps[0].output, D::RGBAF32_LINEAR);
    assert_eq!(plan.steps[1].output, D::RGBA16_SRGB);
    assert_eq!(plan.peak_bytes, Some(4096 * 3072 * 16));
}

#[test]
fn chain_prefers_lower_peak_then_fewer_changes() {
    let source = declared(8);
    let a = [frame(D::RGB16_SRGB), frame(D::RGBF32_LINEAR)];
    let b = [frame(D::RGBF32_LINEAR)];
    let c = [frame(D::RGB8_SRGB)];
    let plan = keep()
        .plan_chain_preset(&source, D::RGB8_SRGB, &[&a, &b, &c], OutputChoice::Hidden)
        .unwrap();
    // RGB16 would be cheaper at the first node, but the peak is set by the second node anyway, so
    // staying in f32 saves a conversion.
    assert_eq!(plan.steps[0].output, D::RGBF32_LINEAR);
}

#[test]
fn chain_named_output_is_consent_only_at_the_last_node() {
    let source = SourcePrecision::from_storage(D::RGB16_SRGB);
    let working = [frame(D::RGB8_SRGB), frame(D::RGBF32_LINEAR)];
    let output = [frame(D::RGB8_SRGB)];
    let plan = keep()
        .plan_chain_preset(
            &source,
            D::RGB16_SRGB,
            &[&working, &output],
            OutputChoice::CallerNamed,
        )
        .unwrap();
    assert_eq!(
        plan.steps[0].output,
        D::RGBF32_LINEAR,
        "working format still holds it"
    );
    assert_eq!(plan.steps[1].effect.bits_dropped, 8);

    // A named output doesn't license narrowing an earlier node.
    let narrow = [frame(D::RGB8_SRGB)];
    let err = keep()
        .plan_chain_preset(
            &source,
            D::RGB16_SRGB,
            &[&narrow, &output],
            OutputChoice::CallerNamed,
        )
        .unwrap_err();
    assert_eq!(err.error().node, 0);
}

#[test]
fn chain_bits_dropped_early_do_not_come_back() {
    let source = SourcePrecision::from_storage(D::RGB16_SRGB);
    let working = [frame(D::RGB8_SRGB)];
    let output = [frame(D::RGB16_SRGB)];
    let plan = keep()
        .with_preset(ConversionPreset::DropPrecisionForMemory)
        .plan_chain_preset(
            &source,
            D::RGB16_SRGB,
            &[&working, &output],
            OutputChoice::Hidden,
        )
        .unwrap();
    assert_eq!(plan.steps[1].effect.bits_kept, 8);
    assert_eq!(plan.steps[1].effect.bits_dropped, 8);
    assert!(!plan.steps[1].effect.is_lossless());
}

#[test]
fn chain_memory_limit_names_the_node() {
    let source = real16();
    let limited = keep().with_max_peak_bytes(10);
    let err = limited
        .plan_chain_preset(
            &source,
            D::RGB16_SRGB,
            &[&[frame(D::RGB16_SRGB)]],
            OutputChoice::Hidden,
        )
        .unwrap_err();
    assert_eq!(err.error().node, 0);
    assert!(matches!(
        err.error().failure,
        ConversionFailure::Refused(ConversionRefusal::NoneAdmitted { .. })
    ));
}

#[test]
fn chain_empty_refuses() {
    let err = keep()
        .plan_chain_preset(&declared(8), D::RGB8_SRGB, &[], OutputChoice::Hidden)
        .unwrap_err();
    assert!(matches!(
        err.error().failure,
        ConversionFailure::Refused(ConversionRefusal::NoOffers)
    ));
}

#[test]
fn chain_memory_preset_narrows_only_because_it_says_so() {
    let source = real16();
    let working = [frame(D::RGB16_SRGB), frame(D::RGB8_SRGB)];
    let output = [frame(D::RGB16_SRGB), frame(D::RGB8_SRGB)];
    let plan = keep()
        .with_preset(ConversionPreset::DropPrecisionForMemory)
        .plan_chain_preset(
            &source,
            D::RGB16_SRGB,
            &[&working, &output],
            OutputChoice::Hidden,
        )
        .unwrap();
    assert_eq!(plan.steps[1].output, D::RGB8_SRGB);
    assert_eq!(plan.steps[1].effect.bits_dropped, 8);
}

#[test]
fn chain_resolver_pins_a_node_and_its_loss_carries() {
    let source = real16();
    let working = [frame(D::RGB8_SRGB), frame(D::RGBF32_LINEAR)];
    let output = [frame(D::RGB16_SRGB)];
    let mut asked = 0;
    let mut pin_first_working =
        |q: ConversionQuestion<'_>| -> Result<ConversionDecision, At<Never>> {
            asked += 1;
            Ok(
                if q.choice == OutputChoice::Hidden && q.admitted().count() == 2 {
                    ConversionDecision::Choose(0)
                } else {
                    ConversionDecision::Defer
                },
            )
        };
    let plan = keep()
        .plan_chain(
            &source,
            D::RGB16_SRGB,
            &[&working, &output],
            OutputChoice::Hidden,
            &mut pin_first_working,
        )
        .unwrap();
    assert_eq!(asked, 2);
    assert_eq!(plan.steps[0].decided_by, DecidedBy::Resolver);
    assert_eq!(plan.steps[0].effect.bits_dropped, 8);
    assert_eq!(plan.steps[1].decided_by, DecidedBy::Preset);
    assert_eq!(
        plan.steps[1].effect.bits_dropped, 8,
        "16-bit output doesn't bring them back"
    );
}

#[test]
fn chain_resolver_failure_and_refusal_name_the_node() {
    let output = [frame(D::RGB8_SRGB)];
    let mut fail = |_: ConversionQuestion<'_>| -> Result<ConversionDecision, At<TenantRule>> {
        Err(whereat::at!(TenantRule("no")))
    };
    let err = keep()
        .plan_chain(
            &declared(8),
            D::RGB8_SRGB,
            &[&output, &output],
            OutputChoice::Hidden,
            &mut fail,
        )
        .unwrap_err();
    assert_eq!(err.error().node, 0);
    assert!(matches!(
        err.error().failure,
        ConversionFailure::Resolver(TenantRule("no"))
    ));

    let mut count = 0;
    let mut refuse_second = |_: ConversionQuestion<'_>| -> Result<ConversionDecision, At<Never>> {
        count += 1;
        Ok(if count == 2 {
            ConversionDecision::Refuse
        } else {
            ConversionDecision::Defer
        })
    };
    let err = keep()
        .plan_chain(
            &declared(8),
            D::RGB8_SRGB,
            &[&output, &output],
            OutputChoice::Hidden,
            &mut refuse_second,
        )
        .unwrap_err();
    assert_eq!(err.error().node, 1);
    assert!(matches!(
        err.error().failure,
        ConversionFailure::Refused(ConversionRefusal::ResolverRefused)
    ));
}

#[test]
fn chain_after_a_pinned_narrowing_or_a_matte_later_nodes_see_what_they_receive() {
    // Pinned to 8 bits, so an 8-bit output further on loses nothing more and needs no consent.
    let source = real16();
    let working = [frame(D::RGB8_SRGB), frame(D::RGBF32_LINEAR)];
    let output = [frame(D::RGB8_SRGB)];
    let mut pin = |q: ConversionQuestion<'_>| -> Result<ConversionDecision, At<Never>> {
        Ok(if q.admitted().count() == 2 {
            ConversionDecision::Choose(0)
        } else {
            ConversionDecision::Defer
        })
    };
    let plan = keep()
        .plan_chain(
            &source,
            D::RGB16_SRGB,
            &[&working, &output],
            OutputChoice::Hidden,
            &mut pin,
        )
        .unwrap();
    assert_eq!(plan.steps[1].effect.bits_dropped, 8);

    // Composited onto a matte, so there's no alpha left to lose at a later RGB node.
    let translucent = declared(8).with_alpha(AlphaContent::Translucent);
    let flatten = [frame(D::RGB8_SRGB).with_matte()];
    let rgb = [frame(D::RGB8_SRGB)];
    let plan = keep()
        .plan_chain_preset(
            &translucent,
            D::RGBA8_SRGB,
            &[&flatten, &rgb],
            OutputChoice::Hidden,
        )
        .unwrap();
    assert!(plan.steps.iter().all(|s| s.effect.is_lossless()));
}

#[test]
fn chain_plan_reports_an_unknown_peak_as_unknown() {
    let plan = keep()
        .plan_chain_preset(
            &declared(8),
            D::RGB8_SRGB,
            &[&[offer(D::RGB8_SRGB)]],
            OutputChoice::Hidden,
        )
        .unwrap();
    assert_eq!(plan.peak_bytes, None);
}

// ---- What the curve tables claim ----

fn srgb(x: f64) -> f64 {
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}
fn bt709(x: f64) -> f64 {
    if x < 0.081 {
        x / 4.5
    } else {
        ((x + 0.099) / 1.099).powf(1.0 / 0.45)
    }
}
fn gamma22(x: f64) -> f64 {
    x.powf(563.0 / 256.0)
}
fn hlg(x: f64) -> f64 {
    let a: f64 = 0.17883277;
    let b = 1.0 - 4.0 * a;
    let c = 0.5 - a * (4.0 * a).ln();
    if x <= 0.5 {
        x * x / 3.0
    } else {
        (((x - c) / a).exp() + b) / 12.0
    }
}
fn pq_eotf(x: f64) -> f64 {
    let (m1, m2): (f64, f64) = (2610.0 / 16384.0, 2523.0 / 4096.0 * 128.0);
    let (c1, c2, c3): (f64, f64, f64) = (
        3424.0 / 4096.0,
        2413.0 / 4096.0 * 32.0,
        2392.0 / 4096.0 * 32.0,
    );
    let p = x.powf(1.0 / m2);
    ((p - c1).max(0.0) / (c2 - c3 * p)).powf(1.0 / m1)
}

type Eotf = fn(f64) -> f64;

const CURVES: [(T, Eotf); 5] = [
    (T::Srgb, srgb),
    (T::Bt709, bt709),
    (T::Gamma22, gamma22),
    (T::Hlg, hlg),
    (T::Pq, pq_eotf),
];

fn largest_source(step_ok: impl Fn(u32) -> bool) -> u8 {
    (1..=64u32).filter(|&n| step_ok(n)).max().unwrap_or(0) as u8
}

#[test]
fn gamma_into_linear_table_matches_the_darkest_step_of_each_curve() {
    for (transfer, eotf) in CURVES {
        for (channel, width) in [
            (ChannelType::U8, 8u32),
            (ChannelType::U16, 16),
            (ChannelType::U32, 32),
            (ChannelType::U64, 64),
        ] {
            let expected = largest_source(|n| {
                eotf(1.0 / (2f64.powi(n as i32) - 1.0)) * (2f64.powi(width as i32) - 1.0) >= 1.0
            });
            let source = SourcePrecision::declared(64, transfer, SignalRange::Full).unwrap();
            let got = ConversionEffect::of(&source, &offer(linear(channel, ChannelLayout::Rgb)))
                .bits_kept;
            assert_eq!(got, expected, "{transfer:?} into {width}-bit linear");
        }
    }
}

#[test]
fn linear_into_gamma_table_matches_the_step_at_white_of_each_curve() {
    let linear_source = SourcePrecision::declared(64, T::Linear, SignalRange::Full).unwrap();
    for (transfer, eotf) in CURVES {
        let h = 1e-7;
        let slope = h / (eotf(1.0) - eotf(1.0 - h));
        for (channel, width) in [
            (ChannelType::U8, 8u32),
            (ChannelType::U16, 16),
            (ChannelType::U32, 32),
            (ChannelType::U64, 64),
        ] {
            let expected = largest_source(|n| {
                slope * (2f64.powi(width as i32) - 1.0) / (2f64.powi(n as i32) - 1.0) >= 1.0
            });
            let output = D::new(channel, ChannelLayout::Rgb, None, transfer);
            let got = ConversionEffect::of(&linear_source, &offer(output)).bits_kept;
            assert_eq!(got, expected, "linear into {width}-bit {transfer:?}");
        }
    }
}

/// Every code through the curve, rounded to storage, stays distinct at the depth the table claims.
fn distinct_through(n: u32, width: u32, curve: impl Fn(f64) -> f64) -> bool {
    let codes = (1u64 << n) - 1;
    let scale = (1u64 << width) as f64 - 1.0;
    let mut previous = None;
    for code in 0..=codes {
        let rounded = (curve(code as f64 / codes as f64) * scale).round() as u64;
        if previous == Some(rounded) {
            return false;
        }
        previous = Some(rounded);
    }
    true
}

#[test]
fn table_entries_keep_every_code_distinct_when_run_exhaustively() {
    for (transfer, eotf) in [
        (T::Srgb, srgb as Eotf),
        (T::Bt709, bt709),
        (T::Gamma22, gamma22),
        (T::Hlg, hlg),
        (T::Pq, pq_eotf),
    ] {
        for (channel, width) in [(ChannelType::U8, 8), (ChannelType::U16, 16)] {
            let source = SourcePrecision::declared(16, transfer, SignalRange::Full).unwrap();
            let kept = ConversionEffect::of(&source, &offer(linear(channel, ChannelLayout::Rgb)))
                .bits_kept;
            assert!(
                distinct_through(u32::from(kept), width, eotf),
                "{transfer:?} {kept} bits into {width}-bit linear collapse"
            );
        }
    }
    // And the ones that matter most, one bit past the entry, do collapse.
    assert!(!distinct_through(13, 16, srgb));
    assert!(!distinct_through(5, 8, srgb));
    // 8-bit linear to sRGB: the highlights collapse, so a linear source doesn't get that for free.
    let oetf = |x: f64| {
        if x <= 0.0031308 {
            12.92 * x
        } else {
            1.055 * x.powf(1.0 / 2.4) - 0.055
        }
    };
    assert!(distinct_through(6, 8, oetf));
    assert!(!distinct_through(8, 8, oetf));
}

#[test]
fn float_capacity_is_the_mantissa_near_white() {
    // n-bit full-range codes in a float of `m` mantissa bits, rounded to its spacing in [0.5, 1).
    let recovers = |n: u32, m: u32| {
        let codes = (1u64 << n) - 1;
        let spacing = 2f64.powi(-(m as i32));
        (0..=codes).all(|code| {
            let v = code as f64 / codes as f64;
            if v < 0.5 {
                return true; // Finer spacing below.
            }
            let stored = (v / spacing).round() * spacing;
            (stored * codes as f64).round() as u64 == code
        })
    };
    assert!(recovers(11, 11), "half holds 11-bit codes");
    assert!(!recovers(12, 11), "half doesn't hold 12-bit codes");
    assert!(recovers(16, 24), "f32 holds u16");
    let half = linear(ChannelType::F16, ChannelLayout::Rgb);
    assert_eq!(
        ConversionEffect::of(&declared(16), &offer(half)).bits_kept,
        11
    );
    assert_eq!(
        ConversionEffect::of(&declared(32), &offer(D::RGBF32_LINEAR)).bits_kept,
        24
    );
}

#[test]
fn storage_alone_is_enough_between_curves_and_when_unknown() {
    // sRGB to BT.709 is close enough that storage answers it, and unknown has no curve to count.
    let bt = D::new(ChannelType::U8, ChannelLayout::Rgb, None, T::Bt709);
    assert!(ConversionEffect::of(&declared(8), &offer(bt)).is_lossless());
    let unknown = D::new(ChannelType::U8, ChannelLayout::Rgb, None, T::Unknown);
    assert!(ConversionEffect::of(&declared(8), &offer(unknown)).is_lossless());
}

#[test]
fn a_full_range_source_does_not_fit_a_narrow_grid_of_the_same_width() {
    let narrow = offer(D::RGB8_SRGB.with_signal_range(SignalRange::Narrow));
    assert_eq!(ConversionEffect::of(&declared(8), &narrow).bits_kept, 7);
    let narrow_source = SourcePrecision::declared(8, T::Srgb, SignalRange::Narrow).unwrap();
    assert!(ConversionEffect::of(&narrow_source, &narrow).is_lossless());
}
