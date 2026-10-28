//! Current-sample precision evidence through end-to-end scenarios.
//!
//! The implemented design carries [`ExactDepth`] on `PixelBuffer`/`PixelSlice`
//! and clears it on every mutable access. Two alternatives are prototyped here,
//! test-only, for comparison: an append-only edit log and an owning typestate
//! wrapper. A model encoder-depth resolver shows how requirements, evidence and
//! preferences combine; it is a proposal for the codec layer, not zenpixels API.

use std::sync::Arc;
use zenpixels::Orientation;
use zenpixels::{
    ChannelType, Cicp, ColorContext, PixelBuffer, PixelCow, PixelDescriptor, PixelSlice,
    SignalRange, TransferFunction,
    sample::{CodePlane, ExactDepth, SampleEncoding},
};
use zenpixels_convert::{
    PixelBufferConvertExt,
    codes::{CodeMap, CodeRole, to_canonical},
    orient::{apply_orientation, apply_orientation_in_place},
};

const W: u32 = 9;
const H: u32 = 5;

fn enc(bits: u8) -> SampleEncoding {
    SampleEncoding::new(ChannelType::U16, bits, 0).unwrap()
}

/// A decoder stand-in: native `bits`-deep interleaved codes widened exactly
/// into a canonical buffer, which states the evidence it just established.
fn decode_native(bits: u8, descriptor: PixelDescriptor, seed: usize) -> PixelBuffer {
    let channels = descriptor.channels();
    let mask = ((1u32 << bits) - 1) as u16;
    let count = (W * H) as usize * channels;
    let codes: Vec<u16> = (0..count)
        .map(|i| (((i + seed).wrapping_mul(2_654_435_761) >> 11) as u16) & mask)
        .collect();
    let planes: Vec<CodePlane<'_>> = (0..channels)
        .map(|c| {
            CodePlane::from_u16_interleaved(
                &codes[c..],
                W as usize,
                H as usize,
                W as usize * channels * 2,
                channels,
                enc(bits),
            )
            .unwrap()
        })
        .collect();
    let mut buffer = PixelBuffer::new(W, H, descriptor);
    let (_, exact) = to_canonical(
        &planes,
        descriptor.signal_range,
        0,
        0,
        buffer.as_slice_mut(),
    )
    .unwrap();
    buffer.set_exact_depth(Some(exact)).unwrap();
    buffer
}

fn samples(slice: &PixelSlice<'_>) -> Vec<u16> {
    let n = slice.width() as usize * slice.descriptor().channels();
    (0..slice.rows())
        .flat_map(|y| {
            let row = slice.row(y);
            (0..n)
                .map(move |i| u16::from_ne_bytes([row[2 * i], row[2 * i + 1]]))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Encoder/decoder stand-in: quantize every canonical sample to `bits` and widen
/// back with the canonical rule. Returns how many samples changed.
fn changed_by_roundtrip(slice: &PixelSlice<'_>, bits: u8) -> usize {
    let d = slice.descriptor();
    let channels = d.channels();
    let alpha = d.has_alpha().then(|| channels - 1);
    let rule = |role| {
        (
            CodeMap::new(enc(16), enc(bits), role, d.signal_range).unwrap(),
            CodeMap::new(enc(bits), enc(16), role, d.signal_range).unwrap(),
        )
    };
    let (color_down, color_up) = rule(CodeRole::Intensity);
    let (alpha_down, alpha_up) = rule(CodeRole::Alpha);
    samples(slice)
        .iter()
        .enumerate()
        .filter(|(i, v)| {
            let back = if Some(i % channels) == alpha {
                alpha_up.code(alpha_down.code(**v))
            } else {
                color_up.code(color_down.code(**v))
            };
            back != **v
        })
        .count()
}

fn write_samples(buffer: &mut PixelBuffer, values: &[u16]) {
    let n = buffer.width() as usize * buffer.descriptor().channels();
    let mut dst = buffer.as_slice_mut();
    for y in 0..dst.rows() {
        let row = dst.row_mut(y);
        let source = &values[y as usize * n..(y as usize + 1) * n];
        for (bytes, value) in row[..2 * n].as_chunks_mut::<2>().0.iter_mut().zip(source) {
            *bytes = value.to_ne_bytes();
        }
    }
}

fn from_samples(descriptor: PixelDescriptor, w: u32, h: u32, values: &[u16]) -> PixelBuffer {
    let mut buffer = PixelBuffer::new(w, h, descriptor);
    write_samples(&mut buffer, values);
    buffer
}

// ---------------------------------------------------------------------------
// Model encoder-depth resolver (proposal for the codec layer)
// ---------------------------------------------------------------------------

/// Mandatory caller requirements, applied before any preference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Requirement {
    /// Decoding must reproduce every current sample.
    Lossless,
    /// Code exactly this many bits (lossy if the samples are not exact there).
    Exactly(u8),
    /// No more than this many bits (for example, a consumer limit).
    AtMost(u8),
}

/// Situational choice among the depths every requirement admitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Preference {
    /// The smallest depth proven exact; otherwise the largest admitted.
    PreserveProvable,
    /// The smallest admitted depth.
    Smallest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Decision {
    bits: u8,
    exact: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Refusal {
    Unmet {
        requirement: Requirement,
        admitted_before: Vec<u8>,
    },
}

fn exact_at(evidence: Option<ExactDepth>, storage_bits: u8, has_alpha: bool, bits: u8) -> bool {
    match evidence {
        Some(e) => e.color_exact_at(bits) && (!has_alpha || e.alpha_exact_at(bits)),
        None => bits >= storage_bits,
    }
}

fn resolve(
    requirements: &[Requirement],
    preference: Preference,
    pixels: &PixelSlice<'_>,
    capabilities: &[u8],
) -> Result<Decision, Refusal> {
    let storage_bits = 8 * pixels.descriptor().channel_type().byte_size() as u8;
    let has_alpha = pixels.descriptor().has_alpha();
    let evidence = pixels.exact_depth();
    let exact = |bits| exact_at(evidence, storage_bits, has_alpha, bits);
    let mut admitted: Vec<u8> = capabilities.to_vec();
    for &requirement in requirements {
        let before = admitted.clone();
        admitted.retain(|&bits| match requirement {
            Requirement::Lossless => exact(bits),
            Requirement::Exactly(b) => bits == b,
            Requirement::AtMost(b) => bits <= b,
        });
        if admitted.is_empty() {
            return Err(Refusal::Unmet {
                requirement,
                admitted_before: before,
            });
        }
    }
    let bits = match preference {
        Preference::PreserveProvable => admitted
            .iter()
            .copied()
            .filter(|&b| exact(b))
            .min()
            .unwrap_or_else(|| *admitted.iter().max().unwrap()),
        Preference::Smallest => *admitted.iter().min().unwrap(),
    };
    Ok(Decision {
        bits,
        exact: exact(bits),
    })
}

/// What a provenance-driven rule does: code the source file's depth.
fn ancestry_depth(source_bits: u8, capabilities: &[u8]) -> u8 {
    capabilities
        .iter()
        .copied()
        .filter(|&b| b >= source_bits)
        .min()
        .unwrap()
}

const AV1: &[u8] = &[8, 10, 12];
const PNG: &[u8] = &[8, 16];

// ---------------------------------------------------------------------------
// Encode without edits, and metadata-only transcodes
// ---------------------------------------------------------------------------

#[test]
fn unedited_native_depth_is_proven_and_encoded_exactly() {
    for (bits, want) in [(12u8, 12u8), (10, 10), (8, 8)] {
        let decoded = decode_native(bits, PixelDescriptor::RGB16_SRGB, 1);
        let pixels = decoded.as_slice();
        assert_eq!(pixels.exact_depth().unwrap().color_bits(), bits);
        let d = resolve(
            &[Requirement::Lossless],
            Preference::PreserveProvable,
            &pixels,
            AV1,
        )
        .unwrap();
        assert_eq!(
            d,
            Decision {
                bits: want,
                exact: true
            }
        );
        assert_eq!(changed_by_roundtrip(&pixels, d.bits), 0);
    }
}

#[test]
fn ten_bit_content_is_not_exact_at_twelve_bits() {
    let decoded = decode_native(10, PixelDescriptor::RGB16_SRGB, 2);
    let pixels = decoded.as_slice();
    // Lossless admits only 10 among {8, 10, 12}: promotion to 12 changes samples.
    let d = resolve(
        &[Requirement::Lossless],
        Preference::PreserveProvable,
        &pixels,
        AV1,
    )
    .unwrap();
    assert_eq!(d.bits, 10);
    assert!(changed_by_roundtrip(&pixels, 12) > 0);
    // A caller can still mandate 12; the decision reports it is not exact.
    let d = resolve(
        &[Requirement::Exactly(12)],
        Preference::PreserveProvable,
        &pixels,
        AV1,
    )
    .unwrap();
    assert_eq!(
        d,
        Decision {
            bits: 12,
            exact: false
        }
    );
    // Narrow-range grids nest, so narrow ten-bit content is exact at 12.
    let narrow = decode_native(
        10,
        PixelDescriptor::RGB16_SRGB.with_signal_range(SignalRange::Narrow),
        2,
    );
    let pixels = narrow.as_slice();
    let d = resolve(
        &[Requirement::Lossless, Requirement::Exactly(12)],
        Preference::Smallest,
        &pixels,
        AV1,
    )
    .unwrap();
    assert_eq!(
        d,
        Decision {
            bits: 12,
            exact: true
        }
    );
    assert_eq!(changed_by_roundtrip(&pixels, 12), 0);
}

#[test]
fn metadata_only_transcode_keeps_evidence() {
    let decoded = decode_native(12, PixelDescriptor::RGBA16_SRGB, 3);
    let evidence = decoded.exact_depth();
    let relabeled = decoded
        .with_color_context(Arc::new(ColorContext::from_cicp(Cicp::DISPLAY_P3)))
        .with_transfer(TransferFunction::Srgb);
    assert_eq!(
        relabeled.exact_depth(),
        evidence,
        "assigning color metadata does not change samples"
    );
    let d = resolve(
        &[Requirement::Lossless],
        Preference::PreserveProvable,
        &relabeled.as_slice(),
        AV1,
    )
    .unwrap();
    assert_eq!(
        d,
        Decision {
            bits: 12,
            exact: true
        }
    );
    // A signal-range relabel changes what the codes mean: evidence ends.
    let narrow = relabeled.with_signal_range(SignalRange::Narrow);
    assert_eq!(narrow.exact_depth(), None);
}

// ---------------------------------------------------------------------------
// Which operations keep the evidence
// ---------------------------------------------------------------------------

#[test]
fn views_crops_clones_and_copies_keep_evidence() {
    let decoded = decode_native(10, PixelDescriptor::RGBA16_SRGB, 4);
    let e = decoded.exact_depth();
    assert!(e.is_some());
    assert_eq!(decoded.as_slice().exact_depth(), e);
    assert_eq!(decoded.rows(1, 2).exact_depth(), e);
    assert_eq!(decoded.crop_view(2, 1, 3, 3).exact_depth(), e);
    assert_eq!(decoded.as_slice().clone().exact_depth(), e);
    let crop = decoded.crop_copy(1, 1, 4, 2);
    assert_eq!(crop.exact_depth(), e);
    assert_eq!(changed_by_roundtrip(&crop.as_slice(), 10), 0);
    // A borrowed cow copied into ownership keeps the same values and evidence.
    let slice = decoded.as_slice();
    let mut cow = PixelCow::from(slice);
    assert_eq!(cow.to_mut().exact_depth(), e);
    let contiguous = decoded.crop_copy(0, 0, W, H).into_contiguous();
    assert_eq!(contiguous.exact_depth(), e);
    // An identity conversion copies the same samples under the same descriptor.
    assert_eq!(
        decoded
            .convert_to(decoded.descriptor())
            .unwrap()
            .exact_depth(),
        e
    );
}

#[test]
fn every_mutable_access_ends_evidence_even_without_a_write() {
    let mut b = decode_native(10, PixelDescriptor::RGB16_SRGB, 5);
    let e = b.exact_depth();
    let _ = b.as_slice_mut();
    assert_eq!(b.exact_depth(), None, "as_slice_mut");
    b.set_exact_depth(e).unwrap();
    let _ = b.rows_mut(0, 1);
    assert_eq!(b.exact_depth(), None, "rows_mut");
    b.set_exact_depth(e).unwrap();
    b.transform_in_place(|p| {
        zenpixels::PixelSliceMut::new(p.bytes, p.width, p.rows, p.stride, p.descriptor).unwrap()
    });
    assert_eq!(b.exact_depth(), None, "transform_in_place");
    // Conservative: an untouched buffer lost evidence; an explicit scan restores it.
    let measured = ExactDepth::measure(&b.as_slice()).unwrap();
    assert_eq!(Some(measured), e);
    b.set_exact_depth(Some(measured)).unwrap();
    // Ownership handoff through public parts drops evidence: parts are editable.
    let parts = b.into_parts();
    let adopted = PixelBuffer::try_from_parts(parts)
        .map_err(|e| e.without_buffer())
        .unwrap();
    assert_eq!(adopted.exact_depth(), None);
}

#[test]
fn relabels_keep_evidence_only_when_storage_layout_and_range_are_unchanged() {
    let b = decode_native(10, PixelDescriptor::RGBA16_SRGB, 6);
    let e = b.exact_depth();
    let s = b.as_slice();
    assert_eq!(
        s.clone()
            .with_transfer(TransferFunction::Linear)
            .exact_depth(),
        e
    );
    assert_eq!(
        s.clone()
            .with_descriptor(s.descriptor().with_transfer(TransferFunction::Pq))
            .exact_depth(),
        e
    );
    assert_eq!(
        s.clone()
            .with_signal_range(SignalRange::Narrow)
            .exact_depth(),
        None
    );
    let mismatch = ExactDepth::new(ChannelType::U8, SignalRange::Full, 8, 8).unwrap();
    assert!(s.clone().with_exact_depth(mismatch).is_err());
}

#[test]
fn resize_filters_color_conversion_and_premultiplication_end_evidence() {
    let decoded = decode_native(10, PixelDescriptor::RGBA16_SRGB, 7);
    let src = samples(&decoded.as_slice());
    let ch = 4usize;
    // Box 2x horizontal downscale: new values, new buffer, no evidence.
    let half_w = W / 2;
    let resized: Vec<u16> = (0..H as usize)
        .flat_map(|y| (0..half_w as usize).flat_map(move |x| (0..ch).map(move |c| (y, x, c))))
        .map(|(y, x, c)| {
            let a = u32::from(src[(y * W as usize + 2 * x) * ch + c]);
            let b = u32::from(src[(y * W as usize + 2 * x + 1) * ch + c]);
            (a + b).div_ceil(2) as u16
        })
        .collect();
    let resized = from_samples(PixelDescriptor::RGBA16_SRGB, half_w, H, &resized);
    assert_eq!(resized.exact_depth(), None);
    assert!(
        changed_by_roundtrip(&resized.as_slice(), 10) > 0,
        "resize created values off the 10-bit grid"
    );

    // In-place filter: +1 on color; mutable access already ended evidence.
    let mut filtered = decode_native(10, PixelDescriptor::RGBA16_SRGB, 7);
    let mut values = samples(&filtered.as_slice());
    for (i, v) in values.iter_mut().enumerate() {
        if i % ch != 3 {
            *v = v.saturating_add(1);
        }
    }
    write_samples(&mut filtered, &values);
    assert_eq!(filtered.exact_depth(), None);
    assert!(changed_by_roundtrip(&filtered.as_slice(), 10) > 0);

    // Color conversion allocates new samples: no evidence crosses it.
    let linear = decoded.convert_to(PixelDescriptor::RGBAF32_LINEAR).unwrap();
    assert_eq!(linear.exact_depth(), None);
    let back = linear.convert_to(PixelDescriptor::RGBA16_SRGB).unwrap();
    assert_eq!(back.exact_depth(), None);

    // Premultiply then unpremultiply with intermediate rounding.
    let pre: Vec<u16> = src
        .chunks_exact(ch)
        .flat_map(|px| {
            let a = u32::from(px[3]);
            let mut out = [px[0], px[1], px[2], px[3]];
            for c in &mut out[..3] {
                *c = ((u32::from(*c) * a + 32767) / 65535) as u16;
            }
            out
        })
        .collect();
    let un: Vec<u16> = pre
        .chunks_exact(ch)
        .flat_map(|px| {
            let a = u32::from(px[3]);
            let mut out = [px[0], px[1], px[2], px[3]];
            for c in &mut out[..3] {
                *c = (u32::from(*c) * 65535 + a / 2)
                    .checked_div(a)
                    .map_or(0, |v| v.min(65535) as u16);
            }
            out
        })
        .collect();
    let round_trip = from_samples(PixelDescriptor::RGBA16_SRGB, W, H, &un);
    assert_eq!(round_trip.exact_depth(), None);
    let measured = ExactDepth::measure(&round_trip.as_slice()).unwrap();
    assert!(
        !measured.color_exact_at(10),
        "premultiplication round trip left the 10-bit grid"
    );
    assert!(measured.alpha_exact_at(10), "alpha itself was untouched");
}

/// Composite evidence: a pure selection keeps values from either source, so its
/// evidence is the smallest grid containing both; blending creates new values.
fn selection_meet(a: ExactDepth, b: ExactDepth) -> Option<ExactDepth> {
    if a.storage() != b.storage() || a.range() != b.range() {
        return None;
    }
    let storage_bits = if a.storage() == ChannelType::U8 {
        8
    } else {
        16
    };
    let join = |x: u8, y: u8, narrow: bool| -> u8 {
        let (x32, y32) = (u32::from(x), u32::from(y));
        let mut g = (x32, y32);
        while g.1 != 0 {
            g = (g.1, g.0 % g.1);
        }
        let combined = if narrow {
            x32.max(y32)
        } else {
            x32 / g.0 * y32
        };
        combined.min(storage_bits) as u8
    };
    let narrow = a.range() == SignalRange::Narrow;
    ExactDepth::new(
        a.storage(),
        a.range(),
        join(a.color_bits(), b.color_bits(), narrow),
        join(a.alpha_bits(), b.alpha_bits(), false),
    )
    .ok()
}

#[test]
fn composites_of_mixed_sources_keep_only_what_both_grids_share() {
    let ten = decode_native(10, PixelDescriptor::RGB16_SRGB, 8);
    let five = decode_native(5, PixelDescriptor::RGB16_SRGB, 9);
    let eight = decode_native(8, PixelDescriptor::RGB16_SRGB, 10);
    let pick = |a: &PixelBuffer, b: &PixelBuffer| -> PixelBuffer {
        let (sa, sb) = (samples(&a.as_slice()), samples(&b.as_slice()));
        let mixed: Vec<u16> = sa
            .iter()
            .zip(&sb)
            .enumerate()
            .map(|(i, (x, y))| if (i / 3).is_multiple_of(2) { *x } else { *y })
            .collect();
        let mut out = from_samples(PixelDescriptor::RGB16_SRGB, W, H, &mixed);
        let meet = selection_meet(a.exact_depth().unwrap(), b.exact_depth().unwrap());
        out.set_exact_depth(meet).unwrap();
        out
    };
    let ten_five = pick(&ten, &five);
    assert_eq!(
        ten_five.exact_depth().unwrap().color_bits(),
        10,
        "5 divides 10"
    );
    assert_eq!(changed_by_roundtrip(&ten_five.as_slice(), 10), 0);
    let ten_eight = pick(&ten, &eight);
    assert_eq!(
        ten_eight.exact_depth().unwrap().color_bits(),
        16,
        "lcm(8, 10) exceeds storage"
    );
    assert!(changed_by_roundtrip(&ten_eight.as_slice(), 10) > 0);
    assert!(changed_by_roundtrip(&ten_eight.as_slice(), 8) > 0);
    // The scan agrees with the meet on the selected samples.
    let m = ExactDepth::measure(&ten_eight.as_slice()).unwrap();
    assert!(!m.color_exact_at(8) && !m.color_exact_at(10));
    // A 50% blend makes new values: no evidence, and the scan confirms 16.
    let (sa, sb) = (samples(&ten.as_slice()), samples(&five.as_slice()));
    let blend: Vec<u16> = sa
        .iter()
        .zip(&sb)
        .map(|(x, y)| (u32::from(*x) + u32::from(*y)).div_ceil(2) as u16)
        .collect();
    let blended = from_samples(PixelDescriptor::RGB16_SRGB, W, H, &blend);
    assert_eq!(blended.exact_depth(), None);
    assert!(changed_by_roundtrip(&blended.as_slice(), 10) > 0);
}

// ---------------------------------------------------------------------------
// Encoder decisions: stale ancestry, unknown provenance, requirement order
// ---------------------------------------------------------------------------

#[test]
fn stale_source_depth_would_discard_new_precision_after_resizing() {
    let decoded = decode_native(8, PixelDescriptor::RGB16_SRGB, 11);
    let src = samples(&decoded.as_slice());
    // A gamma-unaware 2-tap filter creates in-between values.
    let filtered: Vec<u16> = src
        .windows(4)
        .map(|w| (u32::from(w[0]) + u32::from(w[3])).div_ceil(2) as u16)
        .chain(src[src.len() - 3..].iter().copied())
        .collect();
    let edited = from_samples(PixelDescriptor::RGB16_SRGB, W, H, &filtered);
    // Ancestry says "8-bit source"; coding 8 bits silently drops the new precision.
    let by_ancestry = ancestry_depth(8, PNG);
    assert_eq!(by_ancestry, 8);
    assert!(changed_by_roundtrip(&edited.as_slice(), by_ancestry) > 0);
    // Current evidence is absent, so the resolver keeps the samples exact.
    let d = resolve(&[], Preference::PreserveProvable, &edited.as_slice(), PNG).unwrap();
    assert_eq!(
        d,
        Decision {
            bits: 16,
            exact: true
        }
    );
    // Quantizing to 8 bits remains possible, but only as an explicit lossy choice.
    let d = resolve(
        &[Requirement::AtMost(8)],
        Preference::PreserveProvable,
        &edited.as_slice(),
        PNG,
    )
    .unwrap();
    assert_eq!(
        d,
        Decision {
            bits: 8,
            exact: false
        }
    );
}

#[test]
fn unknown_evidence_refuses_lossless_when_no_depth_reaches_storage() {
    let unknown = from_samples(
        PixelDescriptor::RGB16_SRGB,
        W,
        H,
        &samples(&decode_native(10, PixelDescriptor::RGB16_SRGB, 12).as_slice()),
    );
    assert_eq!(unknown.exact_depth(), None);
    assert_eq!(
        resolve(
            &[Requirement::Lossless],
            Preference::PreserveProvable,
            &unknown.as_slice(),
            AV1
        ),
        Err(Refusal::Unmet {
            requirement: Requirement::Lossless,
            admitted_before: vec![8, 10, 12]
        })
    );
    // Without a requirement the resolver loses the least precision it can.
    assert_eq!(
        resolve(&[], Preference::PreserveProvable, &unknown.as_slice(), AV1),
        Ok(Decision {
            bits: 12,
            exact: false
        })
    );
    // An explicit scan is the counted way to prove the depth again.
    let mut proven = unknown;
    let measured = ExactDepth::measure(&proven.as_slice()).unwrap();
    proven.set_exact_depth(Some(measured)).unwrap();
    assert_eq!(
        resolve(
            &[Requirement::Lossless],
            Preference::PreserveProvable,
            &proven.as_slice(),
            AV1
        ),
        Ok(Decision {
            bits: 10,
            exact: true
        })
    );
}

#[test]
fn requirements_precede_preferences_and_conflicts_name_the_requirement() {
    let ten = decode_native(10, PixelDescriptor::RGBA16_SRGB, 13);
    let pixels = ten.as_slice();
    assert_eq!(
        resolve(
            &[Requirement::AtMost(8)],
            Preference::PreserveProvable,
            &pixels,
            AV1
        ),
        Ok(Decision {
            bits: 8,
            exact: false
        })
    );
    assert_eq!(
        resolve(
            &[Requirement::AtMost(8), Requirement::Lossless],
            Preference::PreserveProvable,
            &pixels,
            AV1
        ),
        Err(Refusal::Unmet {
            requirement: Requirement::Lossless,
            admitted_before: vec![8]
        })
    );
    assert_eq!(
        resolve(
            &[Requirement::Exactly(16)],
            Preference::Smallest,
            &pixels,
            AV1
        ),
        Err(Refusal::Unmet {
            requirement: Requirement::Exactly(16),
            admitted_before: vec![8, 10, 12]
        })
    );
    // Binary alpha is exact at any depth; 10-bit color still decides.
    let mut values = samples(&pixels);
    for (i, v) in values.iter_mut().enumerate() {
        if i % 4 == 3 {
            *v = if i % 8 == 3 { 0 } else { 65535 };
        }
    }
    let mut b = from_samples(PixelDescriptor::RGBA16_SRGB, W, H, &values);
    assert_eq!(
        ExactDepth::measure(&b.as_slice()).unwrap(),
        ExactDepth::new(ChannelType::U16, SignalRange::Full, 10, 1).unwrap()
    );
    b.set_exact_depth(Some(
        ExactDepth::new(ChannelType::U16, SignalRange::Full, 10, 1).unwrap(),
    ))
    .unwrap();
    assert_eq!(
        resolve(
            &[Requirement::Lossless],
            Preference::Smallest,
            &b.as_slice(),
            AV1
        ),
        Ok(Decision {
            bits: 10,
            exact: true
        })
    );
}

// ---------------------------------------------------------------------------
// Serialization: evidence read back is a claim until verified
// ---------------------------------------------------------------------------

#[test]
fn serialized_evidence_is_a_claim_that_a_scan_can_refute() {
    let decoded = decode_native(12, PixelDescriptor::RGB16_SRGB, 14);
    let e = decoded.exact_depth().unwrap();
    // Persist pixels and evidence as plain data.
    let mut stored = samples(&decoded.as_slice());
    let header = [e.color_bits(), e.alpha_bits()];
    // Another program edits the stored pixels but not the header.
    stored[7] ^= 1;
    let restored = from_samples(PixelDescriptor::RGB16_SRGB, W, H, &stored);
    let claim = ExactDepth::new(ChannelType::U16, SignalRange::Full, header[0], header[1]).unwrap();
    let verified = ExactDepth::measure(&restored.as_slice()).unwrap();
    assert!(
        !verified.color_exact_at(claim.color_bits()),
        "the stale claim is refuted"
    );
    assert!(changed_by_roundtrip(&restored.as_slice(), 12) > 0);
}

// ---------------------------------------------------------------------------
// Prototype A (rejected): an append-only edit log as precision authority
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
enum Edit {
    Decoded { bits: u8 },
    Crop,
    Resize,
    Composite(Vec<Edit>, Vec<Edit>),
}

fn log_entries(log: &[Edit]) -> usize {
    log.iter()
        .map(|e| match e {
            Edit::Composite(a, b) => 1 + log_entries(a) + log_entries(b),
            _ => 1,
        })
        .sum()
}

/// A log-reading resolver: only entries it understands keep the decoded depth.
fn log_depth(log: &[Edit]) -> Option<u8> {
    let mut bits = None;
    for e in log {
        match e {
            Edit::Decoded { bits: b } => bits = Some(*b),
            Edit::Crop => {}
            Edit::Resize | Edit::Composite(..) => return None,
        }
    }
    bits
}

#[test]
fn an_edit_log_misses_unlogged_mutation_and_grows_with_composition() {
    let mut buffer = decode_native(10, PixelDescriptor::RGB16_SRGB, 15);
    let log = vec![Edit::Decoded { bits: 10 }, Edit::Crop];
    // Code that edits pixels through a mutable view without appending an entry.
    let mut values = samples(&buffer.as_slice());
    values[0] = values[0].wrapping_add(3);
    write_samples(&mut buffer, &values);
    assert_eq!(log_depth(&[Edit::Decoded { bits: 10 }, Edit::Resize]), None);
    assert_eq!(
        log_depth(&log),
        Some(10),
        "the log still claims 10-bit exactness"
    );
    assert!(
        changed_by_roundtrip(&buffer.as_slice(), 10) > 0,
        "trusting it discards the edit"
    );
    assert_eq!(
        buffer.exact_depth(),
        None,
        "buffer evidence ended at the mutable access"
    );

    // Each composite must embed both inputs' histories to stay explainable.
    let frame = vec![Edit::Decoded { bits: 8 }, Edit::Crop];
    let mut accumulated = frame.clone();
    for _ in 0..30 {
        accumulated = vec![Edit::Composite(accumulated, frame.clone())];
    }
    assert_eq!(log_entries(&accumulated), 30 + 31 * 2);
    let mut tree = frame.clone();
    for _ in 0..10 {
        tree = vec![Edit::Composite(tree.clone(), tree)];
    }
    assert_eq!(log_entries(&tree), 1023 + 1024 * 2, "a 1024-tile mosaic");
    // ExactDepth for the same 1024-tile selection is still 4 bytes of state.
    assert_eq!(core::mem::size_of::<Option<ExactDepth>>(), 4);
}

// ---------------------------------------------------------------------------
// Prototype B (not chosen): an owning typestate wrapper
// ---------------------------------------------------------------------------

/// Owns pixels and their evidence; there is no mutable access to keep.
struct Proven {
    buffer: PixelBuffer,
}

impl Proven {
    fn new(mut buffer: PixelBuffer, evidence: ExactDepth) -> Self {
        buffer.set_exact_depth(Some(evidence)).unwrap();
        Self { buffer }
    }
    fn pixels(&self) -> PixelSlice<'_> {
        self.buffer.as_slice()
    }
    /// Every preserving operation needs its own wrapper method.
    fn crop(&self, x: u32, y: u32, w: u32, h: u32) -> Self {
        let evidence = self.buffer.exact_depth().unwrap();
        Self::new(self.buffer.crop_copy(x, y, w, h), evidence)
    }
    fn into_inner(self) -> PixelBuffer {
        let mut b = self.buffer;
        b.set_exact_depth(None).unwrap();
        b
    }
}

#[test]
fn a_typestate_wrapper_guarantees_at_compile_time_but_reduces_to_slice_evidence_at_the_codec() {
    let decoded = decode_native(12, PixelDescriptor::RGB16_SRGB, 16);
    let e = decoded.exact_depth().unwrap();
    let proven = Proven::new(decoded, e).crop(1, 1, 4, 3);
    // An encoder taking `PixelSlice` sees evidence only because the slice carries it.
    let d = resolve(
        &[Requirement::Lossless],
        Preference::PreserveProvable,
        &proven.pixels(),
        AV1,
    )
    .unwrap();
    assert_eq!(
        d,
        Decision {
            bits: 12,
            exact: true
        }
    );
    // Mutation requires leaving the wrapper, which ends the evidence.
    let mut owned = proven.into_inner();
    assert_eq!(owned.exact_depth(), None);
    let _ = owned.as_slice_mut();
}

// ---------------------------------------------------------------------------
// Exact representation changes re-attach transferred evidence
// ---------------------------------------------------------------------------

#[test]
fn reversible_conversions_transfer_evidence_that_survives_a_roundtrip() {
    // U16 RGB → RGBA: color evidence kept, inserted opaque alpha is on every grid.
    let rgb = decode_native(10, PixelDescriptor::RGB16_SRGB, 17);
    let rgba = rgb.convert_to(PixelDescriptor::RGBA16_SRGB).unwrap();
    let e = rgba
        .exact_depth()
        .expect("alpha insertion is a reversible representation change");
    assert_eq!((e.color_bits(), e.alpha_bits()), (10, 1));
    assert_eq!(changed_by_roundtrip(&rgba.as_slice(), 10), 0);

    // U8 → U16 multiplies by 257: 4-bit content keeps its grid, 5-bit lands on the 8-bit grid.
    for (bits, want) in [(4u8, 4u8), (5, 8), (8, 8)] {
        let codes: Vec<u8> = (0..(W * H * 3) as usize)
            .map(|i| ((i * 37) % (1 << bits)) as u8)
            .collect();
        let plane_encoding = SampleEncoding::new(ChannelType::U8, bits, 0).unwrap();
        let planes: Vec<CodePlane<'_>> = (0..3)
            .map(|c| {
                CodePlane::from_u8_interleaved(
                    &codes[c..],
                    W as usize,
                    H as usize,
                    W as usize * 3,
                    3,
                    plane_encoding,
                )
                .unwrap()
            })
            .collect();
        let mut eight = PixelBuffer::new(W, H, PixelDescriptor::RGB8_SRGB);
        let (_, exact) =
            to_canonical(&planes, SignalRange::Full, 0, 0, eight.as_slice_mut()).unwrap();
        eight.set_exact_depth(Some(exact)).unwrap();
        let wide = eight.convert_to(PixelDescriptor::RGB16_SRGB).unwrap();
        let e = wide.exact_depth().unwrap();
        assert_eq!(e.color_bits(), want, "{bits}-bit content after ×257");
        assert_eq!(changed_by_roundtrip(&wide.as_slice(), want), 0);
        if bits == 5 {
            assert!(
                changed_by_roundtrip(&wide.as_slice(), 5) > 0,
                "the 5-bit grid did not survive ×257"
            );
        }
    }
    // A lossy conversion transfers nothing.
    let narrowed = rgb.convert_to(PixelDescriptor::RGB8_SRGB).unwrap();
    assert_eq!(narrowed.exact_depth(), None);
}

#[test]
fn orientation_moves_pixels_and_keeps_evidence() {
    let decoded = decode_native(12, PixelDescriptor::RGBA16_SRGB, 18);
    let e = decoded.exact_depth();
    for orientation in [
        Orientation::Rotate90,
        Orientation::FlipH,
        Orientation::Transverse,
    ] {
        let rotated = apply_orientation(decoded.as_slice(), orientation);
        assert_eq!(rotated.exact_depth(), e, "{orientation:?}");
        assert_eq!(changed_by_roundtrip(&rotated.as_slice(), 12), 0);
    }
    let mut in_place = decoded.crop_copy(0, 0, W, H);
    apply_orientation_in_place(&mut in_place, Orientation::Rotate270).unwrap();
    assert_eq!(in_place.exact_depth(), e);
    assert_eq!(changed_by_roundtrip(&in_place.as_slice(), 12), 0);
}

#[test]
fn narrow_depth_changes_refuse_and_never_carry_full_scale_evidence() {
    use zenpixels_convert::{ConvertError, ConvertPlan};
    // Narrow U8 to U16 must scale anchors by 2^8 (16 -> 4096, 235 -> 60160);
    // the existing depth kernel multiplies by 257 (16 -> 4112). Planning and
    // convert_to refuse the crossing, so the full-scale carry rule never sees
    // narrow evidence. Checked on narrow descriptors, not inferred from full range.
    let narrow = |d: PixelDescriptor| d.with_signal_range(SignalRange::Narrow);
    let rgb8 = narrow(PixelDescriptor::RGB8_SRGB);
    let bytes: Vec<u8> = (0..(W * H * 3) as usize)
        .map(|i| 16 + (i * 37 % 220) as u8)
        .collect();
    let mut eight = PixelBuffer::from_vec(bytes, W, H, rgb8).unwrap();
    let measured = ExactDepth::measure(&eight.as_slice()).unwrap();
    assert_eq!(
        (measured.range(), measured.color_bits()),
        (SignalRange::Narrow, 8)
    );
    eight.set_exact_depth(Some(measured)).unwrap();
    for target in [
        narrow(PixelDescriptor::RGB16_SRGB),
        PixelDescriptor::RGB16_SRGB,
        PixelDescriptor::RGB8_SRGB,
    ] {
        assert!(
            matches!(
                *ConvertPlan::new_preserving_samples(rgb8, target)
                    .unwrap_err()
                    .error(),
                ConvertError::NoPath { .. }
            ),
            "{target:?}"
        );
        assert!(eight.convert_to(target).is_err(), "{target:?}");
    }
    // Full-range samples cannot enter narrow storage either.
    assert!(
        ConvertPlan::new_preserving_samples(
            PixelDescriptor::RGB8_SRGB,
            narrow(PixelDescriptor::RGB16_SRGB)
        )
        .is_err()
    );
    // A narrow layout change at the same depth moves values unchanged, and the
    // evidence it carries equals a fresh scan of the result.
    let rgba = eight
        .convert_to(narrow(PixelDescriptor::RGBA8_SRGB))
        .unwrap();
    let carried = rgba
        .exact_depth()
        .expect("same-depth narrow layout change keeps evidence");
    assert_eq!(
        (carried.range(), carried.color_bits(), carried.alpha_bits()),
        (SignalRange::Narrow, 8, 1)
    );
    assert_eq!(carried, ExactDepth::measure(&rgba.as_slice()).unwrap());
}
