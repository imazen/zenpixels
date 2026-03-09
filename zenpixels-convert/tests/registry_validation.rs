#![cfg(feature = "pipeline")]
//! Validation tests for codec registry declarations.
//!
//! Ensures format entries are consistent and that negotiation works correctly
//! for every registered codec.

use zenpixels_convert::pipeline::registry::{self, ALL_CODECS};
use zenpixels_convert::{ConvertIntent, PixelDescriptor, best_match};

// ---------------------------------------------------------------------------
// Structural invariants
// ---------------------------------------------------------------------------

#[test]
fn all_codecs_have_unique_names() {
    let names: Vec<&str> = ALL_CODECS.iter().map(|c| c.name).collect();
    for (i, name) in names.iter().enumerate() {
        for (j, other) in names.iter().enumerate() {
            if i != j {
                assert_ne!(
                    name, other,
                    "duplicate codec name: {name}"
                );
            }
        }
    }
}

#[test]
fn no_empty_codec_names() {
    for codec in ALL_CODECS {
        assert!(!codec.name.is_empty(), "codec has empty name");
    }
}

#[test]
fn effective_bits_are_nonzero() {
    for codec in ALL_CODECS {
        for entry in codec.decode_outputs.iter().chain(codec.encode_inputs.iter()) {
            assert!(
                entry.effective_bits > 0,
                "{}: effective_bits is 0 for {:?}",
                codec.name,
                entry.descriptor
            );
        }
    }
}

#[test]
fn no_duplicate_descriptors_in_decode() {
    for codec in ALL_CODECS {
        let descs: Vec<PixelDescriptor> = codec.decode_outputs.iter().map(|e| e.descriptor).collect();
        for (i, d) in descs.iter().enumerate() {
            for (j, other) in descs.iter().enumerate() {
                if i != j {
                    assert_ne!(
                        d, other,
                        "{}: duplicate decode descriptor {:?}",
                        codec.name, d
                    );
                }
            }
        }
    }
}

#[test]
fn no_duplicate_descriptors_in_encode() {
    for codec in ALL_CODECS {
        let descs: Vec<PixelDescriptor> = codec.encode_inputs.iter().map(|e| e.descriptor).collect();
        for (i, d) in descs.iter().enumerate() {
            for (j, other) in descs.iter().enumerate() {
                if i != j {
                    assert_ne!(
                        d, other,
                        "{}: duplicate encode descriptor {:?}",
                        codec.name, d
                    );
                }
            }
        }
    }
}

#[test]
fn overshoot_only_on_float_formats() {
    for codec in ALL_CODECS {
        for entry in codec.decode_outputs.iter().chain(codec.encode_inputs.iter()) {
            if entry.can_overshoot {
                assert!(
                    entry.descriptor.channel_type() == zenpixels_convert::ChannelType::F32
                        || entry.descriptor.channel_type() == zenpixels_convert::ChannelType::F16,
                    "{}: overshoot set on integer format {:?}",
                    codec.name,
                    entry.descriptor
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Negotiation sanity
// ---------------------------------------------------------------------------

#[test]
fn every_codec_decode_output_finds_self_encode() {
    // For each codec, every decode output should negotiate back to the same
    // codec's encode inputs (i.e., best_match returns Some).
    for codec in ALL_CODECS {
        let encode_descs: Vec<PixelDescriptor> =
            codec.encode_inputs.iter().map(|e| e.descriptor).collect();

        for entry in codec.decode_outputs {
            let result = best_match(entry.descriptor, &encode_descs, ConvertIntent::Fastest);
            assert!(
                result.is_some(),
                "{}: decode output {:?} finds no encode match",
                codec.name,
                entry.descriptor
            );
        }
    }
}

#[test]
fn cross_codec_negotiation_always_finds_target() {
    // For every (source codec, target codec) pair, at least one decode output
    // should find a match in the target's encode inputs.
    for src_codec in ALL_CODECS {
        for dst_codec in ALL_CODECS {
            let encode_descs: Vec<PixelDescriptor> =
                dst_codec.encode_inputs.iter().map(|e| e.descriptor).collect();

            let any_match = src_codec.decode_outputs.iter().any(|entry| {
                best_match(entry.descriptor, &encode_descs, ConvertIntent::Fastest).is_some()
            });

            assert!(
                any_match,
                "no conversion path from {} to {}",
                src_codec.name, dst_codec.name
            );
        }
    }
}

#[test]
fn identity_negotiation_prefers_exact() {
    // When source format is in the encode list, negotiation should pick it.
    for codec in ALL_CODECS {
        let encode_descs: Vec<PixelDescriptor> =
            codec.encode_inputs.iter().map(|e| e.descriptor).collect();

        for entry in codec.encode_inputs {
            let result = best_match(entry.descriptor, &encode_descs, ConvertIntent::Fastest);
            assert_eq!(
                result,
                Some(entry.descriptor),
                "{}: identity not preferred for {:?}",
                codec.name,
                entry.descriptor
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Specific codec properties
// ---------------------------------------------------------------------------

#[test]
fn jpeg_has_no_cicp() {
    assert!(!registry::JPEG.cicp);
}

#[test]
fn avif_has_cicp() {
    assert!(registry::AVIF.cicp);
}

#[test]
fn gif_has_no_color_management() {
    assert!(!registry::GIF.icc_decode);
    assert!(!registry::GIF.icc_encode);
    assert!(!registry::GIF.cicp);
}

#[test]
fn png_supports_everything() {
    assert!(registry::PNG.icc_decode);
    assert!(registry::PNG.icc_encode);
    assert!(registry::PNG.cicp);
}

#[test]
fn webp_has_icc_but_no_cicp() {
    assert!(registry::WEBP.icc_decode);
    assert!(registry::WEBP.icc_encode);
    assert!(!registry::WEBP.cicp);
}
