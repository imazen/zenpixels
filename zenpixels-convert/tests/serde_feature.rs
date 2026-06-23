//! Coverage for the `serde` Cargo feature in `zenpixels-convert`.
//!
//! The `serde` feature was fixed in 0.2.15 (commit `5ab70a0`) to actually
//! pull `serde` into `zenpixels-convert` itself — previously it only
//! forwarded to `zenpixels/serde`, leaving
//! `hdr::measure::LightLevelMethod`'s `#[cfg_attr(feature = "serde",
//! derive(Serialize, Deserialize))]` failing to resolve under
//! `--features serde,hdr-experimental`. These tests pin both the build
//! AND the JSON round-trip semantics so a future `serde` feature edit
//! can't silently break either.
//!
//! Gated on `serde,hdr-experimental` together — the LightLevelMethod
//! types only exist behind the experimental gate. The serde
//! `ContentLightLevel` / `DiffuseWhite` round trip lives in `zenpixels`
//! itself; we cross-check from this crate's published surface to
//! confirm the re-export keeps the derives intact.

#![cfg(all(feature = "serde", feature = "hdr-experimental"))]

use zenpixels::hdr::{ContentLightLevel, DiffuseWhite};
use zenpixels_convert::hdr::LightLevelMethod;

#[test]
fn light_level_method_serializes_via_serde() {
    // The `MaxRgb` and `LuminanceBt2020` discriminants both have to
    // round-trip through JSON — the derive is `Serialize +
    // Deserialize` so the serialised form is the variant name (default
    // serde enum representation).
    for &m in &[LightLevelMethod::MaxRgb, LightLevelMethod::LuminanceBt2020] {
        let json = serde_json::to_string(&m).expect("serialize");
        let back: LightLevelMethod = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, m, "round-trip mismatch for {m:?} (json: {json})");
    }
    // Default discriminant serialises identifiably (the value is in the
    // JSON text, not just the round-trip).
    let json_default = serde_json::to_string(&LightLevelMethod::default()).expect("serialize");
    assert!(
        json_default.contains("MaxRgb"),
        "serde default representation should carry the MaxRgb name, got {json_default}"
    );
}

#[test]
fn content_light_level_serializes_via_serde() {
    // `ContentLightLevel` is re-exported from zenpixels — its derive
    // travels with the re-export, so a regression in zenpixels would
    // surface here too.
    let cll = ContentLightLevel::new(1000, 250);
    let json = serde_json::to_string(&cll).expect("serialize");
    let back: ContentLightLevel = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, cll);
    // Field names must be stable — schedulers and log parsers downstream
    // grep them. Pin the two CTA-861.3 codes verbatim.
    assert!(json.contains("max_content_light_level"));
    assert!(json.contains("max_frame_average_light_level"));
}

#[test]
fn diffuse_white_serializes_via_serde() {
    // The anchor type — `pub struct DiffuseWhite(f32)` — serialises as
    // a bare float per serde's tuple-struct convention. Pin that so a
    // schema change here is loud.
    let anchor = DiffuseWhite::BT2408;
    let json = serde_json::to_string(&anchor).expect("serialize");
    // 203.0 → "203.0" textually.
    assert!(
        json.contains("203"),
        "DiffuseWhite::BT2408 should carry the 203-nit value, got {json}"
    );
    let back: DiffuseWhite = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, anchor);
}

#[test]
fn default_percentile_const_serializes_via_serde() {
    // The f32 constant itself isn't a serde-derived type, but the
    // documented usage is "serialise it as part of a Settings struct
    // your application drives". Verify that a bare f32 with that
    // value round-trips through JSON without precision loss past what
    // f32 already imposes.
    let p = ContentLightLevel::DEFAULT_PERCENTILE;
    let json = serde_json::to_string(&p).expect("serialize f32");
    let back: f32 = serde_json::from_str(&json).expect("deserialize f32");
    assert_eq!(back.to_bits(), p.to_bits());
    // And the pinned value is exactly the documented 0.99999 — same as
    // the in-crate constant pin in `src/hdr/measure.rs`.
    assert_eq!(p, 0.99999_f32);
}

#[test]
fn light_level_method_unknown_variant_deserialization_fails_cleanly() {
    // A scheduler reading log data with a future LightLevelMethod
    // variant must fail with a typed serde error, not panic or default
    // silently. Pin that contract here — current serde behaviour for
    // unknown enum variants in the default-tag representation is to
    // return Err.
    let res: Result<LightLevelMethod, _> = serde_json::from_str("\"SomeFutureVariant\"");
    assert!(
        res.is_err(),
        "unknown enum variant must error, got Ok({:?})",
        res.ok()
    );
}
