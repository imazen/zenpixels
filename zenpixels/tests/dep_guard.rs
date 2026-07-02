//! Dependency-lean guard for the `zenpixels` crate.
//!
//! `zenpixels` is the foundational interchange crate of the zen ecosystem and
//! must stay dependency-lean (user directive): NEVER add a dependency to this
//! crate. Measurement / SIMD machinery lives ONLY in `zenpixels-convert`
//! (`CllMeasure`); anything heavier belongs in a downstream crate.
//!
//! Parses `Cargo.toml` with plain string handling (no dev-deps, std-only) and
//! fails if `[dependencies]` names anything outside the frozen allowlist.

/// The 0.2.14 dependency set minus `serde` (optional dep soft-removed in
/// 0.2.16 — the feature is now an inert stub). Frozen.
const ALLOWED: &[&str] = &["bytemuck", "rgb", "imgref", "whereat"];

#[test]
fn dependencies_stay_lean() {
    let manifest_path = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");
    let manifest =
        std::fs::read_to_string(manifest_path).expect("failed to read zenpixels/Cargo.toml");

    let mut deps: Vec<String> = Vec::new();
    let mut in_dependencies = false;
    for raw in manifest.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') {
            // `[dependencies]` table — subsequent `name = ...` lines are deps.
            in_dependencies = line == "[dependencies]";
            // `[dependencies.foo]` dotted-table form also declares a dep.
            if let Some(rest) = line.strip_prefix("[dependencies.") {
                if let Some(name) = rest.strip_suffix(']') {
                    deps.push(name.trim().trim_matches('"').to_string());
                }
            }
            continue;
        }
        if in_dependencies {
            if let Some((name, _)) = line.split_once('=') {
                deps.push(name.trim().trim_matches('"').to_string());
            }
        }
    }

    assert!(
        !deps.is_empty(),
        "parsed zero entries from [dependencies] in {manifest_path} — \
         the manifest layout changed; update this parser rather than bypassing the guard"
    );

    let violations: Vec<&str> = deps
        .iter()
        .map(String::as_str)
        .filter(|d| !ALLOWED.contains(d))
        .collect();
    assert!(
        violations.is_empty(),
        "zenpixels must stay dependency-lean (allowed: {ALLOWED:?}), but [dependencies] \
         also names {violations:?}.\n\
         Rule (user directive): NEVER add a dependency to the zenpixels crate. \
         Measurement/SIMD machinery lives ONLY in zenpixels-convert (CllMeasure); \
         put new functionality in a downstream crate instead."
    );
}
