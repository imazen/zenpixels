//! Dependency-lean guard for the `zenpixels` crate.
//!
//! `zenpixels` is the foundational interchange crate of the zen ecosystem and
//! must stay dependency-lean (user directive): NEVER add a dependency to this
//! crate. Measurement / SIMD machinery lives ONLY in `zenpixels-convert`
//! (`CllMeasure`); anything heavier belongs in a downstream crate.
//!
//! Parses `Cargo.toml` with plain string handling (std-only, no crate deps)
//! and fails if any **runtime or build** dependency table names anything
//! outside the frozen allowlist. Covers `[dependencies]`,
//! `[build-dependencies]`, and their `[target.'cfg(..)'.…]`-scoped forms
//! (both the multi-line table and the `[…dependencies.NAME]` dotted form).
//! `[dev-dependencies]` is intentionally exempt — test/bench-only deps don't
//! ship in the published crate.

/// The 0.2.14 dependency set minus `serde` (optional dep soft-removed in
/// 0.2.16 — the feature is now an inert stub). Frozen.
const ALLOWED: &[&str] = &["bytemuck", "rgb", "imgref", "whereat"];

/// Collect dependency names from every guarded table (runtime + build,
/// including `[target.'cfg(..)'.…]` forms); `[dev-dependencies]` is exempt.
/// Plain string scan — no `toml` crate, keeping this test std-only.
fn guarded_deps(manifest: &str) -> Vec<String> {
    let mut deps: Vec<String> = Vec::new();
    let mut in_guarded_table = false;
    for raw in manifest.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(inner) = line.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
            // Normalize both `[table]` and `[[array-of-tables]]` to the bare
            // header text (`bench`, `dependencies`, `target.'cfg(..)'.dependencies`, …).
            let header = inner.trim_start_matches('[').trim_end_matches(']').trim();
            in_guarded_table = false;
            // dev-dependencies (incl. target-scoped) are exempt.
            if header.contains("dev-dependencies") {
                continue;
            }
            if header.ends_with("dependencies") {
                // `[dependencies]` / `[build-dependencies]` /
                // `[target.'cfg(..)'.dependencies]` — following `k = v` lines
                // are dep names.
                in_guarded_table = true;
            } else if let Some(idx) = header.rfind("dependencies.") {
                // Dotted single-dep form: `[…dependencies.NAME]`.
                let name = &header[idx + "dependencies.".len()..];
                deps.push(name.trim().trim_matches('"').to_string());
            }
            continue;
        }
        if in_guarded_table {
            if let Some((name, _)) = line.split_once('=') {
                deps.push(name.trim().trim_matches('"').to_string());
            }
        }
    }
    deps
}

#[test]
fn dependencies_stay_lean() {
    let manifest_path = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");
    let manifest =
        std::fs::read_to_string(manifest_path).expect("failed to read zenpixels/Cargo.toml");

    let deps = guarded_deps(&manifest);
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
        "zenpixels must stay dependency-lean (allowed: {ALLOWED:?}), but a runtime/build \
         dependency table also names {violations:?}.\n\
         Rule (user directive): NEVER add a dependency to the zenpixels crate. \
         Measurement/SIMD machinery lives ONLY in zenpixels-convert (CllMeasure); \
         put new functionality in a downstream crate instead."
    );
}

#[test]
fn parser_covers_build_and_target_tables_but_not_dev() {
    // Synthetic manifest exercising every table shape the real one might
    // grow. dev-deps must NOT be collected; everything else must be.
    let manifest = r#"
[package]
name = "x"

[dependencies]
bytemuck = "1"

[build-dependencies]
cc = "1"

[dependencies.whereat]
version = "0.1"

[target.'cfg(windows)'.dependencies]
winapi = "0.3"

[target.'cfg(unix)'.build-dependencies]
pkg-config = "0.3"

[target.'cfg(target_arch = "wasm32")'.dependencies.wasm-bindgen]
version = "0.2"

[dev-dependencies]
criterion = "0.5"

[target.'cfg(unix)'.dev-dependencies]
tempfile = "3"

[features]
default = ["std"]
std = []

[[bench]]
name = "b"
"#;
    let mut got = guarded_deps(manifest);
    got.sort();
    // Present: every runtime/build dep (incl. dotted + target-scoped).
    // Absent: criterion, tempfile (dev), and feature names.
    assert_eq!(
        got,
        vec![
            "bytemuck".to_string(),
            "cc".to_string(),
            "pkg-config".to_string(),
            "wasm-bindgen".to_string(),
            "whereat".to_string(),
            "winapi".to_string(),
        ],
        "build/target deps must be caught; dev deps + feature names must not"
    );
}
