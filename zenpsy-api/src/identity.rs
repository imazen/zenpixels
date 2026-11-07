//! Metric identity and provenance.
//!
//! A score is only reusable when you know exactly what produced it. A
//! [`MetricId`] answers that in three layers:
//!
//! - **Semantic identity** — the metric family (`"cvvdp"`), an optional
//!   variant (`"vifvec"`, `"v3"`) and the published reference it
//!   reproduces (`"pycvvdp 0.5.7"`).
//! - **Implementation identity** — which crate computed it, its
//!   [`Implementation::score_revision`], and the [`Backend`].
//! - **Configuration** — every construction-time knob that can change an
//!   output (display model, geometry, preprocessing), as sorted
//!   `key = value` pairs.
//!
//! [`MetricId::fingerprint`] hashes all of that *except* the crate version
//! and the build id: those are provenance (recorded by
//! [`MetricId::canonical`]) but don't change the column, so a release that
//! leaves every output bit-identical keeps feeding the same column.
//!
//! ## The revision rule
//!
//! Bump `score_revision` whenever any output can change: a kernel fix, a
//! new default, a changed fold. Updating parity expectations without
//! bumping it is a provenance bug — old and new numbers would land in one
//! column.
//!
//! ## Column names
//!
//! - [`MetricId::semantic_column`]: `{family}[_{variant}]_{key}`, e.g.
//!   `cvvdp_jod`. For display, or for a table known to hold one
//!   implementation.
//! - [`MetricId::column`]: the fully qualified name,
//!   `{semantic}__{crate}_r{revision}_{backend}_{fp8}`, e.g.
//!   `cvvdp_jod__cvvdp_r1_cpu_3f9a01b2`. `fp8` is the top 32 bits of the
//!   fingerprint, so two configurations of one implementation get distinct
//!   columns. Use this for anything persisted.

use alloc::borrow::Cow;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::{self, Write as _};

/// Where a metric's arithmetic runs.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Backend {
    /// Host CPU (any SIMD tier; tiers must agree bit for bit).
    Cpu,
    /// A GPU through the named API (`"cuda"`, `"wgpu"`, `"metal"`).
    Gpu {
        /// The GPU API, lowercase.
        api: &'static str,
    },
}

impl Backend {
    /// Column token: `cpu`, `gpu_cuda`, …
    pub fn token(&self) -> String {
        match self {
            Backend::Cpu => "cpu".into(),
            Backend::Gpu { api } => {
                let mut s = String::from("gpu_");
                push_sanitized(&mut s, api);
                s
            }
        }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.token())
    }
}

/// Which code computed a score. Build it with [`crate::implementation!`] so the
/// crate name and version come from Cargo, not from hand-typed strings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct Implementation {
    /// The implementing crate (`CARGO_PKG_NAME`).
    pub crate_name: &'static str,
    /// Its version (`CARGO_PKG_VERSION`). Provenance only: not hashed.
    pub crate_version: &'static str,
    /// Output revision. Bump whenever any output can change.
    pub score_revision: u32,
    /// Where it runs.
    pub backend: Backend,
    /// Optional build id (a commit hash), from `ZENMETRICS_BUILD_ID` at
    /// compile time. Provenance only: not hashed.
    pub build: Option<&'static str>,
}

impl Implementation {
    /// A CPU implementation with no build id.
    pub const fn new(
        crate_name: &'static str,
        crate_version: &'static str,
        score_revision: u32,
    ) -> Self {
        Self {
            crate_name,
            crate_version,
            score_revision,
            backend: Backend::Cpu,
            build: None,
        }
    }

    /// Replace the backend.
    #[must_use]
    pub const fn with_backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// Record a build id.
    #[must_use]
    pub const fn with_build(mut self, build: Option<&'static str>) -> Self {
        self.build = build;
        self
    }
}

/// The calling crate's [`Implementation`] at `score_revision`, CPU backend,
/// with the build id from the `ZENMETRICS_BUILD_ID` environment variable
/// when it was set at compile time.
///
/// ```
/// let imp = zenpsy_api::implementation!(3);
/// assert_eq!(imp.crate_name, "zenpsy-api");
/// assert_eq!(imp.score_revision, 3);
/// ```
#[macro_export]
macro_rules! implementation {
    ($rev:expr) => {
        $crate::Implementation::new(
            ::core::env!("CARGO_PKG_NAME"),
            ::core::env!("CARGO_PKG_VERSION"),
            $rev,
        )
        .with_build(::core::option_env!("ZENMETRICS_BUILD_ID"))
    };
}

/// Full identity of one configured metric instance. See the
/// crate documentation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct MetricId {
    family: &'static str,
    variant: Option<&'static str>,
    reference: Cow<'static, str>,
    implementation: Implementation,
    config: Vec<(Cow<'static, str>, Cow<'static, str>)>,
}

const FP_DOMAIN: &str = "zenmetrics-id-v1";

impl MetricId {
    /// `family` (`"fsim"`) as computed by `implementation`.
    pub fn new(family: &'static str, implementation: Implementation) -> Self {
        Self {
            family,
            variant: None,
            reference: Cow::Borrowed(""),
            implementation,
            config: Vec::new(),
        }
    }

    /// Set the variant within the family (`"vifvec"`, `"v3"`).
    #[must_use]
    pub fn with_variant(mut self, variant: &'static str) -> Self {
        self.variant = Some(variant);
        self
    }

    /// Name the published reference this implementation reproduces
    /// (`"pycvvdp 0.5.7"`, `"MATLAB FeatureSIM.m (Zhang 2011)"`).
    #[must_use]
    pub fn with_reference(mut self, reference: impl Into<Cow<'static, str>>) -> Self {
        self.reference = reference.into();
        self
    }

    /// Replace the backend.
    #[must_use]
    pub fn with_backend(mut self, backend: Backend) -> Self {
        self.implementation.backend = backend;
        self
    }

    /// Record one configuration knob. Values render through `Display`
    /// (Rust prints floats as their shortest round-trip form, so `0.5`
    /// stays `0.5`). Setting a key twice keeps the last value. Keys stay
    /// sorted, so insertion order never changes the fingerprint.
    #[must_use]
    pub fn with_config(
        mut self,
        key: impl Into<Cow<'static, str>>,
        value: impl fmt::Display,
    ) -> Self {
        let key = key.into();
        let value: Cow<'static, str> = Cow::Owned(value.to_string());
        match self
            .config
            .binary_search_by(|(k, _)| k.as_ref().cmp(key.as_ref()))
        {
            Ok(i) => self.config[i].1 = value,
            Err(i) => self.config.insert(i, (key, value)),
        }
        self
    }

    /// The metric family.
    pub fn family(&self) -> &'static str {
        self.family
    }

    /// The variant, if any.
    pub fn variant(&self) -> Option<&'static str> {
        self.variant
    }

    /// The reproduced reference (empty when unset).
    pub fn reference(&self) -> &str {
        &self.reference
    }

    /// The implementation.
    pub fn implementation(&self) -> &Implementation {
        &self.implementation
    }

    /// Configuration knobs, sorted by key.
    pub fn config(&self) -> &[(Cow<'static, str>, Cow<'static, str>)] {
        &self.config
    }

    /// One configuration value.
    pub fn config_value(&self, key: &str) -> Option<&str> {
        self.config
            .binary_search_by(|(k, _)| k.as_ref().cmp(key))
            .ok()
            .map(|i| self.config[i].1.as_ref())
    }

    /// 64-bit FNV-1a over every identity field except the crate version
    /// and build id, each field length-prefixed so no two field lists
    /// collide by concatenation. Stable across platforms and releases.
    pub fn fingerprint(&self) -> u64 {
        let mut h = Fnv64::new();
        h.field(FP_DOMAIN);
        h.field(self.family);
        h.field(self.variant.unwrap_or(""));
        h.field(&self.reference);
        h.field(self.implementation.crate_name);
        h.write(&self.implementation.score_revision.to_le_bytes());
        h.field(&self.implementation.backend.token());
        h.write(&(self.config.len() as u64).to_le_bytes());
        for (k, v) in &self.config {
            h.field(k);
            h.field(v);
        }
        h.finish()
    }

    /// `{family}[_{variant}]_{key}`, sanitized to `[a-z0-9_]`.
    pub fn semantic_column(&self, key: &str) -> String {
        let mut s = String::new();
        push_sanitized(&mut s, self.family);
        if let Some(v) = self.variant {
            s.push('_');
            push_sanitized(&mut s, v);
        }
        s.push('_');
        push_sanitized(&mut s, key);
        s
    }

    /// The fully qualified column for output `key`:
    /// `{semantic}__{crate}_r{revision}_{backend}_{fp8}`.
    pub fn column(&self, key: &str) -> String {
        let mut s = self.semantic_column(key);
        s.push_str("__");
        push_sanitized(&mut s, self.implementation.crate_name);
        let _ = write!(
            s,
            "_r{}_{}_{:08x}",
            self.implementation.score_revision,
            self.implementation.backend.token(),
            self.fingerprint() >> 32
        );
        s
    }

    /// Every field, provenance included, as one `;`-separated
    /// `key=value` line (`\`, `;` and `=` escaped with `\`). Store it next
    /// to persisted scores; it reconstructs what produced them.
    ///
    /// `family=cvvdp;variant=;reference=pycvvdp 0.5.7;crate=cvvdp@0.1.0;rev=1;backend=cpu;build=;fingerprint=…;cfg.ppd=60`
    pub fn canonical(&self) -> String {
        let mut s = String::new();
        let mut kv = |k: &str, v: &str| {
            if !s.is_empty() {
                s.push(';');
            }
            push_escaped(&mut s, k);
            s.push('=');
            push_escaped(&mut s, v);
        };
        let imp = &self.implementation;
        kv("family", self.family);
        kv("variant", self.variant.unwrap_or(""));
        kv("reference", &self.reference);
        let mut crate_id = String::from(imp.crate_name);
        crate_id.push('@');
        crate_id.push_str(imp.crate_version);
        kv("crate", &crate_id);
        kv("rev", &imp.score_revision.to_string());
        kv("backend", &imp.backend.token());
        kv("build", imp.build.unwrap_or(""));
        let mut fp = String::new();
        let _ = write!(fp, "{:016x}", self.fingerprint());
        kv("fingerprint", &fp);
        for (k, v) in &self.config {
            let mut key = String::from("cfg.");
            key.push_str(k);
            kv(&key, v);
        }
        s
    }
}

impl fmt::Display for MetricId {
    /// `family[/variant] (crate@version r{rev} {backend})`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.family)?;
        if let Some(v) = self.variant {
            write!(f, "/{v}")?;
        }
        let imp = &self.implementation;
        write!(
            f,
            " ({}@{} r{} {})",
            imp.crate_name, imp.crate_version, imp.score_revision, imp.backend
        )
    }
}

/// Lowercase ASCII alphanumerics; every other run of characters becomes
/// one `_`; no leading or trailing `_` from the run rule.
fn push_sanitized(out: &mut String, s: &str) {
    let start = out.len();
    let mut pending = false;
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            if pending && out.len() > start {
                out.push('_');
            }
            pending = false;
            out.push(c.to_ascii_lowercase());
        } else {
            pending = true;
        }
    }
}

/// Whether `s` is already a sanitized column component or full column:
/// non-empty, `[a-z0-9_]` only.
pub fn is_sanitized_column(s: &str) -> bool {
    !s.is_empty()
        && s.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
}

fn push_escaped(out: &mut String, s: &str) {
    for c in s.chars() {
        if matches!(c, '\\' | ';' | '=') {
            out.push('\\');
        }
        out.push(c);
    }
}

struct Fnv64(u64);

impl Fnv64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    fn new() -> Self {
        Self(Self::OFFSET)
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }

    fn field(&mut self, s: &str) {
        self.write(&(s.len() as u64).to_le_bytes());
        self.write(s.as_bytes());
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> MetricId {
        MetricId::new("cvvdp", Implementation::new("cvvdp", "0.1.0", 1))
            .with_reference("pycvvdp 0.5.7")
    }

    #[test]
    fn fnv_known_vector() {
        let mut h = Fnv64::new();
        h.write(b"a");
        assert_eq!(h.finish(), 0xaf63_dc4c_8601_ec8c);
    }

    #[test]
    fn version_and_build_are_provenance_not_identity() {
        let a = base();
        let mut b = base();
        b.implementation.crate_version = "0.2.0";
        b.implementation.build = Some("deadbeef");
        assert_eq!(a.fingerprint(), b.fingerprint());
        assert_eq!(a.column("jod"), b.column("jod"));
        assert_ne!(a.canonical(), b.canonical());
    }

    #[test]
    fn revision_backend_config_variant_change_the_column() {
        let a = base();
        let mut rev = base();
        rev.implementation.score_revision = 2;
        let gpu = base().with_backend(Backend::Gpu { api: "cuda" });
        let cfg = base().with_config("ppd", 60.0f32);
        let var = base().with_variant("hdr");
        let cols = [&a, &rev, &gpu, &cfg, &var].map(|m| m.column("jod"));
        for i in 0..cols.len() {
            for j in i + 1..cols.len() {
                assert_ne!(cols[i], cols[j]);
            }
        }
        assert!(gpu.column("jod").contains("_gpu_cuda_"));
        assert!(rev.column("jod").contains("_r2_"));
    }

    #[test]
    fn config_order_does_not_matter_and_last_write_wins() {
        let a = base().with_config("b", 2).with_config("a", 1);
        let b = base()
            .with_config("a", 9)
            .with_config("b", 2)
            .with_config("a", 1);
        assert_eq!(a.fingerprint(), b.fingerprint());
        assert_eq!(a.config_value("a"), Some("1"));
        assert_eq!(a.config()[0].0, "a");
    }

    #[test]
    fn field_boundaries_cannot_collide() {
        let a = MetricId::new("ab", Implementation::new("c", "0", 1));
        let b = MetricId::new("a", Implementation::new("bc", "0", 1));
        assert_ne!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn column_shapes() {
        let m =
            MetricId::new("mad-iqa", Implementation::new("mad-iqa", "0.1.0", 1)).with_variant("V3");
        assert_eq!(m.semantic_column("Hi Index"), "mad_iqa_v3_hi_index");
        let col = m.column("hi");
        assert!(col.starts_with("mad_iqa_v3_hi__mad_iqa_r1_cpu_"), "{col}");
        assert_eq!(col.len(), "mad_iqa_v3_hi__mad_iqa_r1_cpu_".len() + 8);
        assert!(is_sanitized_column(&col));
    }

    #[test]
    fn canonical_escapes_and_lists_config() {
        let m = base().with_config("eotf", "a;b=c");
        let c = m.canonical();
        assert!(c.starts_with("family=cvvdp;variant=;reference=pycvvdp 0.5.7;crate=cvvdp@0.1.0;rev=1;backend=cpu;build=;fingerprint="), "{c}");
        assert!(c.ends_with(r"cfg.eotf=a\;b\=c"), "{c}");
    }

    #[test]
    fn macro_reads_cargo() {
        let imp = crate::implementation!(4);
        assert_eq!(imp.crate_name, "zenpsy-api");
        assert_eq!(imp.crate_version, env!("CARGO_PKG_VERSION"));
        assert_eq!(imp.score_revision, 4);
        assert_eq!(imp.backend, Backend::Cpu);
    }
}
