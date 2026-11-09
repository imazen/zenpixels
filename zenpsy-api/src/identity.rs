//! Configured scoring identity and actual implementation provenance.
//! Hashing, persistence columns and cache keys belong to consumers.

use alloc::borrow::Cow;
use alloc::string::ToString;
use alloc::vec::Vec;
use core::fmt;

/// Where a metric's arithmetic runs.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Backend {
    /// Host CPU (any SIMD tier; tiers must agree bit for bit).
    Cpu,
    /// A GPU through the named API (`"cuda"`, `"wgpu"`, `"metal"`).
    #[non_exhaustive]
    Gpu {
        /// The GPU API, lowercase.
        api: &'static str,
    },
}

impl Backend {
    /// GPU through the named API; device details may be added independently.
    pub const fn gpu(api: &'static str) -> Self {
        Self::Gpu { api }
    }
}

/// Which code computed a score. Build it with [`crate::implementation!`] so the
/// crate name and version come from Cargo, not from hand-typed strings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct Implementation {
    /// The implementing crate (`CARGO_PKG_NAME`).
    crate_name: &'static str,
    /// Its version (`CARGO_PKG_VERSION`). Implementation provenance.
    crate_version: &'static str,
    /// Output revision. Bump whenever any output can change.
    score_revision: u32,
    /// Where it runs.
    backend: Backend,
    /// Optional build id (a commit hash), from `ZENMETRICS_BUILD_ID` at
    /// compile time. Implementation provenance.
    build: Option<&'static str>,
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

    /// Implementing crate name.
    pub const fn crate_name(&self) -> &'static str {
        self.crate_name
    }
    /// Implementing crate version.
    pub const fn crate_version(&self) -> &'static str {
        self.crate_version
    }
    /// Output definition revision.
    pub const fn score_revision(&self) -> u32 {
        self.score_revision
    }
    /// Actual compute backend.
    pub const fn backend(&self) -> Backend {
        self.backend
    }
    /// Optional implementation build identity.
    pub const fn build(&self) -> Option<&'static str> {
        self.build
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
/// assert_eq!(imp.crate_name(), "zenpsy-api");
/// assert_eq!(imp.score_revision(), 3);
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
            " ({}@{} r{} {:?})",
            imp.crate_name, imp.crate_version, imp.score_revision, imp.backend
        )
    }
}
