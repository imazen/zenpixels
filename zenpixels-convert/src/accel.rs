//! Dead-stripping SIMD accelerator table (`Accelerators`) and process-wide registry.
//!
//! Each accelerator slot holds a plain function pointer (`Option<fn(...)>`).
//! Any slot left `None` is never referenced by a binary, so the linker
//! dead-strips unused SIMD transfer, Oklab, HLG, and layout code completely.
//!
//! Every SIMD accelerator is required to produce **0-bit identical output** to
//! the universal scalar core (`Accelerators::SCALAR`)!

use alloc::boxed::Box;
use core::fmt;
use once_cell::race::OnceBox;

/// Strip-level SIMD function pointer signatures operating on 64-pixel SoA lanes.
pub type ByteLayoutFn = fn(&[u8], &mut [u8], usize) -> bool;
pub type TransferSliceFn = fn(&mut [f32]);
pub type GamutSoaFn = fn(&[[f32; 3]; 3], &mut [f32], &mut [f32], &mut [f32]);
pub type F16DecodeFn = fn(&[u16], &mut [f32]);
pub type F16EncodeFn = fn(&[f32], &mut [u16]);
pub type OklabSoaFn = fn(&[[f32; 3]; 3], &mut [f32], &mut [f32], &mut [f32]);

/// Dead-stripping bundle of optional SIMD accelerators.
///
/// Build a custom subset at compile time using `const` helpers (`SCALAR`,
/// `LAYOUT`, `WEB_SDR`, `HDR`, `OKLAB`, `ALL`, `.with_*()`, `.union()`).
#[derive(Clone, Copy)]
pub struct Accelerators {
    /// Enable `garb` SIMD byte-only layout fast paths (`RGBA↔BGRA`, `RGB↔RGBA`, `U8↔U16`, premul).
    pub byte_layout: bool,
    /// SIMD sRGB EOTF (`srgb_to_linear`) on SoA `f32` lanes.
    pub srgb_eotf: Option<TransferSliceFn>,
    /// SIMD sRGB OETF (`linear_to_srgb`) on SoA `f32` lanes.
    pub srgb_oetf: Option<TransferSliceFn>,
    /// SIMD 3×3 linear gamut multiply on SoA `r/g/b` slices.
    pub gamut_3x3: Option<GamutSoaFn>,
    /// SIMD `f16 → f32` decode (`F16C` / `NEON-fp16`).
    pub f16_decode: Option<F16DecodeFn>,
    /// SIMD `f32 → f16` encode (`F16C` / `NEON-fp16`).
    pub f16_encode: Option<F16EncodeFn>,
    /// SIMD linear RGB → Oklab on SoA `r/g/b` slices.
    pub rgb_to_oklab: Option<OklabSoaFn>,
    /// SIMD Oklab → linear RGB on SoA `l/a/b` slices.
    pub oklab_to_rgb: Option<OklabSoaFn>,
    /// Enable SIMD path inside `PreparedHlg` BT.2100-3 display rendering.
    pub hlg_simd: bool,
    /// Enable SIMD path inside `Bt2446A` tone mapping and `SoftCompress`.
    pub hdr_simd: bool,
}

impl fmt::Debug for Accelerators {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Accelerators")
            .field("byte_layout", &self.byte_layout)
            .field("srgb_eotf", &self.srgb_eotf.is_some())
            .field("srgb_oetf", &self.srgb_oetf.is_some())
            .field("gamut_3x3", &self.gamut_3x3.is_some())
            .field("f16_decode", &self.f16_decode.is_some())
            .field("f16_encode", &self.f16_encode.is_some())
            .field("rgb_to_oklab", &self.rgb_to_oklab.is_some())
            .field("oklab_to_rgb", &self.oklab_to_rgb.is_some())
            .field("hlg_simd", &self.hlg_simd)
            .field("hdr_simd", &self.hdr_simd)
            .finish()
    }
}

impl PartialEq for Accelerators {
    fn eq(&self, other: &Self) -> bool {
        self.byte_layout == other.byte_layout
            && self.srgb_eotf.is_some() == other.srgb_eotf.is_some()
            && self.srgb_oetf.is_some() == other.srgb_oetf.is_some()
            && self.gamut_3x3.is_some() == other.gamut_3x3.is_some()
            && self.f16_decode.is_some() == other.f16_decode.is_some()
            && self.f16_encode.is_some() == other.f16_encode.is_some()
            && self.rgb_to_oklab.is_some() == other.rgb_to_oklab.is_some()
            && self.oklab_to_rgb.is_some() == other.oklab_to_rgb.is_some()
            && self.hlg_simd == other.hlg_simd
            && self.hdr_simd == other.hdr_simd
    }
}

impl Eq for Accelerators {}

impl Default for Accelerators {
    fn default() -> Self {
        #[cfg(feature = "fast")]
        {
            Self::ALL
        }
        #[cfg(not(feature = "fast"))]
        {
            Self::SCALAR
        }
    }
}

impl Accelerators {
    /// Pure scalar baseline — zero SIMD code linked, exact parity oracle.
    pub const SCALAR: Self = Self {
        byte_layout: false,
        srgb_eotf: None,
        srgb_oetf: None,
        gamut_3x3: None,
        f16_decode: None,
        f16_encode: None,
        rgb_to_oklab: None,
        oklab_to_rgb: None,
        hlg_simd: false,
        hdr_simd: false,
    };

    /// Byte-only layout swizzles & integer depth (`garb`).
    pub const LAYOUT: Self = Self::SCALAR.with_byte_layout();

    /// Common web SDR workload: byte layouts + sRGB SIMD + 3×3 gamut SIMD.
    pub const WEB_SDR: Self = Self::SCALAR.with_byte_layout().with_srgb().with_gamut_3x3();

    /// HDR & half-float workload: F16C/NEON-fp16 + HLG/BT.2446A SIMD + 3×3 gamut.
    pub const HDR: Self = Self::SCALAR.with_gamut_3x3().with_f16().with_hdr_simd();

    /// Oklab perceptual SIMD.
    #[cfg(feature = "oklab")]
    pub const OKLAB: Self = Self::SCALAR.with_oklab();

    /// Every enabled SIMD accelerator in this build.
    pub const ALL: Self = {
        let base = Self::WEB_SDR.union(Self::HDR);
        #[cfg(feature = "oklab")]
        {
            base.with_oklab()
        }
        #[cfg(not(feature = "oklab"))]
        {
            base
        }
    };

    /// Enable `garb` byte-only layout fast paths.
    #[inline]
    #[must_use]
    pub const fn with_byte_layout(mut self) -> Self {
        self.byte_layout = true;
        self
    }

    /// Enable SIMD sRGB EOTF & OETF (`linear-srgb`).
    #[inline]
    #[must_use]
    pub const fn with_srgb(mut self) -> Self {
        self.srgb_eotf = Some(simd_srgb_eotf_slice);
        self.srgb_oetf = Some(simd_srgb_oetf_slice);
        self
    }

    /// Enable SIMD 3×3 linear gamut matrix multiply.
    #[inline]
    #[must_use]
    pub const fn with_gamut_3x3(mut self) -> Self {
        self.gamut_3x3 = Some(simd_gamut_3x3_soa);
        self
    }

    /// Enable hardware `f16 ↔ f32` slice conversion (`F16C` / `NEON-fp16`).
    #[inline]
    #[must_use]
    pub const fn with_f16(mut self) -> Self {
        self.f16_decode = Some(simd_f16_to_f32_slice);
        self.f16_encode = Some(simd_f32_to_f16_slice);
        self
    }

    /// Enable SIMD Oklab conversions (`magemath::Cbrtf`).
    #[cfg(feature = "oklab")]
    #[inline]
    #[must_use]
    pub const fn with_oklab(mut self) -> Self {
        self.rgb_to_oklab = Some(simd_rgb_to_oklab_soa);
        self.oklab_to_rgb = Some(simd_oklab_to_rgb_soa);
        self
    }

    /// Enable SIMD inside HLG BT.2100-3 display rendering and BT.2446A / SoftCompress.
    #[inline]
    #[must_use]
    pub const fn with_hdr_simd(mut self) -> Self {
        self.hlg_simd = true;
        self.hdr_simd = true;
        self
    }

    /// Combine two accelerator bundles (`self` takes precedence where both are `Some`).
    #[inline]
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self {
            byte_layout: self.byte_layout || other.byte_layout,
            srgb_eotf: if self.srgb_eotf.is_some() {
                self.srgb_eotf
            } else {
                other.srgb_eotf
            },
            srgb_oetf: if self.srgb_oetf.is_some() {
                self.srgb_oetf
            } else {
                other.srgb_oetf
            },
            gamut_3x3: if self.gamut_3x3.is_some() {
                self.gamut_3x3
            } else {
                other.gamut_3x3
            },
            f16_decode: if self.f16_decode.is_some() {
                self.f16_decode
            } else {
                other.f16_decode
            },
            f16_encode: if self.f16_encode.is_some() {
                self.f16_encode
            } else {
                other.f16_encode
            },
            rgb_to_oklab: if self.rgb_to_oklab.is_some() {
                self.rgb_to_oklab
            } else {
                other.rgb_to_oklab
            },
            oklab_to_rgb: if self.oklab_to_rgb.is_some() {
                self.oklab_to_rgb
            } else {
                other.oklab_to_rgb
            },
            hlg_simd: self.hlg_simd || other.hlg_simd,
            hdr_simd: self.hdr_simd || other.hdr_simd,
        }
    }
}

// ---------------------------------------------------------------------------
// Lock-free process-wide accelerator registry (no_std compatible via OnceBox)
// ---------------------------------------------------------------------------

static GLOBAL_ACCEL: OnceBox<Accelerators> = OnceBox::new();

/// Register a custom process-wide [`Accelerators`] bundle (first call wins; lock-free).
///
/// Returns `true` if installed, `false` if already initialized.
pub fn install(accel: Accelerators) -> bool {
    GLOBAL_ACCEL.set(Box::new(accel)).is_ok()
}

/// Enable all compiled SIMD accelerators process-wide.
pub fn enable_all() -> bool {
    install(Accelerators::ALL)
}

/// Enable common web SDR accelerators (`garb` layouts + sRGB + 3×3 gamut) process-wide.
pub fn enable_web_sdr() -> bool {
    install(Accelerators::WEB_SDR)
}

/// Resolve the active [`Accelerators`] bundle (policy override → global registry → feature default).
#[inline]
#[must_use]
pub fn resolve(override_accel: Option<Accelerators>) -> Accelerators {
    if let Some(a) = override_accel {
        return a;
    }
    if let Some(global) = GLOBAL_ACCEL.get() {
        return *global;
    }
    Accelerators::default()
}

// ---------------------------------------------------------------------------
// Concrete SIMD SoA micro-kernels (referenced ONLY when the caller builds
// a bundle containing them -> dead-stripped otherwise!)
// ---------------------------------------------------------------------------

fn simd_srgb_eotf_slice(vals: &mut [f32]) {
    linear_srgb::default::srgb_to_linear_extended_slice(vals);
}

fn simd_srgb_oetf_slice(vals: &mut [f32]) {
    linear_srgb::default::linear_to_srgb_extended_slice(vals);
}

fn simd_f16_to_f32_slice(src: &[u16], dst: &mut [f32]) {
    crate::f16_scalar::f16_bits_to_f32_slice(src, dst);
}

fn simd_f32_to_f16_slice(src: &[f32], dst: &mut [u16]) {
    crate::f16_scalar::f32_to_f16_bits_slice(src, dst);
}

/// SIMD 3×3 linear gamut multiply on SoA `r, g, b` strips (bit-identical to scalar `r*m00 + g*m01 + b*m02`!).
fn simd_gamut_3x3_soa(m: &[[f32; 3]; 3], r: &mut [f32], g: &mut [f32], b: &mut [f32]) {
    archmage::attuned!(
        gamut_3x3_soa_tier(m, r, g, b),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

#[archmage::attune(_*_t, _v4x_t(cfg(avx512)), define(f32x8))]
fn gamut_3x3_soa_tier(
    token: Token,
    m: &[[f32; 3]; 3],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
) {
    let m00 = f32x8::splat_t(token, m[0][0]);
    let m01 = f32x8::splat_t(token, m[0][1]);
    let m02 = f32x8::splat_t(token, m[0][2]);
    let m10 = f32x8::splat_t(token, m[1][0]);
    let m11 = f32x8::splat_t(token, m[1][1]);
    let m12 = f32x8::splat_t(token, m[1][2]);
    let m20 = f32x8::splat_t(token, m[2][0]);
    let m21 = f32x8::splat_t(token, m[2][1]);
    let m22 = f32x8::splat_t(token, m[2][2]);

    let len = r.len().min(g.len()).min(b.len());
    let mut i = 0;
    while i + 8 <= len {
        let vr = f32x8::from_array_t(token, *<&[f32; 8]>::try_from(&r[i..i + 8]).unwrap());
        let vg = f32x8::from_array_t(token, *<&[f32; 8]>::try_from(&g[i..i + 8]).unwrap());
        let vb = f32x8::from_array_t(token, *<&[f32; 8]>::try_from(&b[i..i + 8]).unwrap());

        // Plain mul+add (no FMA contraction) guarantees 0-bit parity with scalar `m00*r + m01*g + m02*b`.
        let nr = (m00 * vr + m01 * vg) + m02 * vb;
        let ng = (m10 * vr + m11 * vg) + m12 * vb;
        let nb = (m20 * vr + m21 * vg) + m22 * vb;

        r[i..i + 8].copy_from_slice(&nr.to_array());
        g[i..i + 8].copy_from_slice(&ng.to_array());
        b[i..i + 8].copy_from_slice(&nb.to_array());
        i += 8;
    }
    while i < len {
        let (vr, vg, vb) = (r[i], g[i], b[i]);
        r[i] = (m[0][0] * vr + m[0][1] * vg) + m[0][2] * vb;
        g[i] = (m[1][0] * vr + m[1][1] * vg) + m[1][2] * vb;
        b[i] = (m[2][0] * vr + m[2][1] * vg) + m[2][2] * vb;
        i += 1;
    }
}

/// SIMD linear RGB → Oklab on SoA `r, g, b` strips (`magemath::Cbrtf`, 0-bit identical to scalar!).
#[cfg(feature = "oklab")]
fn simd_rgb_to_oklab_soa(m1: &[[f32; 3]; 3], r: &mut [f32], g: &mut [f32], b: &mut [f32]) {
    archmage::attuned!(
        rgb_to_oklab_soa_tier(m1, r, g, b),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

#[cfg(feature = "oklab")]
#[archmage::attune(_*_t, _v4x_t(cfg(avx512)), define(f32x8))]
fn rgb_to_oklab_soa_tier(
    token: Token,
    m1: &[[f32; 3]; 3],
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
) {
    use crate::oklab::{OKLAB_CBRT_ALL, OKLAB_FROM_LMS_CBRT};
    use magemath::SimdMath;

    let m1_00 = f32x8::splat_t(token, m1[0][0]);
    let m1_01 = f32x8::splat_t(token, m1[0][1]);
    let m1_02 = f32x8::splat_t(token, m1[0][2]);
    let m1_10 = f32x8::splat_t(token, m1[1][0]);
    let m1_11 = f32x8::splat_t(token, m1[1][1]);
    let m1_12 = f32x8::splat_t(token, m1[1][2]);
    let m1_20 = f32x8::splat_t(token, m1[2][0]);
    let m1_21 = f32x8::splat_t(token, m1[2][1]);
    let m1_22 = f32x8::splat_t(token, m1[2][2]);

    let m2 = OKLAB_FROM_LMS_CBRT.rows();
    let m2_00 = f32x8::splat_t(token, m2[0][0]);
    let m2_01 = f32x8::splat_t(token, m2[0][1]);
    let m2_02 = f32x8::splat_t(token, m2[0][2]);
    let m2_10 = f32x8::splat_t(token, m2[1][0]);
    let m2_11 = f32x8::splat_t(token, m2[1][1]);
    let m2_12 = f32x8::splat_t(token, m2[1][2]);
    let m2_20 = f32x8::splat_t(token, m2[2][0]);
    let m2_21 = f32x8::splat_t(token, m2[2][1]);
    let m2_22 = f32x8::splat_t(token, m2[2][2]);

    let len = r.len().min(g.len()).min(b.len());
    let mut i = 0;
    while i + 8 <= len {
        let vr = f32x8::from_array_t(token, *<&[f32; 8]>::try_from(&r[i..i + 8]).unwrap());
        let vg = f32x8::from_array_t(token, *<&[f32; 8]>::try_from(&g[i..i + 8]).unwrap());
        let vb = f32x8::from_array_t(token, *<&[f32; 8]>::try_from(&b[i..i + 8]).unwrap());

        let l = (m1_00 * vr + m1_01 * vg) + m1_02 * vb;
        let m = (m1_10 * vr + m1_11 * vg) + m1_12 * vb;
        let s = (m1_20 * vr + m1_21 * vg) + m1_22 * vb;

        let lc = l.cbrt::<OKLAB_CBRT_ALL>();
        let mc = m.cbrt::<OKLAB_CBRT_ALL>();
        let sc = s.cbrt::<OKLAB_CBRT_ALL>();

        let ol = (m2_00 * lc + m2_01 * mc) + m2_02 * sc;
        let oa = (m2_10 * lc + m2_11 * mc) + m2_12 * sc;
        let ob = (m2_20 * lc + m2_21 * mc) + m2_22 * sc;

        r[i..i + 8].copy_from_slice(&ol.to_array());
        g[i..i + 8].copy_from_slice(&oa.to_array());
        b[i..i + 8].copy_from_slice(&ob.to_array());
        i += 8;
    }
    while i < len {
        let [ol, oa, ob] = crate::oklab::rgb_to_oklab(r[i], g[i], b[i], m1);
        r[i] = ol;
        g[i] = oa;
        b[i] = ob;
        i += 1;
    }
}

/// SIMD Oklab → linear RGB on SoA `l, a, b` strips (0-bit identical to scalar!).
#[cfg(feature = "oklab")]
fn simd_oklab_to_rgb_soa(m1_inv: &[[f32; 3]; 3], l: &mut [f32], a: &mut [f32], b: &mut [f32]) {
    archmage::attuned!(
        oklab_to_rgb_soa_tier(m1_inv, l, a, b),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
}

#[cfg(feature = "oklab")]
#[archmage::attune(_*_t, _v4x_t(cfg(avx512)), define(f32x8))]
fn oklab_to_rgb_soa_tier(
    token: Token,
    m1_inv: &[[f32; 3]; 3],
    l: &mut [f32],
    a: &mut [f32],
    b: &mut [f32],
) {
    use crate::oklab::LMS_CBRT_FROM_OKLAB;

    let m2i = LMS_CBRT_FROM_OKLAB.rows();
    let k00 = f32x8::splat_t(token, m2i[0][0]);
    let k01 = f32x8::splat_t(token, m2i[0][1]);
    let k02 = f32x8::splat_t(token, m2i[0][2]);
    let k10 = f32x8::splat_t(token, m2i[1][0]);
    let k11 = f32x8::splat_t(token, m2i[1][1]);
    let k12 = f32x8::splat_t(token, m2i[1][2]);
    let k20 = f32x8::splat_t(token, m2i[2][0]);
    let k21 = f32x8::splat_t(token, m2i[2][1]);
    let k22 = f32x8::splat_t(token, m2i[2][2]);

    let m00 = f32x8::splat_t(token, m1_inv[0][0]);
    let m01 = f32x8::splat_t(token, m1_inv[0][1]);
    let m02 = f32x8::splat_t(token, m1_inv[0][2]);
    let m10 = f32x8::splat_t(token, m1_inv[1][0]);
    let m11 = f32x8::splat_t(token, m1_inv[1][1]);
    let m12 = f32x8::splat_t(token, m1_inv[1][2]);
    let m20 = f32x8::splat_t(token, m1_inv[2][0]);
    let m21 = f32x8::splat_t(token, m1_inv[2][1]);
    let m22 = f32x8::splat_t(token, m1_inv[2][2]);

    let len = l.len().min(a.len()).min(b.len());
    let mut i = 0;
    while i + 8 <= len {
        let vl = f32x8::from_array_t(token, *<&[f32; 8]>::try_from(&l[i..i + 8]).unwrap());
        let va = f32x8::from_array_t(token, *<&[f32; 8]>::try_from(&a[i..i + 8]).unwrap());
        let vb = f32x8::from_array_t(token, *<&[f32; 8]>::try_from(&b[i..i + 8]).unwrap());

        let lc = (k00 * vl + k01 * va) + k02 * vb;
        let mc = (k10 * vl + k11 * va) + k12 * vb;
        let sc = (k20 * vl + k21 * va) + k22 * vb;

        let ll = lc * lc * lc;
        let mm = mc * mc * mc;
        let ss = sc * sc * sc;

        let rr = (m00 * ll + m01 * mm) + m02 * ss;
        let gg = (m10 * ll + m11 * mm) + m12 * ss;
        let bb = (m20 * ll + m21 * mm) + m22 * ss;

        l[i..i + 8].copy_from_slice(&rr.to_array());
        a[i..i + 8].copy_from_slice(&gg.to_array());
        b[i..i + 8].copy_from_slice(&bb.to_array());
        i += 8;
    }
    while i < len {
        let [rr, gg, bb] = crate::oklab::oklab_to_rgb(l[i], a[i], b[i], m1_inv);
        l[i] = rr;
        a[i] = gg;
        b[i] = bb;
        i += 1;
    }
}
