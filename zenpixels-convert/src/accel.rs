//! Dead-stripping SIMD accelerator table (`Accelerators`) and process-wide registry.
//!
//! Every accelerator slot holds a plain function pointer (`Option<fn(...)>`).
//! Any slot left `None` is never referenced by a binary, so the linker
//! dead-strips unused `garb`, SIMD strip codecs, sRGB, BT.709, Gamma 2.2,
//! F16C, PQ, HLG, and Oklab code completely.
//!
//! Every SIMD accelerator is required to produce **0-bit identical output** to
//! the universal scalar core (`Accelerators::SCALAR`)!

use alloc::boxed::Box;
use core::fmt;
use once_cell::race::OnceBox;
use zenpixels::SampleLayout;

use crate::policy::DepthPolicy;
use crate::strip::{ByteFastPath, StripBuf};

/// Strip-level & row-level SIMD function pointer signatures (crate-internal).
pub(crate) type ByteLayoutDispatchFn = fn(ByteFastPath, &[u8], &mut [u8], usize) -> bool;
pub(crate) type Rgba8PremulFn = fn(&[u8], &mut [u8], usize, bool);
pub(crate) type StripUnpackFn = fn(SampleLayout, &[u8], &mut StripBuf, usize) -> bool;
pub(crate) type StripPackFn = fn(SampleLayout, &StripBuf, &mut [u8], usize, DepthPolicy) -> bool;
pub(crate) type F32ToU8SliceFn = fn(&[f32], &mut [u8], bool);
pub(crate) type U8ToF32SliceFn = fn(&[u8], &mut [f32]);
pub(crate) type F32ToU16SliceFn = fn(&[f32], &mut [u16], bool);
pub(crate) type U16ToF32SliceFn = fn(&[u16], &mut [f32]);
pub(crate) type LinF32ToSrgbU8Fn = fn(&[f32], &mut [u8], bool, bool);
pub(crate) type Srgb8MatrixFn = fn(
    &[u8],
    &mut [u8],
    usize,
    &[[f32; 3]; 3],
    zenpixels::ChannelOrder,
    zenpixels::ChannelOrder,
    bool,
    bool,
);
pub(crate) type Srgb8MatrixQ14Fn = fn(
    &[u8],
    &mut [u8],
    usize,
    &crate::strip_simd::Q14MatrixTables,
    zenpixels::ChannelOrder,
    zenpixels::ChannelOrder,
    bool,
    bool,
);
pub(crate) type TransferSliceFn = fn(&mut [f32]);
pub(crate) type GammaPowSliceFn = fn(&mut [f32], f32, f64);
#[cfg(feature = "hdr-transfers")]
pub(crate) type ScaledTransferSliceFn = fn(&mut [f32], f32);
pub(crate) type GamutSoaFn = fn(&[[f32; 3]; 3], &mut [f32], &mut [f32], &mut [f32]);
pub(crate) type F16DecodeFn = fn(&[u16], &mut [f32]);
pub(crate) type F16EncodeFn = fn(&[f32], &mut [u16]);
#[cfg(feature = "oklab")]
pub(crate) type OklabSoaFn = fn(&[[f32; 3]; 3], &mut [f32], &mut [f32], &mut [f32]);
#[cfg(feature = "oklab")]
pub(crate) type SoftCompressFn = fn(&crate::tonemap::SoftCompress, &mut [[f32; 3]]);
#[cfg(feature = "hdr-transfers")]
pub(crate) type HlgDisplaySoaFn =
    fn(&crate::hlg_render::PreparedHlg, &mut [f32], &mut [f32], &mut [f32], usize);
#[cfg(feature = "gainmap")]
pub(crate) type GainMapBlend2Fn = fn(&[f32], &[f32], f32, f32, &mut [f32]);
#[cfg(feature = "gainmap")]
pub(crate) type GainMapBlend4Fn = fn(&[f32], &[f32], &[f32], &[f32], [f32; 4], &mut [f32]);
#[cfg(feature = "gainmap")]
pub(crate) type GainMapDecodeFn = fn(&[f32], f32, f32, &mut [f32]);
#[cfg(feature = "gainmap")]
pub(crate) type GainMapCombineFn =
    fn(&[f32], &[f32], &[f32], &[f32], [f32; 3], [f32; 3], bool, &mut [f32]);

/// Dead-stripping bundle of optional SIMD accelerators.
///
/// Build a custom subset at compile time using `const` helpers (`SCALAR`,
/// `LAYOUT`, `WEB_SDR`, `HDR`, `OKLAB`, `ALL`, `.with_*()`, `.union()`).
#[derive(Clone, Copy)]
pub struct Accelerators {
    /// `garb` SIMD U8 byte-only layout fast paths (`RGBA↔BGRA`, `RGB↔RGBA`, `Gray→RGBA`).
    pub(crate) byte_layout_exec: Option<ByteLayoutDispatchFn>,
    /// SIMD `RGBA8` / `BGRA8` straight → premultiplied (`round` / `trunc`).
    pub(crate) rgba8_straight_to_premul: Option<Rgba8PremulFn>,
    /// Native-width SIMD AoS → SoA strip unpack (`U8`/`U16`/`F32` 3ch/4ch).
    pub(crate) strip_unpack: Option<StripUnpackFn>,
    /// Native-width SIMD SoA → AoS strip pack (`U8`/`U16`/`F32` 3ch/4ch).
    pub(crate) strip_pack: Option<StripPackFn>,
    /// Direct contiguous `f32 → u8` depth quantization.
    pub(crate) f32_to_u8: Option<F32ToU8SliceFn>,
    /// Direct contiguous `u8 → f32` depth promotion.
    pub(crate) u8_to_f32: Option<U8ToF32SliceFn>,
    /// Direct contiguous `f32 → u16` depth quantization.
    pub(crate) f32_to_u16: Option<F32ToU16SliceFn>,
    /// Direct contiguous `u16 → f32` depth promotion.
    pub(crate) u16_to_f32: Option<U16ToF32SliceFn>,
    /// SIMD sRGB EOTF (`srgb_to_linear_extended`) on `f32` slices.
    pub(crate) srgb_eotf: Option<TransferSliceFn>,
    /// SIMD sRGB OETF (`linear_to_srgb_extended`) on `f32` slices.
    pub(crate) srgb_oetf: Option<TransferSliceFn>,
    /// SIMD sRGB `u8 → linear f32` direct slice LUT (`RGB`/`Gray`).
    pub(crate) srgb_u8_to_lin: Option<U8ToF32SliceFn>,
    /// Fused SIMD linear `f32 → sRGB u8` exact polynomial + quantization.
    pub(crate) lin_f32_to_srgb_u8: Option<LinF32ToSrgbU8Fn>,
    /// Option A: Fused exact SIMD U8 sRGB-TRC → 3×3 matrix → sRGB-TRC U8 (0-bit parity).
    pub(crate) srgb8_matrix_exact: Option<Srgb8MatrixFn>,
    /// Option B: Opt-in fast Q14 fixed-point SIMD U8 sRGB-TRC → 3×3 matrix → sRGB-TRC U8 (<= 1 LSB).
    pub(crate) srgb8_matrix_approx: Option<Srgb8MatrixQ14Fn>,
    /// SIMD BT.709 EOTF on `f32` slices.
    pub(crate) bt709_eotf: Option<TransferSliceFn>,
    /// SIMD BT.709 OETF on `f32` slices.
    pub(crate) bt709_oetf: Option<TransferSliceFn>,
    /// SIMD ITU-T H.273 Gamma 2.2 (`2.2`) EOTF on `f32` slices.
    pub(crate) gamma22_eotf: Option<TransferSliceFn>,
    /// SIMD ITU-T H.273 Gamma 2.2 (`1/2.2`) OETF on `f32` slices.
    pub(crate) gamma22_oetf: Option<TransferSliceFn>,
    /// Shared SIMD power slice (`exp in [0.25, 3.0]`) for Adobe RGB (`563/256`), DCI Cinema (`2.6`),
    /// and gain-map non-unit gamma (reuses `gamma_slice` with zero extra monomorphizations).
    pub(crate) gamma_pow: Option<GammaPowSliceFn>,
    /// SIMD 3×3 linear gamut multiply on SoA `r/g/b` slices.
    pub(crate) gamut_3x3: Option<GamutSoaFn>,
    /// SIMD `f16 → f32` decode (`F16C` / `NEON-fp16`).
    pub(crate) f16_decode: Option<F16DecodeFn>,
    /// SIMD `f32 → f16` encode (`F16C` / `NEON-fp16`).
    pub(crate) f16_encode: Option<F16EncodeFn>,
    /// SIMD linear RGB → Oklab on SoA `r/g/b` slices.
    #[cfg(feature = "oklab")]
    pub(crate) rgb_to_oklab: Option<OklabSoaFn>,
    /// SIMD Oklab → linear RGB on SoA `l/a/b` slices.
    #[cfg(feature = "oklab")]
    pub(crate) oklab_to_rgb: Option<OklabSoaFn>,
    /// SIMD OKLch `SoftCompress` gamut compression.
    #[cfg(feature = "oklab")]
    pub(crate) soft_compress: Option<SoftCompressFn>,
    /// SIMD ST 2084 PQ EOTF on `f32` slices.
    #[cfg(feature = "hdr-transfers")]
    pub(crate) pq_eotf: Option<ScaledTransferSliceFn>,
    /// SIMD ST 2084 PQ OETF on `f32` slices.
    #[cfg(feature = "hdr-transfers")]
    pub(crate) pq_oetf: Option<ScaledTransferSliceFn>,
    /// SIMD ARIB STD-B67 HLG scene EOTF on `f32` slices.
    #[cfg(feature = "hdr-transfers")]
    pub(crate) hlg_scene_eotf: Option<TransferSliceFn>,
    /// SIMD ARIB STD-B67 HLG scene OETF on `f32` slices.
    #[cfg(feature = "hdr-transfers")]
    pub(crate) hlg_scene_oetf: Option<TransferSliceFn>,
    /// SIMD BT.2100-3 display HLG rendering on SoA `r/g/b` slices.
    #[cfg(feature = "hdr-transfers")]
    pub(crate) hlg_display_soa: Option<HlgDisplaySoaFn>,
    /// SIMD 2-row separable vertical blend for gain-map bilinear resampling.
    #[cfg(feature = "gainmap")]
    pub(crate) gainmap_blend_2: Option<GainMapBlend2Fn>,
    /// SIMD 4-row separable vertical blend for gain-map bicubic/Mitchell resampling.
    #[cfg(feature = "gainmap")]
    pub(crate) gainmap_blend_4: Option<GainMapBlend4Fn>,
    /// SIMD `exp2` gain-map sample decoder.
    #[cfg(feature = "gainmap")]
    pub(crate) gainmap_decode: Option<GainMapDecodeFn>,
    /// SIMD linear RGB(A) ISO 21496-1 gain combine kernel.
    #[cfg(feature = "gainmap")]
    pub(crate) gainmap_combine: Option<GainMapCombineFn>,
}

impl fmt::Debug for Accelerators {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut ds = f.debug_struct("Accelerators");
        ds.field("byte_layout", &self.byte_layout_exec.is_some())
            .field("premul_u8", &self.rgba8_straight_to_premul.is_some())
            .field("strip_simd", &self.strip_unpack.is_some())
            .field("srgb_eotf", &self.srgb_eotf.is_some())
            .field("srgb_oetf", &self.srgb_oetf.is_some())
            .field("lin_f32_to_srgb_u8", &self.lin_f32_to_srgb_u8.is_some())
            .field("srgb8_matrix_exact", &self.srgb8_matrix_exact.is_some())
            .field("srgb8_matrix_approx", &self.srgb8_matrix_approx.is_some())
            .field("bt709", &self.bt709_eotf.is_some())
            .field("gamma22", &self.gamma22_eotf.is_some())
            .field("gamut_3x3", &self.gamut_3x3.is_some())
            .field("f16_decode", &self.f16_decode.is_some())
            .field("f16_encode", &self.f16_encode.is_some());
        #[cfg(feature = "oklab")]
        ds.field("oklab", &self.rgb_to_oklab.is_some())
            .field("soft_compress", &self.soft_compress.is_some());
        #[cfg(feature = "hdr-transfers")]
        ds.field("hdr_simd", &self.pq_eotf.is_some());
        #[cfg(feature = "gainmap")]
        ds.field("gainmap_simd", &self.gainmap_combine.is_some());
        ds.finish()
    }
}

impl PartialEq for Accelerators {
    fn eq(&self, other: &Self) -> bool {
        let base = self.byte_layout_exec.is_some() == other.byte_layout_exec.is_some()
            && self.rgba8_straight_to_premul.is_some() == other.rgba8_straight_to_premul.is_some()
            && self.strip_unpack.is_some() == other.strip_unpack.is_some()
            && self.strip_pack.is_some() == other.strip_pack.is_some()
            && self.f32_to_u8.is_some() == other.f32_to_u8.is_some()
            && self.u8_to_f32.is_some() == other.u8_to_f32.is_some()
            && self.f32_to_u16.is_some() == other.f32_to_u16.is_some()
            && self.u16_to_f32.is_some() == other.u16_to_f32.is_some()
            && self.srgb_eotf.is_some() == other.srgb_eotf.is_some()
            && self.srgb_oetf.is_some() == other.srgb_oetf.is_some()
            && self.srgb_u8_to_lin.is_some() == other.srgb_u8_to_lin.is_some()
            && self.lin_f32_to_srgb_u8.is_some() == other.lin_f32_to_srgb_u8.is_some()
            && self.srgb8_matrix_exact.is_some() == other.srgb8_matrix_exact.is_some()
            && self.srgb8_matrix_approx.is_some() == other.srgb8_matrix_approx.is_some()
            && self.bt709_eotf.is_some() == other.bt709_eotf.is_some()
            && self.bt709_oetf.is_some() == other.bt709_oetf.is_some()
            && self.gamma22_eotf.is_some() == other.gamma22_eotf.is_some()
            && self.gamma22_oetf.is_some() == other.gamma22_oetf.is_some()
            && self.gamma_pow.is_some() == other.gamma_pow.is_some()
            && self.gamut_3x3.is_some() == other.gamut_3x3.is_some()
            && self.f16_decode.is_some() == other.f16_decode.is_some()
            && self.f16_encode.is_some() == other.f16_encode.is_some();
        #[cfg(feature = "oklab")]
        let base = base
            && self.rgb_to_oklab.is_some() == other.rgb_to_oklab.is_some()
            && self.oklab_to_rgb.is_some() == other.oklab_to_rgb.is_some()
            && self.soft_compress.is_some() == other.soft_compress.is_some();
        #[cfg(feature = "hdr-transfers")]
        let base = base
            && self.pq_eotf.is_some() == other.pq_eotf.is_some()
            && self.pq_oetf.is_some() == other.pq_oetf.is_some()
            && self.hlg_scene_eotf.is_some() == other.hlg_scene_eotf.is_some()
            && self.hlg_scene_oetf.is_some() == other.hlg_scene_oetf.is_some()
            && self.hlg_display_soa.is_some() == other.hlg_display_soa.is_some();
        #[cfg(feature = "gainmap")]
        let base = base
            && self.gainmap_blend_2.is_some() == other.gainmap_blend_2.is_some()
            && self.gainmap_blend_4.is_some() == other.gainmap_blend_4.is_some()
            && self.gainmap_decode.is_some() == other.gainmap_decode.is_some()
            && self.gainmap_combine.is_some() == other.gainmap_combine.is_some();
        base
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
        byte_layout_exec: None,
        rgba8_straight_to_premul: None,
        strip_unpack: None,
        strip_pack: None,
        f32_to_u8: None,
        u8_to_f32: None,
        f32_to_u16: None,
        u16_to_f32: None,
        srgb_eotf: None,
        srgb_oetf: None,
        srgb_u8_to_lin: None,
        lin_f32_to_srgb_u8: None,
        srgb8_matrix_exact: None,
        srgb8_matrix_approx: None,
        bt709_eotf: None,
        bt709_oetf: None,
        gamma22_eotf: None,
        gamma22_oetf: None,
        gamma_pow: None,
        gamut_3x3: None,
        f16_decode: None,
        f16_encode: None,
        #[cfg(feature = "oklab")]
        rgb_to_oklab: None,
        #[cfg(feature = "oklab")]
        oklab_to_rgb: None,
        #[cfg(feature = "oklab")]
        soft_compress: None,
        #[cfg(feature = "hdr-transfers")]
        pq_eotf: None,
        #[cfg(feature = "hdr-transfers")]
        pq_oetf: None,
        #[cfg(feature = "hdr-transfers")]
        hlg_scene_eotf: None,
        #[cfg(feature = "hdr-transfers")]
        hlg_scene_oetf: None,
        #[cfg(feature = "hdr-transfers")]
        hlg_display_soa: None,
        #[cfg(feature = "gainmap")]
        gainmap_blend_2: None,
        #[cfg(feature = "gainmap")]
        gainmap_blend_4: None,
        #[cfg(feature = "gainmap")]
        gainmap_decode: None,
        #[cfg(feature = "gainmap")]
        gainmap_combine: None,
    };

    /// Byte-only layout swizzles (`garb` U8 layouts).
    pub const LAYOUT: Self = Self::SCALAR.with_byte_layout();

    /// Common web SDR workload: byte layouts + SIMD strip codec + sRGB + 3×3 gamut + fused U8 P3/sRGB matrix.
    ///
    /// Links zero BT.709, Gamma 2.2, F16, PQ, HLG, gain-map, or Oklab SIMD code.
    pub const WEB_SDR: Self = Self::SCALAR
        .with_byte_layout()
        .with_strip_simd()
        .with_srgb()
        .with_gamut_3x3()
        .with_srgb8_matrix_internal();

    /// HDR & half-float workload: SIMD strip codec + F16C/NEON-fp16 + PQ/HLG + 3×3 gamut.
    ///
    /// Links zero gain-map SIMD code; chain [`.with_gainmap()`](Self::with_gainmap) when reconstructing
    /// ISO 21496-1 / UltraHDR gain maps.
    pub const HDR: Self = Self::SCALAR
        .with_strip_simd()
        .with_gamut_3x3()
        .with_f16()
        .with_hdr_simd();

    /// ISO 21496-1 / UltraHDR gain-map SIMD kernels only (opt-in so `RowPlan`-only binaries dead-strip them).
    ///
    /// Combine with `Self::ALL.with_gainmap()` or `Self::WEB_SDR.union(Self::GAINMAP)` when applying gain maps.
    #[cfg(feature = "gainmap")]
    pub const GAINMAP: Self = Self::SCALAR.with_gainmap();

    /// Oklab perceptual SIMD + strip codec.
    #[cfg(feature = "oklab")]
    pub const OKLAB: Self = Self::SCALAR.with_strip_simd().with_oklab();

    /// Every enabled `RowPlan` SIMD accelerator in this build (excludes opt-in gain-map SIMD
    /// so non-gainmap binaries dead-strip it completely; use `Accelerators::ALL.with_gainmap()`
    /// for accelerated gain-map reconstruction).
    pub const ALL: Self = {
        let base = Self::WEB_SDR.union(Self::HDR).with_bt709().with_gamma22();
        #[cfg(feature = "oklab")]
        {
            base.with_oklab()
        }
        #[cfg(not(feature = "oklab"))]
        {
            base
        }
    };

    /// Whether `garb` SIMD byte-layout fast paths are enabled.
    #[inline]
    #[must_use]
    pub const fn has_byte_layout(&self) -> bool {
        self.byte_layout_exec.is_some()
    }

    /// Whether native-width SIMD strip unpack/pack and depth codecs are enabled.
    #[inline]
    #[must_use]
    pub const fn has_strip_simd(&self) -> bool {
        self.strip_unpack.is_some()
    }

    /// Enable `garb` byte-only layout fast paths & SIMD U8 premultiplication.
    #[inline]
    #[must_use]
    pub const fn with_byte_layout(mut self) -> Self {
        self.byte_layout_exec = Some(crate::strip::exec_garb_byte_fast_path);
        self.rgba8_straight_to_premul = Some(crate::strip_simd::simd_rgba8_straight_to_premul);
        self
    }

    /// Enable native-width SIMD AoS↔SoA strip unpack/pack and contiguous depth codecs.
    #[inline]
    #[must_use]
    pub const fn with_strip_simd(mut self) -> Self {
        self.strip_unpack = Some(crate::strip_simd::simd_unpack_strip);
        self.strip_pack = Some(crate::strip_simd::simd_pack_strip);
        self.f32_to_u8 = Some(crate::strip_simd::simd_f32_to_u8_slice);
        self.u8_to_f32 = Some(crate::strip_simd::simd_u8_to_f32_slice);
        self.f32_to_u16 = Some(crate::strip_simd::simd_f32_to_u16_slice);
        self.u16_to_f32 = Some(crate::strip_simd::simd_u16_to_f32_slice);
        self
    }

    /// Enable SIMD sRGB EOTF & OETF (`linear-srgb`) + fused linear F32 → sRGB U8.
    #[inline]
    #[must_use]
    pub const fn with_srgb(mut self) -> Self {
        self.srgb_eotf = Some(simd_srgb_eotf_slice);
        self.srgb_oetf = Some(simd_srgb_oetf_slice);
        self.srgb_u8_to_lin = Some(simd_srgb_u8_to_lin_slice);
        self.lin_f32_to_srgb_u8 = Some(crate::strip_simd::simd_lin_f32_to_srgb_u8);
        self
    }

    /// Enable SIMD BT.709 EOTF & OETF (`magemath::Powf`).
    #[inline]
    #[must_use]
    pub const fn with_bt709(mut self) -> Self {
        self.bt709_eotf = Some(crate::transfers::bt709_eotf_simd_slice);
        self.bt709_oetf = Some(crate::transfers::bt709_oetf_simd_slice);
        self
    }

    /// Enable SIMD Gamma 2.2 (`2.2`), Adobe RGB (`563/256`), and DCI Cinema (`2.6`) EOTF & OETF (`magemath::Powf`).
    #[inline]
    #[must_use]
    pub const fn with_gamma22(mut self) -> Self {
        self.gamma22_eotf = Some(crate::transfers::gamma22_eotf_simd_slice);
        self.gamma22_oetf = Some(crate::transfers::gamma22_oetf_simd_slice);
        self.gamma_pow = Some(crate::transfers::gamma_pow_simd_slice);
        self
    }

    /// Enable SIMD 3×3 linear gamut matrix multiply.
    #[inline]
    #[must_use]
    pub const fn with_gamut_3x3(mut self) -> Self {
        self.gamut_3x3 = Some(simd_gamut_3x3_soa);
        self
    }

    #[inline]
    #[must_use]
    const fn with_srgb8_matrix_internal(mut self) -> Self {
        self.srgb8_matrix_exact = Some(crate::strip_simd::simd_srgb8_matrix_exact);
        self.srgb8_matrix_approx = Some(crate::strip_simd::simd_srgb8_matrix_approx_q14);
        self
    }

    /// Enable ISO 21496-1 / UltraHDR gain-map SIMD kernels (vertical resampling blend,
    /// `magemath::Expf` gain decode, and linear RGB(A) multiply-add combine).
    ///
    /// Kept opt-in (excluded from [`ALL`](Self::ALL) and [`HDR`](Self::HDR)) so non-gainmap
    /// `RowPlan` binaries dead-strip gain-map SIMD completely (`+0 B`).
    #[cfg(feature = "gainmap")]
    #[inline]
    #[must_use]
    pub const fn with_gainmap(mut self) -> Self {
        self.gainmap_blend_2 = Some(crate::gainmap_simd::simd_blend_rows_2);
        self.gainmap_blend_4 = Some(crate::gainmap_simd::simd_blend_rows_4);
        self.gainmap_decode = Some(crate::gainmap_simd::simd_decode_gain_1ch);
        self.gainmap_combine = Some(crate::gainmap_simd::simd_combine_linear_f32);
        self.gamma_pow = Some(crate::transfers::gamma_pow_simd_slice);
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

    /// Enable SIMD Oklab conversions (`magemath::Cbrtf`) & `SoftCompress`.
    #[cfg(feature = "oklab")]
    #[inline]
    #[must_use]
    pub const fn with_oklab(mut self) -> Self {
        self.rgb_to_oklab = Some(simd_rgb_to_oklab_soa);
        self.oklab_to_rgb = Some(simd_oklab_to_rgb_soa);
        self.soft_compress = Some(simd_soft_compress_strip);
        self
    }

    /// Enable SIMD PQ, HLG Scene, and BT.2100-3 HLG display rendering.
    #[inline]
    #[must_use]
    #[allow(unused_mut)]
    pub const fn with_hdr_simd(mut self) -> Self {
        #[cfg(feature = "hdr-transfers")]
        {
            self.pq_eotf = Some(crate::transfers::hdr::pq_eotf_simd_slice);
            self.pq_oetf = Some(crate::transfers::hdr::pq_oetf_simd_slice);
            self.hlg_scene_eotf = Some(crate::transfers::hdr::hlg_scene_eotf_simd_slice);
            self.hlg_scene_oetf = Some(crate::transfers::hdr::hlg_scene_oetf_simd_slice);
            self.hlg_display_soa = Some(simd_hlg_display_soa);
        }
        self
    }

    /// Combine two accelerator bundles (`self` takes precedence where both are `Some`).
    #[inline]
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        macro_rules! pick {
            ($field:ident) => {
                if self.$field.is_some() {
                    self.$field
                } else {
                    other.$field
                }
            };
        }
        Self {
            byte_layout_exec: pick!(byte_layout_exec),
            rgba8_straight_to_premul: pick!(rgba8_straight_to_premul),
            strip_unpack: pick!(strip_unpack),
            strip_pack: pick!(strip_pack),
            f32_to_u8: pick!(f32_to_u8),
            u8_to_f32: pick!(u8_to_f32),
            f32_to_u16: pick!(f32_to_u16),
            u16_to_f32: pick!(u16_to_f32),
            srgb_eotf: pick!(srgb_eotf),
            srgb_oetf: pick!(srgb_oetf),
            srgb_u8_to_lin: pick!(srgb_u8_to_lin),
            lin_f32_to_srgb_u8: pick!(lin_f32_to_srgb_u8),
            srgb8_matrix_exact: pick!(srgb8_matrix_exact),
            srgb8_matrix_approx: pick!(srgb8_matrix_approx),
            bt709_eotf: pick!(bt709_eotf),
            bt709_oetf: pick!(bt709_oetf),
            gamma22_eotf: pick!(gamma22_eotf),
            gamma22_oetf: pick!(gamma22_oetf),
            gamma_pow: pick!(gamma_pow),
            gamut_3x3: pick!(gamut_3x3),
            f16_decode: pick!(f16_decode),
            f16_encode: pick!(f16_encode),
            #[cfg(feature = "oklab")]
            rgb_to_oklab: pick!(rgb_to_oklab),
            #[cfg(feature = "oklab")]
            oklab_to_rgb: pick!(oklab_to_rgb),
            #[cfg(feature = "oklab")]
            soft_compress: pick!(soft_compress),
            #[cfg(feature = "hdr-transfers")]
            pq_eotf: pick!(pq_eotf),
            #[cfg(feature = "hdr-transfers")]
            pq_oetf: pick!(pq_oetf),
            #[cfg(feature = "hdr-transfers")]
            hlg_scene_eotf: pick!(hlg_scene_eotf),
            #[cfg(feature = "hdr-transfers")]
            hlg_scene_oetf: pick!(hlg_scene_oetf),
            #[cfg(feature = "hdr-transfers")]
            hlg_display_soa: pick!(hlg_display_soa),
            #[cfg(feature = "gainmap")]
            gainmap_blend_2: pick!(gainmap_blend_2),
            #[cfg(feature = "gainmap")]
            gainmap_blend_4: pick!(gainmap_blend_4),
            #[cfg(feature = "gainmap")]
            gainmap_decode: pick!(gainmap_decode),
            #[cfg(feature = "gainmap")]
            gainmap_combine: pick!(gainmap_combine),
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

/// Enable common web SDR accelerators (`garb` layouts + strip SIMD + sRGB + 3×3 gamut) process-wide.
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
// Concrete SIMD micro-kernels (referenced ONLY when the caller builds
// a bundle containing them -> dead-stripped otherwise!)
// ---------------------------------------------------------------------------

fn simd_srgb_eotf_slice(vals: &mut [f32]) {
    linear_srgb::default::srgb_to_linear_extended_slice(vals);
}

fn simd_srgb_oetf_slice(vals: &mut [f32]) {
    linear_srgb::default::linear_to_srgb_extended_slice(vals);
}

fn simd_srgb_u8_to_lin_slice(src: &[u8], dst: &mut [f32]) {
    linear_srgb::default::srgb_u8_to_linear_slice(src, dst);
}

fn simd_f16_to_f32_slice(src: &[u16], dst: &mut [f32]) {
    crate::f16_scalar::f16_bits_to_f32_slice(src, dst);
}

fn simd_f32_to_f16_slice(src: &[f32], dst: &mut [u16]) {
    crate::f16_scalar::f32_to_f16_bits_slice(src, dst);
}

#[cfg(feature = "oklab")]
fn simd_soft_compress_strip(sc: &crate::tonemap::SoftCompress, pixels: &mut [[f32; 3]]) {
    sc.apply_strip(pixels);
}

#[cfg(feature = "hdr-transfers")]
fn simd_hlg_display_soa(
    prepared: &crate::hlg_render::PreparedHlg,
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
    len: usize,
) {
    prepared.apply_soa_simd(r, g, b, len);
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
