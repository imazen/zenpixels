//! Pre-computed row converter.
//!
//! [`RowConverter`] wraps a [`ConvertPlan`] with pre-allocated scratch
//! buffers for zero-allocation-per-row streaming conversion.

use crate::convert::{ConvertPlan, ConvertScratch, convert_row_buffered};
use crate::{ChannelLayout, ConvertError, PixelDescriptor};
use whereat::{At, ResultAtExt};

/// Pre-computed pixel format converter with pre-allocated scratch buffers.
///
/// Create once, then call [`convert_row`](Self::convert_row) for each row.
/// Multi-step conversions reuse internal scratch buffers, eliminating
/// per-row heap allocation.
///
/// # Example
///
/// ```rust,ignore
/// use zenpixels::{RowConverter, PixelDescriptor};
///
/// let mut conv = RowConverter::new(
///     PixelDescriptor::RGB8_SRGB,
///     PixelDescriptor::RGBA8_SRGB,
/// )?;
///
/// for y in 0..height {
///     conv.convert_row(&src_row, &mut dst_row, width);
/// }
/// ```
pub struct RowConverter {
    plan: ConvertPlan,
    scratch: ConvertScratch,
    /// External CMS transform that bypasses the plan entirely.
    /// Set by `new_explicit_with_cms` when a plugin accepts the conversion.
    external: Option<Box<dyn crate::cms::RowTransformMut>>,
}

impl RowConverter {
    /// Create a converter from `from` to `to`.
    ///
    /// Returns `Err` if no conversion path exists between the formats.
    #[track_caller]
    pub fn new(from: PixelDescriptor, to: PixelDescriptor) -> Result<Self, At<ConvertError>> {
        let plan = ConvertPlan::new(from, to).at()?;
        Ok(Self {
            plan,
            scratch: ConvertScratch::new(),
            external: None,
        })
    }

    /// Create a converter with explicit policy options.
    ///
    /// Like [`new`](Self::new) but validates [`ConvertOptions`] policies
    /// (alpha removal, depth reduction, RGB→Gray) before creating the plan.
    /// Returns a specific error if a forbidden operation would be required.
    ///
    /// [`ConvertOptions`]: crate::policy::ConvertOptions
    #[track_caller]
    pub fn new_explicit(
        from: PixelDescriptor,
        to: PixelDescriptor,
        options: &crate::policy::ConvertOptions,
    ) -> Result<Self, At<ConvertError>> {
        let plan = ConvertPlan::new_explicit(from, to, options).at()?;
        Ok(Self {
            plan,
            scratch: ConvertScratch::new(),
            external: None,
        })
    }

    /// Create a converter that may delegate the color conversion to a
    /// [`PluggableCms`].
    ///
    /// When `cms` is `Some` and the source and destination have different
    /// primaries or transfer functions, the plugin is asked to supply a
    /// row transform for the full `(from, to)` pair. If it accepts, the
    /// plan becomes a single external-transform step; the built-in gamut
    /// matrix and matlut fast paths are bypassed for that conversion.
    /// If the plugin declines or there is no color work to do, behavior
    /// matches [`new_explicit`](Self::new_explicit).
    ///
    /// [`PluggableCms`]: crate::cms::PluggableCms
    #[track_caller]
    pub fn new_explicit_with_cms(
        from: PixelDescriptor,
        to: PixelDescriptor,
        options: &crate::policy::ConvertOptions,
        cms: Option<&dyn crate::cms::PluggableCms>,
    ) -> Result<Self, At<ConvertError>> {
        use crate::policy::{AlphaPolicy, DepthPolicy};

        // Try plugin first — it may take the whole conversion.
        if let Some(cms) = cms {
            let profiles_differ =
                from.primaries != to.primaries || from.transfer() != to.transfer();
            if profiles_differ {
                let src_src = from.color_profile_source();
                let dst_src = to.color_profile_source();
                if let Some(transform) = cms.build_source_transform(
                    src_src,
                    dst_src,
                    from.pixel_format(),
                    to.pixel_format(),
                ) {
                    // Policy checks still apply.
                    let drops_alpha = from.alpha().is_some() && to.alpha().is_none();
                    if drops_alpha && options.alpha_policy == AlphaPolicy::Forbid {
                        return Err(whereat::at!(ConvertError::AlphaRemovalForbidden));
                    }
                    let reduces_depth =
                        from.channel_type().byte_size() > to.channel_type().byte_size();
                    if reduces_depth && options.depth_policy == DepthPolicy::Forbid {
                        return Err(whereat::at!(ConvertError::DepthReductionForbidden));
                    }
                    let src_is_rgb = matches!(
                        from.layout(),
                        ChannelLayout::Rgb | ChannelLayout::Rgba | ChannelLayout::Bgra
                    );
                    let dst_is_gray =
                        matches!(to.layout(), ChannelLayout::Gray | ChannelLayout::GrayAlpha);
                    if src_is_rgb && dst_is_gray && options.luma.is_none() {
                        return Err(whereat::at!(ConvertError::RgbToGray));
                    }

                    // Plugin accepted — build an identity plan (the external
                    // transform drives the row, plan is just a shell for
                    // from/to metadata).
                    return Ok(Self {
                        plan: ConvertPlan::identity(from, to),
                        scratch: ConvertScratch::new(),
                        external: Some(transform),
                    });
                }
            }
        }

        // Plugin declined or absent — built-in path.
        let plan = ConvertPlan::new_explicit(from, to, options).at()?;
        Ok(Self {
            plan,
            scratch: ConvertScratch::new(),
            external: None,
        })
    }

    /// Create a converter from a pre-computed plan.
    pub fn from_plan(plan: ConvertPlan) -> Self {
        Self {
            plan,
            scratch: ConvertScratch::new(),
            external: None,
        }
    }

    /// Convert one row of `width` pixels.
    ///
    /// `src` must contain at least `width * from.bytes_per_pixel()` bytes.
    /// `dst` must contain at least `width * to.bytes_per_pixel()` bytes.
    ///
    /// Multi-step conversions reuse internal scratch buffers — no heap
    /// allocation after the first call at a given width.
    #[inline]
    pub fn convert_row(&mut self, src: &[u8], dst: &mut [u8], width: u32) {
        if let Some(ref mut ext) = self.external {
            ext.transform_row(src, dst, width);
        } else {
            convert_row_buffered(&self.plan, src, dst, width, &mut self.scratch);
        }
    }

    /// Convert multiple rows from a strided source buffer to a strided destination.
    ///
    /// The source and destination can have different strides.
    #[track_caller]
    pub fn convert_rows(
        &mut self,
        src: &[u8],
        src_stride: usize,
        dst: &mut [u8],
        dst_stride: usize,
        width: u32,
        rows: u32,
    ) -> Result<(), At<ConvertError>> {
        for y in 0..rows {
            let src_start = y as usize * src_stride;
            let src_end = src_start + (width as usize * self.plan.from().bytes_per_pixel());
            let dst_start = y as usize * dst_stride;
            let dst_end = dst_start + (width as usize * self.plan.to().bytes_per_pixel());

            if src_end > src.len() || dst_end > dst.len() {
                return Err(whereat::at!(ConvertError::BufferSize {
                    expected: dst_end,
                    actual: dst.len(),
                }));
            }

            self.convert_row(
                &src[src_start..src_end],
                &mut dst[dst_start..dst_end],
                width,
            );
        }
        Ok(())
    }

    /// True if the conversion is a no-op (formats are identical).
    #[inline]
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.plan.is_identity()
    }

    /// Source pixel format.
    #[inline]
    pub fn from_descriptor(&self) -> PixelDescriptor {
        self.plan.from()
    }

    /// Target pixel format.
    #[inline]
    pub fn to_descriptor(&self) -> PixelDescriptor {
        self.plan.to()
    }

    /// Compose two converters: apply `self` then `other` in a single pass.
    ///
    /// Adjacent inverse steps cancel (e.g., sRGB→linear then linear→sRGB
    /// becomes identity). This eliminates intermediate buffers in zenpipe's
    /// TransformSource when chaining format conversions.
    ///
    /// Returns `None` if the converters are incompatible (self.to != other.from).
    pub fn compose(&self, other: &Self) -> Option<Self> {
        self.plan.compose(&other.plan).map(Self::from_plan)
    }

    /// Access the underlying conversion plan.
    pub fn plan(&self) -> &ConvertPlan {
        &self.plan
    }
}

impl Clone for RowConverter {
    fn clone(&self) -> Self {
        // External transforms are not cloneable — the clone falls back
        // to the built-in plan. Callers with a CMS plugin should build
        // a new RowConverter instead of cloning.
        Self {
            plan: self.plan.clone(),
            scratch: ConvertScratch::new(),
            external: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::convert::ConvertPlan;
    use crate::policy::{AlphaPolicy, ConvertOptions, DepthPolicy};
    use crate::{AlphaMode, ChannelLayout, ChannelType, ConvertError, TransferFunction};

    /// Helper: build a RowConverter and convert a single pixel.
    fn convert_pixel(
        from: PixelDescriptor,
        to: PixelDescriptor,
        src: &[u8],
    ) -> alloc::vec::Vec<u8> {
        let mut conv = RowConverter::new(from, to).unwrap();
        let dst_bpp = to.bytes_per_pixel();
        let mut dst = vec![0u8; dst_bpp];
        conv.convert_row(src, &mut dst, 1);
        dst
    }

    // -----------------------------------------------------------------------
    // Identity / no-op
    // -----------------------------------------------------------------------

    #[test]
    fn identity_conversion() {
        let desc = PixelDescriptor::RGB8_SRGB;
        let mut conv = RowConverter::new(desc, desc).unwrap();
        assert!(conv.is_identity());
        assert_eq!(conv.from_descriptor(), desc);
        assert_eq!(conv.to_descriptor(), desc);

        let src = [10u8, 20, 30, 40, 50, 60];
        let mut dst = [0u8; 6];
        conv.convert_row(&src, &mut dst, 2);
        assert_eq!(dst, src);
    }

    // -----------------------------------------------------------------------
    // Layout conversions
    // -----------------------------------------------------------------------

    #[test]
    fn rgb8_to_rgba8() {
        let dst = convert_pixel(
            PixelDescriptor::RGB8_SRGB,
            PixelDescriptor::RGBA8_SRGB,
            &[100, 150, 200],
        );
        assert_eq!(dst, [100, 150, 200, 255]);
    }

    #[test]
    fn rgba8_to_rgb8() {
        let dst = convert_pixel(
            PixelDescriptor::RGBA8_SRGB,
            PixelDescriptor::RGB8_SRGB,
            &[100, 150, 200, 128],
        );
        assert_eq!(dst, [100, 150, 200]);
    }

    #[test]
    fn bgra8_to_rgba8() {
        let dst = convert_pixel(
            PixelDescriptor::BGRA8_SRGB,
            PixelDescriptor::RGBA8_SRGB,
            &[200, 150, 100, 255], // BGRA
        );
        assert_eq!(dst, [100, 150, 200, 255]); // RGBA
    }

    #[test]
    fn rgba8_to_bgra8() {
        let dst = convert_pixel(
            PixelDescriptor::RGBA8_SRGB,
            PixelDescriptor::BGRA8_SRGB,
            &[100, 150, 200, 255],
        );
        assert_eq!(dst, [200, 150, 100, 255]);
    }

    #[test]
    fn rgb8_to_bgra8() {
        // RGB → RGBA → BGRA (two-step).
        let dst = convert_pixel(
            PixelDescriptor::RGB8_SRGB,
            PixelDescriptor::BGRA8_SRGB,
            &[100, 150, 200],
        );
        assert_eq!(dst, [200, 150, 100, 255]);
    }

    #[test]
    fn bgra8_to_rgb8() {
        // BGRA → RGBA → RGB (two-step).
        let dst = convert_pixel(
            PixelDescriptor::BGRA8_SRGB,
            PixelDescriptor::RGB8_SRGB,
            &[200, 150, 100, 255],
        );
        assert_eq!(dst, [100, 150, 200]);
    }

    #[test]
    fn gray8_to_rgb8() {
        let dst = convert_pixel(
            PixelDescriptor::GRAY8_SRGB,
            PixelDescriptor::RGB8_SRGB,
            &[128],
        );
        assert_eq!(dst, [128, 128, 128]);
    }

    #[test]
    fn gray8_to_rgba8() {
        let dst = convert_pixel(
            PixelDescriptor::GRAY8_SRGB,
            PixelDescriptor::RGBA8_SRGB,
            &[200],
        );
        assert_eq!(dst, [200, 200, 200, 255]);
    }

    #[test]
    fn gray8_to_bgra8() {
        // Gray → RGBA → BGRA (two-step).
        let dst = convert_pixel(
            PixelDescriptor::GRAY8_SRGB,
            PixelDescriptor::BGRA8_SRGB,
            &[128],
        );
        // Gray broadcasts to (128,128,128) then swizzles — all same so still (128,128,128,255).
        assert_eq!(dst, [128, 128, 128, 255]);
    }

    #[test]
    fn rgb8_to_gray8() {
        // BT.709 luma: (54*R + 183*G + 19*B + 128) >> 8
        let dst = convert_pixel(
            PixelDescriptor::RGB8_SRGB,
            PixelDescriptor::GRAY8_SRGB,
            &[255, 0, 0],
        );
        // (54*255 + 0 + 0 + 128) >> 8 = (13770 + 128) >> 8 = 13898 >> 8 = 54
        assert_eq!(dst, [54]);
    }

    #[test]
    fn rgba8_to_gray8() {
        let dst = convert_pixel(
            PixelDescriptor::RGBA8_SRGB,
            PixelDescriptor::GRAY8_SRGB,
            &[0, 255, 0, 255],
        );
        // (0 + 183*255 + 0 + 128) >> 8 = (46665 + 128) >> 8 = 46793 >> 8 = 182
        assert_eq!(dst, [182]);
    }

    #[test]
    fn bgra8_to_gray8() {
        // BGRA → RGBA → Gray (two-step).
        let dst = convert_pixel(
            PixelDescriptor::BGRA8_SRGB,
            PixelDescriptor::GRAY8_SRGB,
            &[0, 255, 0, 255], // BGRA: B=0, G=255, R=0
        );
        // After swizzle: RGBA = [0, 255, 0, 255], then gray = 182.
        assert_eq!(dst, [182]);
    }

    // -----------------------------------------------------------------------
    // GrayAlpha conversions
    // -----------------------------------------------------------------------

    #[test]
    fn gray8_to_grayalpha8() {
        let from = PixelDescriptor::new(
            ChannelType::U8,
            ChannelLayout::Gray,
            None,
            TransferFunction::Srgb,
        );
        let to = PixelDescriptor::new(
            ChannelType::U8,
            ChannelLayout::GrayAlpha,
            Some(AlphaMode::Straight),
            TransferFunction::Srgb,
        );
        let dst = convert_pixel(from, to, &[100]);
        assert_eq!(dst, [100, 255]);
    }

    #[test]
    fn grayalpha8_to_gray8() {
        let from = PixelDescriptor::new(
            ChannelType::U8,
            ChannelLayout::GrayAlpha,
            Some(AlphaMode::Straight),
            TransferFunction::Srgb,
        );
        let to = PixelDescriptor::new(
            ChannelType::U8,
            ChannelLayout::Gray,
            None,
            TransferFunction::Srgb,
        );
        let dst = convert_pixel(from, to, &[100, 200]);
        assert_eq!(dst, [100]); // Alpha dropped.
    }

    #[test]
    fn grayalpha8_to_rgba8() {
        let from = PixelDescriptor::new(
            ChannelType::U8,
            ChannelLayout::GrayAlpha,
            Some(AlphaMode::Straight),
            TransferFunction::Srgb,
        );
        let dst = convert_pixel(from, PixelDescriptor::RGBA8_SRGB, &[128, 200]);
        assert_eq!(dst, [128, 128, 128, 200]);
    }

    #[test]
    fn grayalpha8_to_rgb8() {
        let from = PixelDescriptor::new(
            ChannelType::U8,
            ChannelLayout::GrayAlpha,
            Some(AlphaMode::Straight),
            TransferFunction::Srgb,
        );
        let dst = convert_pixel(from, PixelDescriptor::RGB8_SRGB, &[128, 200]);
        assert_eq!(dst, [128, 128, 128]);
    }

    #[test]
    fn grayalpha8_to_bgra8() {
        // GrayAlpha → RGBA → BGRA (two-step).
        let from = PixelDescriptor::new(
            ChannelType::U8,
            ChannelLayout::GrayAlpha,
            Some(AlphaMode::Straight),
            TransferFunction::Srgb,
        );
        let dst = convert_pixel(from, PixelDescriptor::BGRA8_SRGB, &[128, 200]);
        // Gray broadcasts to (128,128,128,200) then BGRA swizzle — all same so (128,128,128,200).
        assert_eq!(dst, [128, 128, 128, 200]);
    }

    // -----------------------------------------------------------------------
    // Depth conversions
    // -----------------------------------------------------------------------

    #[test]
    fn u8_to_u16_roundtrip() {
        let u8_desc = PixelDescriptor::RGB8_SRGB;
        let u16_desc = PixelDescriptor::new(
            ChannelType::U16,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Srgb,
        );
        let src = [0u8, 128, 255];
        let wide = convert_pixel(u8_desc, u16_desc, &src);
        let wide16: &[u16] = bytemuck::cast_slice(&wide);
        assert_eq!(wide16[0], 0); // 0 * 257 = 0
        assert_eq!(wide16[1], 128 * 257); // 32896
        assert_eq!(wide16[2], 255 * 257); // 65535

        // Narrow back to u8.
        let narrow = convert_pixel(u16_desc, u8_desc, &wide);
        assert_eq!(narrow, [0, 128, 255]);
    }

    #[test]
    fn naive_u8_to_f32() {
        // Non-sRGB transfer → naive path.
        let u8_desc = PixelDescriptor::new(
            ChannelType::U8,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
        );
        let f32_desc = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
        );
        let dst = convert_pixel(u8_desc, f32_desc, &[0, 128, 255]);
        let f: &[f32] = bytemuck::cast_slice(&dst);
        assert!((f[0] - 0.0).abs() < 1e-6);
        assert!((f[1] - 128.0 / 255.0).abs() < 1e-5);
        assert!((f[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn naive_f32_to_u8() {
        let f32_desc = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
        );
        let u8_desc = PixelDescriptor::new(
            ChannelType::U8,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
        );
        let src_f: [f32; 3] = [0.0, 0.5, 1.0];
        let src: &[u8] = bytemuck::cast_slice(&src_f);
        let dst = convert_pixel(f32_desc, u8_desc, src);
        assert_eq!(dst[0], 0);
        assert_eq!(dst[1], 128); // 0.5 * 255 + 0.5 = 128.0 → 128
        assert_eq!(dst[2], 255);
    }

    #[test]
    fn srgb_u8_to_linear_f32() {
        let dst = convert_pixel(
            PixelDescriptor::RGB8_SRGB,
            PixelDescriptor::new(
                ChannelType::F32,
                ChannelLayout::Rgb,
                None,
                TransferFunction::Linear,
            ),
            &[0, 128, 255],
        );
        let f: &[f32] = bytemuck::cast_slice(&dst);
        assert!((f[0] - 0.0).abs() < 1e-6);
        // sRGB 128/255 ≈ 0.2158 linear
        assert!((f[1] - 0.2158).abs() < 0.01);
        assert!((f[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn linear_f32_to_srgb_u8() {
        let f32_lin = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
        );
        let src_f: [f32; 3] = [0.0, 0.5, 1.0];
        let src: &[u8] = bytemuck::cast_slice(&src_f);
        let dst = convert_pixel(f32_lin, PixelDescriptor::RGB8_SRGB, src);
        assert_eq!(dst[0], 0);
        // linear 0.5 ≈ sRGB 188
        assert!((dst[1] as i32 - 188).abs() <= 1);
        assert_eq!(dst[2], 255);
    }

    #[test]
    fn u16_to_f32_and_back() {
        let u16_desc = PixelDescriptor::new(
            ChannelType::U16,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Srgb,
        );
        let f32_desc = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Srgb,
        );
        let src16: [u16; 3] = [0, 32768, 65535];
        let src: &[u8] = bytemuck::cast_slice(&src16);
        let mid = convert_pixel(u16_desc, f32_desc, src);
        let f: &[f32] = bytemuck::cast_slice(&mid);
        assert!((f[0] - 0.0).abs() < 1e-6);
        assert!((f[1] - 0.5000076).abs() < 1e-4);
        assert!((f[2] - 1.0).abs() < 1e-6);

        // Round-trip back.
        let back = convert_pixel(f32_desc, u16_desc, &mid);
        let back16: &[u16] = bytemuck::cast_slice(&back);
        assert_eq!(back16[0], 0);
        assert!((back16[1] as i32 - 32768).abs() <= 1);
        assert_eq!(back16[2], 65535);
    }

    // -----------------------------------------------------------------------
    // HDR transfer function conversions
    // -----------------------------------------------------------------------

    #[test]
    fn pq_u16_to_linear_f32_roundtrip() {
        let pq_u16 = PixelDescriptor::new(
            ChannelType::U16,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Pq,
        );
        let lin_f32 = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
        );
        let src16: [u16; 3] = [0, 32768, 65535];
        let src: &[u8] = bytemuck::cast_slice(&src16);
        let mid = convert_pixel(pq_u16, lin_f32, src);
        let f: &[f32] = bytemuck::cast_slice(&mid);
        assert_eq!(f[0], 0.0);
        assert!(f[1] > 0.0 && f[1] < 1.0);

        // Round-trip.
        let back = convert_pixel(lin_f32, pq_u16, &mid);
        let back16: &[u16] = bytemuck::cast_slice(&back);
        assert_eq!(back16[0], 0);
        assert!((back16[1] as i32 - 32768).abs() <= 2);
    }

    #[test]
    fn pq_f32_to_linear_f32_roundtrip() {
        let pq_f32 = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Pq,
        );
        let lin_f32 = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
        );
        let src_f: [f32; 3] = [0.0, 0.5, 1.0];
        let src: &[u8] = bytemuck::cast_slice(&src_f);
        let mid = convert_pixel(pq_f32, lin_f32, src);
        let back = convert_pixel(lin_f32, pq_f32, &mid);
        let back_f: &[f32] = bytemuck::cast_slice(&back);
        for i in 0..3 {
            assert!(
                (back_f[i] - src_f[i]).abs() < 1e-4,
                "PQ F32 roundtrip ch{i}: {:.6} vs {:.6}",
                back_f[i],
                src_f[i]
            );
        }
    }

    #[test]
    fn hlg_u16_to_linear_f32_roundtrip() {
        let hlg_u16 = PixelDescriptor::new(
            ChannelType::U16,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Hlg,
        );
        let lin_f32 = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
        );
        let src16: [u16; 3] = [0, 32768, 65535];
        let src: &[u8] = bytemuck::cast_slice(&src16);
        let mid = convert_pixel(hlg_u16, lin_f32, src);
        let f: &[f32] = bytemuck::cast_slice(&mid);
        assert_eq!(f[0], 0.0);
        assert!(f[1] > 0.0);

        let back = convert_pixel(lin_f32, hlg_u16, &mid);
        let back16: &[u16] = bytemuck::cast_slice(&back);
        assert_eq!(back16[0], 0);
        assert!((back16[1] as i32 - 32768).abs() <= 2);
    }

    #[test]
    fn hlg_f32_to_linear_f32_roundtrip() {
        let hlg_f32 = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Hlg,
        );
        let lin_f32 = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
        );
        let src_f: [f32; 3] = [0.0, 0.5, 1.0];
        let src: &[u8] = bytemuck::cast_slice(&src_f);
        let mid = convert_pixel(hlg_f32, lin_f32, src);
        let back = convert_pixel(lin_f32, hlg_f32, &mid);
        let back_f: &[f32] = bytemuck::cast_slice(&back);
        for i in 0..3 {
            assert!(
                (back_f[i] - src_f[i]).abs() < 1e-4,
                "HLG F32 roundtrip ch{i}: {:.6} vs {:.6}",
                back_f[i],
                src_f[i]
            );
        }
    }

    #[test]
    fn pq_to_hlg_via_linear_f32() {
        let pq = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Pq,
        );
        let hlg = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Hlg,
        );
        let src_f: [f32; 3] = [0.0, 0.5, 1.0];
        let src: &[u8] = bytemuck::cast_slice(&src_f);
        let dst = convert_pixel(pq, hlg, src);
        let f: &[f32] = bytemuck::cast_slice(&dst);
        // PQ 0 → HLG 0.
        assert_eq!(f[0], 0.0);
        // PQ 0.5 → some HLG value.
        assert!(f[1] > 0.0 && f[1] <= 1.0);
    }

    #[test]
    fn hdr_u16_to_sdr_u8_pq() {
        // PQ U16 → sRGB U8 (two-step: EOTF + OETF).
        let pq_u16 = PixelDescriptor::new(
            ChannelType::U16,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Pq,
        );
        let src16: [u16; 3] = [32768, 32768, 32768];
        let src: &[u8] = bytemuck::cast_slice(&src16);
        let dst = convert_pixel(pq_u16, PixelDescriptor::RGB8_SRGB, src);
        // Should produce some valid sRGB value.
        assert!(dst[0] > 0 && dst[0] < 255);
    }

    #[test]
    fn hdr_u16_to_sdr_u8_hlg() {
        let hlg_u16 = PixelDescriptor::new(
            ChannelType::U16,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Hlg,
        );
        let src16: [u16; 3] = [32768, 32768, 32768];
        let src: &[u8] = bytemuck::cast_slice(&src16);
        let dst = convert_pixel(hlg_u16, PixelDescriptor::RGB8_SRGB, src);
        assert!(dst[0] > 0 && dst[0] < 255);
    }

    // -----------------------------------------------------------------------
    // Alpha premultiplication
    // -----------------------------------------------------------------------

    #[test]
    fn straight_to_premul_u8() {
        let straight = PixelDescriptor::RGBA8_SRGB;
        let premul = PixelDescriptor::new(
            ChannelType::U8,
            ChannelLayout::Rgba,
            Some(AlphaMode::Premultiplied),
            TransferFunction::Srgb,
        );
        // 50% alpha: RGB channels halved.
        let dst = convert_pixel(straight, premul, &[200, 100, 50, 128]);
        // (200 * 128 + 128) / 255 = 100, (100 * 128 + 128) / 255 = 50, (50 * 128 + 128) / 255 = 25
        assert!((dst[0] as i32 - 100).abs() <= 1);
        assert!((dst[1] as i32 - 50).abs() <= 1);
        assert!((dst[2] as i32 - 25).abs() <= 1);
        assert_eq!(dst[3], 128);
    }

    #[test]
    fn premul_to_straight_u8() {
        let premul = PixelDescriptor::new(
            ChannelType::U8,
            ChannelLayout::Rgba,
            Some(AlphaMode::Premultiplied),
            TransferFunction::Srgb,
        );
        let straight = PixelDescriptor::RGBA8_SRGB;
        // Premul with alpha=128: channels are already halved.
        let dst = convert_pixel(premul, straight, &[100, 50, 25, 128]);
        // Unpremultiply: (100 * 255 + 64) / 128 = 199, etc.
        assert!((dst[0] as i32 - 200).abs() <= 1);
        assert!((dst[1] as i32 - 100).abs() <= 1);
        assert!((dst[2] as i32 - 50).abs() <= 1);
        assert_eq!(dst[3], 128);
    }

    #[test]
    fn premul_to_straight_zero_alpha() {
        let premul = PixelDescriptor::new(
            ChannelType::U8,
            ChannelLayout::Rgba,
            Some(AlphaMode::Premultiplied),
            TransferFunction::Srgb,
        );
        let straight = PixelDescriptor::RGBA8_SRGB;
        let dst = convert_pixel(premul, straight, &[0, 0, 0, 0]);
        assert_eq!(dst, [0, 0, 0, 0]);
    }

    #[test]
    fn straight_to_premul_f32() {
        let straight = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgba,
            Some(AlphaMode::Straight),
            TransferFunction::Linear,
        );
        let premul = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgba,
            Some(AlphaMode::Premultiplied),
            TransferFunction::Linear,
        );
        let src_f: [f32; 4] = [1.0, 0.5, 0.25, 0.5];
        let src: &[u8] = bytemuck::cast_slice(&src_f);
        let dst = convert_pixel(straight, premul, src);
        let f: &[f32] = bytemuck::cast_slice(&dst);
        assert!((f[0] - 0.5).abs() < 1e-6);
        assert!((f[1] - 0.25).abs() < 1e-6);
        assert!((f[2] - 0.125).abs() < 1e-6);
        assert!((f[3] - 0.5).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Oklab conversions
    // -----------------------------------------------------------------------

    #[test]
    fn rgb8_to_oklabf32_does_not_panic() {
        let mut conv =
            RowConverter::new(PixelDescriptor::RGB8_SRGB, PixelDescriptor::OKLABF32).unwrap();
        assert!(!conv.is_identity());

        let src = [128u8, 64, 200];
        let mut dst = [0u8; 12];
        conv.convert_row(&src, &mut dst, 1);

        let oklab: [f32; 3] = bytemuck::cast(dst);
        assert!(
            oklab[0] >= 0.0 && oklab[0] <= 1.0,
            "L out of range: {}",
            oklab[0]
        );
    }

    #[test]
    fn oklabf32_roundtrip() {
        // Linear RGB F32 → Oklab F32 → Linear RGB F32.
        let lin = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
        );
        let oklab = PixelDescriptor::OKLABF32;

        let src_f: [f32; 3] = [0.5, 0.3, 0.8];
        let src: &[u8] = bytemuck::cast_slice(&src_f);
        let mid = convert_pixel(lin, oklab, src);
        let back = convert_pixel(oklab, lin, &mid);
        let back_f: &[f32] = bytemuck::cast_slice(&back);
        for i in 0..3 {
            assert!(
                (back_f[i] - src_f[i]).abs() < 1e-4,
                "Oklab roundtrip ch{i}: {:.6} vs {:.6}",
                back_f[i],
                src_f[i]
            );
        }
    }

    #[test]
    fn oklabaf32_preserves_alpha() {
        let lin_rgba = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgba,
            Some(AlphaMode::Straight),
            TransferFunction::Linear,
        );
        let oklaba = PixelDescriptor::OKLABAF32;
        let src_f: [f32; 4] = [0.5, 0.3, 0.8, 0.7];
        let src: &[u8] = bytemuck::cast_slice(&src_f);
        let mid = convert_pixel(lin_rgba, oklaba, src);
        let mid_f: &[f32] = bytemuck::cast_slice(&mid);
        assert!(
            (mid_f[3] - 0.7).abs() < 1e-6,
            "Alpha not preserved in Oklaba"
        );

        let back = convert_pixel(oklaba, lin_rgba, &mid);
        let back_f: &[f32] = bytemuck::cast_slice(&back);
        assert!(
            (back_f[3] - 0.7).abs() < 1e-6,
            "Alpha not preserved on round-trip"
        );
        for i in 0..3 {
            assert!(
                (back_f[i] - src_f[i]).abs() < 1e-4,
                "Oklaba roundtrip ch{i}: {:.6} vs {:.6}",
                back_f[i],
                src_f[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // Multi-row conversion
    // -----------------------------------------------------------------------

    #[test]
    fn convert_rows_basic() {
        let mut conv =
            RowConverter::new(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBA8_SRGB).unwrap();

        let src = [10u8, 20, 30, 40, 50, 60];
        let src_stride = 3; // tight, 1 pixel per row
        let mut dst = [0u8; 8]; // 2 rows × 4 bytes
        let dst_stride = 4;

        conv.convert_rows(&src, src_stride, &mut dst, dst_stride, 1, 2)
            .unwrap();
        assert_eq!(&dst[0..4], &[10, 20, 30, 255]);
        assert_eq!(&dst[4..8], &[40, 50, 60, 255]);
    }

    #[test]
    fn convert_rows_buffer_too_small() {
        let mut conv =
            RowConverter::new(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBA8_SRGB).unwrap();

        let src = [10u8, 20, 30];
        let mut dst = [0u8; 4];
        let err = conv.convert_rows(&src, 3, &mut dst, 4, 1, 2).unwrap_err();
        assert!(matches!(*err.error(), ConvertError::BufferSize { .. }));
    }

    // -----------------------------------------------------------------------
    // ConvertPlan::new_explicit policy checks
    // -----------------------------------------------------------------------

    #[test]
    fn new_explicit_alpha_forbid() {
        let from = PixelDescriptor::RGBA8_SRGB;
        let to = PixelDescriptor::RGB8_SRGB;
        let opts = ConvertOptions::forbid_lossy().with_depth_policy(DepthPolicy::Round);
        let err = ConvertPlan::new_explicit(from, to, &opts).unwrap_err();
        assert_eq!(*err.error(), ConvertError::AlphaRemovalForbidden);
    }

    #[test]
    fn new_explicit_depth_forbid() {
        let from = PixelDescriptor::new(
            ChannelType::U16,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Srgb,
        );
        let to = PixelDescriptor::RGB8_SRGB;
        let opts = ConvertOptions::forbid_lossy().with_alpha_policy(AlphaPolicy::DiscardUnchecked);
        let err = ConvertPlan::new_explicit(from, to, &opts).unwrap_err();
        assert_eq!(*err.error(), ConvertError::DepthReductionForbidden);
    }

    #[test]
    fn new_explicit_rgb_to_gray_requires_luma() {
        let opts = ConvertOptions::forbid_lossy()
            .with_alpha_policy(AlphaPolicy::DiscardUnchecked)
            .with_depth_policy(DepthPolicy::Round);
        let err = ConvertPlan::new_explicit(
            PixelDescriptor::RGB8_SRGB,
            PixelDescriptor::GRAY8_SRGB,
            &opts,
        )
        .unwrap_err();
        assert_eq!(*err.error(), ConvertError::RgbToGray);
    }

    #[test]
    fn new_explicit_allows_when_policies_permit() {
        let opts = ConvertOptions::permissive().with_alpha_policy(AlphaPolicy::DiscardUnchecked);
        let plan = ConvertPlan::new_explicit(
            PixelDescriptor::RGBA8_SRGB,
            PixelDescriptor::GRAY8_SRGB,
            &opts,
        )
        .unwrap();
        assert!(!plan.is_identity());
    }

    #[test]
    fn clip_out_of_gamut_false_preserves_negatives() {
        // P3 pure green → sRGB produces negative red (out of sRGB gamut).
        // With clip_out_of_gamut=false, the extended-range transfer must
        // preserve those negatives instead of clamping to zero.
        let p3 = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Srgb,
        )
        .with_primaries(zenpixels::ColorPrimaries::DisplayP3);
        let srgb = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Srgb,
        );

        let opts = ConvertOptions::permissive().with_clip_out_of_gamut(false);
        let mut conv = crate::RowConverter::new_explicit(p3, srgb, &opts).unwrap();

        let src: [f32; 3] = [0.0, 1.0, 0.0];
        let mut dst = [0.0f32; 3];
        conv.convert_row(
            bytemuck::cast_slice(&src),
            bytemuck::cast_slice_mut(&mut dst),
            1,
        );
        assert!(
            dst[0] < 0.0,
            "extended range should preserve negative red, got {}",
            dst[0]
        );
    }

    #[test]
    fn clip_out_of_gamut_true_clamps_negatives() {
        // Default clip_out_of_gamut=true clamps sRGB transfer to [0, 1].
        let p3 = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Srgb,
        )
        .with_primaries(zenpixels::ColorPrimaries::DisplayP3);
        let srgb = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Srgb,
        );

        let opts = ConvertOptions::permissive();
        assert!(opts.clip_out_of_gamut);
        let mut conv = crate::RowConverter::new_explicit(p3, srgb, &opts).unwrap();

        let src: [f32; 3] = [0.0, 1.0, 0.0];
        let mut dst = [0.0f32; 3];
        conv.convert_row(
            bytemuck::cast_slice(&src),
            bytemuck::cast_slice_mut(&mut dst),
            1,
        );
        assert!(
            dst[0] >= 0.0,
            "clamped path should not produce negatives, got {}",
            dst[0]
        );
    }

    // -----------------------------------------------------------------------
    // Pluggable CMS
    // -----------------------------------------------------------------------

    /// Mock CMS that paints every output pixel red. Used to verify that the
    /// plugin gets hooked into the plan and actually drives the row.
    struct PaintRedCms {
        accepted: core::sync::atomic::AtomicUsize,
    }

    struct PaintRedTransform;

    impl crate::cms::RowTransformMut for PaintRedTransform {
        fn transform_row(&mut self, _src: &[u8], dst: &mut [u8], width: u32) {
            for px in dst.chunks_exact_mut(3).take(width as usize) {
                px[0] = 255;
                px[1] = 0;
                px[2] = 0;
            }
        }
    }

    impl crate::cms::PluggableCms for PaintRedCms {
        fn build_source_transform(
            &self,
            _src: zenpixels::ColorProfileSource<'_>,
            _dst: zenpixels::ColorProfileSource<'_>,
            _src_format: zenpixels::PixelFormat,
            _dst_format: zenpixels::PixelFormat,
        ) -> Option<Box<dyn crate::cms::RowTransformMut>> {
            self.accepted
                .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            Some(Box::new(PaintRedTransform))
        }
    }

    #[test]
    fn pluggable_cms_drives_row_when_profiles_differ() {
        // P3 RGB8 → sRGB RGB8: profiles differ, plugin must be asked and
        // its transform must be the one that runs.
        let p3 = PixelDescriptor::RGB8_SRGB.with_primaries(zenpixels::ColorPrimaries::DisplayP3);
        let srgb = PixelDescriptor::RGB8_SRGB;

        let cms = PaintRedCms {
            accepted: core::sync::atomic::AtomicUsize::new(0),
        };
        let opts = ConvertOptions::permissive();
        let mut conv =
            crate::RowConverter::new_explicit_with_cms(p3, srgb, &opts, Some(&cms)).unwrap();

        assert_eq!(cms.accepted.load(core::sync::atomic::Ordering::Relaxed), 1);

        let src = [10u8, 20, 30, 40, 50, 60];
        let mut dst = [0u8; 6];
        conv.convert_row(&src, &mut dst, 2);
        assert_eq!(dst, [255, 0, 0, 255, 0, 0]);
    }

    #[test]
    fn pluggable_cms_skipped_when_profiles_match() {
        // Same profile on both sides — plan is identity, plugin is never
        // consulted.
        let cms = PaintRedCms {
            accepted: core::sync::atomic::AtomicUsize::new(0),
        };
        let opts = ConvertOptions::permissive();
        let conv = crate::RowConverter::new_explicit_with_cms(
            PixelDescriptor::RGB8_SRGB,
            PixelDescriptor::RGB8_SRGB,
            &opts,
            Some(&cms),
        )
        .unwrap();
        assert!(conv.is_identity());
        assert_eq!(cms.accepted.load(core::sync::atomic::Ordering::Relaxed), 0);
    }

    #[test]
    fn pluggable_cms_declines_falls_back_to_builtin() {
        // Plugin returns None — must fall back to the built-in gamut path.
        struct DeclineCms;
        impl crate::cms::PluggableCms for DeclineCms {
            fn build_source_transform(
                &self,
                _src: zenpixels::ColorProfileSource<'_>,
                _dst: zenpixels::ColorProfileSource<'_>,
                _src_format: zenpixels::PixelFormat,
                _dst_format: zenpixels::PixelFormat,
            ) -> Option<Box<dyn crate::cms::RowTransformMut>> {
                None
            }
        }

        let p3 = PixelDescriptor::RGB8_SRGB.with_primaries(zenpixels::ColorPrimaries::DisplayP3);
        let srgb = PixelDescriptor::RGB8_SRGB;
        let opts = ConvertOptions::permissive();
        let mut conv =
            crate::RowConverter::new_explicit_with_cms(p3, srgb, &opts, Some(&DeclineCms)).unwrap();

        // Built-in path should produce non-red output from grey source.
        let src = [128u8, 128, 128];
        let mut dst = [0u8; 3];
        conv.convert_row(&src, &mut dst, 1);
        // P3 grey → sRGB grey: R ≈ G ≈ B. Not the all-red sentinel the
        // plugin would have written, proving we took the built-in path.
        assert_ne!(dst, [255, 0, 0]);
    }

    // -----------------------------------------------------------------------
    // Plan accessors
    // -----------------------------------------------------------------------

    #[test]
    fn plan_accessors() {
        let from = PixelDescriptor::RGB8_SRGB;
        let to = PixelDescriptor::RGBA8_SRGB;
        let conv = RowConverter::new(from, to).unwrap();
        let plan = conv.plan();
        assert_eq!(plan.from(), from);
        assert_eq!(plan.to(), to);
        assert!(!plan.is_identity());
    }

    #[test]
    fn from_plan() {
        let plan =
            ConvertPlan::new(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBA8_SRGB).unwrap();
        let conv = RowConverter::from_plan(plan);
        assert!(!conv.is_identity());
    }

    // -----------------------------------------------------------------------
    // Multi-pixel conversion
    // -----------------------------------------------------------------------

    #[test]
    fn convert_multiple_pixels() {
        let mut conv =
            RowConverter::new(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBA8_SRGB).unwrap();
        let src = [10, 20, 30, 40, 50, 60, 70, 80, 90];
        let mut dst = [0u8; 12];
        conv.convert_row(&src, &mut dst, 3);
        assert_eq!(dst, [10, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255]);
    }

    // -----------------------------------------------------------------------
    // Combined layout + depth conversions
    // -----------------------------------------------------------------------

    #[test]
    fn gray8_to_rgbaf32_linear() {
        // Gray U8 sRGB → RGBA F32 Linear: depth expands, then layout expands.
        let from = PixelDescriptor::GRAY8_SRGB;
        let to = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgba,
            Some(AlphaMode::Straight),
            TransferFunction::Linear,
        );
        let dst = convert_pixel(from, to, &[128]);
        let f: &[f32] = bytemuck::cast_slice(&dst);
        // sRGB 128 ≈ linear 0.2158, broadcasted, alpha = 1.0.
        assert!((f[0] - 0.2158).abs() < 0.01);
        assert!((f[0] - f[1]).abs() < 1e-6);
        assert!((f[0] - f[2]).abs() < 1e-6);
        assert!((f[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rgba8_to_rgb16() {
        // Layout contracts (drop alpha), then depth expands.
        let from = PixelDescriptor::RGBA8_SRGB;
        let to = PixelDescriptor::new(
            ChannelType::U16,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Srgb,
        );
        let dst = convert_pixel(from, to, &[100, 150, 200, 255]);
        let u16s: &[u16] = bytemuck::cast_slice(&dst);
        assert_eq!(u16s[0], 100 * 257);
        assert_eq!(u16s[1], 150 * 257);
        assert_eq!(u16s[2], 200 * 257);
    }

    // -----------------------------------------------------------------------
    // U16 and F32 layout conversion kernels
    // -----------------------------------------------------------------------

    #[test]
    fn gray_u16_to_rgb_u16() {
        let from = PixelDescriptor::new(
            ChannelType::U16,
            ChannelLayout::Gray,
            None,
            TransferFunction::Srgb,
        );
        let to = PixelDescriptor::new(
            ChannelType::U16,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Srgb,
        );
        let src16: [u16; 1] = [40000];
        let src: &[u8] = bytemuck::cast_slice(&src16);
        let dst = convert_pixel(from, to, src);
        let out: &[u16] = bytemuck::cast_slice(&dst);
        assert_eq!(out, [40000, 40000, 40000]);
    }

    #[test]
    fn gray_f32_to_rgba_f32() {
        let from = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Gray,
            None,
            TransferFunction::Linear,
        );
        let to = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgba,
            Some(AlphaMode::Straight),
            TransferFunction::Linear,
        );
        let src_f: [f32; 1] = [0.6];
        let src: &[u8] = bytemuck::cast_slice(&src_f);
        let dst = convert_pixel(from, to, src);
        let out: &[f32] = bytemuck::cast_slice(&dst);
        assert!((out[0] - 0.6).abs() < 1e-6);
        assert!((out[1] - 0.6).abs() < 1e-6);
        assert!((out[2] - 0.6).abs() < 1e-6);
        assert!((out[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rgb_f32_to_rgba_f32() {
        let from = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
        );
        let to = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgba,
            Some(AlphaMode::Straight),
            TransferFunction::Linear,
        );
        let src_f: [f32; 3] = [0.2, 0.4, 0.8];
        let src: &[u8] = bytemuck::cast_slice(&src_f);
        let dst = convert_pixel(from, to, src);
        let out: &[f32] = bytemuck::cast_slice(&dst);
        assert!((out[0] - 0.2).abs() < 1e-6);
        assert!((out[1] - 0.4).abs() < 1e-6);
        assert!((out[2] - 0.8).abs() < 1e-6);
        assert!((out[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rgba_u16_to_rgb_u16() {
        let from = PixelDescriptor::new(
            ChannelType::U16,
            ChannelLayout::Rgba,
            Some(AlphaMode::Straight),
            TransferFunction::Srgb,
        );
        let to = PixelDescriptor::new(
            ChannelType::U16,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Srgb,
        );
        let src16: [u16; 4] = [10000, 20000, 30000, 65535];
        let src: &[u8] = bytemuck::cast_slice(&src16);
        let dst = convert_pixel(from, to, src);
        let out: &[u16] = bytemuck::cast_slice(&dst);
        assert_eq!(out, [10000, 20000, 30000]);
    }

    #[test]
    fn gray_alpha_u16_to_rgba_u16() {
        let from = PixelDescriptor::new(
            ChannelType::U16,
            ChannelLayout::GrayAlpha,
            Some(AlphaMode::Straight),
            TransferFunction::Srgb,
        );
        let to = PixelDescriptor::new(
            ChannelType::U16,
            ChannelLayout::Rgba,
            Some(AlphaMode::Straight),
            TransferFunction::Srgb,
        );
        let src16: [u16; 2] = [50000, 32768];
        let src: &[u8] = bytemuck::cast_slice(&src16);
        let dst = convert_pixel(from, to, src);
        let out: &[u16] = bytemuck::cast_slice(&dst);
        assert_eq!(out, [50000, 50000, 50000, 32768]);
    }

    #[test]
    fn gray_alpha_f32_to_rgb_f32() {
        let from = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::GrayAlpha,
            Some(AlphaMode::Straight),
            TransferFunction::Linear,
        );
        let to = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
        );
        let src_f: [f32; 2] = [0.75, 0.5];
        let src: &[u8] = bytemuck::cast_slice(&src_f);
        let dst = convert_pixel(from, to, src);
        let out: &[f32] = bytemuck::cast_slice(&dst);
        assert!((out[0] - 0.75).abs() < 1e-6);
        assert!((out[1] - 0.75).abs() < 1e-6);
        assert!((out[2] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn gray_u16_to_gray_alpha_u16() {
        let from = PixelDescriptor::new(
            ChannelType::U16,
            ChannelLayout::Gray,
            None,
            TransferFunction::Srgb,
        );
        let to = PixelDescriptor::new(
            ChannelType::U16,
            ChannelLayout::GrayAlpha,
            Some(AlphaMode::Straight),
            TransferFunction::Srgb,
        );
        let src16: [u16; 1] = [12345];
        let src: &[u8] = bytemuck::cast_slice(&src16);
        let dst = convert_pixel(from, to, src);
        let out: &[u16] = bytemuck::cast_slice(&dst);
        assert_eq!(out, [12345, 65535]);
    }

    #[test]
    fn gray_alpha_f32_to_gray_f32() {
        let from = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::GrayAlpha,
            Some(AlphaMode::Straight),
            TransferFunction::Linear,
        );
        let to = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Gray,
            None,
            TransferFunction::Linear,
        );
        let src_f: [f32; 2] = [0.33, 0.9];
        let src: &[u8] = bytemuck::cast_slice(&src_f);
        let dst = convert_pixel(from, to, src);
        let out: &[f32] = bytemuck::cast_slice(&dst);
        assert!((out[0] - 0.33).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Transfer function roundtrips (ext.rs branches)
    // -----------------------------------------------------------------------

    #[test]
    fn bt709_linear_f32_roundtrip() {
        use crate::TransferFunctionExt;
        let tf = TransferFunction::Bt709;
        let values = [0.0f32, 0.1, 0.25, 0.5, 0.75, 1.0];
        for &v in &values {
            let linear = tf.linearize(v);
            let back = tf.delinearize(linear);
            assert!(
                (back - v).abs() < 1e-5,
                "Bt709 roundtrip failed for {v}: linearize={linear}, delinearize={back}"
            );
        }
    }

    #[test]
    fn unknown_transfer_roundtrip() {
        use crate::TransferFunctionExt;
        let tf = TransferFunction::Unknown;
        let values = [0.0f32, 0.1, 0.5, 0.99, 1.0];
        for &v in &values {
            let linear = tf.linearize(v);
            assert!(
                (linear - v).abs() < 1e-7,
                "Unknown linearize should be identity: {v} -> {linear}"
            );
            let back = tf.delinearize(linear);
            assert!(
                (back - v).abs() < 1e-7,
                "Unknown delinearize should be identity: {linear} -> {back}"
            );
        }
    }

    #[test]
    fn oklab_unknown_primaries_returns_error() {
        use crate::ColorPrimaries;
        let from = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgb,
            None,
            TransferFunction::Linear,
        )
        .with_primaries(ColorPrimaries::Unknown);
        let to = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Oklab,
            None,
            TransferFunction::Linear,
        )
        .with_primaries(ColorPrimaries::Unknown);
        let result = RowConverter::new(from, to);
        assert!(result.is_err(), "Oklab with Unknown primaries should fail");
    }

    // -----------------------------------------------------------------------
    // Compose
    // -----------------------------------------------------------------------

    #[test]
    fn compose_roundtrip_cancels_to_identity() {
        let a = RowConverter::new(PixelDescriptor::RGBA8_SRGB, PixelDescriptor::RGBAF32_LINEAR)
            .unwrap();
        let b = RowConverter::new(PixelDescriptor::RGBAF32_LINEAR, PixelDescriptor::RGBA8_SRGB)
            .unwrap();
        let composed = a.compose(&b).unwrap();
        assert!(composed.is_identity(), "sRGB→linear→sRGB should cancel");
    }

    #[test]
    fn compose_chain_reduces_steps() {
        // RGBA8 sRGB → RGBAF32 linear → RGBAF32 sRGB
        // Steps: SrgbU8ToLinearF32 then LinearF32ToSrgbF32
        // The sRGB→linear step from plan A and linear→sRGB from plan B
        // should NOT cancel (different depth target), but composing avoids
        // an intermediate allocation.
        let a = RowConverter::new(PixelDescriptor::RGBA8_SRGB, PixelDescriptor::RGBAF32_LINEAR)
            .unwrap();
        let srgb_f32 = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgba,
            Some(AlphaMode::Straight),
            TransferFunction::Srgb,
        );
        let b = RowConverter::new(PixelDescriptor::RGBAF32_LINEAR, srgb_f32).unwrap();
        let composed = a.compose(&b).unwrap();
        assert!(!composed.is_identity());
        assert_eq!(composed.from_descriptor(), PixelDescriptor::RGBA8_SRGB);
        assert_eq!(composed.to_descriptor(), srgb_f32);
    }

    #[test]
    fn compose_incompatible_returns_none() {
        let a = RowConverter::new(PixelDescriptor::RGBA8_SRGB, PixelDescriptor::RGBAF32_LINEAR)
            .unwrap();
        let b = RowConverter::new(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBA8_SRGB).unwrap();
        assert!(a.compose(&b).is_none(), "RGBAF32_LINEAR != RGB8_SRGB");
    }

    #[test]
    fn compose_premul_roundtrip_cancels() {
        let premul = PixelDescriptor::new(
            ChannelType::F32,
            ChannelLayout::Rgba,
            Some(AlphaMode::Premultiplied),
            TransferFunction::Linear,
        );
        let a = RowConverter::new(PixelDescriptor::RGBAF32_LINEAR, premul).unwrap();
        let b = RowConverter::new(premul, PixelDescriptor::RGBAF32_LINEAR).unwrap();
        let composed = a.compose(&b).unwrap();
        assert!(
            composed.is_identity(),
            "straight→premul→straight should cancel"
        );
    }

    // -----------------------------------------------------------------------
    // CMYK guard
    // -----------------------------------------------------------------------

    #[test]
    #[should_panic(expected = "CMYK pixel data cannot be processed")]
    fn cmyk_rejected_by_row_converter() {
        let _ = RowConverter::new(PixelDescriptor::CMYK8, PixelDescriptor::RGB8_SRGB);
    }
}
