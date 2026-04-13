//! Lightweight CMS backend using fused SIMD gamut kernels.
//!
//! [`ZenCmsLite`] implements [`ColorManagement`] for named-profile conversions
//! (sRGB, Display P3, BT.2020, Adobe RGB, DCI-P3) without ICC profile parsing.
//! It delegates to [`fast_gamut`](crate::fast_gamut) kernels for the actual
//! conversion work — fused TRC + matrix + TRC in a single SIMD pass.
//!
//! # When to use
//!
//! Use `ZenCmsLite` when:
//! - All source/destination color spaces are known named profiles or CICP codes
//! - No custom ICC profiles are involved
//! - You want maximum conversion speed (1.5–2x faster than moxcms for f32)
//!
//! Use `MoxCms` (or another full CMS) when:
//! - Source or destination is a custom ICC profile
//! - Profile identification from ICC bytes is needed
//!
//! # Supported conversions
//!
//! Any combination of `ColorProfileSource::PrimariesTransferPair`,
//! `ColorProfileSource::Named`, or `ColorProfileSource::Cicp` where both
//! primaries sets have known chromaticities. ICC profile sources are not
//! supported — `build_transform` returns an error, `identify_profile`
//! returns `None`.
//!
//! # Example
//!
//! ```rust,ignore
//! use zenpixels_convert::cms_lite::ZenCmsLite;
//! use zenpixels_convert::output::{finalize_for_output, OutputProfile};
//! use zenpixels_convert::Cicp;
//!
//! let ready = finalize_for_output(
//!     &buffer, &origin,
//!     OutputProfile::Named(Cicp::SRGB),
//!     PixelFormat::Rgb8,
//!     &ZenCmsLite,
//! )?;
//! ```

use alloc::boxed::Box;
use alloc::format;

use crate::cms::{ColorManagement, RowTransform};
use crate::fast_gamut;
use crate::{ChannelType, Cicp, ColorPrimaries, PixelFormat, TransferFunction};

/// Lightweight CMS using fused SIMD gamut conversion kernels.
///
/// Handles named-profile conversions without ICC parsing. Stateless and
/// zero-cost to construct.
#[derive(Debug, Clone, Copy, Default)]
pub struct ZenCmsLite;

/// Error from the lightweight CMS.
#[derive(Debug, Clone)]
pub struct ZenCmsLiteError(pub alloc::string::String);

impl core::fmt::Display for ZenCmsLiteError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Extract (primaries, transfer) from a `ColorProfileSource`, if possible.
fn extract_primaries_transfer(
    src: &crate::ColorProfileSource<'_>,
) -> Option<(ColorPrimaries, TransferFunction)> {
    src.primaries_transfer()
}

impl ColorManagement for ZenCmsLite {
    type Error = ZenCmsLiteError;

    fn build_transform(
        &self,
        _src_icc: &[u8],
        _dst_icc: &[u8],
    ) -> Result<Box<dyn RowTransform>, Self::Error> {
        Err(ZenCmsLiteError(
            "ZenCmsLite does not support ICC profile transforms".into(),
        ))
    }

    fn identify_profile(&self, _icc: &[u8]) -> Option<Cicp> {
        // ZenCmsLite cannot parse ICC profiles.
        None
    }

    fn build_source_transform(
        &self,
        src: crate::ColorProfileSource<'_>,
        dst: crate::ColorProfileSource<'_>,
        src_format: PixelFormat,
        dst_format: PixelFormat,
    ) -> Option<Result<Box<dyn RowTransform>, Self::Error>> {
        let (src_p, src_t) = extract_primaries_transfer(&src)?;
        let (dst_p, dst_t) = extract_primaries_transfer(&dst)?;

        // Same color space — no conversion needed.
        if src_p as u8 == dst_p as u8 && src_t as u8 == dst_t as u8 {
            return None;
        }

        // Compute the gamut matrix. If primaries are the same but TRC differs,
        // the matrix is identity — but we still need TRC conversion.
        let matrix = if src_p as u8 == dst_p as u8 {
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        } else {
            match src_p.gamut_matrix_to(dst_p) {
                Some(m) => m,
                None => {
                    return Some(Err(ZenCmsLiteError(format!(
                        "no gamut matrix for {src_p:?} → {dst_p:?}"
                    ))));
                }
            }
        };

        // Verify TRC functions are supported.
        if fast_gamut::scalar_linearize(src_t).is_none() {
            return Some(Err(ZenCmsLiteError(format!(
                "unsupported source transfer function: {src_t:?}"
            ))));
        }
        if fast_gamut::scalar_encode(dst_t).is_none() {
            return Some(Err(ZenCmsLiteError(format!(
                "unsupported destination transfer function: {dst_t:?}"
            ))));
        }

        // Determine pixel layout from the format we'll actually process.
        // Use src_format to determine channel type and alpha.
        let channel_type = src_format.channel_type();
        let has_alpha = src_format.has_alpha_bytes();

        Some(Ok(Box::new(LiteTransform {
            matrix,
            src_trc: src_t,
            dst_trc: dst_t,
            linearize: fast_gamut::scalar_linearize(src_t).unwrap(),
            encode: fast_gamut::scalar_encode(dst_t).unwrap(),
            has_alpha,
            channel_type,
            _dst_format: dst_format,
        })))
    }
}

struct LiteTransform {
    matrix: [[f32; 3]; 3],
    src_trc: TransferFunction,
    dst_trc: TransferFunction,
    linearize: fn(f32) -> f32,
    encode: fn(f32) -> f32,
    has_alpha: bool,
    channel_type: ChannelType,
    _dst_format: PixelFormat,
}

// LiteTransform contains only Copy types and fn pointers, which are all Send.
// Rust auto-derives Send for this, but static_assertions would catch regressions.

impl RowTransform for LiteTransform {
    fn transform_row(&self, src: &[u8], dst: &mut [u8], width: u32) {
        match self.channel_type {
            ChannelType::U8 => self.transform_u8(src, dst, width),
            ChannelType::U16 => self.transform_u16(src, dst, width),
            ChannelType::F32 | ChannelType::F16 | _ => self.transform_f32(src, dst),
        }
    }
}

impl LiteTransform {
    fn transform_u8(&self, src: &[u8], dst: &mut [u8], _width: u32) {
        if self.has_alpha {
            fast_gamut::convert_u8_rgba(&self.matrix, src, dst, self.linearize, self.encode);
        } else {
            fast_gamut::convert_u8_rgb(&self.matrix, src, dst, self.linearize, self.encode);
        }
    }

    fn transform_u16(&self, src: &[u8], dst: &mut [u8], _width: u32) {
        let src_u16: &[u16] = bytemuck::cast_slice(src);
        let dst_u16: &mut [u16] = bytemuck::cast_slice_mut(dst);
        if self.has_alpha {
            convert_u16_rgba(&self.matrix, src_u16, dst_u16, self.linearize, self.encode);
        } else {
            fast_gamut::convert_u16_rgb(
                &self.matrix,
                src_u16,
                dst_u16,
                self.linearize,
                self.encode,
            );
        }
    }

    fn transform_f32(&self, src: &[u8], dst: &mut [u8]) {
        // Copy src → dst, then transform in place.
        dst.copy_from_slice(src);
        let dst_f32: &mut [f32] = bytemuck::cast_slice_mut(dst);
        if self.has_alpha {
            fast_gamut::convert_f32_rgba_dispatch(
                &self.matrix,
                dst_f32,
                self.src_trc,
                self.dst_trc,
            );
        } else {
            fast_gamut::convert_f32_rgb_dispatch(&self.matrix, dst_f32, self.src_trc, self.dst_trc);
        }
    }
}

/// Convert u16 RGBA source to u16 RGBA dest via gamut conversion. Alpha copied.
fn convert_u16_rgba(
    m: &[[f32; 3]; 3],
    src: &[u16],
    dst: &mut [u16],
    linearize_fn: fn(f32) -> f32,
    encode_fn: fn(f32) -> f32,
) {
    debug_assert_eq!(src.len() % 4, 0);
    debug_assert_eq!(src.len(), dst.len());
    for (src_px, dst_px) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        let r = linearize_fn(src_px[0] as f32 / 65535.0);
        let g = linearize_fn(src_px[1] as f32 / 65535.0);
        let b = linearize_fn(src_px[2] as f32 / 65535.0);
        let (nr, ng, nb) = fast_gamut::mat3x3_pub(&m, r, g, b);
        dst_px[0] = (encode_fn(nr) * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
        dst_px[1] = (encode_fn(ng) * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
        dst_px[2] = (encode_fn(nb) * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
        dst_px[3] = src_px[3];
    }
}
