//! ICC-to-ICC transforms through `PluggableCms`, with the call shape these tests use.

use zenpixels::{ColorProfileSource, PixelFormat};
use zenpixels_convert::cms::RowTransformMut;
use zenpixels_convert::cms_moxcms::MoxCms;
use zenpixels_convert::policy::ConvertOptions;

pub trait IccTransforms {
    fn build_transform(&self, src: &[u8], dst: &[u8]) -> Result<Box<dyn RowTransformMut>, String> {
        self.build_transform_for_format(src, dst, PixelFormat::Rgb8, PixelFormat::Rgb8)
    }

    fn build_transform_for_format(
        &self,
        src: &[u8],
        dst: &[u8],
        src_format: PixelFormat,
        dst_format: PixelFormat,
    ) -> Result<Box<dyn RowTransformMut>, String>;

    /// A CICP for the profile, from its table entry or its own CICP tag.
    fn identify_profile(&self, icc: &[u8]) -> Option<zenpixels::Cicp> {
        zenpixels::icc::IccIdentification::from_icc(icc)
            .and_then(|id| id.to_cicp())
            .or_else(|| zenpixels::Cicp::from_icc(icc))
    }
}

impl IccTransforms for MoxCms {
    fn build_transform_for_format(
        &self,
        src: &[u8],
        dst: &[u8],
        src_format: PixelFormat,
        dst_format: PixelFormat,
    ) -> Result<Box<dyn RowTransformMut>, String> {
        match self.build_source_transform(
            ColorProfileSource::Icc(src),
            ColorProfileSource::Icc(dst),
            src_format,
            dst_format,
            &ConvertOptions::retain_channels()
                .with_alpha_policy(zenpixels_convert::AlphaPolicy::DiscardIfOpaque)
                .with_depth_policy(zenpixels_convert::DepthPolicy::Round)
                .with_luma(Some(zenpixels_convert::LumaCoefficients::Bt709)),
        ) {
            Some(Ok(transform)) => Ok(transform),
            Some(Err(error)) => Err(error.to_string()),
            None => Err("declined".into()),
        }
    }
}

#[path = "../support/cms.rs"]
mod test_cms;
pub(crate) use test_cms::TestCmsExt;
