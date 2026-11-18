//! Test adapter exposing the accepted CMS worker uniformly to corpus fixtures.

use std::sync::Arc;
use whereat::At;
use zenpixels_convert::cms::{
    CmsPluginError, CmsRequest, CmsSupport, PluggableCms, RowTransform, RowTransformMut,
};
use zenpixels_convert::{ColorProfileSource, ConvertOptions, PixelFormat};

struct SharedWorker(Arc<dyn RowTransform>);
impl RowTransformMut for SharedWorker {
    fn try_transform_row(
        &mut self,
        src: &[u8],
        dst: &mut [u8],
        width: u32,
    ) -> Result<(), At<CmsPluginError>> {
        self.0.try_transform_row(src, dst, width)
    }

    fn prepare(&mut self, width: u32) -> Result<(), At<CmsPluginError>> {
        self.0.prepare(width)
    }
}

pub trait TestCmsExt {
    fn build_source_transform(
        &self,
        src: ColorProfileSource<'_>,
        dst: ColorProfileSource<'_>,
        from: PixelFormat,
        to: PixelFormat,
        options: &ConvertOptions,
    ) -> Option<Result<Box<dyn RowTransformMut>, At<CmsPluginError>>>;
}

impl<T: PluggableCms + ?Sized> TestCmsExt for T {
    fn build_source_transform(
        &self,
        src: ColorProfileSource<'_>,
        dst: ColorProfileSource<'_>,
        from: PixelFormat,
        to: PixelFormat,
        options: &ConvertOptions,
    ) -> Option<Result<Box<dyn RowTransformMut>, At<CmsPluginError>>> {
        match self.build(&CmsRequest::new(
            src,
            dst,
            from.descriptor(),
            to.descriptor(),
            *options,
        )) {
            Ok(CmsSupport::Declined) => None,
            Ok(CmsSupport::Owned(worker)) => Some(Ok(worker)),
            Ok(CmsSupport::Shared(worker)) => Some(Ok(Box::new(SharedWorker(worker)))),
            Err(e) => Some(Err(e)),
        }
    }
}
