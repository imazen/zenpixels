//! Kernel-level transfer tests keep their original thresholds and byte goldens.
//! HLG's scene transfer primitives remain testable internally even though the
//! public light-domain API refuses HLG until its display OOTF is implemented.
use crate::{ConvertOptions, ConvertPlan, PixelDescriptor, TransferFunction};
use whereat::At;

#[derive(Debug)]
struct KernelConverter(crate::RowConverter);
impl KernelConverter {
    fn new(from: PixelDescriptor, to: PixelDescriptor) -> Result<Self, At<crate::ConvertError>> {
        Self::new_explicit(
            from,
            to,
            &ConvertOptions::permissive().with_alpha_policy(crate::AlphaPolicy::DiscardUnchecked),
        )
    }
    fn new_explicit(
        from: PixelDescriptor,
        to: PixelDescriptor,
        options: &ConvertOptions,
    ) -> Result<Self, At<crate::ConvertError>> {
        if crate::convert::is_hdr_to_sdr(from.transfer(), to.transfer())
            || from.transfer() != to.transfer()
                && (from.transfer() == TransferFunction::Hlg
                    || to.transfer() == TransferFunction::Hlg)
        {
            return ConvertPlan::plan_native(from, to)
                .map(crate::RowConverter::from_plan)
                .map(Self);
        }
        let options = options.with_sdr_white(zenpixels::hdr::DiffuseWhite::new(10_000.0));
        crate::RowConverter::new_explicit(from, to, &options).map(Self)
    }
    #[cfg(feature = "hdr-experimental")]
    fn from_plan(plan: ConvertPlan) -> Self {
        Self(crate::RowConverter::from_plan(plan))
    }
}
impl core::ops::Deref for KernelConverter {
    type Target = crate::RowConverter;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl core::ops::DerefMut for KernelConverter {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

mod cicp_transfer_correctness;
mod roundtrip;
mod row_partition;
mod transfer_alpha;
mod transfer_row_kernels;
mod ulp_exhaustive;

mod gamut_across_layouts;
mod planner_silent_passthrough;
