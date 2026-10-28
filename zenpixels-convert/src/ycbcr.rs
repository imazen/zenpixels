//! Borrowed native component conversion, independent of codec ownership and I/O.
//!
//! The first qualified route is centered BT.709-matrix/sRGB, 8/10/12-bit,
//! full or limited range. It preserves excursions in RGB F32 and leaves the
//! transfer unchanged. It is neither a lossless transform nor HDR rendering.
use alloc::sync::Arc;
use core::fmt;
use whereat::At;
use zenpixels::{
    Cicp, ColorContext, PixelDescriptor, PixelSliceMut, TransferFunction,
    sample::{ChromaSiting, ChromaSubsampling, CodePlane, SampleEncoding},
};

/// A prepared native component route cannot honor its interpretation or output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum YcbcrError {
    /// Primaries, transfer or matrix is outside the implemented route.
    UnsupportedColor { actual: Cicp },
    /// Bit depth or per-plane sample encodings disagree or are unsupported.
    UnsupportedEncoding {
        plane: usize,
        actual: SampleEncoding,
    },
    /// Chroma sampling/location has no qualified kernel. Unknown is not Center.
    UnsupportedChroma {
        subsampling: ChromaSubsampling,
        siting: ChromaSiting,
    },
    /// Source planes do not describe a nonempty full coded image.
    SourceGeometry {
        plane: usize,
        actual: (usize, usize),
        expected: Option<(usize, usize)>,
    },
    /// Requested output rows/columns extend beyond the complete source image.
    OutputGeometry {
        origin: (usize, usize),
        extent: (usize, usize),
        source: (usize, usize),
    },
    /// Destination does not have the exact promised output descriptor.
    OutputDescriptor { actual: PixelDescriptor },
}
impl fmt::Display for YcbcrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedColor { actual } => write!(
                f,
                "unsupported YCbCr color {actual:?}; requires BT.709 matrix and sRGB"
            ),
            Self::UnsupportedEncoding { plane, actual } => write!(
                f,
                "unsupported YCbCr plane {plane} encoding {actual:?}; requires matching 8/10/12-bit codes"
            ),
            Self::UnsupportedChroma {
                subsampling,
                siting,
            } => write!(f, "unsupported YCbCr chroma {subsampling:?} at {siting:?}"),
            Self::SourceGeometry {
                plane,
                actual,
                expected,
            } => write!(
                f,
                "YCbCr plane {plane} extent {actual:?}, expected {expected:?} (None requires nonempty luma)"
            ),
            Self::OutputGeometry {
                origin,
                extent,
                source,
            } => write!(
                f,
                "YCbCr output {extent:?} at {origin:?} exceeds source {source:?}"
            ),
            Self::OutputDescriptor { actual } => {
                write!(f, "YCbCr output {actual:?} must be full-range sRGB RGB F32")
            }
        }
    }
}
impl core::error::Error for YcbcrError {}

/// Retain validated, borrowed Y/Cb/Cr planes and their selected RGB F32 recipe.
///
/// The plan keeps the original coded-coordinate system: crop origins and row
/// partitions cannot reset chroma phase. Input storage stays borrowed, never
/// copied. Construction allocates one shared output color context; execution
/// allocates no payload or scratch and does not publish to a downstream sink.
/// Callers supply destination storage and can check cancellation between batches.
///
/// Pixel equality across crop/batch partitions is exact. Numerical agreement
/// with another implementation is a separate qualification, not bit preservation
/// through an irreversible chroma reconstruction or matrix conversion.
pub struct YcbcrToRgbF32<'a> {
    planes: [CodePlane<'a>; 3],
    sub: ChromaSubsampling,
    y_offset: f32,
    y_scale: f32,
    uv_center: f32,
    uv_scale: f32,
    vr: f32,
    ug: f32,
    vg: f32,
    ub: f32,
    context: Arc<ColorContext>,
}
impl<'a> YcbcrToRgbF32<'a> {
    /// Prepare the exact held source in Y, Cb, Cr order.
    ///
    /// Accepts BT.709 primaries, sRGB transfer and BT.709 matrix, at 8/10/12 bits.
    /// Both declared ranges and valid word placements are supported. Chroma
    /// extents round up at odd edges. Subsampled sources require explicit Center;
    /// 4:4:4 does not use chroma siting. Resolve unknown source facts before this
    /// call; this constructor never invents a policy or inspects format metadata.
    pub fn new(
        planes: [CodePlane<'a>; 3],
        sub: ChromaSubsampling,
        siting: ChromaSiting,
        color: Cicp,
    ) -> Result<Self, At<YcbcrError>> {
        if color.color_primaries != 1
            || color.transfer_characteristics != 13
            || color.matrix_coefficients != 1
        {
            return Err(whereat::at!(YcbcrError::UnsupportedColor { actual: color }));
        }
        let encoding = planes[0].encoding();
        let bits = encoding.code_bits();
        for (plane, p) in planes.iter().enumerate() {
            if !matches!(bits, 8 | 10 | 12) || p.encoding() != encoding {
                return Err(whereat::at!(YcbcrError::UnsupportedEncoding {
                    plane,
                    actual: p.encoding()
                }));
            }
        }
        let (w, h) = (planes[0].width(), planes[0].height());
        let (cw, ch) = match sub {
            ChromaSubsampling::Yuv444 => (w, h),
            ChromaSubsampling::Yuv422 => (w.div_ceil(2), h),
            ChromaSubsampling::Yuv420 => (w.div_ceil(2), h.div_ceil(2)),
            _ => {
                return Err(whereat::at!(YcbcrError::UnsupportedChroma {
                    subsampling: sub,
                    siting
                }));
            }
        };
        if sub != ChromaSubsampling::Yuv444 && siting != ChromaSiting::Center {
            return Err(whereat::at!(YcbcrError::UnsupportedChroma {
                subsampling: sub,
                siting
            }));
        }
        if w == 0 || h == 0 {
            return Err(whereat::at!(YcbcrError::SourceGeometry {
                plane: 0,
                actual: (w, h),
                expected: None
            }));
        }
        for (plane, p) in planes.iter().enumerate().skip(1) {
            let actual = (p.width(), p.height());
            if actual != (cw, ch) {
                return Err(whereat::at!(YcbcrError::SourceGeometry {
                    plane,
                    actual,
                    expected: Some((cw, ch))
                }));
            }
        }
        // Recipe from imazen/zenavif feadc4c94b19 src/yuv_convert.rs, qualified
        // without integer saturation in zencodec c7e76b23. Keep its arithmetic.
        let shift = u32::from(bits) - 8;
        let max = ((1u32 << bits) - 1) as f32;
        let (kr, kb) = (0.2126f32, 0.0722f32);
        let kg = 1.0 - kr - kb;
        Ok(Self {
            planes,
            sub,
            y_offset: if color.full_range {
                0.0
            } else {
                (16u32 << shift) as f32
            },
            y_scale: 1.0
                / if color.full_range {
                    max
                } else {
                    (219u32 << shift) as f32
                },
            uv_center: (128u32 << shift) as f32,
            uv_scale: 1.0
                / if color.full_range {
                    max
                } else {
                    (224u32 << shift) as f32
                },
            vr: 2.0 * (1.0 - kr),
            ug: -2.0 * kb * (1.0 - kb) / kg,
            vg: -2.0 * kr * (1.0 - kr) / kg,
            ub: 2.0 * (1.0 - kb),
            context: Arc::new(ColorContext::from_cicp(Cicp::SRGB)),
        })
    }

    /// Actual output meaning: full-range RGB F32 with unchanged sRGB transfer.
    pub const fn output_descriptor(&self) -> PixelDescriptor {
        PixelDescriptor::RGBF32_LINEAR.with_transfer(TransferFunction::Srgb)
    }

    /// Convert a strided region starting at coded coordinates (x, y).
    ///
    /// All extent/descriptor checks precede writes. Output row padding stays
    /// untouched. Returned color context matches the output; any previous context
    /// on the supplied destination view is replaced. No transfer/gamut conversion,
    /// integer quantization, nominal-range clipping or source-role inference occurs.
    pub fn convert_rows<'b>(
        &self,
        x: usize,
        y: usize,
        mut destination: PixelSliceMut<'b>,
    ) -> Result<PixelSliceMut<'b>, At<YcbcrError>> {
        if destination.descriptor() != self.output_descriptor() {
            return Err(whereat::at!(YcbcrError::OutputDescriptor {
                actual: destination.descriptor()
            }));
        }
        let width = destination.width() as usize;
        let height = destination.rows() as usize;
        if x.checked_add(width)
            .is_none_or(|v| v > self.planes[0].width())
            || y.checked_add(height)
                .is_none_or(|v| v > self.planes[0].height())
        {
            return Err(whereat::at!(YcbcrError::OutputGeometry {
                origin: (x, y),
                extent: (width, height),
                source: (self.planes[0].width(), self.planes[0].height())
            }));
        }
        for row in 0..height {
            for (i, dst) in destination
                .row_mut(row as u32)
                .as_chunks_mut::<12>()
                .0
                .iter_mut()
                .enumerate()
            {
                let luma = (f32::from(self.planes[0].code(x + i, y + row).unwrap())
                    - self.y_offset)
                    * self.y_scale;
                let u = (self.chroma(1, x + i, y + row) - self.uv_center) * self.uv_scale;
                let v = (self.chroma(2, x + i, y + row) - self.uv_center) * self.uv_scale;
                let rgb = [
                    fma(v, self.vr, luma),
                    fma(v, self.vg, fma(u, self.ug, luma)),
                    fma(u, self.ub, luma),
                ];
                for (bytes, value) in dst.as_chunks_mut::<4>().0.iter_mut().zip(rgb) {
                    bytes.copy_from_slice(&value.to_ne_bytes());
                }
            }
        }
        Ok(destination.with_color_context(self.context.clone()))
    }

    fn chroma(&self, index: usize, x: usize, y: usize) -> f32 {
        let p = self.planes[index];
        let (x0, x1, wx) = axis(x, p.width(), self.sub != ChromaSubsampling::Yuv444);
        let (y0, y1, wy) = axis(y, p.height(), self.sub == ChromaSubsampling::Yuv420);
        let sample = |x, y| u32::from(p.code(x, y).unwrap());
        let top = (4 - wx) * sample(x0, y0) + wx * sample(x1, y0);
        let bottom = (4 - wx) * sample(x0, y1) + wx * sample(x1, y1);
        ((4 - wy) * top + wy * bottom) as f32 * (1.0 / 16.0)
    }
}
fn axis(position: usize, length: usize, subsampled: bool) -> (usize, usize, u32) {
    if !subsampled {
        return (position, position, 0);
    }
    let cell = position / 2;
    if position.is_multiple_of(2) {
        (cell.saturating_sub(1), cell, 3)
    } else {
        (cell, (cell + 1).min(length - 1), 1)
    }
}
fn fma(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::fmaf(a, b, c)
    }
}
