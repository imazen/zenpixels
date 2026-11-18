//! Pixel hashing. The inclusion contract is in `docs/hashing.md`.

use core::hash::{Hash, Hasher};

use crate::AlphaMode;
use crate::buffer::{ByteStorage, PixelBuffer, PixelCow, PixelSlice, PixelSliceMut};
#[cfg(feature = "planar")]
use crate::plane::{AnyPlane, PlanarSlice, Plane, PlaneEncoding, PlaneSample, SampleStorage};
#[cfg(feature = "planar")]
use crate::sample::{CodePlane, SampleEncoding};

/// Fields included in a pixel fingerprint. Storage padding is always excluded.
///
/// Geometry, sample type and component/plane boundaries are always included.
/// See the hashing reference for floating-point and metadata byte semantics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum HashInclusions {
    /// Active sample bits and structural geometry, without color interpretation.
    SamplesOnly,
    /// Samples, format/layout, encoding, alpha and all current color signaling.
    Pixels,
    /// Everything in `Pixels`, plus declared exact-depth evidence where present.
    #[default]
    All,
}

// Normalize numeric Hash writes from derived metadata impls. In particular,
// enum discriminants and slice lengths must not acquire the target's usize
// width or native byte order. Sample payloads are serialized separately.
struct Portable<'a, H>(&'a mut H);
macro_rules! integer_writes {
    ($($method:ident($ty:ty)),* $(,)?) => {$ (
        fn $method(&mut self, value: $ty) { self.write(&value.to_le_bytes()); }
    )*};
}
impl<H: Hasher> Hasher for Portable<'_, H> {
    fn write(&mut self, bytes: &[u8]) {
        self.0.write(bytes);
    }
    fn finish(&self) -> u64 {
        self.0.finish()
    }
    integer_writes! { write_u8(u8), write_u16(u16), write_u32(u32), write_u64(u64), write_u128(u128), write_i8(i8), write_i16(i16), write_i32(i32), write_i64(i64), write_i128(i128) }
    fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }
    fn write_isize(&mut self, value: isize) {
        self.write_i64(value as i64);
    }
}

fn begin<H: Hasher>(state: &mut H, domain: &[u8], inclusions: HashInclusions) {
    state.write(b"zenpixels/hash/v1\0");
    state.write(domain);
    state.write(&[match inclusions {
        HashInclusions::SamplesOnly => 0,
        HashInclusions::Pixels => 1,
        HashInclusions::All => 2,
    }]);
}

// Hasher::write need not concatenate separate calls. Serialize the same fixed
// blocks for tight and strided views, including with non-streaming hashers.
struct FixedWrites<'a, H> {
    state: &'a mut H,
    bytes: [u8; 1024],
    len: usize,
}
impl<H: Hasher> FixedWrites<'_, H> {
    fn flush(&mut self) {
        if self.len != 0 {
            self.state.write(&self.bytes[..self.len]);
            self.len = 0;
        }
    }
}
impl<H: Hasher> Hasher for FixedWrites<'_, H> {
    fn write(&mut self, mut bytes: &[u8]) {
        if self.len != 0 {
            let n = (self.bytes.len() - self.len).min(bytes.len());
            self.bytes[self.len..self.len + n].copy_from_slice(&bytes[..n]);
            self.len += n;
            bytes = &bytes[n..];
            if self.len == self.bytes.len() {
                self.flush();
            }
        }
        while bytes.len() >= self.bytes.len() {
            let (head, rest) = bytes.split_at(self.bytes.len());
            self.state.write(head);
            bytes = rest;
        }
        if !bytes.is_empty() {
            self.bytes[..bytes.len()].copy_from_slice(bytes);
            self.len = bytes.len();
        }
    }
    fn finish(&self) -> u64 {
        // This private adapter only serializes our closed, derived metadata
        // Hash impls. A Hash serializer must not ask its state for a digest.
        unreachable!("pixel Hash serialization only writes to its state")
    }
}
fn fixed<H: Hasher>(state: &mut H, f: impl FnOnce(&mut FixedWrites<'_, H>)) {
    let mut state = FixedWrites {
        state,
        bytes: [0; 1024],
        len: 0,
    };
    f(&mut state);
    state.flush();
}

fn digest(f: impl FnOnce(&mut FixedWrites<'_, zenhash::Xxh3>)) -> u64 {
    let mut state = zenhash::Xxh3::new();
    fixed(&mut state, f);
    state.digest()
}

// Only active samples reach this function. Little-endian tightly packed rows
// feed the hasher directly. Big-endian targets use a bounded stack buffer.
fn samples<H: Hasher>(bytes: &[u8], size: usize, state: &mut H) {
    if cfg!(target_endian = "little") || size == 1 {
        state.write(bytes);
    } else {
        let mut buf = [0u8; 256];
        for chunk in bytes.chunks(buf.len()) {
            for (src, dst) in chunk.chunks_exact(size).zip(buf.chunks_exact_mut(size)) {
                for i in 0..size {
                    dst[i] = src[size - 1 - i];
                }
            }
            state.write(&buf[..chunk.len()]);
        }
    }
}

fn packed<H: Hasher>(view: &PixelSlice<'_>, inclusions: HashInclusions, state: &mut H) {
    packed_header(
        (view.width(), view.rows()),
        view.encoding(),
        view.exact_depth(),
        inclusions,
        state,
    );
    packed_samples(view, state);
}

fn packed_header(
    (width, height): (u32, u32),
    encoding: crate::PixelEncoding<'_>,
    exact: Option<crate::sample::ExactDepth>,
    inclusions: HashInclusions,
    state: &mut impl Hasher,
) {
    begin(state, b"packed\0", inclusions);
    let descriptor = encoding.descriptor();
    let channels =
        descriptor.channels() - usize::from(descriptor.alpha == Some(AlphaMode::Undefined));
    let mut p = Portable(state);
    width.hash(&mut p);
    height.hash(&mut p);
    descriptor.channel_type().hash(&mut p);
    channels.hash(&mut p);
    if inclusions != HashInclusions::SamplesOnly {
        descriptor.hash(&mut p);
        encoding.color_context().hash(&mut p);
    }
    if inclusions == HashInclusions::All {
        exact.hash(&mut p);
    }
}

fn packed_samples(view: &PixelSlice<'_>, state: &mut impl Hasher) {
    let descriptor = view.descriptor();
    let size = descriptor.channel_type().byte_size();
    let padding = descriptor.alpha == Some(AlphaMode::Undefined);
    let channels = descriptor.channels() - usize::from(padding);
    if view.width() == 0 || view.rows() == 0 {
        return;
    }
    let row_bytes = view.width() as usize * descriptor.bytes_per_pixel();
    if !padding && view.stride() == row_bytes {
        samples(
            &view.as_strided_bytes()[..row_bytes * view.rows() as usize],
            size,
            state,
        );
    } else if !padding {
        for y in 0..view.rows() {
            samples(view.row(y), size, state);
        }
    } else {
        // RGBX/BGRX's fourth component is undefined, even inside active rows.
        let mut buf = [0u8; 256];
        let active = channels * size;
        let bpp = descriptor.bytes_per_pixel();
        let batch = (buf.len() / active) * bpp;
        for y in 0..view.rows() {
            for group in view.row(y).chunks(batch) {
                let mut len = 0;
                for pixel in group.chunks_exact(bpp) {
                    buf[len..len + active].copy_from_slice(&pixel[..active]);
                    len += active;
                }
                samples(&buf[..len], size, state);
            }
        }
    }
}

#[cfg(feature = "planar")]
fn plane_header(
    state: &mut impl Hasher,
    inclusions: HashInclusions,
    width: usize,
    height: usize,
    ty: crate::ChannelType,
) {
    begin(state, b"plane\0", inclusions);
    let mut p = Portable(state);
    width.hash(&mut p);
    height.hash(&mut p);
    ty.hash(&mut p);
}

#[cfg(feature = "planar")]
fn plane_bytes<H: Hasher>(
    width: usize,
    height: usize,
    stride_samples: usize,
    ty: crate::ChannelType,
    strided_bytes: &[u8],
    inclusions: HashInclusions,
    state: &mut H,
) {
    plane_header(state, inclusions, width, height, ty);
    plane_samples(width, height, stride_samples, ty, strided_bytes, state);
}

#[cfg(feature = "planar")]
fn plane_samples(
    width: usize,
    height: usize,
    stride_samples: usize,
    ty: crate::ChannelType,
    strided_bytes: &[u8],
    state: &mut impl Hasher,
) {
    if width == 0 || height == 0 {
        return;
    }
    let size = ty.byte_size();
    if stride_samples == width {
        samples(strided_bytes, size, state);
    } else {
        let row_bytes = width * size;
        let stride_bytes = stride_samples * size;
        for y in 0..height {
            let off = y * stride_bytes;
            samples(&strided_bytes[off..off + row_bytes], size, state);
        }
    }
}

#[cfg(feature = "planar")]
#[inline]
fn plane<C: SampleStorage, H: Hasher>(view: &Plane<C>, inclusions: HashInclusions, state: &mut H) {
    plane_bytes(
        view.width(),
        view.height(),
        view.stride(),
        <C::Sample as PlaneSample>::CHANNEL_TYPE,
        bytemuck::cast_slice(view.strided_samples()),
        inclusions,
        state,
    );
}

#[cfg(feature = "planar")]
fn any_plane<H: Hasher>(
    view: &AnyPlane<'_>,
    inclusions: HashInclusions,
    encoding: Option<SampleEncoding>,
    state: &mut H,
) {
    plane_header(
        state,
        inclusions,
        view.width(),
        view.height(),
        view.channel_type(),
    );
    any_plane_samples(view, encoding, state);
}

#[cfg(feature = "planar")]
fn any_plane_samples(
    view: &AnyPlane<'_>,
    encoding: Option<SampleEncoding>,
    state: &mut impl Hasher,
) {
    let size = view.channel_type().byte_size();
    if let Some(e) = encoding.filter(|e| usize::from(e.code_bits()) != size * 8) {
        // Reuse the checked native-code reader; padding rules have one owner.
        let codes = CodePlane::new(*view, e);
        if let Ok(codes) = codes {
            code_samples(&codes, size, state);
        } else {
            // 64-bit native carriers are hashable without claiming
            // support in the 8/16/32-bit numerical CodePlane reader.
            match view {
                AnyPlane::U64(p) => masked_plane_bytes(
                    p.width(),
                    p.height(),
                    p.stride(),
                    8,
                    bytemuck::cast_slice(p.strided_samples()),
                    e,
                    state,
                ),
                AnyPlane::I64(p) => masked_plane_bytes(
                    p.width(),
                    p.height(),
                    p.stride(),
                    8,
                    bytemuck::cast_slice(p.strided_samples()),
                    e,
                    state,
                ),
                _ => unreachable!("validated planar integer encoding"),
            }
        }
        return;
    }
    let stride_samples = match view {
        AnyPlane::U8(v) => v.stride(),
        AnyPlane::I8(v) => v.stride(),
        AnyPlane::U16(v) => v.stride(),
        AnyPlane::I16(v) => v.stride(),
        AnyPlane::F16(v) => v.stride(),
        AnyPlane::U32(v) => v.stride(),
        AnyPlane::I32(v) => v.stride(),
        AnyPlane::F32(v) => v.stride(),
        AnyPlane::U64(v) => v.stride(),
        AnyPlane::I64(v) => v.stride(),
        AnyPlane::F64(v) => v.stride(),
    };
    plane_samples(
        view.width(),
        view.height(),
        stride_samples,
        view.channel_type(),
        view.as_strided_bytes(),
        state,
    );
}

#[cfg(feature = "planar")]
fn masked_plane_bytes(
    width: usize,
    height: usize,
    stride_samples: usize,
    size: usize,
    strided_bytes: &[u8],
    encoding: SampleEncoding,
    state: &mut impl Hasher,
) {
    if width == 0 || height == 0 {
        return;
    }
    let mask = u64::MAX >> (64 - encoding.code_bits());
    let mut bytes = [0u8; 1024];
    let row_bytes = width * size;
    let stride_bytes = stride_samples * size;
    for y in 0..height {
        let off = y * stride_bytes;
        let row = &strided_bytes[off..off + row_bytes];
        for block in row.chunks(bytes.len()) {
            for (input, output) in block.chunks_exact(size).zip(bytes.chunks_exact_mut(size)) {
                let word = match size {
                    1 => u64::from(input[0]),
                    2 => u64::from(u16::from_ne_bytes(input.try_into().expect("sample width"))),
                    4 => u64::from(u32::from_ne_bytes(input.try_into().expect("sample width"))),
                    8 => u64::from_ne_bytes(input.try_into().expect("sample width")),
                    _ => unreachable!("native integer sample width"),
                };
                let code = (word >> encoding.bit_shift()) & mask;
                output.copy_from_slice(&code.to_le_bytes()[..size]);
            }
            state.write(&bytes[..block.len()]);
        }
    }
}

#[cfg(feature = "planar")]
fn planar_header(
    view: &PlanarSlice<'_>,
    width: u32,
    height: u32,
    inclusions: HashInclusions,
    state: &mut impl Hasher,
) {
    begin(state, b"planar\0", inclusions);
    {
        let mut p = Portable(state);
        width.hash(&mut p);
        height.hash(&mut p);
        view.layout().planes().len().hash(&mut p);
        if inclusions != HashInclusions::SamplesOnly {
            view.layout().hash(&mut p);
            view.color_context().hash(&mut p);
            view.alpha_mode().hash(&mut p);
        }
    }
}

#[cfg(feature = "planar")]
fn planar(view: &PlanarSlice<'_>, inclusions: HashInclusions, state: &mut impl Hasher) {
    planar_header(view, view.width(), view.height(), inclusions, state);
    for (view, spec) in view.planes().zip(view.layout().planes()) {
        let encoding = match spec.encoding() {
            PlaneEncoding::Code(e) => Some(e),
            PlaneEncoding::Float(_) | PlaneEncoding::Float64(_) | PlaneEncoding::Float16(_) => None,
        };
        any_plane(&view, inclusions, encoding, state);
    }
}

impl<P> Hash for PixelSlice<'_, P> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        let erased = self.reborrow_erased();
        fixed(state, |h| packed(&erased, HashInclusions::default(), h));
    }
}
impl<P> PixelSlice<'_, P> {
    /// Fingerprint active pixels with the selected inclusion policy.
    #[inline]
    #[must_use]
    pub fn hash_with(&self, inclusions: HashInclusions) -> u64 {
        let erased = self.reborrow_erased();
        digest(|h| packed(&erased, inclusions, h))
    }
}

macro_rules! packed_owner {
    ($ty:ty, $view:ident $(, $storage:ident : $bound:path)?) => {
        impl<P $(, $storage: $bound)?> Hash for $ty {
            #[inline]
            fn hash<H: Hasher>(&self, state: &mut H) { self.$view().erase().hash(state); }
        }
        impl<P $(, $storage: $bound)?> $ty {
            /// Fingerprint the current active view, excluding storage padding.
            #[inline]
            #[must_use]
            pub fn hash_with(&self, inclusions: HashInclusions) -> u64 { self.$view().erase().hash_with(inclusions) }
        }
    };
}
packed_owner!(PixelBuffer<P, S>, as_slice, S: ByteStorage);
packed_owner!(PixelCow<'_, P>, as_slice);
packed_owner!(PixelSliceMut<'_, P>, as_pixel_slice);

#[cfg(feature = "planar")]
impl<C: SampleStorage> Hash for Plane<C> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        fixed(state, |h| plane(self, HashInclusions::default(), h));
    }
}
#[cfg(feature = "planar")]
impl<C: SampleStorage> Plane<C> {
    /// Fingerprint active samples and geometry, excluding stride and storage.
    #[must_use]
    pub fn hash_with(&self, inclusions: HashInclusions) -> u64 {
        digest(|h| plane(self, inclusions, h))
    }
}
#[cfg(feature = "planar")]
impl Hash for AnyPlane<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        fixed(state, |h| {
            any_plane(self, HashInclusions::default(), None, h)
        });
    }
}
#[cfg(feature = "planar")]
impl AnyPlane<'_> {
    /// Fingerprint active samples identically to the corresponding typed plane.
    #[must_use]
    pub fn hash_with(&self, inclusions: HashInclusions) -> u64 {
        digest(|h| any_plane(self, inclusions, None, h))
    }
}
#[cfg(feature = "planar")]
impl Hash for PlanarSlice<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        fixed(state, |h| planar(self, HashInclusions::default(), h));
    }
}
#[cfg(feature = "planar")]
impl PlanarSlice<'_> {
    /// Fingerprint active planes and the selected picture metadata.
    #[must_use]
    pub fn hash_with(&self, inclusions: HashInclusions) -> u64 {
        digest(|h| planar(self, inclusions, h))
    }
}

#[cfg(feature = "planar")]
impl Hash for CodePlane<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        fixed(state, |h| codes(self, HashInclusions::default(), h));
    }
}
#[cfg(feature = "planar")]
impl CodePlane<'_> {
    /// Fingerprint component codes, ignoring unused bits and other components.
    #[must_use]
    pub fn hash_with(&self, inclusions: HashInclusions) -> u64 {
        digest(|h| codes(self, inclusions, h))
    }
}
#[cfg(feature = "planar")]
fn codes(view: &CodePlane<'_>, inclusions: HashInclusions, state: &mut impl Hasher) {
    begin(state, b"codes\0", inclusions);
    {
        let mut p = Portable(state);
        view.width().hash(&mut p);
        view.height().hash(&mut p);
        if inclusions != HashInclusions::SamplesOnly {
            view.encoding().hash(&mut p);
        }
    }
    let size = if matches!(
        view.encoding().storage(),
        crate::ChannelType::U32 | crate::ChannelType::I32
    ) {
        4
    } else {
        2
    };
    code_samples(view, size, state);
}
#[cfg(feature = "planar")]
fn code_samples(view: &CodePlane<'_>, size: usize, state: &mut impl Hasher) {
    if view.width() == 0 {
        return;
    }
    let signed = view.encoding().is_signed();
    let mask = u32::MAX >> (32 - u32::from(view.encoding().code_bits()));
    if size == 4 {
        let mut codes32 = [0u32; 128];
        let mut signed32 = [0i32; 128];
        for y in 0..view.height() {
            for x in (0..view.width()).step_by(codes32.len()) {
                let n = (view.width() - x).min(codes32.len());
                if signed {
                    view.read_codes_i32(x, y, &mut signed32[..n])
                        .expect("checked active code plane extent");
                    for (dst, &src) in codes32[..n].iter_mut().zip(&signed32[..n]) {
                        *dst = (src as u32) & mask;
                    }
                } else {
                    view.read_codes_u32(x, y, &mut codes32[..n])
                        .expect("checked active code plane extent");
                }
                samples(bytemuck::cast_slice(&codes32[..n]), 4, state);
            }
        }
        return;
    }
    let mut codes = [0u16; 128];
    let mut signed32 = [0i32; 128];
    let mut bytes = [0u8; 128];
    for y in 0..view.height() {
        for x in (0..view.width()).step_by(codes.len()) {
            let n = (view.width() - x).min(codes.len());
            if signed {
                view.read_codes_i32(x, y, &mut signed32[..n])
                    .expect("checked active code plane extent");
                for (dst, &src) in codes[..n].iter_mut().zip(&signed32[..n]) {
                    *dst = ((src as u32) & mask) as u16;
                }
            } else {
                view.read_codes(x, y, &mut codes[..n])
                    .expect("checked active code plane extent");
            }
            match size {
                // Reuse the wide-sample serializer. Little-endian hosts can
                // hash the complete code block directly, without per-code copies.
                2 => samples(bytemuck::cast_slice(&codes[..n]), 2, state),
                1 => {
                    for (byte, code) in bytes[..n].iter_mut().zip(&codes[..n]) {
                        *byte = *code as u8;
                    }
                    state.write(&bytes[..n]);
                }
                _ => unreachable!("CodePlane supports 8/16/32-bit containers"),
            }
        }
    }
}

impl crate::banded::BandedPixelBuffer {
    fn hash_into(&self, inclusions: HashInclusions, state: &mut impl Hasher) {
        let geometry = self.layout();
        packed_header(
            (geometry.width(), geometry.height()),
            self.encoding(),
            self.exact_depth(),
            inclusions,
            state,
        );
        for view in self.bands() {
            packed_samples(&view, state);
        }
    }

    /// Hash the same byte stream as a contiguous packed view of the whole image.
    pub fn hash_with(&self, inclusions: HashInclusions) -> u64 {
        digest(|h| self.hash_into(inclusions, h))
    }
}

impl Hash for crate::banded::BandedPixelBuffer {
    fn hash<H: Hasher>(&self, state: &mut H) {
        fixed(state, |h| self.hash_into(HashInclusions::default(), h));
    }
}

#[cfg(feature = "planar")]
impl<T: PlaneSample> crate::plane::BandedPlane<T> {
    fn hash_into(&self, inclusions: HashInclusions, state: &mut impl Hasher) {
        let geometry = self.layout();
        plane_header(
            state,
            inclusions,
            geometry.width() as usize,
            geometry.height() as usize,
            T::CHANNEL_TYPE,
        );
        for band in self.bands() {
            plane_samples(
                band.width(),
                band.height(),
                band.stride(),
                T::CHANNEL_TYPE,
                bytemuck::cast_slice(band.strided_samples()),
                state,
            );
        }
    }

    /// Hash the same byte stream as a contiguous plane of the whole image.
    pub fn hash_with(&self, inclusions: HashInclusions) -> u64 {
        digest(|h| self.hash_into(inclusions, h))
    }
}

#[cfg(feature = "planar")]
impl<T: PlaneSample> Hash for crate::plane::BandedPlane<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        fixed(state, |h| self.hash_into(HashInclusions::default(), h));
    }
}

#[cfg(feature = "planar")]
impl crate::plane::BandedPlanarBuffer {
    fn hash_into(&self, inclusions: HashInclusions, state: &mut impl Hasher) {
        let geometry = self.geometry();
        let (width, height) = (geometry.width(), geometry.height());
        let first = self.bands().next().expect("nonempty geometry");
        planar_header(&first, width, height, inclusions, state);
        for (index, spec) in first.layout().planes().iter().enumerate() {
            let (w, h) = first
                .layout()
                .plane_size(index, width, height)
                .expect("valid plane index");
            let samples = w as usize * spec.role().components();
            plane_header(
                state,
                inclusions,
                samples,
                h as usize,
                spec.encoding().channel_type(),
            );
            let encoding = match spec.encoding() {
                PlaneEncoding::Code(e) => Some(e),
                _ => None,
            };
            for band in self.bands() {
                any_plane_samples(
                    &band.plane(index).expect("validated band layout"),
                    encoding,
                    state,
                );
            }
        }
    }

    /// Hash samples and interpretation, with the contiguous planar byte stream.
    pub fn hash_with(&self, inclusions: HashInclusions) -> u64 {
        digest(|h| self.hash_into(inclusions, h))
    }
}

#[cfg(feature = "planar")]
impl Hash for crate::plane::BandedPlanarBuffer {
    fn hash<H: Hasher>(&self, state: &mut H) {
        fixed(state, |h| self.hash_into(HashInclusions::default(), h));
    }
}
