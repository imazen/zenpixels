//! Pixel hashing. The inclusion contract is in `docs/hashing.md`.

use core::hash::{Hash, Hasher};

use crate::AlphaMode;
use crate::buffer::{ByteStorage, PixelBuffer, PixelCow, PixelSlice, PixelSliceMut};
use crate::plane::{AnyPlane, PlanarSlice, Plane, PlaneEncoding, SampleStorage};
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

fn digest(f: impl FnOnce(&mut zenhash::RapidhashV3)) -> u64 {
    let mut state = zenhash::RapidhashV3::new();
    f(&mut state);
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

fn packed<P, H: Hasher>(view: &PixelSlice<'_, P>, inclusions: HashInclusions, state: &mut H) {
    begin(state, b"packed\0", inclusions);
    let descriptor = view.descriptor();
    let size = descriptor.channel_type().byte_size();
    let padding = matches!(descriptor.alpha, Some(AlphaMode::Undefined));
    let channels = descriptor.channels() - usize::from(padding);
    {
        let mut p = Portable(state);
        view.width().hash(&mut p);
        view.rows().hash(&mut p);
        descriptor.channel_type().hash(&mut p);
        channels.hash(&mut p);
        if inclusions != HashInclusions::SamplesOnly {
            descriptor.hash(&mut p);
            view.color_context().hash(&mut p);
        }
        if inclusions == HashInclusions::All {
            view.exact_depth().hash(&mut p);
        }
    }
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

fn plane<C: SampleStorage, H: Hasher>(
    view: &Plane<C>,
    inclusions: HashInclusions,
    encoding: Option<SampleEncoding>,
    state: &mut H,
) {
    begin(state, b"plane\0", inclusions);
    {
        let mut p = Portable(state);
        view.width().hash(&mut p);
        view.height().hash(&mut p);
        <C::Sample as crate::plane::PlaneSample>::CHANNEL_TYPE.hash(&mut p);
    }
    if view.width() == 0 || view.height() == 0 {
        return;
    }
    let size = core::mem::size_of::<C::Sample>();
    let partial = encoding.filter(|e| usize::from(e.code_bits()) != size * 8);
    if let Some(e) = partial {
        let mut buf = [0u8; 256];
        let mask = (1u32 << e.code_bits()) - 1;
        for row in view.rows() {
            let bytes: &[u8] = bytemuck::cast_slice(row);
            for group in bytes.chunks(buf.len()) {
                for (src, dst) in group.chunks_exact(size).zip(buf.chunks_exact_mut(size)) {
                    let value = if size == 1 {
                        u32::from(src[0])
                    } else {
                        u32::from(u16::from_ne_bytes([src[0], src[1]]))
                    };
                    let code = (value >> e.bit_shift()) & mask;
                    dst.copy_from_slice(&code.to_le_bytes()[..size]);
                }
                state.write(&buf[..group.len()]);
            }
        }
    } else if view.stride() == view.width() {
        samples(bytemuck::cast_slice(view.strided_samples()), size, state);
    } else {
        for row in view.rows() {
            samples(bytemuck::cast_slice(row), size, state);
        }
    }
}

fn any_plane<H: Hasher>(
    view: &AnyPlane<'_>,
    inclusions: HashInclusions,
    encoding: Option<SampleEncoding>,
    state: &mut H,
) {
    match view {
        AnyPlane::U8(v) => plane(v, inclusions, encoding, state),
        AnyPlane::U16(v) => plane(v, inclusions, encoding, state),
        AnyPlane::F32(v) => plane(v, inclusions, None, state),
    }
}

fn planar(view: &PlanarSlice<'_>, inclusions: HashInclusions, state: &mut impl Hasher) {
    begin(state, b"planar\0", inclusions);
    {
        let mut p = Portable(state);
        view.width().hash(&mut p);
        view.height().hash(&mut p);
        view.planes().len().hash(&mut p);
        if inclusions != HashInclusions::SamplesOnly {
            view.layout().hash(&mut p);
            view.color_context().hash(&mut p);
            view.alpha_mode().hash(&mut p);
        }
    }
    for (view, spec) in view.planes().iter().zip(view.layout().planes()) {
        let encoding = match spec.encoding() {
            PlaneEncoding::Code(e) => Some(e),
            PlaneEncoding::Float(_) => None,
        };
        any_plane(view, inclusions, encoding, state);
    }
}

impl<P> Hash for PixelSlice<'_, P> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        packed(self, HashInclusions::default(), state);
    }
}
impl<P> PixelSlice<'_, P> {
    /// Fingerprint active pixels with the selected inclusion policy.
    #[must_use]
    pub fn hash_with(&self, inclusions: HashInclusions) -> u64 {
        digest(|h| packed(self, inclusions, h))
    }
}

macro_rules! packed_owner {
    ($ty:ty, $view:ident $(, $storage:ident : $bound:path)?) => {
        impl<P $(, $storage: $bound)?> Hash for $ty {
            fn hash<H: Hasher>(&self, state: &mut H) { self.$view().hash(state); }
        }
        impl<P $(, $storage: $bound)?> $ty {
            /// Fingerprint the current active view, excluding storage padding.
            #[must_use]
            pub fn hash_with(&self, inclusions: HashInclusions) -> u64 { self.$view().hash_with(inclusions) }
        }
    };
}
packed_owner!(PixelBuffer<P, S>, as_slice, S: ByteStorage);
packed_owner!(PixelCow<'_, P>, as_slice);
packed_owner!(PixelSliceMut<'_, P>, as_pixel_slice);

impl<C: SampleStorage> Hash for Plane<C> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        plane(self, HashInclusions::default(), None, state);
    }
}
impl<C: SampleStorage> Plane<C> {
    /// Fingerprint active samples and geometry, excluding stride and storage.
    #[must_use]
    pub fn hash_with(&self, inclusions: HashInclusions) -> u64 {
        digest(|h| plane(self, inclusions, None, h))
    }
}
impl Hash for AnyPlane<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        any_plane(self, HashInclusions::default(), None, state);
    }
}
impl AnyPlane<'_> {
    /// Fingerprint active samples identically to the corresponding typed plane.
    #[must_use]
    pub fn hash_with(&self, inclusions: HashInclusions) -> u64 {
        digest(|h| any_plane(self, inclusions, None, h))
    }
}
impl Hash for PlanarSlice<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        planar(self, HashInclusions::default(), state);
    }
}
impl PlanarSlice<'_> {
    /// Fingerprint active planes and the selected picture metadata.
    #[must_use]
    pub fn hash_with(&self, inclusions: HashInclusions) -> u64 {
        digest(|h| planar(self, inclusions, h))
    }
}

impl Hash for CodePlane<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        codes(self, HashInclusions::default(), state);
    }
}
impl CodePlane<'_> {
    /// Fingerprint component codes, ignoring unused bits and other components.
    #[must_use]
    pub fn hash_with(&self, inclusions: HashInclusions) -> u64 {
        digest(|h| codes(self, inclusions, h))
    }
}
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
    if view.width() == 0 {
        return;
    }
    let mut codes = [0u16; 128];
    let mut bytes = [0u8; 256];
    for y in 0..view.height() {
        for x in (0..view.width()).step_by(codes.len()) {
            let n = (view.width() - x).min(codes.len());
            view.read_codes(x, y, &mut codes[..n])
                .expect("checked active code plane extent");
            for (code, bytes) in codes[..n].iter().zip(bytes.chunks_exact_mut(2)) {
                bytes.copy_from_slice(&code.to_le_bytes());
            }
            state.write(&bytes[..n * 2]);
        }
    }
}
