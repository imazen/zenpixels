//! Portable, endian-independent, stride- and band-invariant XXH3 pixel fingerprinting.
//!
//! Enabled via the `hash` Cargo feature (`xxhash-rust` `xxh3`, `no_std`).
//!
//! Guarantees:
//! - Identical `u64` digests on 32-bit, 64-bit, little-endian, and big-endian targets.
//! - Identical digests regardless of row stride, trailing buffer slop, backing storage
//!   variant (`Vec`, `Aligned64`, `External`, `ExternalMut`), or [`BandedBuf`] band sizing.
//! - Undefined padding channels (`Rgbx`, `Bgrx`), [`AlphaMode::Opaque`] stored alpha lanes,
//!   and unused high/low bits in non-`Full` [`BitPacking`] / [`crate::SampleEncoding`] layouts are
//!   excluded from the hash so padding garbage never perturbs fingerprints.

use core::hash::{Hash, Hasher};

use crate::buffer::{
    BandedBuf, PixelBuf, PixelCow, PixelSource, PixelView, PixelViewMut, PlanarBuf, PlanarView,
    PlanarViewMut, PlaneBuf, PlaneView, PlaneViewMut,
};
use crate::color::{AlphaMode, ImageSpec};
use crate::facts::SampleFacts;
use crate::layout::{BitPacking, ChannelOrder, SampleType};
use crate::sample::{CodePlane, CodePlaneMut};

/// Fields included in a portable pixel fingerprint. Storage padding is always excluded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum HashInclusions {
    /// Active sample values and structural geometry only, without color interpretation.
    SamplesOnly,
    /// Active samples, geometry, layout, bit packing, alpha mode, and current color space.
    Pixels,
    /// Everything in [`Pixels`](Self::Pixels), plus cached [`SampleFacts`] where present.
    #[default]
    All,
}

// Normalize numeric Hash writes from derived metadata impls so enum discriminants and
// slice lengths never acquire the target's `usize` width or native byte order.
struct Portable<'a, H>(&'a mut H);

macro_rules! integer_writes {
    ($($method:ident($ty:ty)),* $(,)?) => {$(
        #[inline]
        fn $method(&mut self, value: $ty) {
            self.write(&value.to_le_bytes());
        }
    )*};
}

impl<H: Hasher> Hasher for Portable<'_, H> {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.0.write(bytes);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0.finish()
    }

    integer_writes! {
        write_u8(u8), write_u16(u16), write_u32(u32), write_u64(u64), write_u128(u128),
        write_i8(i8), write_i16(i16), write_i32(i32), write_i64(i64), write_i128(i128),
    }

    #[inline]
    fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }

    #[inline]
    fn write_isize(&mut self, value: isize) {
        self.write_i64(value as i64);
    }
}

// Feed the underlying XXH3 hasher in fixed 1024-byte chunks so row/band splits
// never alter chunking boundaries even for hashers sensitive to call boundaries.
struct FixedWrites<'a, H> {
    state: &'a mut H,
    bytes: [u8; 1024],
    len: usize,
}

impl<H: Hasher> FixedWrites<'_, H> {
    #[inline]
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
        unreachable!("pixel hash serialization only writes into state")
    }
}

fn digest(f: impl FnOnce(&mut FixedWrites<'_, xxhash_rust::xxh3::Xxh3>)) -> u64 {
    let mut state = xxhash_rust::xxh3::Xxh3::new();
    let mut fixed = FixedWrites {
        state: &mut state,
        bytes: [0; 1024],
        len: 0,
    };
    f(&mut fixed);
    fixed.flush();
    state.digest()
}

fn begin<H: Hasher>(state: &mut H, domain: &[u8], inclusions: HashInclusions) {
    state.write(b"zenpixels/hash/v2\0");
    state.write(domain);
    state.write(&[match inclusions {
        HashInclusions::SamplesOnly => 0,
        HashInclusions::Pixels => 1,
        HashInclusions::All => 2,
    }]);
}

/// Write tightly packed active sample bytes in canonical little-endian order.
fn write_samples_le<H: Hasher>(bytes: &[u8], elem_size: usize, state: &mut H) {
    if cfg!(target_endian = "little") || elem_size <= 1 {
        state.write(bytes);
        return;
    }
    let mut buf = [0u8; 256];
    let chunk_bytes = (buf.len() / elem_size) * elem_size;
    for chunk in bytes.chunks(chunk_bytes) {
        let mut out = 0usize;
        for elem in chunk.chunks_exact(elem_size) {
            for i in 0..elem_size {
                buf[out + i] = elem[elem_size - 1 - i];
            }
            out += elem_size;
        }
        state.write(&buf[..out]);
    }
}

fn read_ne_u32(sample: SampleType, bytes: &[u8]) -> u32 {
    match sample {
        SampleType::U8 => u32::from(bytes[0]),
        SampleType::U16 => u32::from(u16::from_ne_bytes([bytes[0], bytes[1]])),
        SampleType::U32 => u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        _ => 0,
    }
}

fn hash_packed_header<H: Hasher>(
    state: &mut H,
    width: u32,
    height: u32,
    spec: &ImageSpec,
    facts: SampleFacts,
    inclusions: HashInclusions,
) {
    begin(state, b"packed\0", inclusions);
    let mut p = Portable(state);
    width.hash(&mut p);
    height.hash(&mut p);
    spec.layout.sample.hash(&mut p);
    spec.layout.order.hash(&mut p);
    spec.layout.packing.hash(&mut p);
    match inclusions {
        HashInclusions::SamplesOnly => {}
        HashInclusions::Pixels => {
            spec.color.hash(&mut p);
        }
        HashInclusions::All => {
            spec.color.hash(&mut p);
            facts.hash(&mut p);
        }
    }
}

fn hash_packed_row<H: Hasher>(row: &[u8], width: u32, spec: &ImageSpec, state: &mut H) {
    let layout = spec.layout;
    let bpp = layout.bytes_per_pixel();
    let channels = layout.order.channels();
    let ignore_last = matches!(layout.order, ChannelOrder::Rgbx | ChannelOrder::Bgrx)
        || (layout.order.has_alpha_slot() && spec.color.alpha() == AlphaMode::Opaque);
    let active_channels = channels - usize::from(ignore_last);

    if layout.packing == BitPacking::Full {
        let elem_size = layout.sample.byte_size();
        if !ignore_last {
            write_samples_le(row, elem_size, state);
        } else {
            let active_bytes = active_channels * elem_size;
            let mut buf = [0u8; 256];
            let batch_px = (buf.len() / active_bytes).max(1);
            for group in row.chunks(batch_px * bpp) {
                let mut len = 0usize;
                for px in group.chunks_exact(bpp) {
                    buf[len..len + active_bytes].copy_from_slice(&px[..active_bytes]);
                    len += active_bytes;
                }
                write_samples_le(&buf[..len], elem_size, state);
            }
        }
    } else if layout.packing.is_packed_word() {
        let word_bytes = layout.sample.byte_size();
        for px in row.chunks_exact(word_bytes).take(width as usize) {
            let w = read_ne_u32(layout.sample, px);
            for c in 0..active_channels {
                let code = layout.packing.extract_code(layout.sample, w, c);
                state.write(&code.to_le_bytes());
            }
        }
    } else {
        let elem_size = layout.sample.byte_size();
        for px in row.chunks_exact(bpp).take(width as usize) {
            for c in 0..active_channels {
                let w = read_ne_u32(layout.sample, &px[c * elem_size..]);
                let code = layout.packing.extract_code(layout.sample, w, c);
                state.write(&code.to_le_bytes());
            }
        }
    }
}

fn hash_packed_source<S: PixelSource + ?Sized>(src: &S, inclusions: HashInclusions) -> u64 {
    let width = src.source_width();
    let height = src.source_height();
    let spec = src.source_spec();
    let facts = src.source_facts();
    digest(|state| {
        hash_packed_header(state, width, height, spec, facts, inclusions);
        for y in 0..height {
            hash_packed_row(src.source_row(y), width, spec, state);
        }
    })
}

fn hash_plane_into<H: Hasher>(plane: &PlaneView<'_>, inclusions: HashInclusions, state: &mut H) {
    let spec = plane.spec();
    begin(state, b"plane\0", inclusions);
    {
        let mut p = Portable(state);
        plane.width().hash(&mut p);
        plane.height().hash(&mut p);
        spec.sample.hash(&mut p);
        spec.components.hash(&mut p);
        spec.packing.hash(&mut p);
        if !matches!(inclusions, HashInclusions::SamplesOnly) {
            spec.role.hash(&mut p);
            spec.subsampling.hash(&mut p);
        }
    }
    let elem_size = spec.sample.byte_size();
    let comps = spec.components as usize;
    let elem_bytes = spec.bytes_per_element();
    for y in 0..plane.height() {
        let row = plane.row(y);
        if spec.packing == BitPacking::Full {
            write_samples_le(row, elem_size, state);
        } else if spec.packing.is_packed_word() {
            for px in row.chunks_exact(elem_size).take(plane.width() as usize) {
                let w = read_ne_u32(spec.sample, px);
                for c in 0..comps {
                    let code = spec.packing.extract_code(spec.sample, w, c);
                    state.write(&code.to_le_bytes());
                }
            }
        } else {
            for px in row.chunks_exact(elem_bytes).take(plane.width() as usize) {
                for c in 0..comps {
                    let w = read_ne_u32(spec.sample, &px[c * elem_size..]);
                    let code = spec.packing.extract_code(spec.sample, w, c);
                    state.write(&code.to_le_bytes());
                }
            }
        }
    }
}

fn hash_planar_into<H: Hasher>(view: &PlanarView<'_>, inclusions: HashInclusions, state: &mut H) {
    begin(state, b"planar\0", inclusions);
    {
        let mut p = Portable(state);
        view.width().hash(&mut p);
        view.height().hash(&mut p);
        view.planes().len().hash(&mut p);
        if !matches!(inclusions, HashInclusions::SamplesOnly) {
            view.crop_origin().hash(&mut p);
            view.chroma_siting().hash(&mut p);
            view.color().hash(&mut p);
            view.ycbcr_matrix_code().hash(&mut p);
            view.range().hash(&mut p);
            view.alpha().hash(&mut p);
        }
    }
    for plane in view.planes() {
        hash_plane_into(plane, inclusions, state);
    }
}

fn hash_code_plane_into<H: Hasher>(
    plane: CodePlane<'_>,
    inclusions: HashInclusions,
    state: &mut H,
) {
    begin(state, b"code_plane\0", inclusions);
    {
        let mut p = Portable(state);
        plane.width().hash(&mut p);
        plane.height().hash(&mut p);
        plane.encoding().storage().hash(&mut p);
        plane.encoding().code_bits().hash(&mut p);
    }
    let mut row_codes = [0u16; 128];
    for y in 0..plane.height() {
        let mut x = 0usize;
        while x < plane.width() {
            let n = (plane.width() - x).min(row_codes.len());
            let _ = plane.read_codes(x, y, &mut row_codes[..n]);
            for &c in &row_codes[..n] {
                state.write(&c.to_le_bytes());
            }
            x += n;
        }
    }
}

impl PixelView<'_> {
    /// Compute a portable, endian- and stride-independent XXH3 fingerprint of this view.
    #[must_use]
    pub fn hash_with(&self, inclusions: HashInclusions) -> u64 {
        hash_packed_source(self, inclusions)
    }
}

impl PixelViewMut<'_> {
    /// Compute a portable, endian- and stride-independent XXH3 fingerprint of this mutable view.
    #[must_use]
    pub fn hash_with(&self, inclusions: HashInclusions) -> u64 {
        self.as_view().hash_with(inclusions)
    }
}

impl PixelBuf {
    /// Compute a portable, endian- and stride-independent XXH3 fingerprint of this buffer.
    #[must_use]
    pub fn hash_with(&self, inclusions: HashInclusions) -> u64 {
        hash_packed_source(self, inclusions)
    }
}

impl PixelCow<'_> {
    /// Compute a portable, endian- and stride-independent XXH3 fingerprint.
    #[must_use]
    pub fn hash_with(&self, inclusions: HashInclusions) -> u64 {
        self.as_view().hash_with(inclusions)
    }
}

impl BandedBuf {
    /// Compute a portable, endian-, stride-, and band-size-independent XXH3 fingerprint.
    ///
    /// Guaranteed to equal `PixelBuf::hash_with` for the same pixels and metadata.
    #[must_use]
    pub fn hash_with(&self, inclusions: HashInclusions) -> u64 {
        hash_packed_source(self, inclusions)
    }
}

impl PlaneView<'_> {
    /// Compute a portable, endian- and stride-independent XXH3 fingerprint of this plane.
    #[must_use]
    pub fn hash_with(&self, inclusions: HashInclusions) -> u64 {
        digest(|state| hash_plane_into(self, inclusions, state))
    }
}

impl PlaneViewMut<'_> {
    /// Compute a portable, endian- and stride-independent XXH3 fingerprint of this mutable plane.
    #[must_use]
    pub fn hash_with(&self, inclusions: HashInclusions) -> u64 {
        self.as_view().hash_with(inclusions)
    }
}

impl PlaneBuf {
    /// Compute a portable, endian- and stride-independent XXH3 fingerprint of this owned plane.
    #[must_use]
    pub fn hash_with(&self, inclusions: HashInclusions) -> u64 {
        self.as_view().hash_with(inclusions)
    }
}

impl PlanarView<'_> {
    /// Compute a portable, endian- and stride-independent XXH3 fingerprint of this planar view.
    #[must_use]
    pub fn hash_with(&self, inclusions: HashInclusions) -> u64 {
        digest(|state| hash_planar_into(self, inclusions, state))
    }
}

impl PlanarViewMut<'_> {
    /// Compute a portable, endian- and stride-independent XXH3 fingerprint of this mutable planar view.
    #[must_use]
    pub fn hash_with(&self, inclusions: HashInclusions) -> u64 {
        self.as_view().hash_with(inclusions)
    }
}

impl PlanarBuf {
    /// Compute a portable, endian- and stride-independent XXH3 fingerprint of this planar buffer.
    #[must_use]
    pub fn hash_with(&self, inclusions: HashInclusions) -> u64 {
        self.as_view().hash_with(inclusions)
    }
}

impl CodePlane<'_> {
    /// Compute a portable XXH3 fingerprint of the unpacked integer codes in this plane.
    #[must_use]
    pub fn hash_with(&self, inclusions: HashInclusions) -> u64 {
        digest(|state| hash_code_plane_into(*self, inclusions, state))
    }
}

impl CodePlaneMut<'_> {
    /// Compute a portable XXH3 fingerprint of the unpacked integer codes in this mutable plane.
    #[must_use]
    pub fn hash_with(&self, inclusions: HashInclusions) -> u64 {
        self.as_ref().hash_with(inclusions)
    }
}
