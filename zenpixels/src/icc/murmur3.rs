// Adapted from Quickwit Inc.'s MurmurHash3_x86_32 implementation:
// https://github.com/quickwit-oss/murmurhash32/blob/23fff70ddd628a1f47db5616f01e3bb3a9e8b1e7/src/murmurhash3.rs
// Streaming and const evaluation added for normalized ICC fingerprints.
//
// Copyright (c) 2024 by Quickwit Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

const fn mix(mut k: u32) -> u32 {
    k = k.wrapping_mul(0xcc9e_2d51);
    k = k.rotate_left(15);
    k.wrapping_mul(0x1b87_3593)
}

const fn round(h: u32, k: u32) -> u32 {
    (h ^ mix(k))
        .rotate_left(13)
        .wrapping_mul(5)
        .wrapping_add(0xe654_6b64)
}

const fn finalize(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^ (h >> 16)
}

/// MurmurHash3_x86_32 of `data` with an explicit seed.
///
/// This always uses the 32-bit algorithm and little-endian words, on every
/// CPU and byte order. It is a non-cryptographic fingerprint. The length is
/// folded modulo 2^32, as specified by this variant.
#[must_use]
pub const fn murmur3_32(data: &[u8], seed: u32) -> u32 {
    let mut state = Murmur3_32::with_seed(seed);
    state.update(data);
    state.digest()
}

/// Allocation-free streaming MurmurHash3_x86_32.
///
/// Update boundaries do not affect the result. This implements the same fixed
/// variant on 32-bit, 64-bit, little-endian and big-endian targets.
#[derive(Clone)]
pub struct Murmur3_32 {
    state: u32,
    len: u32,
    tail: u32,
    tail_len: u8,
}

impl Murmur3_32 {
    /// Start a byte stream with `seed`.
    #[must_use]
    pub const fn with_seed(seed: u32) -> Self {
        Self {
            state: seed,
            len: 0,
            tail: 0,
            tail_len: 0,
        }
    }

    /// Append bytes without allocating. Empty updates have no effect.
    pub const fn update(&mut self, data: &[u8]) {
        self.len = self.len.wrapping_add(data.len() as u32);
        let mut i = 0;
        while self.tail_len != 0 && i < data.len() {
            self.tail |= (data[i] as u32) << (self.tail_len as u32 * 8);
            self.tail_len += 1;
            i += 1;
            if self.tail_len == 4 {
                self.state = round(self.state, self.tail);
                self.tail = 0;
                self.tail_len = 0;
            }
        }
        while data.len() - i >= 4 {
            let k = u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]);
            self.state = round(self.state, k);
            i += 4;
        }
        while i < data.len() {
            self.tail |= (data[i] as u32) << (self.tail_len as u32 * 8);
            self.tail_len += 1;
            i += 1;
        }
    }

    /// Snapshot the digest without consuming or modifying the byte stream.
    #[must_use]
    pub const fn digest(&self) -> u32 {
        let h = if self.tail_len == 0 {
            self.state
        } else {
            self.state ^ mix(self.tail)
        };
        finalize(h ^ self.len)
    }
}

impl core::fmt::Debug for Murmur3_32 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Murmur3_32")
            .field("bytes_mod_2_32", &self.len)
            .finish_non_exhaustive()
    }
}
