# U16 signaling and narrowing: recommended next chunks

**Implementation update (2026-09-27):** narrow-range depth changes refuse after
PR #75. U16 analysis now fuses opacity/chroma/replication checks, and
`RowConverter::prepare` explicitly initializes selected transfer tables. See
[local x86 measurements](../benchmarks/bridge-u16-analysis-2026-09-27.md).
The raw-sample vocabulary and video adapters below remain proposals; the
production narrowing kernel was not changed without cross-platform evidence.

2026-09-27. Review of [PR #74 at `100d8ed`](https://github.com/imazen/zenpixels/blob/100d8ed7307487b58ae7beb8640a18cdc7265cc5/docs/final-0.2-and-0.3-sample-signaling.md).
The API names below are sketches, not added public APIs. The benchmark candidate
is implemented in the shootout, not selected by the production converter.

For the full representation/operation/rounding/support tables, start with the
[U16 contract matrix](u16-contract-matrix.md). The [implementation ledger](implementation-status-0.2-and-0.3.md)
tracks the broader work still outstanding.

## 1. Keep existing U16 meaning; bind raw codes to their encoding

Adopt PR #74's separation of storage width, code depth, padding and signal range.
Existing RGB/gray U16 keeps its 16-bit numerical domain. Source bit depth belongs
to provenance after normalization; it must not override current interpretation.

| Meaning | White / maximum word, full range | U16 interpretation |
|---|---:|---|
| Native 10-bit code | 1023 | Code depth 10, shift 0 |
| Left-aligned 10-bit code | 65472 | Code depth 10, shift 6 |
| Native 12-bit code | 4095 | Code depth 12, shift 0 |
| Left-aligned 12-bit code | 65520 | Code depth 12, shift 4 |
| Normalized image sample | 65535 | Code depth 16, shift 0 |

The following is legal today but declares the wrong meaning:

```rust
use zenpixels::{PixelDescriptor, PixelSlice};
let native_10bit = [1023u16];
let wrong = PixelSlice::new(
    bytemuck::cast_slice(&native_10bit), 1, 1, 2,
    PixelDescriptor::GRAY16_SRGB,
).unwrap();
// The ordinary converter interprets 1023 / 65535, not 1023 / 1023.
// A correct full-range U16→U8 kernel therefore produces 4, not 255.
assert_eq!(wrong.row(0).len(), 2);
```

An ordinary constructor cannot detect this mistake from bytes. Do not deprecate
all U16 input or add a guessing scan. Migrate codec-facing native-code entry
points to a distinct view that requires encoding, and deprecate ambiguous native
sample entry points where they actually exist.

```rust,ignore
// Proposed shape, to prove in the AOM/SVT adapter before publishing names:
let encoding = U16Encoding::new(10, 0)?; // code_bits, low padding bits
let plane = RawU16Plane::new(words, width, height, stride_samples, encoding)?;
// The enclosing signal supplies range, component roles, matrix and chroma layout.
let frame = /* codec-owned Y/Cb/Cr views with resolved signal interpretation */;
```

Construction checks structure, not values: `1 <= bits <= 16`, `bits + shift <= 16`,
extent/stride/alignment. Typed `&[u16]` is native-endian. Byte-backed external
surfaces declare actual endianness at their byte-view boundary. Existing
`ByteOrder` means RGB/BGR channel order and is unsuitable here.

Keep the small encoding primitive reusable in core; prove the plane carrier in
the media prototype. Reuse `SignalRange`. Do not add `U10`/`U12` pixel variants,
an optional `original_bit_depth` conversion override, or a second universal
frame owner. Keep layout, sample encoding and color signal separate in docs.rs,
with cross-linked codec examples. Ordinary image users retain their small tree.

Reads mask padding, writes zero padding, as PR #74 proposes. An explicit
conformance scan can reject nonzero padding when needed; ordinary conversion
does not scan first. Padding masking does not establish nominal-range conformity.

## 2. Make the numeric contract exact before optimizing

For full-range code values, with `max = (1 << bits) - 1` and wide arithmetic:

```text
native code → normalized U16: (q * 65535 + max / 2) / max
native code → U8 directly:    (q *   255 + max / 2) / max
normalized U16 → U8:          (v + 128) / 257
```

These are nearest rounding; the full-range odd denominators have no half ties.
For 10 and 12 bits, exhaustive checks performed for this review establish:

- Expansion to normalized U16 and inverse rescaling recovers every input code.
- Direct U8 rescaling equals expansion followed by exact U16 narrowing.
- These results apply to this depth-only mapping, not arbitrary intervening color
  transforms. Transfer changes should quantize once at the final output.

Left shifting is packing, not full-range normalization. In narrow-range
conversion, scale code anchors by powers of two instead:

```text
8-bit narrow black 16 → 16-bit narrow black 4096 (16 << 8)
Full-range widening gives 4112 (16 * 257), which is the wrong narrow anchor.
10-bit narrow white 940 → 16-bit narrow white 60160 (940 << 6)
```

Narrowing narrow-range samples needs explicit tie rounding and excursion/clipping
policy. Do not silently substitute a full-scale kernel. Chroma has its own offset
and span; alpha has independent full-scale endpoints and no transfer/matrix
operation. This is a correctness gate for the bridge: fix supported cases or
refuse them before conversion, rather than promising approximate correctness.

## 3. Narrowing speed: first compare a simpler exact kernel

The current full-range kernel is exact; keep that contract. Its comment's older
4.5× speed claim describes an M4 Pro comparison against garb's scalar fallback.
The [September x86 report](../benchmarks/u16_narrow_2026-09-24_x86.txt) instead
shows 1080p at about 2.54 ms for the shipped kernel versus 0.82 ms for the exact
u32 arithmetic candidate. Its [metadata](../benchmarks/u16_narrow_2026-09-24_x86.meta)
records a busy host and limited/noisy rounds. Those timings are diagnostic, not
a portable performance guarantee. Garb's faster approximation differs on 127
inputs and must not replace the exact kernel.

I added this candidate to the existing exhaustive-checking shootout:

```rust
fn narrow_full_range(v: u16) -> u8 {
    let t = v.saturating_add(128);
    ((t - (t >> 8)) >> 8) as u8
}
```

Why it is exact: for unsaturated `t`, write `t = 256*h + l`. Both division by
257 and the shift expression yield `h` when `l >= h`, and `h - 1` otherwise.
Saturation changes only inputs that already round to 255. All 65,536 inputs
pass against `(u32::from(v) + 128) / 257`. It uses 16-bit saturation/subtraction/
shifts, avoiding widening multiplies and the current byte-lane shuffles.

The [exploratory x86 run](../benchmarks/u16_narrow_2026-09-27_x86_candidate.txt)
measured the candidate at 0.9 µs versus 7.0 µs shipped for 64² RGB pixels,
and 4.1 ms versus 20.3 ms for 4096². Only one round per size was accepted
on the busy host: these are promising diagnostics, not a validated speedup.
The [run metadata](../benchmarks/u16_narrow_2026-09-27_x86_candidate.meta)
records the command, compiler, source hash and filtering limitation.

Benchmark this on default-target x86 and ARM, including short/tail rows,
unaligned byte slices and large images. Use existing archmage/magetypes dispatch
only if measurement justifies it; start with portable safe Rust, no new core
dependency or public API. Preserve exact parity for every input and all tails.

## 4. Separate three different meanings of “narrowing”

1. **Ordinary quantization:** any normalized U16 value may be rounded to U8.
   One traversal, no losslessness scan.
2. **Exact lossless compaction:** current `uses_low_bits == Some(false)` proves
   byte replication (`0xABAB`), not “values fit in eight low bits.” Its existing
   rewrite may select either identical byte. Keep the legacy field, clarify its
   documentation; don't add a duplicative `bit_depth` field to the report.
3. **Raw-code conversion:** 10/12-bit packing plus code-domain range conversion.
   Choose the kernel once from validated metadata; unpack and scale within the
   same traversal, without allocating a normalized U16 image first.

For exact compaction, U16 RGBA analysis currently invokes separate opacity,
grayscale and bit-replication scans, up to three image reads before rewriting.
The U8 RGBA path already has fused analysis. Extend that approach to U16 with
one requested analysis traversal, then fuse channel selection and depth packing
in the rewrite. Analysis remains explicitly requested. Do not silently scan
ordinary narrowing inputs in hopes of selecting a faster kernel.

Producer facts can avoid scanning only when they establish the current value
domain. “Decoded from 8-bit” is invalid proof after resizing or color conversion.
Keep any checked proof tied to unchanged storage; no persistent boolean that
survives mutable access.

## 5. Expose transfer-table preparation cost

`sdr_u16_to_u8` lazily builds a 65,536-entry, 64 KiB table for each distinct
supported SDR transfer pair. First execution allocates and computes all entries;
warm conversion performs indexed loads. Concurrent initialization can duplicate
construction under `OnceBox`'s race semantics.

Move that initialization into explicit prepared-converter setup. Keep preparation
separate from execution timing. For an explicitly described native 10/12-bit
domain, compare 1/4 KiB direct-code tables against arithmetic; do not expand to a
65,536-entry domain solely to reuse this lookup. Benchmark cache effects and
total setup-plus-execution for small images. Alpha must bypass color transfer.

## Release order and compatibility

1. **Now:** correct U16 documentation and narrow-range behavior; benchmark and
   select exact narrowing improvements independently of a new sample API.
2. **Bridge:** prove one checked encoding primitive with borrowed AOM/SVT views,
   then publish the common spelling in the designated final 0.2 patch if migrated
   video consumers must build unchanged against either core line. Keep codec
   plane pointers and strides; no 40–100 MB normalization copy at handoff.
3. **0.3.1:** retain that common surface, remove already-migrated deprecated APIs,
   and expand video conversions behind precise capability checks. Raw YCbCr
   handoff can remain borrowed; CVVDP's RGB input requires explicit conversion.

If the raw-sample surface ships only in 0.3.1, new video clients using it cannot
claim same-source compatibility with the 0.2 bridge. Existing image clients can.
PR #74 should make that distinction explicit before calling the bridge final.

The owner chose final-output optimization for composition, and explicit
preflight or opt-in fused checks for value-dependent validation. These choices
also apply here; see the [performance review](performance-review-0.2-and-0.3.md).
