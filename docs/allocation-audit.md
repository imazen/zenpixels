# Allocation-failure audit (issue #151)

Status: covers every allocation reachable from a `Result`-returning public API
in `zenpixels` and `zenpixels-convert` on the default feature set plus the
optional `rgb`/`imgref`/`image`/`bt2446a`/`hdr-experimental`/`cms-*` surfaces.
Each entry names the API, the allocations it can hit, and the observed
guarantee. "Fallible" means a refused reserve returns the crate's typed
allocation error (`BufferError::AllocationFailed`,
`ConvertError::AllocationFailed` / `ConvertError::Buffer(..)`, or
`PlaneError::AllocationFailed`) with a `whereat` location — no panic, no abort
— and no partially-built output escapes.

This is not a claim that every allocation in the crate is fallible. The
"Named exceptions" section is the definitive scope boundary.

## zenpixels — storage and typed constructors

| API | Allocation path | Guarantee |
|---|---|---|
| `PixelBuffer::new` / `new_typed` | `alloc_storage` | **Intentionally panics** on OOM; use `try_new` / `try_new_typed`. |
| `PixelBuffer::try_new` / `try_new_typed` | `alloc_storage` → `PixelStorage::try_zeroed` → `bytemuck::allocation::try_zeroed_vec` | Fallible; dimensions checked before allocating (`InvalidDimensions` on `width*height*bpp`/`stride` overflow). |
| `PixelBuffer::try_new_aligned` | `AlignedSamples::try_zeroed` → `try_zeroed_vec` per alignment-chunk | Fallible. |
| `PixelBuffer::from_vec` | misaligned-input copy → `PixelStorage::try_zeroed` | Fallible; validates geometry and length before effects. |
| `PixelBuffer::from_pixels` | `try_cast_vec` move (no alloc) → `pixels_to_bytes` copy arm → `try_reserve_exact` + `resize` | Fallible; on reserve failure the source `Vec<P>` is returned by the helper and dropped by the API — no half-built buffer. The copy arm is defensive: `try_cast_vec::<P, u8>` rejects only allocations whose address is not representable in the target layout, which no correct allocator returns for `u8`. |
| `PixelBuffer::from_imgvec` | same helpers via `pixels_into_storage` | **Documented panic** on OOM (see `# Panics` docs). |
| `PixelBuffer::try_from_parts` / `try_from_raw_parts` | misalignment copy → `try_zeroed` | Fallible. |
| `crop_copy` | `alloc_storage` | Fallible. |
| `PixelStorage::try_zeroed` / `try_from_rows` / `alloc_storage` | `try_zeroed_vec`, `try_reserve_exact` | Fallible. |
| `Plane`/`PlaneMut` `try_*` constructors, `AlignedSamples::try_zeroed` | `try_zeroed_vec` | Fallible (`PlaneError::AllocationFailed`). |

## zenpixels — extraction, copies, ownership handoff

| API | Allocation path | Guarantee |
|---|---|---|
| `PixelSlice::try_to_tight_buffer` | `PixelStorage::try_from_rows` → `try_reserve_exact` | Fallible; borrowed input untouched; padded rows and short final rows covered by `tests/alloc_failure.rs`. |
| `PixelSlice::contiguous_bytes` | `Vec::with_capacity` row copy | **Intentionally panics** on OOM (documented); use `try_contiguous_bytes`. |
| `PixelSlice::try_contiguous_bytes` | `try_reserve_exact` (only when padding must be stripped; borrows otherwise) | Fallible. |
| `PixelBuffer::copy_to_contiguous_bytes` | `to_vec`/`with_capacity` | **Intentionally panics** on OOM (documented); use `try_copy_to_contiguous_bytes`. |
| `PixelBuffer::try_copy_to_contiguous_bytes` | `try_reserve_exact` | Fallible. |
| `PixelBuffer::into_contiguous_pixels` | `try_cast_vec`, then `Vec::with_capacity`/iterator `collect` copy on alignment mismatch | **Panics on the copy arm** (documented `# Panics`); `Option` result is for descriptor/layout refusal, not OOM. No `try_` sibling today — flag if a consumer needs one. |
| `PixelCow::to_mut` / `into_owned` | `pixel_slice_to_owned` → `PixelBuffer::new` | **Same contract as `Cow`**: panics on OOM (documented). |
| `PixelSlice::into_buffer` | `try_to_tight_buffer` | Fallible. |

## zenpixels-convert — conversion pipeline

| API | Allocation path | Guarantee |
|---|---|---|
| `PixelSliceConvertExt::convert_to{,_with_options}` | output via `PixelBuffer::try_new`; identity-copy and `try_convert_row` rows | Output fallible; `Arc::clone` of the color context is refcount-only. |
| `RowConverter::try_convert_row` / `convert_rows` | `ConvertScratch::try_ensure_capacity` → `try_reserve`; `hdr.prepare` → `try_reserve` | Fallible, including when `prepare` was never called or `width` grew past it (`tests/alloc_failure.rs`). |
| `ConvertScratch::prepare` | `try_ensure_capacity`, `hdr.prepare` (only when `plan.needs_hdr_scratch()`), warmup row vecs via `try_reserve_exact` | Fallible. |
| `PixelSliceConvertExt::convert_into` | no output alloc (caller-provided dst); same row machinery | Fallible for scratch. |
| `PixelSliceConvertExt::oriented`, `PixelBufferConvertExt::into_oriented` | `apply_orientation` → `try_new`; `apply_orientation_in_place` → `try_reserve_exact` `moved` bitvec (non-square transposes only) | Fallible; in-place variant refuses allocation rather than allocating inside the `transform_in_place` closure. |
| `convert_to_with_hdr_config` (bt2446a) | `try_zeroed_vec` output | Fallible. |
| `convert_buffer` (crate-internal) | same-format `try_reserve_exact` + `extend_from_slice`; `try_zeroed_vec` output | Fallible. |
| `convert_buffer_with_anchor` / `adapt_raw_rows_for_encode` | `PixelBuffer::try_new` | Fallible. |
| `PixelSliceConvertExt::quantize_sdr` / `PixelBufferConvertExt::quantized` | `PixelBuffer::try_new` | Fallible. |
| `hdr::quantize_to` (internal) | `convert_buffer_with_anchor` → `try_new` | Fallible. |
| `matte::matte_background`/`matte_white` | fixed `[f32;3]`/`[u8;4]` + `convert_row_buffered` | Fallible. |

## Named exceptions — intentionally or unavoidably infallible

These are allocation sites that remain infallible. They are **outside** the
guarantee above and each is scoped small, bounded, dependency-owned, or behind
an explicitly documented panic contract:

- **Error-path trace boxing**: `whereat::at!` boxes an `AtTrace` (~1.2 KB)
  with `Box::new` — whereat documents `Box::try_new` as unstable, so the
  error *construction* can still abort under extreme exhaustion. A dependency
  limitation, not a path the audit can fix here.
- **Control `Arc::new`/`Box::new`**: color-context attachment
  (`PixelBuffer::with_color_context` ~48 B), `AnyColor::try_new`'s fixed
  `Box<[u8]>` (~48 B), `ConvertError` boxing in `cms.rs`, `Arc::new` in
  `ycbcr.rs`/`interpretation.rs`/`output/plan.rs`/`hdr/mod.rs` — a handful of
  fixed-size control allocs per call, no stable fallible `Arc`/`Box` ctor
  exists. `convert_to` attaches a color `Arc::new` only on the already
  infallible `matte` route.
- **Process caches** (alloc on first use, then shared, unbounded lifetime):
  `cicp_bundle::GROUP_CACHE` (`vec![0; len]` + `Box` inside `OnceBox`, and
  its `expect` on corrupt committed bundles), `fast_gamut` LUT `OnceBox`es,
  `icc_profiles` `OnceBox` tables, `ToneMapStep.prepared` `OnceBox` (warmed
  by `ConvertScratch::prepare`). These are deliberately lazy caches; they are
  listed so no reader assumes a cold first call is allocation-free.
- **Planner/metadata vectors, bounded by descriptor algebra**: `ConvertPlan::build`'s
  `Vec`/`vec!` of steps (≤ ~10), `precision.rs` candidate lists
  (`Vec<Vec<Candidate>>`, ≤ ~30 nodes), `negotiate::pick_target` `collect` of
  caller-provided candidates, `operations` `push`/`insert`, `exact_depth`'s
  `to_vec`/`to_owned` (~50 B), `try_reduce_to_load_bearing_format` ICC
  `into_owned` (≤ profile size, ~4 KB) — caller-count-bounded and not
  image-sized; all return `Option`/`Result` for their real failure modes.
- **`ToneMapper::prepare` → `PreparedMapper` / `StripMapper`** (`Box`/`Arc`
  inside the `OnceBox`): the pluggable-curve contract is infallible by
  signature — a `Result`-returning prepare would be an API change to the
  extension point itself.
- **Option-collapsed allocation causes**: `try_reduce_to_load_bearing_format`
  maps `try_new` failure to `None` (the `Option` contract can't carry a
  cause — documented on the method); `negotiate`/`scan`/ICC readers return
  `Option` for unsupported-input as well as allocation refusal.
- **`hdr::measure` histogram** `vec![0u32; 256]` — fixed 1 KB bins.
- **Pixel hashing** (`hash` feature): validated views use bounded stack
  buffers and borrowed samples — no image-sized allocation on the serialize
  path.
- **Doc/test-only** `vec!`/`with_capacity` in examples and `#[cfg(test)]`
  modules: not production paths.
- **`try_cast_vec` ownership moves** (`from_pixels`, `from_vec`, `into_parts`,
  `Plane::try_into_vec`): when the source `Vec`'s layout is incompatible
  the cast returns the `Vec` back — no allocation on that path at all; the
  *copy* fallback is the arm audited above.

## Memory-budget admission

No configured operation-level memory cap exists in either crate: allocation
fallibility returns `AllocationFailed`; it does not implement an admission
budget. There is nothing to "charge/release" — the earlier #150 estimate
concerned accuracy of the *reported* estimate, which is separate.

## Regression controls

`zenpixels/tests/alloc_failure.rs` and `zenpixels-convert/tests/alloc_failure.rs`
install a threshold `GlobalAlloc` (fails allocations above N bytes) inside
their own test binaries and lock-test every `try_*`/fallible path listed
above: refused reserves, reuse-after-refusal, zero extents, overflow →
`InvalidDimensions`, padded/short-final-row copies, `crop_copy`, unprepared
`try_convert_row` scratch growth, `into_oriented`'s non-square bitvec, and
the square-transpose no-scratch case. The tests intentionally cover the
fallible contract only; the named exceptions above are not gated.
