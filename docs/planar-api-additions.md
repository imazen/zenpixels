# Borrowed planar picture additions

`zenpixels::plane::Planes` validates borrowed views against `PlanarLayout`,
retaining picture dimensions, CICP and optional `Arc<ColorContext>` without
copying or converting samples. Its concrete consumers are the still-image IQA
contract in zenpixels PR 111 and the native frame mapping in zencodec PR 197.
Those consumers are migrated in their own lanes; this PR changes neither.

`PlanarLayout` holds up to four specs inline and implements `Copy`. Its
constructor takes `impl AsRef<[PlaneSpec]>`: arrays, slices and existing Vec
calls work. Untyped `collect()` expressions need `Vec<_>` or another explicit
collection type. Layout validation still rejects mixed or incomplete families.
`roles()` iterates storage order, and `check_plane(index, plane, width, height)`
validates individual mappings with an explicit out-of-range error.

`AnyPlane::as_strided_bytes` borrows native-endian samples with intervening
padding and no trailing padding. `stride_bytes` returns a Result because a
valid single-row sample view may have a sample stride that overflows bytes.
The byte conversion never copies and never silently saturates a stride.

The typed `PixelBuffer::from_pixels(Vec<P>, width, height)` constructor already
exists with the rgb feature. Its documentation now shows typed-vector use;
no duplicate constructor is added. Default storage is correctly documented as
`PixelStorage`. The approved luminance `Nits` type belongs to HDR PR 120,
where its consumers are the conversion options and source-peak context.

Validation: all zenpixels feature tests and doctests, strict all-target
Clippy, no-default-feature check, and public API snapshots. New integration
controls cover mixed 10-bit/U8 alpha, odd NV12 dimensions, native byte/pointer
identity, missing final-row padding and byte-stride overflow.
