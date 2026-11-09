# Planar accessors and validation

The integrated revision retains `PlanarLayout`'s owned `Vec<PlaneSpec>` and
`Clone` contract. It does not promise `Copy` or a fixed four-slot representation.
Current RGB/YUV role validation remains unchanged, including duplicate-role
and mixed-family rejection. `planes()` already provides a borrowed layout slice.

`AnyPlane::stride_bytes` checks multiplication overflow; `as_strided_bytes`
borrows native-endian samples including intervening padding and excludes final
padding. `PlanarLayout::check_plane` validates one view without assembling an
entire picture; `check` uses the same validation. `roles` follows storage order.
The existing `PixelBuffer::from_pixels` documentation now makes typed ownership
construction explicit.

The original proposed `Planes` wrapper remains in review history. Its claimed
consumers no longer require it: zenpsy-api accepts packed PixelSlice views and
zencodec's merged plane contract uses PlanarLayout directly. It is not introduced
as a permanent public type without an adopting consumer.

Tests retain odd-size subsampling, mixed depths, padding, zero-copy pointers,
type/geometry mismatch and overflowing byte strides. Constructor inputs and the
owned layout representation remain compatible with main.
