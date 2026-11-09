# Zero-cost interoperability probes (2026-11-03)

Host r7900x, Ubuntu GLIBC 2.43-2ubuntu2.4, rustc 1.99.1; `allocator_beta` on rustc
1.100.0-beta.9 (d05f596bd 2026-11-03), because the `Allocator` API is stable from 1.100.
Crates: v_frame 0.7.0 and 0.3.9, image 0.25.10.

`same-ptr` means the destination reads the source's first sample. `allocs` counts heap
allocations during the conversion through a counting global allocator, except for the
one mutable `FlatSamples` route noted in the output. Pixels are asserted row by row
after the conversions that read them; `zenpixels/tests/image_interop.rs` has the full
coverage across formats.

```sh
cd benchmarks/plane-interop
cargo run --bin interop
cargo run --bin packed_storage
cargo run --bin stride_rule
cargo +beta run --features allocator-beta --bin allocator_beta
```

```text
### interop
--- v_frame 0.7 (padding_api) -> zenpixels
    FrameBuilder 37x11 4:2:0: Some(UnsupportedResolution)
    FrameBuilder 37x11 4:4:4: ok=true
borrow v_frame07::Plane<u8> as PlaneRef                        same-ptr  allocs=0
borrow v_frame07::Plane<u8> as PlaneMut                        same-ptr  allocs=0
move v_frame07::Plane<u8> into Plane<custom storage>           same-ptr  allocs=0
    v_frame row 0 address mod 64 with pad_left=16: 16
PlaneBuf -> existing v_frame07::Plane (row copy)               COPIED    allocs=0
    copy_from_u8_slice_with_stride on a trimmed last row: Err(DataLength { expected: 768, found: 742 })
--- v_frame 0.3.9 (pinned by zenrav1e) -> zenpixels
borrow v_frame03::Plane<u16> as PlaneRef                       same-ptr  allocs=0
    stride=96 xorigin=32 yorigin=13
borrow v_frame03::Plane<u16> as PlaneMut                       same-ptr  allocs=0
move v_frame03::Plane<u16> into Plane<custom storage>          same-ptr  allocs=0
slice -> v_frame03::Plane::from_slice (copies)                 COPIED    allocs=1
--- image: gray planes
strided PlaneRef<u8> -> image::flat::View<Luma<u8>>            same-ptr  allocs=0
    as_view_mut::<Luma<u8>> on a strided plane: ok=true
strided PlaneMut<u8> -> FlatSamples<&mut [u8]>                 same-ptr  allocs=0
    (the mutable route above has no allocation counter; it only stores a slice and a layout)
    TRAP from_raw on strided samples: accepted=true wrong_pixels=true
tight PlaneRef<u8> -> ImageBuffer<Luma<u8>, &[u8]>             same-ptr  allocs=0
ImageBuffer<Luma<u16>, Vec> -> PlaneRef<u16>                   same-ptr  allocs=0
move ImageBuffer<Luma<f32>, Vec> -> PlaneBuf<f32>              same-ptr  allocs=0
move tight PlaneBuf<f32> -> ImageBuffer<Luma<f32>, Vec>        same-ptr  allocs=0
    alloc(37,11) stride=64 addr%64=0 image_tight=false
    alloc(64,11) stride=64 addr%64=0 image_tight=true
    alloc(4096,4) stride=4160 addr%64=0 image_tight=false
    alloc(4096,64) stride=4160 addr%64=0 image_tight=false
--- image: packed RGB <-> PixelSlice / PixelBuffer
move ImageBuffer<Rgb<u8>, Vec> -> PixelBuffer                  same-ptr  allocs=0
move tight PixelBuffer<RGB8> -> ImageBuffer<Rgb<u8>, Vec>      same-ptr  allocs=0
ImageBuffer<Rgb<u16>> -> PixelSlice (borrow)                   same-ptr  allocs=0
    Vec<u16> -> Vec<u8>: refused (AlignmentMismatch), so Vec<u8> storage cannot adopt it; PixelStorage holds the Vec<u16> instead
    Vec<u8> -> Vec<u16>: refused (AlignmentMismatch), so bytes copy to reach ImageBuffer<Rgb<u16>, Vec<u16>>
    new_simd_aligned stride_bytes=384 (next multiple of 64 = 256)
    as_view::<Rgb<u16>> ok=true
strided PixelSlice<RGB16> -> image::flat::View<Rgb<u16>>       same-ptr  allocs=0
    from_raw color after handing over linear data: Cicp { primaries: SRgb, transfer: SRgb, matrix: Identity, full_range: FullRange }
### packed_storage
move ImageBuffer<Rgb<u16>, Vec<u16>> -> PixelBuffer            same-ptr  allocs=0
move PixelBuffer<RGB16, Vec<u16>> -> ImageBuffer<Rgb<u16>>     same-ptr  allocs=0
move ImageBuffer<Rgb<f32>, Vec<f32>> -> PixelBuffer            same-ptr  allocs=0
adopt strided Vec<u16> with offset                             same-ptr  allocs=0
as_slice_mut over Vec<u16>                                     same-ptr  allocs=0
adopt Box<[u8]>                                                same-ptr  allocs=0
adopt &[u8]                                                    same-ptr  allocs=0
adopt custom owner (pool slot)                                 same-ptr  allocs=0
TryFrom ImageBuffer<Rgb<u16>, Vec<u16>> -> PixelBuffer         same-ptr  allocs=0
TryFrom PixelBuffer -> ImageBuffer<Rgb<u16>, Vec<u16>>         same-ptr  allocs=0
TryFrom &ImageBuffer<Rgb<u16>> -> PixelSlice                   same-ptr  allocs=0
    try_new_aligned RGB16: addr%64=0 stride=384 allocs=1
    last row as &[[u16; 3]]: 37 pixels
    misaligned u8 storage for RGB16: refused, parts_returned=true
    size_of PixelBuffer: Vec<u8>=72 Vec<u16>=64 Box<[u8]>=56 &[u8]=56
### stride_rule
RGB8   tight         stride=111  PixelSlice: ok
RGB8   4-byte rows   stride=112  PixelSlice: ok
RGB8   16-byte rows  stride=112  PixelSlice: ok
RGB8   64-byte rows  stride=128  PixelSlice: ok
RGBA8  tight         stride=148  PixelSlice: ok
RGBA8  4-byte rows   stride=148  PixelSlice: ok
RGBA8  16-byte rows  stride=160  PixelSlice: ok
RGBA8  64-byte rows  stride=192  PixelSlice: ok
RGB16  tight         stride=222  PixelSlice: ok
RGB16  4-byte rows   stride=224  PixelSlice: ok
RGB16  16-byte rows  stride=224  PixelSlice: ok
RGB16  64-byte rows  stride=256  PixelSlice: ok
PlaneRef<u8> width 37 stride 64: ok=true
### allocator_beta
Vec<u8, Align64> adopted: offset=0 addr%64=0
-> ImageBuffer<Rgb<u8>, Vec<u8, Align64>>: same-ptr
Vec<u16, Align64> adopted: offset=0 addr%64=0
Vec<f32, Align64> adopted: offset=0 addr%64=0
Plane<Vec<u16, Align64>>::try_into_vec: addr%64=0
size_of PixelBuffer: default=72 Vec<u8, Align64>=64
```

Interpretation is in `docs/planar-types.md`.
