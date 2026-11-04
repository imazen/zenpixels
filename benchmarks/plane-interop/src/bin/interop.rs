//! Zero-cost interop probe: zenpixels planes and packed buffers against
//! v_frame 0.7, v_frame 0.3.9 and image 0.25.10. Every route reports pointer
//! identity and heap allocations during the conversion; pixels are compared
//! after it. Asserts stop the run on a wrong result.
use image::flat::{FlatSamples, SampleLayout};
use image::{GenericImageView, ImageBuffer, Luma, Rgb};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use zenpixels::plane::{Plane, PlaneBuf, PlaneMut, PlaneRef, PlaneStorage};
use zenpixels::{PixelBuffer, PixelBufferParts, PixelDescriptor, PixelSlice};

struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(p, l, n) }
    }
}
#[global_allocator]
static A: Counting = Counting;

fn counted<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let before = ALLOCS.load(Ordering::Relaxed);
    let r = std::hint::black_box(f());
    (r, ALLOCS.load(Ordering::Relaxed) - before)
}
fn report(name: &str, same_ptr: Option<bool>, allocs: usize) {
    let p = match same_ptr {
        Some(true) => "same-ptr",
        Some(false) => "COPIED",
        None => "-",
    };
    println!("{name:<62} {p:<9} allocs={allocs}");
}

const W: usize = 37;
const H: usize = 11;

fn main() {
    println!("--- v_frame 0.7 (padding_api) -> zenpixels");
    {
        use v_frame07::chroma::ChromaSubsampling;
        use v_frame07::frame::FrameBuilder;
        let odd = FrameBuilder::new(37, 11, ChromaSubsampling::Yuv420, 8).build::<u8>();
        println!("    FrameBuilder 37x11 4:2:0: {:?}", odd.as_ref().err());
        let odd444 = FrameBuilder::new(37, 11, ChromaSubsampling::Yuv444, 8).build::<u8>();
        println!("    FrameBuilder 37x11 4:4:4: ok={}", odd444.is_ok());
        const W: usize = 38;
        const H: usize = 12;
        let mut f = FrameBuilder::new(W, H, ChromaSubsampling::Yuv420, 8)
            .luma_padding_left(16)
            .luma_padding_right(10)
            .luma_padding_top(4)
            .luma_padding_bottom(4)
            .build::<u8>()
            .unwrap();
        for (y, row) in f.y_plane.rows_mut().enumerate() {
            for (x, s) in row.iter_mut().enumerate() {
                *s = (x * 3 + y * 7) as u8;
            }
        }
        // Borrowed, active area only, padding on all four sides.
        let g = f.y_plane.geometry();
        let origin = g.pad_top() * g.stride() + g.pad_left();
        let (view, a) =
            counted(|| PlaneRef::new(&f.y_plane.data()[origin..], W, H, g.stride()).unwrap());
        for y in 0..H {
            assert_eq!(view.row(y).unwrap(), f.y_plane.row(y).unwrap());
        }
        report(
            "borrow v_frame07::Plane<u8> as PlaneRef",
            Some(view.row(0).unwrap().as_ptr() == f.y_plane.row(0).unwrap().as_ptr()),
            a,
        );
        let p0 = f.y_plane.row(0).unwrap().as_ptr();
        let (mut m, a) = counted(|| {
            PlaneMut::new(&mut f.y_plane.data_mut()[origin..], W, H, g.stride()).unwrap()
        });
        m.row_mut(1).unwrap()[2] = 201;
        report(
            "borrow v_frame07::Plane<u8> as PlaneMut",
            Some(m.row(0).unwrap().as_ptr() == p0),
            a,
        );
        assert_eq!(f.y_plane.row(1).unwrap()[2], 201);

        // Owned: move the whole v_frame plane behind our storage trait.
        struct OwnedV(v_frame07::plane::Plane<u8>, usize);
        impl PlaneStorage for OwnedV {
            type Sample = u8;
            fn samples(&self) -> &[u8] {
                &self.0.data()[self.1..]
            }
        }
        let f2 = FrameBuilder::new(W, H, ChromaSubsampling::Yuv420, 8)
            .luma_padding_left(16)
            .luma_padding_right(10)
            .luma_padding_top(4)
            .luma_padding_bottom(4)
            .build::<u8>()
            .unwrap();
        let p0 = f2.y_plane.row(0).unwrap().as_ptr();
        let stride = f2.y_plane.geometry().stride();
        let (owned, a) = counted(|| {
            let y = f2.y_plane;
            Plane::new(OwnedV(y, origin), W, H, stride).unwrap()
        });
        report(
            "move v_frame07::Plane<u8> into Plane<custom storage>",
            Some(owned.row(0).unwrap().as_ptr() == p0),
            a,
        );
        println!(
            "    v_frame row 0 address mod 64 with pad_left=16: {}",
            p0 as usize % 64
        );

        // Zenpixels -> v_frame has no adopting constructor; rows are copied.
        let src = PlaneBuf::<u8>::alloc(W, H).unwrap();
        let mut dst = FrameBuilder::new(W, H, ChromaSubsampling::Monochrome, 8)
            .build::<u8>()
            .unwrap();
        let (_, a) = counted(|| {
            for (d, y) in dst.y_plane.rows_mut().zip(0..H) {
                d.copy_from_slice(src.row(y).unwrap());
            }
        });
        report(
            "PlaneBuf -> existing v_frame07::Plane (row copy)",
            Some(false),
            a,
        );
        // Its byte-slice stride copy demands stride*height exactly.
        let trimmed_buf = vec![0u8; 64 * (H - 1) + W];
        let trimmed = PlaneRef::new(&trimmed_buf[..], W, H, 64).unwrap();
        let r = dst
            .y_plane
            .copy_from_u8_slice_with_stride(trimmed.into_strided_samples(), 64);
        println!("    copy_from_u8_slice_with_stride on a trimmed last row: {r:?}");
    }

    println!("--- v_frame 0.3.9 (pinned by zenrav1e) -> zenpixels");
    {
        let mut p = v_frame03::plane::Plane::<u16>::new(W, H, 0, 0, 13, 13);
        let stride = p.cfg.stride;
        for y in 0..H {
            for x in 0..W {
                p.data_origin_mut()[y * stride + x] = (x * 5 + y * 9) as u16;
            }
        }
        let p0 = p.data_origin().as_ptr();
        let (view, a) = counted(|| PlaneRef::new(p.data_origin(), W, H, stride).unwrap());
        for y in 0..H {
            assert_eq!(
                view.row(y).unwrap(),
                &p.data_origin()[y * stride..y * stride + W]
            );
        }
        report(
            "borrow v_frame03::Plane<u16> as PlaneRef",
            Some(view.row(0).unwrap().as_ptr() == p0),
            a,
        );
        println!(
            "    stride={stride} xorigin={} yorigin={}",
            p.cfg.xorigin, p.cfg.yorigin
        );
        let (m, a) = counted(|| PlaneMut::new(p.data_origin_mut(), W, H, stride).unwrap());
        report(
            "borrow v_frame03::Plane<u16> as PlaneMut",
            Some(m.row(0).unwrap().as_ptr() == p0),
            a,
        );
        struct OwnedV3(v_frame03::plane::Plane<u16>);
        impl PlaneStorage for OwnedV3 {
            type Sample = u16;
            fn samples(&self) -> &[u16] {
                self.0.data_origin()
            }
        }
        let (o, a) = counted(|| Plane::new(OwnedV3(p), W, H, stride).unwrap());
        report(
            "move v_frame03::Plane<u16> into Plane<custom storage>",
            Some(o.row(0).unwrap().as_ptr() == p0),
            a,
        );
        let tight = vec![7u16; W * H];
        let (_, a) = counted(|| v_frame03::plane::Plane::from_slice(&tight, W));
        report(
            "slice -> v_frame03::Plane::from_slice (copies)",
            Some(false),
            a,
        );
    }

    println!("--- image: gray planes");
    {
        let stride = 64;
        let mut buf = vec![0u8; stride * (H - 1) + W];
        for y in 0..H {
            for x in 0..W {
                buf[y * stride + x] = (x * 3 + y * 7) as u8;
            }
        }
        let plane = PlaneRef::new(&buf[..], W, H, stride).unwrap();

        // Strided plane -> image's strided view.
        let layout = SampleLayout {
            channels: 1,
            channel_stride: 1,
            width: W as u32,
            width_stride: 1,
            height: H as u32,
            height_stride: stride,
        };
        let samples = plane.view().into_strided_samples();
        let (flat, a) = counted(|| FlatSamples {
            samples,
            layout,
            color_hint: None,
        });
        let view = flat.as_view::<Luma<u8>>().unwrap();
        for y in 0..H {
            for x in 0..W {
                assert_eq!(
                    view.get_pixel(x as u32, y as u32).0[0],
                    plane.row(y).unwrap()[x]
                );
            }
        }
        report(
            "strided PlaneRef<u8> -> image::flat::View<Luma<u8>>",
            Some(flat.samples.as_ptr() == buf.as_ptr()),
            a,
        );

        // Strided, mutable.
        let mut mbuf = buf.clone();
        let m0 = mbuf.as_ptr();
        let mp = PlaneMut::new(&mut mbuf[..], W, H, stride).unwrap();
        let mut mflat = FlatSamples {
            samples: mp.into_strided_samples_mut(),
            layout,
            color_hint: None,
        };
        let vm = mflat.as_view_mut::<Luma<u8>>();
        println!(
            "    as_view_mut::<Luma<u8>> on a strided plane: ok={}",
            vm.is_ok()
        );
        report(
            "strided PlaneMut<u8> -> FlatSamples<&mut [u8]>",
            Some(mflat.samples.as_ptr() == m0),
            0,
        );
        println!(
            "    (the mutable route above has no allocation counter; it only stores a slice and a layout)"
        );

        // Trap: from_raw only checks length, so strided samples pass and read wrong.
        let wrong = ImageBuffer::<Luma<u8>, &[u8]>::from_raw(
            W as u32,
            H as u32,
            plane.view().into_strided_samples(),
        );
        let bad = wrong
            .as_ref()
            .is_some_and(|i| i.get_pixel(0, 1).0[0] != plane.row(1).unwrap()[0]);
        println!(
            "    TRAP from_raw on strided samples: accepted={} wrong_pixels={}",
            wrong.is_some(),
            bad
        );

        // Tight borrow, both ways.
        let tight: Vec<u8> = (0..W * H).map(|i| i as u8).collect();
        let tp = PlaneRef::new(&tight[..], W, H, W).unwrap();
        let (ib, a) = counted(|| {
            ImageBuffer::<Luma<u8>, &[u8]>::from_raw(W as u32, H as u32, tp.into_strided_samples())
                .unwrap()
        });
        report(
            "tight PlaneRef<u8> -> ImageBuffer<Luma<u8>, &[u8]>",
            Some(ib.as_raw().as_ptr() == tight.as_ptr()),
            a,
        );
        let owned_img = ImageBuffer::<Luma<u16>, Vec<u16>>::new(W as u32, H as u32);
        let (b, a) = counted(|| PlaneRef::new(owned_img.as_raw().as_slice(), W, H, W).unwrap());
        report(
            "ImageBuffer<Luma<u16>, Vec> -> PlaneRef<u16>",
            Some(b.row(0).unwrap().as_ptr() == owned_img.as_raw().as_ptr()),
            a,
        );

        // Owned, both ways.
        let fimg = ImageBuffer::<Luma<f32>, Vec<f32>>::new(W as u32, H as u32);
        let p0 = fimg.as_raw().as_ptr();
        let (pb, a) = counted(|| PlaneBuf::new(fimg.into_raw(), W, H, W).unwrap());
        report(
            "move ImageBuffer<Luma<f32>, Vec> -> PlaneBuf<f32>",
            Some(pb.row(0).unwrap().as_ptr() == p0),
            a,
        );
        let (back, a) = counted(|| {
            ImageBuffer::<Luma<f32>, Vec<f32>>::from_raw(W as u32, H as u32, pb.into_vec().unwrap())
                .unwrap()
        });
        report(
            "move tight PlaneBuf<f32> -> ImageBuffer<Luma<f32>, Vec>",
            Some(back.as_raw().as_ptr() == p0),
            a,
        );

        // PlaneBuf::alloc rounds stride up, so its Vec is rarely tight.
        for (w, h) in [(37usize, 11usize), (64, 11), (4096, 4), (4096, 64)] {
            let pb = PlaneBuf::<u8>::alloc(w, h).unwrap();
            let s = pb.stride();
            match pb.into_vec() {
                Ok(_) => println!(
                    "    alloc({w},{h}) stride={s} start0=yes image_tight={}",
                    s == w
                ),
                Err(_) => println!("    alloc({w},{h}) stride={s} start0=no  image_tight=false"),
            }
        }
    }

    println!("--- image: packed RGB <-> PixelSlice / PixelBuffer");
    {
        let (w, h) = (W as u32, H as u32);
        // u8 owned, both ways.
        let img = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(w, h);
        let p0 = img.as_raw().as_ptr();
        let (pb, a) = counted(|| {
            PixelBuffer::try_from_parts(PixelBufferParts::new(
                img.into_raw(),
                w,
                h,
                W * 3,
                PixelDescriptor::RGB8_SRGB,
            ))
            .unwrap()
        });
        report(
            "move ImageBuffer<Rgb<u8>, Vec> -> PixelBuffer",
            Some(pb.as_slice().as_strided_bytes().as_ptr() == p0),
            a,
        );
        let (back, a) = counted(|| {
            let parts = pb.into_parts();
            assert_eq!((parts.offset, parts.stride_bytes), (0, W * 3));
            ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(w, h, parts.data).unwrap()
        });
        report(
            "move tight PixelBuffer<RGB8> -> ImageBuffer<Rgb<u8>, Vec>",
            Some(back.as_raw().as_ptr() == p0),
            a,
        );

        // u16 borrowed.
        let img16 = ImageBuffer::<Rgb<u16>, Vec<u16>>::new(w, h);
        let (s, a) = counted(|| {
            PixelSlice::new(
                bytemuck::cast_slice(img16.as_raw().as_slice()),
                w,
                h,
                W * 6,
                PixelDescriptor::RGB16_SRGB,
            )
            .unwrap()
        });
        report(
            "ImageBuffer<Rgb<u16>> -> PixelSlice (borrow)",
            Some(s.as_strided_bytes().as_ptr() as *const u16 == img16.as_raw().as_ptr()),
            a,
        );

        // u16 owned: PixelBuffer's Vec<u8> cannot adopt a Vec<u16>.
        let raw: Vec<u16> = img16.into_raw();
        match bytemuck::allocation::try_cast_vec::<u16, u8>(raw) {
            Ok(_) => println!("    Vec<u16> -> Vec<u8>: adopted"),
            Err((e, _)) => println!(
                "    Vec<u16> -> Vec<u8>: refused ({e:?}), so Vec<u8> storage cannot adopt it"
            ),
        }
        let bytes = vec![0u8; 12];
        match bytemuck::allocation::try_cast_vec::<u8, u16>(bytes) {
            Ok(_) => println!("    Vec<u8> -> Vec<u16>: adopted"),
            Err((e, _)) => println!(
                "    Vec<u8> -> Vec<u16>: refused ({e:?}), so a Vec<u8>-backed buffer copies to reach ImageBuffer<Rgb<u16>, Vec<u16>>"
            ),
        }

        // PixelBuffer u16, padded rows, to image's strided view (borrow).
        let stride_bytes = (W * 6).next_multiple_of(64);
        let aligned =
            PixelBuffer::try_new_simd_aligned(w, h, PixelDescriptor::RGB16_SRGB, 64).unwrap();
        let sl = aligned.as_slice();
        let bytes = sl.as_strided_bytes();
        println!(
            "    new_simd_aligned stride_bytes={} (next multiple of 64 = {stride_bytes})",
            sl.stride()
        );
        let layout = SampleLayout {
            channels: 3,
            channel_stride: 1,
            width: w,
            width_stride: 3,
            height: h,
            height_stride: sl.stride() / 2,
        };
        let (flat, a) = counted(|| FlatSamples {
            samples: bytemuck::cast_slice::<u8, u16>(bytes),
            layout,
            color_hint: None,
        });
        let v = flat.as_view::<Rgb<u16>>();
        println!("    as_view::<Rgb<u16>> ok={}", v.is_ok());
        report(
            "strided PixelSlice<RGB16> -> image::flat::View<Rgb<u16>>",
            Some(flat.samples.as_ptr() as *const u8 == bytes.as_ptr()),
            a,
        );

        // Color: from_raw labels everything sRGB; nothing carries our descriptor.
        let linear = vec![0u8; 3 * W * H];
        let i = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(w, h, linear).unwrap();
        println!(
            "    from_raw color after handing over linear data: {:?}",
            i.color_space()
        );
    }
}
