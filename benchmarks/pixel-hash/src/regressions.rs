//! Paired published-source baselines. Construction/parity checks precede timing.
use std::hash::Hash;
use zenbench::prelude::*;
use zenpixels as current;
use zenpixels_before as before;

fn digest<T: Hash>(view: &T) -> u64 {
    let mut state = zenhash::Xxh3::new();
    view.hash(&mut state);
    state.digest()
}
fn digest_next<T: Hash>(view: &T) -> u64 {
    let mut state = zenhash_next::Xxh3::new();
    view.hash(&mut state);
    state.digest()
}
fn compare<B: Fn() -> u64 + Send + 'static, A: Fn() -> u64 + Send + 'static>(
    suite: &mut Suite,
    name: String,
    bytes: u64,
    old: B,
    new: A,
) {
    assert_eq!(old(), new(), "output parity before timing: {name}");
    suite.group(name, |g| {
        g.throughput(Throughput::Bytes(bytes));
        g.bench("before", move |b| b.iter(&old));
        g.bench("after", move |b| b.iter(&new));
    });
}
fn words(n: usize) -> &'static [u16] {
    (0..n)
        .map(|i| (((i as u16) & 1023) << 6) | 63)
        .collect::<Vec<_>>()
        .leak()
}
macro_rules! plane {
    ($api:ident, $data:expr, $w:expr, $h:expr, $stride:expr) => {
        $api::plane::AnyPlane::U16($api::plane::PlaneRef::new($data, $w, $h, $stride).unwrap())
    };
}
macro_rules! picture {
    ($api:ident, $y:expr, $uv:expr, $w:expr, $h:expr, $pad:expr) => {{
        use $api::plane::{PlanarLayout, PlanarSlice, PlaneEncoding, PlaneRole, PlaneSpec};
        use $api::sample::{ChromaSiting, ChromaSubsampling, SampleEncoding};
        let enc = PlaneEncoding::Code(SampleEncoding::new($api::ChannelType::U16, 10, 6).unwrap());
        let layout = Box::leak(Box::new(
            PlanarLayout::new(
                vec![
                    PlaneSpec::new(PlaneRole::Luma, enc),
                    PlaneSpec::new(PlaneRole::CbCr, enc),
                ],
                ChromaSubsampling::Yuv420,
                ChromaSiting::Left,
            )
            .unwrap(),
        ));
        let planes = Box::leak(Box::new([
            plane!($api, $y, $w, $h, $w + $pad),
            plane!(
                $api,
                $uv,
                $w.div_ceil(2) * 2,
                $h.div_ceil(2),
                $w.div_ceil(2) * 2 + $pad
            ),
        ]));
        let color = Box::leak(Box::new($api::ColorContext::default()));
        PlanarSlice::new(planes, layout, $w as u32, $h as u32, color, None).unwrap()
    }};
}
macro_rules! component {
    ($api:ident, $data:expr, $w:expr, $h:expr, $step:expr) => {
        $api::sample::CodePlane::component(
            plane!($api, $data, $w * $step, $h, $w * $step + 7),
            $step,
            0,
            $api::sample::SampleEncoding::new($api::ChannelType::U16, 10, 6).unwrap(),
        )
        .unwrap()
    };
}
fn bench(suite: &mut Suite) {
    for (w, h) in [(1usize, 1usize), (64, 64), (3840, 2160)] {
        for pad in [0, 32] {
            let y = words((w + pad) * h);
            let uv = words((w.div_ceil(2) * 2 + pad) * h.div_ceil(2));
            let a = picture!(before, y, uv, w, h, pad);
            let b = picture!(current, y, uv, w, h, pad);
            let bytes = (w * h + w.div_ceil(2) * 2 * h.div_ceil(2)) as u64 * 2;
            compare(
                suite,
                format!("reader/p010/pad{pad}/{w}x{h}"),
                bytes,
                move || digest(black_box(&a)),
                move || digest(black_box(&b)),
            );
        }
        for step in [1, 2] {
            let data = words((w * step + 7) * h);
            let a = component!(before, data, w, h, step);
            let b = component!(current, data, w, h, step);
            compare(
                suite,
                format!("reader/component{step}/{w}x{h}"),
                (w * h * 2) as u64,
                move || digest(black_box(&a)),
                move || digest(black_box(&b)),
            );
        }
        let data = words(w * h * 4);
        let a = before::PixelSlice::<()>::new(
            bytemuck::cast_slice(data),
            w as u32,
            h as u32,
            w * 8,
            before::PixelDescriptor::RGBA16,
        )
        .unwrap();
        let b = current::PixelSlice::<()>::new(
            bytemuck::cast_slice(data),
            w as u32,
            h as u32,
            w * 8,
            current::PixelDescriptor::RGBA16,
        )
        .unwrap();
        let a: &_ = Box::leak(Box::new(a));
        let b: &_ = Box::leak(Box::new(b));
        compare(
            suite,
            format!("packed_control/{w}x{h}"),
            (w * h * 8) as u64,
            move || digest(black_box(a)),
            move || digest(black_box(b)),
        );
        compare(
            suite,
            format!("backend_xxh3/{w}x{h}"),
            (w * h * 8) as u64,
            move || digest(black_box(b)),
            move || digest_next(black_box(b)),
        );
    }
    for n in [1, 64, 4096, 131072] {
        let input: &'static [u8] = (0..n)
            .map(|i| (i * 37 + 11) as u8)
            .collect::<Vec<_>>()
            .leak();
        compare(
            suite,
            format!("murmur32_control/{n}"),
            n as u64,
            move || u64::from(zenhash::murmur3_32(black_box(input), 42)),
            move || u64::from(zenhash_next::murmur3_32(black_box(input), 42)),
        );
        compare(
            suite,
            format!("fnv_fmix_control/{n}"),
            n as u64,
            move || zenhash::fnv1a64_fmix(black_box(input)),
            move || zenhash_next::fnv1a64_fmix(black_box(input)),
        );
        // Different output widths/algorithms: a cost comparison, not output
        // parity or a regression against an existing 128-bit implementation.
        suite.group(format!("murmur_width_cost/{n}"), |g| {
            g.throughput(Throughput::Bytes(n as u64));
            g.bench("x86_32", move |b| {
                b.iter(|| zenhash_next::murmur3_32(black_box(input), 42))
            });
            g.bench("x64_128", move |b| {
                b.iter(|| zenhash_next::murmur3_x64_128(black_box(input), 42))
            });
        });
    }
}
zenbench::main!(bench);
