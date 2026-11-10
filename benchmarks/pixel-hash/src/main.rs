use std::hash::Hash;
use zenbench::prelude::*;
use zenpixels::plane::{
    AnyPlane, PlanarLayout, PlanarSlice, PlaneEncoding, PlaneRef, PlaneRole, PlaneSpec,
};
use zenpixels::sample::{ChromaSiting, ChromaSubsampling, SampleEncoding};
use zenpixels::{ChannelType, ColorContext, PixelDescriptor, PixelSlice};

fn compare<T: Hash + Sync + 'static>(suite: &mut Suite, name: String, bytes: u64, view: T) {
    let view: &'static T = Box::leak(Box::new(view));
    suite.group(name, |g| {
        g.throughput(Throughput::Bytes(bytes));
        g.bench("rapidhash_v3", move |b| {
            b.iter(|| {
                let mut state = zenhash::RapidhashV3::new();
                black_box(view).hash(&mut state);
                state.digest()
            })
        });
        g.bench("xxh3_64", move |b| {
            b.iter(|| {
                let mut state = zenhash::Xxh3::new();
                black_box(view).hash(&mut state);
                state.digest()
            })
        });
    });
}

fn input_words(n: usize) -> &'static [u64] {
    (0..n)
        .map(|i| (i as u64).wrapping_mul(0x9e3779b97f4a7c15).rotate_left(17))
        .collect::<Vec<_>>()
        .leak()
}

fn bench(suite: &mut Suite) {
    // Entry, cache and large-image sizes. All allocation is outside timing.
    for (w, h) in [
        (0, 0),
        (1, 1),
        (8, 8),
        (64, 64),
        (256, 256),
        (1024, 1024),
        (3840, 2160),
    ] {
        for (name, desc) in [
            ("rgba8", PixelDescriptor::RGBA8),
            ("rgba16", PixelDescriptor::RGBA16),
            ("rgbf32", PixelDescriptor::RGBF32),
            ("rgbx8", PixelDescriptor::RGBX8),
            (
                "grayf64",
                PixelDescriptor::try_new(
                    ChannelType::F64,
                    zenpixels::ChannelLayout::Gray,
                    None,
                    zenpixels::TransferFunction::Linear,
                )
                .unwrap(),
            ),
        ] {
            for pad in [0, 64] {
                let stride = w as usize * desc.bytes_per_pixel() + pad;
                let size = stride * h as usize;
                let bytes = bytemuck::cast_slice(input_words(size.div_ceil(8)));
                let view = PixelSlice::<()>::new(bytes, w, h, stride, desc).unwrap();
                let meaningful = u64::from(w)
                    * u64::from(h)
                    * (desc.channels()
                        - usize::from(desc.alpha == Some(zenpixels::AlphaMode::Undefined)))
                        as u64
                    * desc.channel_type().byte_size() as u64;
                if name == "rgba8" && pad == 0 && [1, 64, 1024].contains(&w) {
                    for bytes in [512, 4096, 131072] {
                        let color = ColorContext::from_icc(bytemuck::cast_slice::<_, u8>(
                            input_words(bytes / 8),
                        ))
                        .with_cicp(zenpixels::Cicp::SRGB)
                        .with_diffuse_white(zenpixels::hdr::DiffuseWhite::BT2408)
                        .with_hdr_peak(zenpixels::hdr::Nits::new(1000.0));
                        compare(
                            suite,
                            format!("metadata/packed/icc{bytes}/{w}x{h}"),
                            meaningful,
                            view.clone().with_color_context(std::sync::Arc::new(color)),
                        );
                    }
                }
                compare(
                    suite,
                    format!("packed/{name}/pad{pad}/{w}x{h}"),
                    meaningful,
                    view,
                );
            }
        }
        for pad in [0, 64] {
            let stride = w as usize + pad;
            let count = stride * h as usize;
            let layout = Box::leak(Box::new(
                PlanarLayout::new(
                    [PlaneRole::Red, PlaneRole::Green, PlaneRole::Blue]
                        .into_iter()
                        .map(|r| {
                            PlaneSpec::new(
                                r,
                                PlaneEncoding::Code(
                                    SampleEncoding::new(ChannelType::U8, 8, 0).unwrap(),
                                ),
                            )
                        })
                        .collect(),
                    ChromaSubsampling::Yuv444,
                    ChromaSiting::Center,
                )
                .unwrap(),
            ));
            let planes = (0..3)
                .map(|_| {
                    let bytes: &'static [u8] = bytemuck::cast_slice(input_words(count.div_ceil(8)));
                    AnyPlane::U8(PlaneRef::new(bytes, w as usize, h as usize, stride).unwrap())
                })
                .collect::<Vec<_>>()
                .leak();
            let color = Box::leak(Box::new(ColorContext::default()));
            let view = PlanarSlice::new(planes, layout, w, h, color, None).unwrap();
            if pad == 0 && [1, 64, 1024].contains(&w) {
                for bytes in [512, 4096, 131072] {
                    let color = Box::leak(Box::new(
                        ColorContext::from_icc(bytemuck::cast_slice::<_, u8>(input_words(
                            bytes / 8,
                        )))
                        .with_cicp(zenpixels::Cicp::SRGB)
                        .with_diffuse_white(zenpixels::hdr::DiffuseWhite::BT2408)
                        .with_hdr_peak(zenpixels::hdr::Nits::new(1000.0)),
                    ));
                    let tagged = PlanarSlice::new(planes, layout, w, h, color, None).unwrap();
                    compare(
                        suite,
                        format!("metadata/planar/icc{bytes}/{w}x{h}"),
                        u64::from(w) * u64::from(h) * 3,
                        tagged,
                    );
                }
            }
            compare(
                suite,
                format!("planar/rgb8/pad{pad}/{w}x{h}"),
                u64::from(w) * u64::from(h) * 3,
                view,
            );
        }
        for (name, ty, bits, shift) in [
            ("nv12", ChannelType::U8, 8, 0),
            ("p010", ChannelType::U16, 10, 6),
        ] {
            let enc = PlaneEncoding::Code(SampleEncoding::new(ty, bits, shift).unwrap());
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
            let (cw, ch) = ((w as usize).div_ceil(2) * 2, (h as usize).div_ceil(2));
            let planes: Vec<_> = [(w as usize, h as usize), (cw, ch)]
                .into_iter()
                .map(|(width, height)| {
                    let words = input_words((width * height * ty.byte_size()).div_ceil(8));
                    if ty == ChannelType::U8 {
                        AnyPlane::U8(
                            PlaneRef::new(bytemuck::cast_slice(words), width, height, width)
                                .unwrap(),
                        )
                    } else {
                        AnyPlane::U16(
                            PlaneRef::new(bytemuck::cast_slice(words), width, height, width)
                                .unwrap(),
                        )
                    }
                })
                .collect();
            let color = Box::leak(Box::new(ColorContext::default()));
            let view = PlanarSlice::new(planes.leak(), layout, w, h, color, None).unwrap();
            compare(
                suite,
                format!("planar/{name}/{w}x{h}"),
                (w as usize * h as usize + cw * ch) as u64 * ty.byte_size() as u64,
                view,
            );
        }
    }
}
zenbench::main!(bench);
