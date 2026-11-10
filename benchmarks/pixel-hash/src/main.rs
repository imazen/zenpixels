use std::hash::Hash;
use zenbench::prelude::*;
use zenpixels::{ColorContext, PixelDescriptor, PixelSlice, ChannelType};
use zenpixels::plane::{AnyPlane, PlaneRef, PlanarLayout, PlanarSlice, PlaneSpec, PlaneRole, PlaneEncoding};
use zenpixels::sample::{SampleEncoding, ChromaSubsampling, ChromaSiting};

fn compare<T: Hash + Sync + 'static>(suite: &mut Suite, name: String, bytes: u64, view: T) {
    let view: &'static T = Box::leak(Box::new(view));
    suite.group(name, |g| {
        g.throughput(Throughput::Bytes(bytes));
        g.bench("rapidhash_v3", move |b| b.iter(|| {
            let mut state = zenhash::RapidhashV3::new();
            black_box(view).hash(&mut state);
            state.digest()
        }));
        g.bench("xxh3_64", move |b| b.iter(|| {
            let mut state = zenhash::Xxh3::new();
            black_box(view).hash(&mut state);
            state.digest()
        }));
    });
}

fn input_words(n: usize) -> &'static [u64] {
    (0..n).map(|i| (i as u64).wrapping_mul(0x9e3779b97f4a7c15).rotate_left(17)).collect::<Vec<_>>().leak()
}

fn bench(suite: &mut Suite) {
    // Entry, cache and large-image sizes. All allocation is outside timing.
    for (w,h) in [(0,0), (1,1), (8,8), (64,64), (256,256), (1024,1024), (3840,2160)] {
        for (name,desc) in [("rgba8",PixelDescriptor::RGBA8), ("rgba16",PixelDescriptor::RGBA16), ("rgbf32",PixelDescriptor::RGBF32)] {
            for pad in [0,64] {
                let stride = w as usize * desc.bytes_per_pixel() + pad;
                let size = stride * h as usize;
                let bytes = bytemuck::cast_slice(input_words(size.div_ceil(8)));
                let view = PixelSlice::<()>::new(bytes, w, h, stride, desc).unwrap();
                compare(suite, format!("packed/{name}/pad{pad}/{w}x{h}"), u64::from(w)*u64::from(h)*desc.bytes_per_pixel() as u64, view);
            }
        }
        for pad in [0,64] {
            let stride = w as usize + pad;
            let count = stride * h as usize;
            let layout = Box::leak(Box::new(PlanarLayout::new(
                [PlaneRole::Red, PlaneRole::Green, PlaneRole::Blue].into_iter().map(|r| PlaneSpec::new(r, PlaneEncoding::Code(SampleEncoding::new(ChannelType::U8,8,0).unwrap()))).collect(),
                ChromaSubsampling::Yuv444, ChromaSiting::Center,
            ).unwrap()));
            let planes = (0..3).map(|_| {
                let bytes: &'static [u8] = bytemuck::cast_slice(input_words(count.div_ceil(8)));
                AnyPlane::U8(PlaneRef::new(bytes,w as usize,h as usize,stride).unwrap())
            }).collect::<Vec<_>>().leak();
            let color = Box::leak(Box::new(ColorContext::default()));
            let view = PlanarSlice::new(planes,layout,w,h,color,None).unwrap();
            compare(suite,format!("planar/rgb8/pad{pad}/{w}x{h}"),u64::from(w)*u64::from(h)*3,view);
        }
        for (name,ty,bits,shift) in [("nv12",ChannelType::U8,8,0),("p010",ChannelType::U16,10,6)] {
            let enc = PlaneEncoding::Code(SampleEncoding::new(ty,bits,shift).unwrap());
            let layout=Box::leak(Box::new(PlanarLayout::new(vec![PlaneSpec::new(PlaneRole::Luma,enc),PlaneSpec::new(PlaneRole::CbCr,enc)],ChromaSubsampling::Yuv420,ChromaSiting::Left).unwrap()));
            let (cw,ch)=((w as usize).div_ceil(2)*2,(h as usize).div_ceil(2));
            let planes: Vec<_> = [(w as usize,h as usize),(cw,ch)].into_iter().map(|(width,height)| {
                let words=input_words((width*height*ty.byte_size()).div_ceil(8));
                if ty==ChannelType::U8 { AnyPlane::U8(PlaneRef::new(bytemuck::cast_slice(words),width,height,width).unwrap()) }
                else { AnyPlane::U16(PlaneRef::new(bytemuck::cast_slice(words),width,height,width).unwrap()) }
            }).collect();
            let color=Box::leak(Box::new(ColorContext::default()));
            let view=PlanarSlice::new(planes.leak(),layout,w,h,color,None).unwrap();
            compare(suite,format!("planar/{name}/{w}x{h}"),(w as usize*h as usize+cw*ch) as u64*ty.byte_size() as u64,view);
        }
    }
}
zenbench::main!(bench);
