//! Conversions to and from `image`, across every format they cover, with
//! `image` itself reading the results back. These run on plain `cargo test`
//! because the crate's dev-dependency on itself enables the feature.

use image::flat::FlatSamples;
use image::metadata::{CicpColorPrimaries as P, CicpTransferCharacteristics as Tf};
use image::{
    ColorType, DynamicImage, ExtendedColorType, GenericImage, GenericImageView, ImageBuffer, Luma,
    LumaA, Pixel, Rgb, Rgba,
};
use zenpixels::plane::{
    AlignedPlane, AlignedSamples, Plane, PlaneBuf, PlaneMut, PlaneRef, SampleStorage,
};
use zenpixels::{
    AlphaMode, ChannelLayout, ChannelType, ColorPrimaries, PixelBuffer, PixelBufferParts,
    PixelDescriptor, PixelFormat, PixelSlice, PixelSliceMut, SignalRange, TransferFunction,
};

const SIZES: [(u32, u32); 5] = [(1, 1), (2, 3), (37, 11), (64, 2), (65, 7)];

/// Samples where every position differs, so a wrong stride cannot pass.
fn fill_u8(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i % 251) as u8 + 1).collect()
}
fn fill_u16(n: usize) -> Vec<u16> {
    (0..n).map(|i| (i % 65_521) as u16 + 1).collect()
}
fn fill_f32(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i % 8191) as f32 / 8191.0 + 0.5).collect()
}

macro_rules! every_format {
    ($($name:ident: $pixel:ident<$sample:ty>, $fill:ident, $channel:ident, $layout:ident, $alpha:expr;)*) => {$(
        mod $name {
            use super::*;
            type Px = $pixel<$sample>;
            const CHANNELS: usize = <Px as Pixel>::CHANNEL_COUNT as usize;
            const BPP: usize = CHANNELS * size_of::<$sample>();

            fn image(w: u32, h: u32) -> ImageBuffer<Px, Vec<$sample>> {
                ImageBuffer::from_raw(w, h, $fill(w as usize * h as usize * CHANNELS)).unwrap()
            }

            #[test]
            fn owned_round_trip_moves_the_allocation() {
                for (w, h) in SIZES {
                    let img = image(w, h);
                    let want = img.as_raw().clone();
                    let ptr = img.as_raw().as_ptr();
                    let color = img.color_space();

                    let buffer: PixelBuffer<(), Vec<$sample>> = PixelBuffer::try_from(img).unwrap();
                    let d = buffer.descriptor();
                    assert_eq!(d.channel_type(), ChannelType::$channel);
                    assert_eq!(d.layout(), ChannelLayout::$layout);
                    assert_eq!(d.alpha(), $alpha);
                    assert_eq!(buffer.stride(), w as usize * BPP);
                    assert_eq!(buffer.as_slice().as_strided_bytes().as_ptr().cast(), ptr);
                    assert_eq!(buffer.as_slice().as_strided_bytes(), bytemuck::cast_slice::<$sample, u8>(&want));

                    let back = ImageBuffer::<Px, Vec<$sample>>::try_from(buffer).unwrap();
                    assert_eq!(back.as_raw().as_ptr(), ptr);
                    assert_eq!(back.as_raw(), &want);
                    assert_eq!(back.color_space(), color);
                }
            }

            /// The default storage, which crosses `dyn` boundaries, moves too.
            #[test]
            fn default_storage_round_trip_moves_the_allocation() {
                for (w, h) in SIZES {
                    let img = image(w, h);
                    let ptr = img.as_raw().as_ptr();
                    let want = img.as_raw().clone();
                    let buffer: PixelBuffer = PixelBuffer::try_from(img).unwrap();
                    assert_eq!(buffer.as_slice().as_strided_bytes().as_ptr().cast(), ptr);
                    let back = ImageBuffer::<Px, Vec<$sample>>::try_from(buffer).unwrap();
                    assert_eq!(back.as_raw().as_ptr(), ptr);
                    assert_eq!(back.as_raw(), &want);
                }
            }

            /// A buffer we allocated reaches `image` without a copy.
            #[test]
            fn allocated_buffers_hand_off() {
                let d = PixelDescriptor::new_full(
                    ChannelType::$channel, ChannelLayout::$layout, $alpha,
                    TransferFunction::Srgb, ColorPrimaries::Bt709,
                );
                let buffer = PixelBuffer::new(9, 4, d);
                let ptr = buffer.as_slice().as_strided_bytes().as_ptr();
                let img = ImageBuffer::<Px, Vec<$sample>>::try_from(buffer).unwrap();
                assert_eq!(img.as_raw().as_ptr().cast(), ptr);
            }

            #[test]
            fn borrows_read_and_write_the_image() {
                for (w, h) in SIZES {
                    let mut img = image(w, h);
                    {
                        let s = PixelSlice::try_from(&img).unwrap();
                        assert_eq!(s.as_strided_bytes().as_ptr().cast(), img.as_raw().as_ptr());
                        assert_eq!((s.width(), s.rows(), s.stride()), (w, h, w as usize * BPP));
                    }
                    let mut m = PixelSliceMut::try_from(&mut img).unwrap();
                    let last = m.rows() - 1;
                    m.row_mut(last)[..BPP].copy_from_slice(&[0x5A; BPP]);
                    let px = img.get_pixel(0, h - 1);
                    assert_eq!(bytemuck::cast_slice::<$sample, u8>(px.channels()), &[0x5A; BPP]);
                }
            }

            /// `image` reads a strided slice through `FlatSamples`, and its own
            /// crop agrees with ours, with non-zero bytes in every row's padding.
            #[test]
            fn strided_views_read_the_right_pixels() {
                for (w, h) in SIZES {
                    for extra_pixels in [1usize, 2, 9] {
                        let stride_samples = (w as usize + extra_pixels) * CHANNELS;
                        let mut raw = vec![<$sample>::from(7u8); stride_samples * h as usize];
                        let tight = $fill(w as usize * h as usize * CHANNELS);
                        for (row, src) in raw
                            .chunks_exact_mut(stride_samples)
                            .zip(tight.chunks_exact(w as usize * CHANNELS))
                        {
                            row[..src.len()].copy_from_slice(src);
                        }
                        let d = PixelDescriptor::new_full(
                            ChannelType::$channel, ChannelLayout::$layout, $alpha,
                            TransferFunction::Srgb, ColorPrimaries::Bt709,
                        );
                        let buffer = PixelBuffer::try_from_parts(PixelBufferParts::new(
                            raw, w, h, stride_samples * size_of::<$sample>(), d,
                        )).unwrap();
                        let slice = buffer.as_slice();
                        let flat = FlatSamples::<&[$sample]>::try_from(slice).unwrap();
                        assert_eq!(flat.samples.as_ptr().cast(), buffer.as_slice().as_strided_bytes().as_ptr());
                        let view = flat.as_view::<Px>().unwrap();
                        let reference = ImageBuffer::<Px, _>::from_raw(w, h, tight).unwrap();
                        assert_eq!(view.dimensions(), (w, h));
                        for y in 0..h {
                            for x in 0..w {
                                assert_eq!(view.get_pixel(x, y), *reference.get_pixel(x, y), "{w}x{h} {x},{y}");
                            }
                        }
                        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
                        let cropped = image::imageops::crop_imm(&view, w - cw, h - ch, cw, ch);
                        let mut theirs = Vec::new();
                        for y in 0..ch {
                            for x in 0..cw {
                                theirs.extend_from_slice(cropped.get_pixel(x, y).channels());
                            }
                        }
                        let ours = buffer.crop_copy(w - cw, h - ch, cw, ch);
                        assert_eq!(bytemuck::cast_slice::<$sample, u8>(&theirs), ours.copy_to_contiguous_bytes());
                    }
                }
            }

            #[test]
            fn padded_rows_are_refused_not_misread() {
                let (w, h) = (5u32, 3u32);
                let d = PixelDescriptor::new_full(
                    ChannelType::$channel, ChannelLayout::$layout, $alpha,
                    TransferFunction::Srgb, ColorPrimaries::Bt709,
                );
                let stride = (w as usize + 1) * BPP;
                let raw = $fill(stride * h as usize / size_of::<$sample>());
                let ptr = raw.as_ptr();
                let buffer = PixelBuffer::try_from_parts(PixelBufferParts::new(raw, w, h, stride, d)).unwrap();
                let back = ImageBuffer::<Px, Vec<$sample>>::try_from(buffer).unwrap_err();
                assert_eq!(back.as_slice().as_strided_bytes().as_ptr().cast(), ptr);
            }

            #[test]
            fn aligned_storage_is_an_image_container() {
                let (w, h) = (37u32, 11u32);
                let d = PixelDescriptor::new_full(
                    ChannelType::$channel, ChannelLayout::$layout, $alpha,
                    TransferFunction::Srgb, ColorPrimaries::Bt709,
                );
                let mut samples = AlignedSamples::<$sample>::try_zeroed(w as usize * h as usize * CHANNELS).unwrap();
                let len = samples.len();
                samples.copy_from_slice(&$fill(len));
                let ptr = samples.as_ptr();
                assert_eq!(ptr as usize % 64, 0);
                let buffer = PixelBuffer::try_from_parts(PixelBufferParts::new(samples, w, h, w as usize * BPP, d)).unwrap();
                let img = ImageBuffer::<Px, AlignedSamples<$sample>>::try_from(buffer).unwrap();
                assert_eq!(img.as_raw().as_ptr(), ptr);
                let again: PixelBuffer<(), AlignedSamples<$sample>> = PixelBuffer::try_from(img).unwrap();
                assert_eq!(again.as_slice().as_strided_bytes().as_ptr().cast(), ptr);
            }
        }
    )*};
}

every_format! {
    luma8: Luma<u8>, fill_u8, U8, Gray, None;
    luma16: Luma<u16>, fill_u16, U16, Gray, None;
    lumaf: Luma<f32>, fill_f32, F32, Gray, None;
    luma_a8: LumaA<u8>, fill_u8, U8, GrayAlpha, Some(AlphaMode::Straight);
    luma_a16: LumaA<u16>, fill_u16, U16, GrayAlpha, Some(AlphaMode::Straight);
    luma_af: LumaA<f32>, fill_f32, F32, GrayAlpha, Some(AlphaMode::Straight);
    rgb8: Rgb<u8>, fill_u8, U8, Rgb, None;
    rgb16: Rgb<u16>, fill_u16, U16, Rgb, None;
    rgbf: Rgb<f32>, fill_f32, F32, Rgb, None;
    rgba8: Rgba<u8>, fill_u8, U8, Rgba, Some(AlphaMode::Straight);
    rgba16: Rgba<u16>, fill_u16, U16, Rgba, Some(AlphaMode::Straight);
    rgbaf: Rgba<f32>, fill_f32, F32, Rgba, Some(AlphaMode::Straight);
}

fn rgb8_labeled(prim: P, tf: Tf) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let mut img = ImageBuffer::from_raw(4, 2, fill_u8(24)).unwrap();
    img.set_rgb_primaries(prim);
    img.set_transfer_function(tf);
    img
}

#[test]
fn color_survives_both_directions() {
    let cases = [
        (
            P::SRgb,
            Tf::SRgb,
            ColorPrimaries::Bt709,
            TransferFunction::Srgb,
        ),
        (
            P::SRgb,
            Tf::Linear,
            ColorPrimaries::Bt709,
            TransferFunction::Linear,
        ),
        (
            P::SRgb,
            Tf::Bt709,
            ColorPrimaries::Bt709,
            TransferFunction::Bt709,
        ),
        (
            P::Rgb2020,
            Tf::Smpte2084,
            ColorPrimaries::Bt2020,
            TransferFunction::Pq,
        ),
        (
            P::Rgb2020,
            Tf::Bt2100Hlg,
            ColorPrimaries::Bt2020,
            TransferFunction::Hlg,
        ),
        (
            P::SmpteRp432,
            Tf::SRgb,
            ColorPrimaries::DisplayP3,
            TransferFunction::Srgb,
        ),
    ];
    for (prim, tf, want_prim, want_tf) in cases {
        let img = rgb8_labeled(prim, tf);
        let color = img.color_space();
        let slice = PixelSlice::try_from(&img).unwrap();
        assert_eq!(slice.descriptor().primaries, want_prim);
        assert_eq!(slice.descriptor().transfer, want_tf);
        assert_eq!(slice.descriptor().signal_range, SignalRange::Full);
        assert!(slice.color_context().is_none());
        let buffer: PixelBuffer = PixelBuffer::try_from(img).unwrap();
        assert_eq!(buffer.descriptor().primaries, want_prim);
        assert_eq!(buffer.descriptor().transfer, want_tf);
        assert!(buffer.color_context().is_none());
        let back = ImageBuffer::<Rgb<u8>, Vec<u8>>::try_from(buffer).unwrap();
        assert_eq!(back.color_space(), color);
    }
}

#[test]
fn color_zenpixels_cannot_name_is_kept_and_blocks_the_return_trip() {
    let img = rgb8_labeled(P::Rgb240m, Tf::Smpte240m);
    let ptr = img.as_raw().as_ptr();
    let buffer: PixelBuffer = PixelBuffer::try_from(img).unwrap();
    assert_eq!(buffer.descriptor().primaries, ColorPrimaries::Unknown);
    assert_eq!(buffer.descriptor().transfer, TransferFunction::Unknown);
    let cicp = buffer.color_context().unwrap().cicp.unwrap();
    assert_eq!(
        (cicp.color_primaries, cicp.transfer_characteristics),
        (7, 7)
    );
    let refused = ImageBuffer::<Rgb<u8>, Vec<u8>>::try_from(buffer).unwrap_err();
    assert_eq!(refused.as_slice().as_strided_bytes().as_ptr(), ptr);
}

fn rgb8_parts(d: PixelDescriptor) -> PixelBuffer {
    PixelBuffer::try_from_parts(PixelBufferParts::new(fill_u8(24).into(), 4, 2, 12, d)).unwrap()
}

#[test]
fn a_buffer_image_cannot_label_is_refused_until_relabeled() {
    // Unknown color: image would call it sRGB.
    let unknown = rgb8_parts(PixelDescriptor::RGB8);
    assert_eq!(unknown.descriptor().transfer, TransferFunction::Unknown);
    let unknown = ImageBuffer::<Rgb<u8>, Vec<u8>>::try_from(unknown).unwrap_err();
    let labeled = unknown
        .with_transfer(TransferFunction::Linear)
        .with_primaries(ColorPrimaries::Bt709);
    let img = ImageBuffer::<Rgb<u8>, Vec<u8>>::try_from(labeled).unwrap();
    assert_eq!(img.color_space().transfer, Tf::Linear);

    let srgb = PixelDescriptor::new_full(
        ChannelType::U8,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Srgb,
        ColorPrimaries::Bt709,
    );
    // Narrow range, an ICC profile, primaries image lacks, Gamma22.
    assert!(
        ImageBuffer::<Rgb<u8>, Vec<u8>>::try_from(rgb8_parts(
            srgb.with_signal_range(SignalRange::Narrow)
        ))
        .is_err()
    );
    assert!(
        ImageBuffer::<Rgb<u8>, Vec<u8>>::try_from(rgb8_parts(srgb).with_icc(vec![0u8; 4])).is_err()
    );
    assert!(
        ImageBuffer::<Rgb<u8>, Vec<u8>>::try_from(rgb8_parts(
            srgb.with_primaries(ColorPrimaries::AdobeRgb)
        ))
        .is_err()
    );
    assert!(
        ImageBuffer::<Rgb<u8>, Vec<u8>>::try_from(rgb8_parts(
            srgb.with_transfer(TransferFunction::Gamma22)
        ))
        .is_err()
    );
    // CICP that disagrees with the descriptor.
    let cicp = zenpixels::Cicp::new(9, 16, 0, true);
    assert!(ImageBuffer::<Rgb<u8>, Vec<u8>>::try_from(rgb8_parts(srgb).with_cicp(cicp)).is_err());
    // CICP that agrees passes.
    let cicp = zenpixels::Cicp::new(1, 13, 0, true);
    assert!(ImageBuffer::<Rgb<u8>, Vec<u8>>::try_from(rgb8_parts(srgb).with_cicp(cicp)).is_ok());
}

#[test]
fn format_and_alpha_mismatches_are_refused() {
    let rgba = |alpha| {
        PixelDescriptor::new_full(
            ChannelType::U8,
            ChannelLayout::Rgba,
            alpha,
            TransferFunction::Srgb,
            ColorPrimaries::Bt709,
        )
    };
    let buffer =
        |d| PixelBuffer::try_from_parts(PixelBufferParts::new(fill_u8(32), 4, 2, 16, d)).unwrap();
    assert!(
        ImageBuffer::<Rgba<u8>, Vec<u8>>::try_from(buffer(rgba(Some(AlphaMode::Straight)))).is_ok()
    );
    assert!(
        ImageBuffer::<Rgba<u8>, Vec<u8>>::try_from(buffer(rgba(Some(AlphaMode::Opaque)))).is_ok()
    );
    assert!(
        ImageBuffer::<Rgba<u8>, Vec<u8>>::try_from(buffer(rgba(Some(AlphaMode::Premultiplied))))
            .is_err()
    );
    assert!(
        ImageBuffer::<Rgba<u8>, Vec<u8>>::try_from(buffer(rgba(Some(AlphaMode::Undefined))))
            .is_err()
    );
    assert!(
        ImageBuffer::<Luma<u8>, Vec<u8>>::try_from(buffer(rgba(Some(AlphaMode::Straight))))
            .is_err()
    );
    assert!(
        ImageBuffer::<LumaA<u8>, Vec<u8>>::try_from(buffer(rgba(Some(AlphaMode::Straight))))
            .is_err()
    );
}

#[test]
fn a_prefix_before_the_first_pixel_is_refused() {
    let d = PixelDescriptor::new_full(
        ChannelType::U8,
        ChannelLayout::Rgb,
        None,
        TransferFunction::Srgb,
        ColorPrimaries::Bt709,
    );
    let mut parts = PixelBufferParts::new(fill_u8(30).into(), 4, 2, 12, d);
    parts.offset = 3;
    let buffer: PixelBuffer = PixelBuffer::try_from_parts(parts).unwrap();
    assert!(ImageBuffer::<Rgb<u8>, Vec<u8>>::try_from(buffer).is_err());
}

#[test]
fn flat_views_refuse_what_image_cannot_read() {
    let bgra = PixelDescriptor::new(
        ChannelType::U8,
        ChannelLayout::Bgra,
        Some(AlphaMode::Straight),
        TransferFunction::Srgb,
    );
    let raw = fill_u8(32);
    let slice = PixelSlice::new(&raw, 4, 2, 16, bgra).unwrap();
    assert!(FlatSamples::<&[u8]>::try_from(slice).is_err());
    let rgb16 = PixelBuffer::try_new_aligned(4, 2, PixelDescriptor::RGB16_SRGB)
        .map(|b: PixelBuffer<(), AlignedSamples<u8>>| b)
        .unwrap();
    assert!(FlatSamples::<&[u8]>::try_from(rgb16.as_slice()).is_err());
    assert!(FlatSamples::<&[f32]>::try_from(rgb16.as_slice()).is_err());
    assert!(FlatSamples::<&[u16]>::try_from(rgb16.as_slice()).is_ok());
}

macro_rules! gray_planes {
    ($($name:ident: $sample:ty, $fill:ident;)*) => {$(
        mod $name {
            use super::*;

            #[test]
            fn owned_round_trip_moves_the_allocation() {
                let img = ImageBuffer::<Luma<$sample>, _>::from_raw(9, 4, $fill(36)).unwrap();
                let ptr = img.as_raw().as_ptr();
                let plane = PlaneBuf::try_from(img).unwrap();
                assert_eq!(plane.stride(), 9);
                assert_eq!(plane.row(1).unwrap(), &$fill(36)[9..18]);
                assert_eq!(plane.row(0).unwrap().as_ptr(), ptr);
                let back = ImageBuffer::<Luma<$sample>, Vec<$sample>>::try_from(plane).unwrap();
                assert_eq!(back.as_raw().as_ptr(), ptr);
            }

            #[test]
            fn borrows_read_and_write_the_image() {
                let mut img = ImageBuffer::<Luma<$sample>, _>::from_raw(9, 4, $fill(36)).unwrap();
                assert_eq!(PlaneRef::try_from(&img).unwrap().row(0).unwrap().as_ptr(), img.as_raw().as_ptr());
                let mut plane = PlaneMut::try_from(&mut img).unwrap();
                let one = plane.row(1).unwrap()[2];
                plane.row_mut(0).unwrap()[3] = one;
                assert_eq!(img.get_pixel(3, 0).0[0], one);
            }

            #[test]
            fn strided_planes_read_through_flat_samples() {
                for width in [1usize, 2, 3, 31, 64, 65] {
                    for pad in [0usize, 1, 7, 64] {
                        let (height, stride) = (5usize, width + pad);
                        let mut raw = vec![<$sample>::from(9u8); stride * (height - 1) + width];
                        let tight = $fill(width * height);
                        for y in 0..height {
                            raw[y * stride..y * stride + width].copy_from_slice(&tight[y * width..(y + 1) * width]);
                        }
                        let plane = PlaneRef::new(&raw[..], width, height, stride).unwrap();
                        let flat = FlatSamples::<&[$sample]>::try_from(plane).unwrap();
                        assert_eq!(flat.samples.as_ptr(), raw.as_ptr());
                        let view = flat.as_view::<Luma<$sample>>().unwrap();
                        for y in 0..height {
                            for x in 0..width {
                                assert_eq!(view.get_pixel(x as u32, y as u32).0[0], tight[y * width + x]);
                            }
                        }
                    }
                }
            }

            #[test]
            fn mutable_flat_samples_write_the_plane() {
                let mut raw = vec![<$sample>::from(1u8); 5 * 16];
                let plane = PlaneMut::new(&mut raw[..], 5, 5, 16).unwrap();
                let mut flat = FlatSamples::<&mut [$sample]>::try_from(plane).unwrap();
                let mut view = flat.as_view_mut::<Luma<$sample>>().unwrap();
                view.put_pixel(4, 3, Luma([<$sample>::from(200u8)]));
                assert_eq!(raw[3 * 16 + 4], <$sample>::from(200u8));
                assert_eq!(raw[3 * 16 + 5], <$sample>::from(1u8));
            }

            /// `ImageBuffer::from_raw` alone would accept these and read wrong pixels.
            #[test]
            fn strided_and_offset_planes_are_refused_not_misread() {
                let strided = PlaneBuf::new($fill(64), 5, 4, 16).unwrap();
                let back = ImageBuffer::<Luma<$sample>, Vec<$sample>>::try_from(strided).unwrap_err();
                assert_eq!(back.stride(), 16);

                let raw = $fill(16);
                let tight = PlaneRef::new(&raw[..], 4, 4, 4).unwrap();
                let offset = tight.sub(1, 1, 2, 2).unwrap();
                assert!(ImageBuffer::<Luma<$sample>, &[$sample]>::try_from(offset).is_err());
                let whole = PlaneRef::new(&raw[..], 4, 4, 4).unwrap();
                let img = ImageBuffer::<Luma<$sample>, &[$sample]>::try_from(whole).unwrap();
                assert_eq!(img.as_raw().as_ptr(), raw.as_ptr());
            }

            #[test]
            fn aligned_planes_reach_image_when_tight() {
                let width = 64 / size_of::<$sample>();
                let plane = AlignedPlane::<$sample>::alloc(width, 3).unwrap();
                let ptr = plane.row(0).unwrap().as_ptr();
                let img = ImageBuffer::<Luma<$sample>, AlignedSamples<$sample>>::try_from(plane).unwrap();
                assert_eq!(img.as_raw().as_ptr(), ptr);
                let again: Plane<AlignedSamples<$sample>> = Plane::try_from(img).unwrap();
                assert_eq!(again.row(0).unwrap().as_ptr(), ptr);

                let padded = AlignedPlane::<$sample>::alloc(width - 1, 3).unwrap();
                assert!(ImageBuffer::<Luma<$sample>, AlignedSamples<$sample>>::try_from(padded).is_err());
            }
        }
    )*};
}

gray_planes! {
    plane8: u8, fill_u8;
    plane16: u16, fill_u16;
}

mod planef {
    use super::*;

    #[test]
    fn owned_borrowed_and_strided() {
        let img = ImageBuffer::<Luma<f32>, _>::from_raw(9, 4, fill_f32(36)).unwrap();
        let ptr = img.as_raw().as_ptr();
        assert_eq!(
            PlaneRef::try_from(&img).unwrap().row(0).unwrap().as_ptr(),
            ptr
        );
        let plane = PlaneBuf::try_from(img).unwrap();
        assert_eq!(plane.row(0).unwrap().as_ptr(), ptr);
        let back = ImageBuffer::<Luma<f32>, Vec<f32>>::try_from(plane).unwrap();
        assert_eq!(back.as_raw().as_ptr(), ptr);

        let mut raw = vec![f32::NAN; 16 * 3 + 5];
        for y in 0..4 {
            raw[y * 16..y * 16 + 5].copy_from_slice(&fill_f32(20)[y * 5..(y + 1) * 5]);
        }
        let plane = PlaneRef::new(&raw[..], 5, 4, 16).unwrap();
        let flat = FlatSamples::<&[f32]>::try_from(plane).unwrap();
        let view = flat.as_view::<Luma<f32>>().unwrap();
        assert_eq!(view.get_pixel(4, 3).0[0], fill_f32(20)[19]);
    }
}

#[test]
fn storage_trait_is_met_by_the_image_container_types() {
    fn storage<S: SampleStorage>(_: &S) {}
    storage(&vec![0u8; 1]);
    storage(&AlignedSamples::<f32>::try_zeroed(1).unwrap());
}

macro_rules! dynamic_images {
    ($($variant:ident($pixel:ident<$sample:ty>), $fill:ident, $color:ident;)*) => {
        #[test]
        fn dynamic_images_move_borrow_and_return_in_every_variant() {
            $({
                let channels = usize::from(<$pixel<$sample> as Pixel>::CHANNEL_COUNT);
                let raw = $fill(9 * 4 * channels);
                let mut image = DynamicImage::$variant(ImageBuffer::from_raw(9, 4, raw.clone()).unwrap());
                assert_eq!(image.color(), ColorType::$color);
                let slice = PixelSlice::try_from(&image).unwrap();
                assert_eq!(slice.as_strided_bytes(), bytemuck::cast_slice::<$sample, u8>(&raw));
                assert_eq!(slice.descriptor().format, PixelFormat::try_from(ColorType::$color).unwrap());
                {
                    let mut m = PixelSliceMut::try_from(&mut image).unwrap();
                    assert_eq!(m.row_mut(0).len(), 9 * channels * size_of::<$sample>());
                }
                let moved_from = match &image {
                    DynamicImage::$variant(image) => image.as_raw().as_ptr(),
                    _ => panic!("wrong variant"),
                };
                let buffer = PixelBuffer::try_from(image).unwrap();
                assert_eq!(buffer_ptr::<$sample>(&buffer), moved_from);
                let back = DynamicImage::try_from(buffer).unwrap();
                assert_eq!(back.color(), ColorType::$color);
                match back {
                    DynamicImage::$variant(image) => assert_eq!(image.as_raw(), &raw),
                    _ => panic!("wrong variant"),
                }
            })*
        }
    };
}

fn buffer_ptr<T>(buffer: &PixelBuffer) -> *const T {
    buffer.as_slice().as_strided_bytes().as_ptr().cast()
}

dynamic_images! {
    ImageLuma8(Luma<u8>), fill_u8, L8;
    ImageLumaA8(LumaA<u8>), fill_u8, La8;
    ImageRgb8(Rgb<u8>), fill_u8, Rgb8;
    ImageRgba8(Rgba<u8>), fill_u8, Rgba8;
    ImageLuma16(Luma<u16>), fill_u16, L16;
    ImageLumaA16(LumaA<u16>), fill_u16, La16;
    ImageRgb16(Rgb<u16>), fill_u16, Rgb16;
    ImageRgba16(Rgba<u16>), fill_u16, Rgba16;
    ImageRgb32F(Rgb<f32>), fill_f32, Rgb32F;
    ImageRgba32F(Rgba<f32>), fill_f32, Rgba32F;
}

#[test]
fn dynamic_image_refuses_a_format_it_has_no_variant_for() {
    let d = PixelDescriptor::new_full(
        ChannelType::F32,
        ChannelLayout::Gray,
        None,
        TransferFunction::Linear,
        ColorPrimaries::Bt709,
    );
    assert!(DynamicImage::try_from(PixelBuffer::new(2, 2, d)).is_err());
}

#[test]
fn every_color_type_maps_to_a_format_and_back() {
    use ColorType::*;
    let all = [
        L8, La8, Rgb8, Rgba8, L16, La16, Rgb16, Rgba16, Rgb32F, Rgba32F,
    ];
    for color in all {
        let format = PixelFormat::try_from(color).unwrap();
        assert_eq!(ColorType::try_from(format).unwrap(), color);
        assert_eq!(
            usize::from(color.bytes_per_pixel()),
            format.bytes_per_pixel(),
            "{color:?}"
        );
        let extended = ExtendedColorType::try_from(format).unwrap();
        assert_eq!(extended, ExtendedColorType::from(color));
    }
}

#[test]
fn extended_color_types_map_or_refuse_explicitly() {
    use ExtendedColorType as E;
    let mapped = [
        (E::L8, PixelFormat::Gray8),
        (E::La8, PixelFormat::GrayA8),
        (E::Rgb8, PixelFormat::Rgb8),
        (E::Rgba8, PixelFormat::Rgba8),
        (E::L16, PixelFormat::Gray16),
        (E::La16, PixelFormat::GrayA16),
        (E::Rgb16, PixelFormat::Rgb16),
        (E::Rgba16, PixelFormat::Rgba16),
        (E::Bgra8, PixelFormat::Bgra8),
        (E::Rgb32F, PixelFormat::RgbF32),
        (E::Rgba32F, PixelFormat::RgbaF32),
        (E::Cmyk8, PixelFormat::Cmyk8),
    ];
    for (color, format) in mapped {
        assert_eq!(PixelFormat::try_from(color).unwrap(), format);
        assert_eq!(E::try_from(format).unwrap(), color);
    }
    let refused = [
        E::A8,
        E::L1,
        E::La1,
        E::Rgb1,
        E::Rgba1,
        E::L2,
        E::La2,
        E::Rgb2,
        E::Rgba2,
        E::L4,
        E::La4,
        E::Rgb4,
        E::Rgba4,
        E::Rgb5x1,
        E::Bgr8,
        E::Cmyk16,
        E::Unknown(7),
    ];
    for color in refused {
        assert!(PixelFormat::try_from(color).is_err(), "{color:?}");
    }
    assert!(ColorType::try_from(PixelFormat::GrayF32).is_err());
    assert!(ColorType::try_from(PixelFormat::Bgra8).is_err());
}
