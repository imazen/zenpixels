//! Conversion, orientation and scanning over rows whose stride isn't a whole number of pixels
//! (imazen/zenpixels#91): the same pixels, packed, are the oracle for every result, and padding
//! is poisoned so a kernel that reads it changes the answer.

use whereat::At;
use zenpixels::plane::{AnyPlane, PlaneRef};
use zenpixels::sample::{ChromaSiting, ChromaSubsampling, CodePlane, SampleEncoding};
use zenpixels::{
    AlphaMode, ChannelType, Cicp, ColorOrigin, Orientation, PixelBuffer, PixelBufferParts,
    PixelDescriptor as D, PixelFormat, PixelSlice, PixelSliceMut, SignalRange, TransferFunction,
};
use zenpixels_convert::adapt::adapt_raw_rows_for_encode;
use zenpixels_convert::codes::to_canonical;
use zenpixels_convert::ycbcr::YcbcrToRgbF32;
use zenpixels_convert::{
    ConvertError, ConvertOptions, OutputPlan, OutputProfile, PixelBufferConvertExt,
    PixelBufferLoadBearingExt, PixelSliceConvertExt, PixelSliceLoadBearingExt, RowConverter,
};

const PAD: u8 = 0xEE;

/// A stride with `extra` bytes after each row, at the sample size or a multiple of it.
fn stride_of(d: D, width: u32, extra: usize) -> usize {
    width as usize * d.bytes_per_pixel() + extra * d.min_alignment()
}

/// `(descriptor, width, height, stride)`. Sizes reach the SIMD and tiled kernels.
fn cases() -> Vec<(D, u32, u32, usize)> {
    let mut cases = vec![
        // Both real cases in the issue.
        (D::RGB8_SRGB, 2, 3, 8),
        (D::RGBF32_LINEAR, 5, 3, 76),
    ];
    for &(width, height) in &[(5u32, 3u32), (37, 19), (67, 67)] {
        for d in [D::RGB8_SRGB, D::RGB16_SRGB, D::RGBF32_LINEAR] {
            for extra in [1, 2, 5] {
                let stride = stride_of(d, width, extra);
                if !is_whole_pixels(d, stride) {
                    cases.push((d, width, height, stride));
                }
            }
        }
    }
    cases
}

fn is_whole_pixels(d: D, stride: usize) -> bool {
    stride.is_multiple_of(d.bytes_per_pixel())
}

/// Samples for `d`: varied, and in range for a float.
fn packed_pixels(d: D, width: u32, height: u32) -> Vec<u32> {
    let row = width as usize * d.bytes_per_pixel();
    let samples = row * height as usize / d.min_alignment();
    let mut state = 0x2545_F491u32;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let mut words = vec![0u32; (row * height as usize).div_ceil(4)];
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut words);
    for i in 0..samples {
        match d.channel_type() {
            ChannelType::U8 => bytes[i] = next() as u8,
            ChannelType::U16 => {
                bytes[i * 2..i * 2 + 2].copy_from_slice(&(next() as u16).to_ne_bytes())
            }
            ChannelType::F32 => bytes[i * 4..i * 4 + 4]
                .copy_from_slice(&(((next() >> 8) as f32) / 16_777_215.0).to_ne_bytes()),
            other => panic!("no fixture for {other:?}"),
        }
    }
    words
}

/// The packed pixels at `stride`, with `PAD` in every other byte.
fn pad(packed: &[u32], d: D, width: u32, height: u32, stride: usize) -> Vec<u32> {
    let row = width as usize * d.bytes_per_pixel();
    let total = (height as usize - 1) * stride + row;
    let mut words = vec![u32::from_ne_bytes([PAD; 4]); total.div_ceil(4)];
    let source: &[u8] = bytemuck::cast_slice(packed);
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut words);
    for y in 0..height as usize {
        bytes[y * stride..y * stride + row].copy_from_slice(&source[y * row..(y + 1) * row]);
    }
    words
}

fn view<'a>(words: &'a [u32], d: D, width: u32, height: u32, stride: usize) -> PixelSlice<'a> {
    let row = width as usize * d.bytes_per_pixel();
    let bytes = bytemuck::cast_slice::<u32, u8>(words);
    PixelSlice::new(
        &bytes[..(height as usize - 1) * stride + row],
        width,
        height,
        stride,
        d,
    )
    .unwrap()
}

fn owned(words: Vec<u32>, d: D, width: u32, height: u32, stride: usize) -> PixelBuffer {
    PixelBuffer::try_from_parts(PixelBufferParts::new(
        words.into(),
        width,
        height,
        stride,
        d,
    ))
    .unwrap()
}

fn tight(buffer: &PixelBuffer) -> Vec<u8> {
    buffer.copy_to_contiguous_bytes()
}

fn result_bytes(result: Result<PixelBuffer, At<ConvertError>>) -> Result<Vec<u8>, ConvertError> {
    result.map(|b| tight(&b)).map_err(|e| e.error().clone())
}

fn targets(d: D) -> Vec<D> {
    let mut targets = vec![
        D::RGBA8_SRGB,
        D::BGRA8_SRGB,
        D::RGB8_SRGB,
        D::RGB16_SRGB,
        D::RGBA16_SRGB,
        D::RGBF32_LINEAR,
        D::RGBAF32_LINEAR,
        D::GRAY8_SRGB,
    ];
    targets.retain(|t| *t != d);
    targets.push(d);
    targets
}

fn options() -> ConvertOptions {
    ConvertOptions::permissive()
}

#[test]
fn conversion_of_a_byte_stride_matches_packed() {
    for (d, w, h, stride) in cases() {
        assert!(!is_whole_pixels(d, stride), "{d:?} {w} {stride}");
        let packed = packed_pixels(d, w, h);
        let padded = pad(&packed, d, w, h, stride);
        let row = w as usize * d.bytes_per_pixel();
        let a = view(&packed, d, w, h, row);
        let b = view(&padded, d, w, h, stride);
        for target in targets(d) {
            let want = result_bytes(a.convert_to(target));
            assert_eq!(
                result_bytes(b.convert_to(target)),
                want,
                "{d:?} -> {target:?} {w}x{h}"
            );
        }
    }
}

/// Converting pieces of the image, split at any row, gives the same pixels as the whole.
#[test]
fn row_splits_do_not_change_conversion() {
    for (d, w, h, stride) in cases() {
        let packed = packed_pixels(d, w, h);
        let padded = pad(&packed, d, w, h, stride);
        let b = view(&padded, d, w, h, stride);
        for target in [D::RGBA8_SRGB, D::RGBF32_LINEAR] {
            if target == d {
                continue;
            }
            let whole = tight(&b.convert_to(target).unwrap());
            for split in [1, h / 2, h - 1] {
                let mut joined = tight(&b.sub_rows(0, split).convert_to(target).unwrap());
                joined.extend(tight(
                    &b.sub_rows(split, h - split).convert_to(target).unwrap(),
                ));
                assert_eq!(joined, whole, "{d:?} -> {target:?} split {split}");
            }
        }
    }
}

/// A byte-strided destination gets the same pixels and keeps its padding.
#[test]
fn convert_into_a_byte_strided_destination_leaves_padding() {
    for (d, w, h, stride) in cases() {
        let packed = packed_pixels(d, w, h);
        let padded = pad(&packed, d, w, h, stride);
        let source = view(&padded, d, w, h, stride);
        for (target, extra) in [(D::RGB8_SRGB, 7), (D::RGB16_SRGB, 2), (D::RGBF32_LINEAR, 5)] {
            let want = tight(
                &view(&packed, d, w, h, w as usize * d.bytes_per_pixel())
                    .convert_to(target)
                    .unwrap(),
            );
            let row = w as usize * target.bytes_per_pixel();
            let dst_stride = stride_of(target, w, extra);
            let total = (h as usize - 1) * dst_stride + row;
            let mut out = vec![u32::from_ne_bytes([PAD; 4]); total.div_ceil(4)];
            let bytes = bytemuck::cast_slice_mut::<u32, u8>(&mut out);
            let dst = PixelSliceMut::new(&mut bytes[..total], w, h, dst_stride, target).unwrap();
            let done = source.convert_into(dst).unwrap();
            assert_eq!(done.stride(), dst_stride);
            drop(done);
            for y in 0..h as usize {
                let line = &bytes[y * dst_stride..][..row];
                assert_eq!(
                    line,
                    &want[y * row..(y + 1) * row],
                    "{d:?} -> {target:?} row {y}"
                );
                if y + 1 < h as usize {
                    assert!(
                        bytes[y * dst_stride + row..(y + 1) * dst_stride]
                            .iter()
                            .all(|&b| b == PAD)
                    );
                }
            }
        }
    }
}

#[test]
fn row_converter_rows_match_packed() {
    for (d, w, h, stride) in cases() {
        let packed = packed_pixels(d, w, h);
        let padded = pad(&packed, d, w, h, stride);
        let row = w as usize * d.bytes_per_pixel();
        for (target, dst_stride) in [
            (D::RGBA8_SRGB, 4 * w as usize + 3),
            (D::RGB8_SRGB, 3 * w as usize + 1),
            (D::RGBF32_LINEAR, 12 * w as usize + 4),
        ] {
            let mut converter = RowConverter::new_explicit(d, target, &options()).unwrap();
            let out_row = w as usize * target.bytes_per_pixel();
            let total = (h as usize - 1) * dst_stride + out_row;
            let mut got = vec![u32::from_ne_bytes([PAD; 4]); total.div_ceil(4)];
            let got_bytes = bytemuck::cast_slice_mut::<u32, u8>(&mut got);
            converter
                .convert_rows(
                    &bytemuck::cast_slice::<u32, u8>(&padded)[..(h as usize - 1) * stride + row],
                    stride,
                    &mut got_bytes[..total],
                    dst_stride,
                    w,
                    h,
                )
                .unwrap();
            let mut want = vec![0u32; (out_row * h as usize).div_ceil(4)];
            let mut converter = RowConverter::new_explicit(d, target, &options()).unwrap();
            converter
                .convert_rows(
                    bytemuck::cast_slice(&packed),
                    row,
                    bytemuck::cast_slice_mut(&mut want),
                    out_row,
                    w,
                    h,
                )
                .unwrap();
            let want_bytes = bytemuck::cast_slice::<u32, u8>(&want);
            for y in 0..h as usize {
                assert_eq!(
                    &got_bytes[y * dst_stride..][..out_row],
                    &want_bytes[y * out_row..(y + 1) * out_row],
                    "{d:?} -> {target:?} row {y}"
                );
                if y + 1 < h as usize {
                    assert!(
                        got_bytes[y * dst_stride + out_row..(y + 1) * dst_stride]
                            .iter()
                            .all(|&b| b == PAD)
                    );
                }
            }
        }
    }
}

#[test]
fn every_orientation_matches_packed() {
    for (d, w, h, stride) in cases() {
        let packed = packed_pixels(d, w, h);
        let padded = pad(&packed, d, w, h, stride);
        let row = w as usize * d.bytes_per_pixel();
        let a = view(&packed, d, w, h, row);
        let b = view(&padded, d, w, h, stride);
        for orientation in Orientation::ALL {
            let want = a.oriented(orientation).unwrap();
            let got = b.oriented(orientation).unwrap();
            assert_eq!(got.width(), want.width());
            assert_eq!(
                tight(&got),
                tight(&want),
                "{d:?} {w}x{h} {stride} {orientation:?}"
            );

            // Into a caller's byte-strided buffer, with padding left alone.
            let (ow, oh) = (want.width(), want.height());
            let extra = if d.min_alignment() == 1 { 5 } else { 1 };
            let out_stride = stride_of(d, ow, extra);
            let out_row = ow as usize * d.bytes_per_pixel();
            let total = (oh as usize - 1) * out_stride + out_row;
            let mut out = vec![u32::from_ne_bytes([PAD; 4]); total.div_ceil(4)];
            let bytes = bytemuck::cast_slice_mut::<u32, u8>(&mut out);
            let dst = PixelSliceMut::new(&mut bytes[..total], ow, oh, out_stride, d).unwrap();
            b.oriented_into(orientation, dst).unwrap();
            let want_bytes = tight(&want);
            for y in 0..oh as usize {
                assert_eq!(
                    &bytes[y * out_stride..][..out_row],
                    &want_bytes[y * out_row..(y + 1) * out_row]
                );
                if y + 1 < oh as usize {
                    assert!(
                        bytes[y * out_stride + out_row..(y + 1) * out_stride]
                            .iter()
                            .all(|&b| b == PAD)
                    );
                }
            }

            // In place, in the buffer's own storage.
            let buffer = owned(padded.clone(), d, w, h, stride);
            let moved = buffer.into_oriented(orientation).unwrap();
            assert_eq!(tight(&moved), want_bytes, "{d:?} in place {orientation:?}");
        }
    }
}

#[test]
fn owned_conversions_match_packed() {
    for (d, w, h, stride) in cases() {
        let packed = packed_pixels(d, w, h);
        let padded = pad(&packed, d, w, h, stride);
        let row = w as usize * d.bytes_per_pixel();
        let a = owned(packed, d, w, h, row);
        let b = owned(padded, d, w, h, stride);
        assert_eq!(
            result_bytes(b.try_add_alpha()),
            result_bytes(a.try_add_alpha()),
            "{d:?}"
        );
        assert_eq!(
            result_bytes(b.try_widen_to_u16()),
            result_bytes(a.try_widen_to_u16())
        );
        assert_eq!(
            result_bytes(b.try_narrow_to_u8()),
            result_bytes(a.try_narrow_to_u8())
        );
        assert_eq!(result_bytes(b.linearize()), result_bytes(a.linearize()));
        assert_eq!(
            result_bytes(b.delinearize(TransferFunction::Srgb)),
            result_bytes(a.delinearize(TransferFunction::Srgb))
        );
        assert_eq!(
            result_bytes(b.convert_to(D::RGBA8_SRGB)),
            result_bytes(a.convert_to(D::RGBA8_SRGB))
        );
    }
}

/// Fills the alpha or chroma test with padding that would flip the answer if read.
#[test]
fn scans_ignore_padding() {
    // Opaque gray pixels in rows that aren't a whole number of pixels apart, and padding of 1,
    // which is neither opaque nor gray next to its neighbors.
    let (w, h) = (4u32, 3u32);
    for stride in [17usize, 18, 19, 21, 67] {
        let mut bytes = vec![0u8; 2 * stride + 16];
        for y in 0..h as usize {
            bytes[y * stride..y * stride + 16]
                .as_chunks_mut::<4>()
                .0
                .fill([9, 9, 9, 255]);
            if y + 1 < h as usize {
                bytes[y * stride + 16..(y + 1) * stride].fill(1);
            }
        }
        let opaque = PixelSlice::new(&bytes, w, h, stride, D::RGBA8_SRGB).unwrap();
        opaque.check_opaque().unwrap();
        let report = opaque.determine_load_bearing();
        assert_eq!(report.uses_alpha, Some(false), "stride {stride}");
        assert_eq!(report.uses_chroma, Some(false), "stride {stride}");
        let reduced = opaque.try_reduce_to_load_bearing_format().unwrap();
        assert_eq!(reduced.descriptor().layout(), D::GRAY8_SRGB.layout());
        for y in 0..h {
            assert_eq!(reduced.as_slice().row(y), &[9u8; 4]);
        }

        // Change the last pixel only: each answer must see it.
        let last = 2 * stride + 12;
        bytes[last + 3] = 254;
        bytes[last] = 200;
        let view = PixelSlice::new(&bytes, w, h, stride, D::RGBA8_SRGB).unwrap();
        assert!(view.check_opaque().is_err());
        let report = view.determine_load_bearing();
        assert_eq!(report.uses_alpha, Some(true));
        assert_eq!(report.uses_chroma, Some(true));
    }
}

#[test]
fn in_place_reduction_matches_the_allocating_path() {
    for stride in [17usize, 18, 19, 21, 67] {
        let (w, h) = (4u32, 3u32);
        let mut words = vec![u32::from_ne_bytes([PAD; 4]); (2 * stride + 16).div_ceil(4)];
        let bytes = bytemuck::cast_slice_mut::<u32, u8>(&mut words);
        for y in 0..h as usize {
            for x in 0..w as usize {
                let v = (y * 4 + x + 20) as u8;
                bytes[y * stride + x * 4..y * stride + x * 4 + 4].copy_from_slice(&[v, v, v, 255]);
            }
        }
        let want = view(&words, D::RGBA8_SRGB, w, h, stride)
            .try_reduce_to_load_bearing_format()
            .unwrap();
        let mut buffer = owned(words, D::RGBA8_SRGB, w, h, stride);
        buffer.reduce_to_load_bearing_format_in_place(true);
        assert_eq!(buffer.descriptor(), want.descriptor());
        assert_eq!(tight(&buffer), tight(&want), "stride {stride}");
    }
}

/// RGBA↔BGRA swaps through garb's strided kernel: it must not refuse this geometry, because the
/// descriptor would change while the bytes didn't.
#[test]
fn in_place_swap_moves_the_samples_and_not_the_padding() {
    for (w, h, stride) in [
        (2u32, 3u32, 9usize),
        (5, 4, 21),
        (37, 3, 150),
        (37, 3, 151),
        (37, 3, 155),
    ] {
        let d = D::RGBA8_SRGB;
        let row = w as usize * 4;
        let packed = packed_pixels(d, w, h);
        let padded = pad(&packed, d, w, h, stride);
        let want = tight(
            &view(&packed, d, w, h, row)
                .convert_to(D::BGRA8_SRGB)
                .unwrap(),
        );
        let mut buffer = owned(padded, d, w, h, stride);
        buffer.try_adapt_in_place(D::BGRA8_SRGB).unwrap();
        assert_eq!(buffer.descriptor(), D::BGRA8_SRGB);
        assert_eq!(buffer.stride(), stride);
        assert_eq!(tight(&buffer), want, "{w}x{h} stride {stride}");
        let slice = buffer.as_slice();
        let bytes = slice.as_strided_bytes();
        for y in 0..h as usize - 1 {
            assert!(
                bytes[y * stride + row..(y + 1) * stride]
                    .iter()
                    .all(|&b| b == PAD)
            );
        }
    }
}

/// Dropping a lane rounds the stride down to whole target pixels, so what could be viewed
/// through `imgref` before still can.
#[test]
fn in_place_lane_drop_from_a_byte_stride() {
    let source = D::RGBA8_SRGB.with_alpha(Some(AlphaMode::Opaque));
    for (w, h, stride) in [(2u32, 3u32, 9usize), (2, 3, 10), (5, 4, 21), (37, 3, 150)] {
        let mut words = vec![
            u32::from_ne_bytes([PAD; 4]);
            ((h as usize - 1) * stride + w as usize * 4).div_ceil(4)
        ];
        let bytes = bytemuck::cast_slice_mut::<u32, u8>(&mut words);
        let mut want = Vec::new();
        for y in 0..h as usize {
            for x in 0..w as usize {
                let p = [(y * 9 + x) as u8, (y + x * 7) as u8, (y * x + 3) as u8];
                want.extend_from_slice(&p);
                bytes[y * stride + x * 4..y * stride + x * 4 + 4]
                    .copy_from_slice(&[p[0], p[1], p[2], 255]);
            }
        }
        let mut buffer = owned(words, source, w, h, stride);
        buffer.try_adapt_in_place(D::RGB8_SRGB).unwrap();
        assert_eq!(buffer.descriptor(), D::RGB8_SRGB);
        assert_eq!(buffer.stride(), stride - stride % 3);
        assert_eq!(tight(&buffer), want, "{w}x{h} stride {stride}");
    }
}

#[test]
fn adapt_for_encode_borrows_a_byte_stride_and_converts_it_correctly() {
    for (d, w, h, stride) in cases() {
        let packed = packed_pixels(d, w, h);
        let padded = pad(&packed, d, w, h, stride);
        let row = w as usize * d.bytes_per_pixel();
        let a = view(&packed, d, w, h, row);
        let b = view(&padded, d, w, h, stride);
        let same = b.adapt_for_encode(&[d], &options()).unwrap();
        assert!(
            same.is_borrowed(),
            "{d:?}: a format the encoder takes needs no copy"
        );
        let same = same.as_slice();
        assert_eq!(same.stride(), stride);
        assert_eq!(
            same.as_strided_bytes().as_ptr(),
            b.as_strided_bytes().as_ptr()
        );

        let target = if d == D::RGBA8_SRGB {
            D::RGB8_SRGB
        } else {
            D::RGBA8_SRGB
        };
        let got = b.adapt_for_encode(&[target], &options()).unwrap();
        let want = a.adapt_for_encode(&[target], &options()).unwrap();
        assert_eq!(
            got.as_slice().contiguous_bytes().as_ref(),
            want.as_slice().contiguous_bytes().as_ref(),
            "{d:?} -> {target:?}"
        );
    }
}

/// Rows off a sample boundary have no view at all, so the raw path copies them.
#[test]
fn raw_rows_off_a_sample_boundary_are_copied() {
    let d = D::RGB16_SRGB;
    let (w, h) = (3u32, 3u32);
    let row = w as usize * 6;
    let stride = row + 3;
    let packed = packed_pixels(d, w, h);
    let source = &bytemuck::cast_slice::<u32, u8>(&packed)[..row * h as usize];
    let mut raw = vec![PAD; 2 * stride + row];
    // Rows start at 0, 21 and 42: whole bytes, odd and even, so some u16 can't sit there.
    for y in 0..h as usize {
        raw[y * stride..y * stride + row].copy_from_slice(&source[y * row..(y + 1) * row]);
    }
    let adapted = adapt_raw_rows_for_encode(&raw, d, w, h, stride, &[d], &options()).unwrap();
    assert!(adapted.is_owned());
    assert_eq!(adapted.as_slice().contiguous_bytes().as_ref(), source);

    // The same bytes, one sample of padding later, are a view.
    let stride = row + 2;
    let words = pad(&packed, d, w, h, stride);
    let bytes = &bytemuck::cast_slice::<u32, u8>(&words)[..2 * stride + row];
    let adapted = adapt_raw_rows_for_encode(bytes, d, w, h, stride, &[d], &options()).unwrap();
    assert!(adapted.is_borrowed());
    assert_eq!(adapted.as_slice().contiguous_bytes().as_ref(), source);
}

#[test]
fn output_plan_rows_match_packed() {
    let profile = OutputProfile::Named(Cicp::SRGB);
    for (d, w, h, stride) in cases() {
        let source = d.with_transfer(TransferFunction::Srgb);
        let packed = packed_pixels(source, w, h);
        let padded = pad(&packed, source, w, h, stride);
        let row = w as usize * source.bytes_per_pixel();
        let a = view(&packed, source, w, h, row);
        let b = view(&padded, source, w, h, stride);
        let format = match source.channel_type() {
            ChannelType::U8 => PixelFormat::Rgba8,
            ChannelType::U16 => PixelFormat::Rgba16,
            _ => PixelFormat::RgbaF32,
        };
        let mut plan = OutputPlan::new(
            source,
            None,
            &ColorOrigin::assumed(),
            profile.clone(),
            format,
            &ConvertOptions::retain_channels(),
            None,
        )
        .unwrap();
        let target = plan.output_descriptor();
        let out_row = w as usize * target.bytes_per_pixel();
        let mut want = vec![0u32; (out_row * h as usize).div_ceil(4)];
        plan.convert_rows(
            a.clone(),
            PixelSliceMut::new(bytemuck::cast_slice_mut(&mut want), w, h, out_row, target).unwrap(),
        )
        .unwrap();
        let dst_stride = stride_of(target, w, 3);
        let total = (h as usize - 1) * dst_stride + out_row;
        let mut got = vec![u32::from_ne_bytes([PAD; 4]); total.div_ceil(4)];
        let got_bytes = bytemuck::cast_slice_mut::<u32, u8>(&mut got);
        plan.convert_rows(
            b.clone(),
            PixelSliceMut::new(&mut got_bytes[..total], w, h, dst_stride, target).unwrap(),
        )
        .unwrap();
        let want_bytes = bytemuck::cast_slice::<u32, u8>(&want);
        for y in 0..h as usize {
            assert_eq!(
                &got_bytes[y * dst_stride..][..out_row],
                &want_bytes[y * out_row..(y + 1) * out_row],
                "{source:?} row {y}"
            );
            if y + 1 < h as usize {
                assert!(
                    got_bytes[y * dst_stride + out_row..(y + 1) * dst_stride]
                        .iter()
                        .all(|&b| b == PAD)
                );
            }
        }

        // An identity plan hands the rows straight back, padding and all.
        let plan = OutputPlan::new_with_descriptor(
            source,
            None,
            &ColorOrigin::assumed(),
            profile.clone(),
            source,
            &ConvertOptions::retain_channels(),
            None,
        )
        .unwrap();
        let borrowed = plan.borrow_rows(b.clone()).unwrap().unwrap();
        assert_eq!(borrowed.stride(), stride);
        assert_eq!(
            borrowed.as_strided_bytes().as_ptr(),
            b.as_strided_bytes().as_ptr()
        );
    }
}

/// Native components widen into a destination whose rows aren't whole pixels.
#[test]
fn to_canonical_writes_samples_and_leaves_padding() {
    let (w, h) = (5usize, 3usize);
    let d = D::RGB16;
    let ten = SampleEncoding::new(ChannelType::U16, 10, 0).unwrap();
    let source: Vec<u16> = (0..w * h * 3).map(|i| ((i * 37) % 1024) as u16).collect();
    let planes = [0usize, 1, 2].map(|c| {
        CodePlane::component(
            AnyPlane::U16(PlaneRef::new(&source[..], w * 3, h, w * 3).unwrap()),
            3,
            c,
            ten,
        )
        .unwrap()
    });

    let mut want = vec![0u16; w * h * 3];
    to_canonical(
        &planes,
        SignalRange::Full,
        0,
        0,
        PixelSliceMut::new(
            bytemuck::cast_slice_mut(&mut want),
            w as u32,
            h as u32,
            w * 6,
            d,
        )
        .unwrap(),
    )
    .unwrap();

    let stride = w * 6 + 2;
    let mut out = vec![0xABABu16; (h - 1) * stride / 2 + w * 3];
    to_canonical(
        &planes,
        SignalRange::Full,
        0,
        0,
        PixelSliceMut::new(
            bytemuck::cast_slice_mut(&mut out),
            w as u32,
            h as u32,
            stride,
            d,
        )
        .unwrap(),
    )
    .unwrap();
    for y in 0..h {
        let start = y * stride / 2;
        assert_eq!(
            &out[start..start + w * 3],
            &want[y * w * 3..(y + 1) * w * 3]
        );
        if y + 1 < h {
            assert_eq!(out[start + w * 3], 0xABAB);
        }
    }
}

/// YCbCr writes into a destination with the native JXL shape: f32 RGB with 16 bytes of padding.
#[test]
fn ycbcr_writes_samples_and_leaves_padding() {
    let (w, h) = (5usize, 3usize);
    let y: Vec<u8> = (0..w * h).map(|i| (i * 17) as u8).collect();
    let u: Vec<u8> = (0..w * h).map(|i| (i * 41) as u8).collect();
    let v: Vec<u8> = (0..w * h)
        .map(|i| 255usize.wrapping_sub(i * 31) as u8)
        .collect();
    let enc = SampleEncoding::new(ChannelType::U8, 8, 0).unwrap();
    let planes = [&y, &u, &v].map(|p| {
        CodePlane::new(AnyPlane::U8(PlaneRef::new(&p[..], w, h, w).unwrap()), enc).unwrap()
    });
    let plan = YcbcrToRgbF32::new(
        planes,
        ChromaSubsampling::Yuv444,
        ChromaSiting::Center,
        Cicp::new(1, 13, 1, true),
    )
    .unwrap();
    let d = plan.output_descriptor();
    let row = w * 12;
    let mut want = vec![0f32; w * h * 3];
    plan.convert_rows(
        0,
        0,
        PixelSliceMut::new(
            bytemuck::cast_slice_mut(&mut want),
            w as u32,
            h as u32,
            row,
            d,
        )
        .unwrap(),
    )
    .unwrap();
    let stride = row + 16;
    let sentinel = f32::from_bits(0x7fc1_2345);
    let mut out = vec![sentinel; (h - 1) * stride / 4 + w * 3];
    plan.convert_rows(
        0,
        0,
        PixelSliceMut::new(
            bytemuck::cast_slice_mut(&mut out),
            w as u32,
            h as u32,
            stride,
            d,
        )
        .unwrap(),
    )
    .unwrap();
    for line in 0..h {
        let start = line * stride / 4;
        for (a, b) in out[start..start + w * 3].iter().zip(&want[line * w * 3..]) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        if line + 1 < h {
            assert!(
                out[start + w * 3..start + stride / 4]
                    .iter()
                    .all(|v| v.to_bits() == 0x7fc1_2345)
            );
        }
    }
}

#[cfg(feature = "hdr-experimental")]
mod hdr {
    use super::*;
    use zenpixels::hdr::{ContentLightLevel, DiffuseWhite};
    use zenpixels_convert::hdr::{CllMeasure, LightLevelMethod};

    /// Padding holds the largest value in the image, so a scan that reads it reports it.
    #[test]
    fn light_level_measures_ignore_padding() {
        let d = D::RGBF32_LINEAR;
        let (w, h) = (5u32, 3u32);
        let row = w as usize * 12;
        let packed: Vec<f32> = (0..15 * 3).map(|i| 0.01 * i as f32).collect();
        let a = PixelSlice::new(bytemuck::cast_slice(&packed), w, h, row, d).unwrap();
        for stride in [row + 4, row + 8, row + 16, row + 40] {
            let mut words = vec![1000.0f32; (2 * stride + row) / 4];
            for y in 0..h as usize {
                words[y * stride / 4..y * stride / 4 + 15]
                    .copy_from_slice(&packed[y * 15..(y + 1) * 15]);
            }
            let b = PixelSlice::new(bytemuck::cast_slice(&words), w, h, stride, d).unwrap();
            let white = DiffuseWhite::new(203.0);
            for method in [LightLevelMethod::MaxRgb, LightLevelMethod::LuminanceBt2020] {
                let same = |x: Option<ContentLightLevel>, y: Option<ContentLightLevel>| {
                    assert_eq!(x.is_some(), y.is_some());
                    if let (Some(x), Some(y)) = (x, y) {
                        assert_eq!(x.max_content_light_level, y.max_content_light_level);
                        assert_eq!(
                            x.max_frame_average_light_level,
                            y.max_frame_average_light_level
                        );
                    }
                };
                same(
                    ContentLightLevel::measure_max(b.clone(), white, method),
                    ContentLightLevel::measure_max(a.clone(), white, method),
                );
                same(
                    ContentLightLevel::measure_robust(b.clone(), white, method),
                    ContentLightLevel::measure_robust(a.clone(), white, method),
                );
                same(
                    ContentLightLevel::measure_max_smoothed(b.clone(), white, method),
                    ContentLightLevel::measure_max_smoothed(a.clone(), white, method),
                );
            }
        }
    }
}
