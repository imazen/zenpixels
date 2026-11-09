//! The precision rule holds up against the real kernels: where it says a format keeps a source,
//! every code comes back through it, and where it says bits go, codes collapse.

use zenpixels::{
    ChannelLayout, ChannelType, PixelDescriptor as D, SignalRange, TransferFunction as T,
};
use zenpixels_convert::RowConverter;
use zenpixels_convert::precision::{ConversionEffect, ConversionOffer, SourcePrecision};

fn rgb(channel: ChannelType, transfer: T) -> D {
    D::new(channel, ChannelLayout::Rgb, None, transfer)
}

/// Push rows of gray codes through `via` and back to `from`, and count those that changed.
fn codes_lost(from: D, via: D, codes: &[[u8; 6]]) -> usize {
    let width = codes.len() as u32;
    let src: Vec<u8> = codes.iter().flatten().copied().collect();
    let mut there = vec![0u8; codes.len() * via.bytes_per_pixel()];
    let mut back = vec![0u8; src.len()];
    RowConverter::new(from, via)
        .unwrap()
        .try_convert_row(&src, &mut there, width)
        .unwrap();
    RowConverter::new(via, from)
        .unwrap()
        .try_convert_row(&there, &mut back, width)
        .unwrap();
    let (src, back) = (src.as_chunks::<6>().0, back.as_chunks::<6>().0);
    src.iter().zip(back).filter(|(a, b)| a != b).count()
}

fn sixteen_bit_codes(codes: impl Iterator<Item = u16>) -> Vec<[u8; 6]> {
    codes
        .map(|v| {
            let b = v.to_ne_bytes();
            [b[0], b[1], b[0], b[1], b[0], b[1]]
        })
        .collect()
}

fn lost_eight_bit(via: D) -> usize {
    let width = 256;
    let src: Vec<u8> = (0..=255u8).flat_map(|v| [v, v, v]).collect();
    let mut there = vec![0u8; width * via.bytes_per_pixel()];
    let mut back = vec![0u8; src.len()];
    RowConverter::new(D::RGB8_SRGB, via)
        .unwrap()
        .try_convert_row(&src, &mut there, width as u32)
        .unwrap();
    RowConverter::new(via, D::RGB8_SRGB)
        .unwrap()
        .try_convert_row(&there, &mut back, width as u32)
        .unwrap();
    let (src, back) = (src.as_chunks::<3>().0, back.as_chunks::<3>().0);
    src.iter().zip(back).filter(|(a, b)| a != b).count()
}

fn says_keeps(source: &SourcePrecision, via: D) -> bool {
    ConversionEffect::of(source, &ConversionOffer::new(via)).bits_dropped == 0
}

#[test]
fn eight_bit_srgb_through_each_working_format() {
    let source = SourcePrecision::declared(8, T::Srgb, SignalRange::Full).unwrap();
    for via in [
        rgb(ChannelType::U16, T::Linear),
        rgb(ChannelType::F32, T::Linear),
        rgb(ChannelType::U16, T::Srgb),
        rgb(ChannelType::F32, T::Srgb),
    ] {
        assert!(says_keeps(&source, via), "{via:?}");
        assert_eq!(lost_eight_bit(via), 0, "{via:?} keeps every code");
    }
    let linear8 = rgb(ChannelType::U8, T::Linear);
    assert!(!says_keeps(&source, linear8));
    assert!(
        lost_eight_bit(linear8) > 0,
        "8-bit linear collapses dark codes"
    );
}

#[test]
fn sixteen_bit_srgb_through_float_keeps_and_through_u8_or_u16_linear_does_not() {
    let every = sixteen_bit_codes(0..=u16::MAX);
    let source = SourcePrecision::from_storage(D::RGB16_SRGB);
    let float = rgb(ChannelType::F32, T::Linear);
    assert!(says_keeps(&source, float));
    assert_eq!(codes_lost(D::RGB16_SRGB, float, &every), 0);

    let eight = D::RGB8_SRGB;
    assert!(!says_keeps(&source, eight));
    assert!(codes_lost(D::RGB16_SRGB, eight, &every) > 0);

    let linear16 = rgb(ChannelType::U16, T::Linear);
    assert!(!says_keeps(&source, linear16));
    assert!(codes_lost(D::RGB16_SRGB, linear16, &every) > 0);
}

#[test]
fn twelve_bit_codes_in_u16_survive_u16_linear() {
    // The table's 12 for sRGB in 16-bit linear, on the kernels that would run. A 12-bit source
    // kept in canonical U16 has only these codes.
    let codes = sixteen_bit_codes((0..4096u32).map(|c| ((c * 65535 + 2047) / 4095) as u16));
    let source = SourcePrecision::declared(12, T::Srgb, SignalRange::Full).unwrap();
    let linear16 = rgb(ChannelType::U16, T::Linear);
    assert!(says_keeps(&source, linear16));
    // Come back to the same 12-bit code, after rounding what U16 returns to 12 bits.
    let width = codes.len() as u32;
    let src: Vec<u8> = codes.iter().flatten().copied().collect();
    let mut there = vec![0u8; src.len()];
    let mut back = vec![0u8; src.len()];
    RowConverter::new(D::RGB16_SRGB, linear16)
        .unwrap()
        .try_convert_row(&src, &mut there, width)
        .unwrap();
    RowConverter::new(linear16, D::RGB16_SRGB)
        .unwrap()
        .try_convert_row(&there, &mut back, width)
        .unwrap();
    let wrong = back
        .as_chunks::<6>()
        .0
        .iter()
        .enumerate()
        .filter(|(code, pixel)| {
            let v = u32::from(u16::from_ne_bytes([pixel[0], pixel[1]]));
            (v * 4095 + 32767) / 65535 != *code as u32
        })
        .count();
    assert_eq!(wrong, 0);
}

/// Row "Real 16-bit source, output format chosen automatically: pick a format that holds it".
#[test]
fn encode_adapter_picks_a_format_that_holds_the_source_when_one_is_offered() {
    use zenpixels::PixelSlice;
    use zenpixels_convert::PixelSliceConvertExt;
    let samples: Vec<u16> = vec![1, 2, 3, 65534, 65533, 65532];
    let pixels = PixelSlice::new(bytemuck::cast_slice(&samples), 2, 1, 12, D::RGB16_SRGB).unwrap();
    let options = zenpixels::ConvertOptions::permissive();
    let float = rgb(ChannelType::F32, T::Linear);
    for supported in [[D::RGB8_SRGB, float], [float, D::RGB8_SRGB]] {
        let adapted = pixels.adapt_for_encode(&supported, &options).unwrap();
        assert_eq!(adapted.as_slice().descriptor(), float, "{supported:?}");
    }
    // A source that needs none of it still takes the cheap path.
    let eight = [10u8, 20, 30];
    let pixels = PixelSlice::new(&eight, 1, 1, 3, D::RGB8_SRGB).unwrap();
    let adapted = pixels
        .adapt_for_encode(&[D::RGB8_SRGB, float], &options)
        .unwrap();
    assert_eq!(adapted.as_slice().descriptor(), D::RGB8_SRGB);
    // With nothing that holds it, a narrowing the options permit still happens: this adapter's
    // options are how a caller names it.
    let samples: Vec<u16> = vec![1, 2, 3];
    let pixels = PixelSlice::new(bytemuck::cast_slice(&samples), 1, 1, 6, D::RGB16_SRGB).unwrap();
    let adapted = pixels.adapt_for_encode(&[D::RGB8_SRGB], &options).unwrap();
    assert_eq!(adapted.as_slice().descriptor(), D::RGB8_SRGB);
}

// ---- The scan ----

mod scan {
    use super::D;
    use enough::{Stop, StopReason, Unstoppable};
    use zenpixels::PixelSlice;
    use zenpixels_convert::PixelSliceConvertExt;
    use zenpixels_convert::precision::{
        ConversionEffect, ConversionOffer, ConversionPolicy, OutputChoice, SourcePrecision,
    };

    fn gray16(samples: &[u16], width: u32, stride: usize) -> PixelSlice<'_> {
        let rows = (samples.len() * 2 / stride) as u32;
        PixelSlice::new(
            bytemuck::cast_slice(samples),
            width,
            rows,
            stride,
            D::GRAY16_SRGB,
        )
        .unwrap()
    }

    #[test]
    fn finds_8bit_content_in_u16_and_clears_the_narrowing() {
        let samples: Vec<u16> = (0..=255u16).map(|v| v * 257).collect();
        let scanned = gray16(&samples, 256, 512)
            .scan_source_precision(&Unstoppable)
            .unwrap();
        assert_eq!(scanned.significant_bits(), 8);
        let offers = [ConversionOffer::new(D::GRAY8_SRGB)];
        let record = ConversionPolicy::new()
            .resolve_preset(&scanned, OutputChoice::Hidden, &offers)
            .unwrap();
        assert!(record.effect.preserves_modeled_capacity());
        // Before the scan, the same conversion is refused.
        assert!(
            ConversionPolicy::new()
                .resolve_preset(
                    &SourcePrecision::from_storage(D::GRAY16_SRGB),
                    OutputChoice::Hidden,
                    &offers
                )
                .is_err()
        );
    }

    #[test]
    fn finds_real_16bit_and_10bit_content() {
        let real = [0u16, 1, 65534, 65535];
        let scanned = gray16(&real, 4, 8)
            .scan_source_precision(&Unstoppable)
            .unwrap();
        assert_eq!(scanned.significant_bits(), 16);

        // Canonical 10-bit codes in U16: 8 bits can't hold them.
        let ten: Vec<u16> = (0..1024u32)
            .map(|code| ((code * 65535 + 511) / 1023) as u16)
            .collect();
        let scanned = gray16(&ten, 1024, 2048)
            .scan_source_precision(&Unstoppable)
            .unwrap();
        assert_eq!(scanned.significant_bits(), 10);
        let effect = ConversionEffect::of(&scanned, &ConversionOffer::new(D::GRAY8_SRGB));
        assert_eq!(effect.bits_dropped, 2);
    }

    #[test]
    fn rows_on_different_grids_need_a_depth_that_holds_both() {
        // 2-bit codes on one row and 3-bit codes on the next share no grid below 6 bits.
        let two = 21845u16;
        let three = 9362u16;
        let scanned = gray16(&[two, three], 1, 2)
            .scan_source_precision(&Unstoppable)
            .unwrap();
        assert_eq!(scanned.significant_bits(), 6);
    }

    #[test]
    fn reads_alpha_and_counts_its_depth() {
        let rgba = |samples: &'static [u16]| {
            PixelSlice::new(bytemuck::cast_slice(samples), 1, 1, 8, D::RGBA16_SRGB)
                .unwrap()
                .scan_source_precision(&Unstoppable)
                .unwrap()
        };
        let opaque = rgba(&[257, 514, 771, 65535]);
        assert_eq!(opaque.significant_bits(), 8);
        let rgb8 = ConversionOffer::new(D::RGB8_SRGB);
        assert!(ConversionEffect::of(&opaque, &rgb8).preserves_modeled_capacity());

        let translucent = rgba(&[257, 514, 771, 1235]);
        assert_eq!(
            translucent.significant_bits(),
            16,
            "the alpha sample is off the 8-bit grid"
        );
        let rgb16 = ConversionOffer::new(D::RGB16_SRGB);
        assert!(ConversionEffect::of(&translucent, &rgb16).alpha_dropped);
    }

    #[test]
    fn ignores_row_padding() {
        let rows = [257u16, 514, 0x1234, 771, 1028, 0x5678];
        let scanned = gray16(&rows, 2, 6)
            .scan_source_precision(&Unstoppable)
            .unwrap();
        assert_eq!(scanned.significant_bits(), 8);
    }

    #[test]
    fn a_float_is_storage() {
        let floats = [0.5f32, 0.25, 0.75];
        let pixels =
            PixelSlice::new(bytemuck::cast_slice(&floats), 1, 1, 12, D::RGBF32_LINEAR).unwrap();
        assert_eq!(
            pixels.scan_source_precision(&Unstoppable).unwrap(),
            SourcePrecision::from_storage(D::RGBF32_LINEAR)
        );
    }

    struct StopAfter(core::sync::atomic::AtomicU32);
    impl Stop for StopAfter {
        fn check(&self) -> Result<(), StopReason> {
            use core::sync::atomic::Ordering::Relaxed;
            let left = self.0.load(Relaxed);
            if left == 0 {
                return Err(StopReason::Cancelled);
            }
            self.0.store(left - 1, Relaxed);
            Ok(())
        }
    }

    #[test]
    fn stops_between_rows_with_a_trace() {
        let rows = [257u16; 8];
        let stop = StopAfter(core::sync::atomic::AtomicU32::new(2));
        let err = gray16(&rows, 2, 4)
            .scan_source_precision(&stop)
            .unwrap_err();
        assert_eq!(*err.error(), StopReason::Cancelled);
        assert!(err.frame_count() >= 1);
    }
}
