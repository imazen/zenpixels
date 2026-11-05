//! SDR samples don't say where their white belongs on PQ or HLG. Passing the signal through
//! puts it at the transfer's peak, 10,000 cd/m² on PQ, so every entry point refuses.
use zenpixels::{ChannelLayout, ChannelType, ColorPrimaries, PixelBuffer, PixelDescriptor};
use zenpixels::{TransferFunction as Tf, hdr::DiffuseWhite};
use zenpixels_convert::{
    ConvertError, ConvertOptions, ConvertPlan, PixelBufferConvertExt, PixelSliceConvertExt,
    RowConverter,
};

const SDR: [Tf; 3] = [Tf::Srgb, Tf::Bt709, Tf::Gamma22];
const HDR: [Tf; 2] = [Tf::Pq, Tf::Hlg];

fn rgb(channel: ChannelType, transfer: Tf, primaries: ColorPrimaries) -> PixelDescriptor {
    PixelDescriptor::new(channel, ChannelLayout::Rgb, None, transfer).with_primaries(primaries)
}

/// Every depth pair, with and without a gamut change, since ZenCmsLite takes the latter.
fn pairs() -> Vec<(PixelDescriptor, PixelDescriptor)> {
    let depths = [ChannelType::U8, ChannelType::U16, ChannelType::F32];
    let mut all = Vec::new();
    for sdr in SDR {
        for hdr in HDR {
            for from in depths {
                for to in [ChannelType::U16, ChannelType::F32] {
                    for primaries in [ColorPrimaries::Bt709, ColorPrimaries::Bt2020] {
                        all.push((
                            rgb(from, sdr, ColorPrimaries::Bt709),
                            rgb(to, hdr, primaries),
                        ));
                    }
                }
            }
        }
    }
    all
}

#[test]
fn row_converter_refuses_sdr_into_pq_and_hlg() {
    for (from, to) in pairs() {
        let e = RowConverter::new(from, to).map(|_| ()).unwrap_err();
        assert!(
            matches!(e.error(), ConvertError::NoPath { .. }),
            "{from:?} to {to:?}: {e:?}"
        );
    }
}

#[test]
fn plans_and_buffers_refuse_sdr_into_pq_and_hlg() {
    let options = ConvertOptions::permissive();
    for (from, to) in pairs() {
        assert!(ConvertPlan::new(from, to).is_err(), "{from:?} to {to:?}");
        assert!(
            ConvertPlan::new_explicit(from, to, &options).is_err(),
            "{from:?} to {to:?}"
        );
        assert!(
            RowConverter::new_explicit(from, to, &options).is_err(),
            "{from:?} to {to:?}"
        );
        let buffer = PixelBuffer::try_new(1, 1, from).unwrap();
        assert!(buffer.convert_to(to).is_err(), "{from:?} to {to:?}");
    }
}

/// Linear has no SDR question: PQ decodes to it and comes back with 1.0 at the peak.
#[test]
fn linear_and_hdr_sources_still_reach_pq_and_hlg() {
    let linear = rgb(ChannelType::F32, Tf::Linear, ColorPrimaries::Bt2020);
    for hdr in HDR {
        let to = rgb(ChannelType::U16, hdr, ColorPrimaries::Bt2020);
        assert!(RowConverter::new(linear, to).is_ok());
        assert!(RowConverter::new(to, linear).is_ok());
        assert!(RowConverter::new(to, to).is_ok());
    }
}

/// The route for a caller who does know: decode to linear, then quantize with the anchor.
#[test]
fn an_explicit_anchor_places_sdr_white_where_css_puts_it() {
    let source = PixelBuffer::from_vec(vec![255u8; 3], 1, 1, PixelDescriptor::RGB8_SRGB).unwrap();
    let linear = source
        .convert_to(PixelDescriptor::RGBF32_LINEAR)
        .unwrap()
        .with_color_context(std::sync::Arc::new(
            zenpixels::ColorContext::default().with_diffuse_white(DiffuseWhite::BT2408),
        ));
    let target = rgb(ChannelType::F32, Tf::Pq, ColorPrimaries::Bt709);
    let pq = linear.as_slice().quantize_to(target).unwrap();
    let view = pq.as_slice();
    let signal: &[f32] = bytemuck::cast_slice(view.row(0));
    for &c in signal {
        // docs/css-color-parity.md: 203 cd/m² is 0.5807 of PQ's signal.
        assert!((c - 0.5807).abs() < 1e-3, "{c}");
    }
}
