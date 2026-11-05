//! A [`PluggableCms`] over the pure-Rust lcms port, the second real backend behind the trait.
//!
//! Test equipment, not a product backend: the port is unreleased and its own README calls it early.
//! It tracks lcms2 2.20 bit for bit, so it also stands in for lcms2 as a reference here. Its transform
//! isn't `Send`, so it lives on a thread of its own.

use std::sync::mpsc::{Receiver, Sender, channel};

use lcms::types::{FLAGS_BLACKPOINTCOMPENSATION, FLAGS_COPY_ALPHA, FLAGS_NOOPTIMIZE};
use zenpixels::{ChannelType, Cicp, ColorProfileSource, PixelFormat};
use zenpixels_convert::cms::{CmsPluginError, PluggableCms, RowTransformMut};
use zenpixels_convert::icc_profiles::SynthesizedIcc;
use zenpixels_convert::policy::ConvertOptions;

whereat::define_at_crate_info!();

const FLOAT: u32 = 1 << 22;
const PT_GRAY: u32 = 3;
const PT_RGB: u32 = 4;
const PT_CMYK: u32 = 6;

const fn lcms_format(space: u32, channels: u32, bytes: u32, extra: u32) -> u32 {
    let float = if bytes == 4 { FLOAT } else { 0 };
    // lcms encodes 4-byte samples as bytes = 4; 8-byte doubles as bytes = 0.
    float | space << 16 | extra << 7 | channels << 3 | bytes
}

/// lcms float CMYK runs 0..=100; these formats run 0..=1.
#[derive(Clone, Copy, PartialEq)]
enum Ink {
    None,
    Float,
    FloatInverted,
}

fn describe(format: PixelFormat) -> Option<(u32, Ink)> {
    use PixelFormat::*;
    Some(match format {
        Rgb8 => (lcms_format(PT_RGB, 3, 1, 0), Ink::None),
        Rgba8 => (lcms_format(PT_RGB, 3, 1, 1), Ink::None),
        Rgb16 => (lcms_format(PT_RGB, 3, 2, 0), Ink::None),
        Rgba16 => (lcms_format(PT_RGB, 3, 2, 1), Ink::None),
        RgbF32 => (lcms_format(PT_RGB, 3, 4, 0), Ink::None),
        RgbaF32 => (lcms_format(PT_RGB, 3, 4, 1), Ink::None),
        Gray8 => (lcms_format(PT_GRAY, 1, 1, 0), Ink::None),
        Gray16 => (lcms_format(PT_GRAY, 1, 2, 0), Ink::None),
        GrayF32 => (lcms_format(PT_GRAY, 1, 4, 0), Ink::None),
        GrayA8 => (lcms_format(PT_GRAY, 1, 1, 1), Ink::None),
        GrayA16 => (lcms_format(PT_GRAY, 1, 2, 1), Ink::None),
        GrayAF32 => (lcms_format(PT_GRAY, 1, 4, 1), Ink::None),
        Cmyk8 => (lcms_format(PT_CMYK, 4, 1, 0), Ink::None),
        CmykF32 => (lcms_format(PT_CMYK, 4, 4, 0), Ink::Float),
        CmykF32Inverted => (lcms_format(PT_CMYK, 4, 4, 0), Ink::FloatInverted),
        _ => return None,
    })
}

#[derive(Clone)]
enum Profile {
    Icc(Vec<u8>),
    Srgb,
}

impl Profile {
    fn of(source: ColorProfileSource<'_>) -> Option<Self> {
        let cicp = match source {
            ColorProfileSource::Icc(bytes) => return Some(Self::Icc(bytes.to_vec())),
            ColorProfileSource::Cicp(cicp) => cicp,
            ColorProfileSource::PrimariesTransferPair {
                primaries,
                transfer,
            } => Cicp::new(primaries.to_cicp()?, transfer.to_cicp()?, 0, true),
            _ => return None,
        };
        if cicp.color_primaries == 1 && cicp.transfer_characteristics == 13 {
            return Some(Self::Srgb);
        }
        match SynthesizedIcc::for_cicp(cicp) {
            SynthesizedIcc::Profile(bytes) => Some(Self::Icc(bytes.into_owned())),
            _ => None,
        }
    }

    fn open(&self) -> Option<Box<lcms::profile::IccProfile>> {
        match self {
            Self::Icc(bytes) => lcms::profile::open_profile_from_mem(None, bytes),
            Self::Srgb => lcms::virt::create_srgb_profile(None),
        }
    }
}

struct Job {
    src: Vec<u8>,
    dst: Vec<u8>,
    width: u32,
}

/// Builds the transform on a thread of its own and serves rows from it. `lcms::xform::Transform`
/// isn't `Send`, while `RowTransformMut` must be, so nothing else can hold it. This only costs speed,
/// and nothing here measures speed.
fn spawn_worker(
    source: Profile,
    target: Profile,
    from: u32,
    to: u32,
    flags: u32,
    bpc: bool,
) -> Result<(Sender<Job>, Receiver<Job>), &'static str> {
    let (jobs, inbox) = channel::<Job>();
    let (outbox, done) = channel::<Job>();
    let (ready, built) = channel::<Result<(), &'static str>>();
    std::thread::spawn(move || {
        let Some((source, target)) = source.open().zip(target.open()) else {
            let _ = ready.send(Err("lcms could not open a profile"));
            return;
        };
        let mut profiles = vec![source, target];
        let intents = [lcms::virt::INTENT_RELATIVE_COLORIMETRIC; 2];
        let transform = lcms::xform::create_extended_transform(
            None,
            &mut profiles,
            &mut [bpc, bpc],
            &intents,
            &[1.0, 1.0],
            None,
            0,
            from,
            to,
            flags,
        );
        let Some(mut transform) = transform else {
            let _ = ready.send(Err("lcms refused the transform"));
            return;
        };
        let _ = ready.send(Ok(()));
        for mut job in inbox {
            transform.do_transform(&job.src, &mut job.dst, job.width);
            if outbox.send(job).is_err() {
                return;
            }
        }
    });
    built
        .recv()
        .map_err(|_| "the lcms worker panicked")?
        .map(|()| (jobs, done))
}

/// Bytes of a gamma 2.2 gray profile, written by the port.
pub fn gray_gamma22_icc() -> Vec<u8> {
    let curve = lcms::gamma::build_gamma(None, 2.2).expect("gamma curve");
    let mut profile = lcms::virt::create_gray_profile(None, None, Some(&curve)).expect("gray");
    lcms::profile::save_profile_to_mem(&mut profile).expect("saved profile")
}

/// The lcms port as a [`PluggableCms`].
pub struct LcmsPort;

impl PluggableCms for LcmsPort {
    fn build_source_transform(
        &self,
        src: ColorProfileSource<'_>,
        dst: ColorProfileSource<'_>,
        src_format: PixelFormat,
        dst_format: PixelFormat,
        options: &ConvertOptions,
    ) -> Option<Result<Box<dyn RowTransformMut>, whereat::At<CmsPluginError>>> {
        let (from, from_ink) = describe(src_format)?;
        let (to, to_ink) = describe(dst_format)?;
        if to_ink != Ink::None {
            return Some(Err(whereat::at!(CmsPluginError::msg(
                "conversion into F32 CMYK is not qualified"
            ))));
        }
        // A source this backend can't express isn't its problem.
        let source = Profile::of(src)?;
        let target = Profile::of(dst)?;
        let ink_source = from_ink != Ink::None;
        // The optimizer resamples 16-bit pipelines, up to 7.5 of 255 from the unoptimized result.
        let flags = FLAGS_COPY_ALPHA
            | FLAGS_NOOPTIMIZE
            | if ink_source {
                FLAGS_BLACKPOINTCOMPENSATION
            } else {
                0
            };
        let (jobs, done) = match spawn_worker(source, target, from, to, flags, ink_source) {
            Ok(worker) => worker,
            Err(reason) => return Some(Err(whereat::at!(CmsPluginError::msg(reason)))),
        };
        Some(Ok(Box::new(Row {
            jobs,
            done,
            ink: from_ink,
            clip: options.clip_out_of_gamut() && dst_format.channel_type() == ChannelType::F32,
            source_bytes: src_format.bytes_per_pixel(),
            target_bytes: dst_format.bytes_per_pixel(),
            target_color: dst_format.channels().min(3),
            target_channels: dst_format.channels(),
        })))
    }
}

struct Row {
    jobs: Sender<Job>,
    done: Receiver<Job>,
    ink: Ink,
    clip: bool,
    source_bytes: usize,
    target_bytes: usize,
    target_color: usize,
    target_channels: usize,
}

impl RowTransformMut for Row {
    fn transform_row(&mut self, src: &[u8], dst: &mut [u8], width: u32) {
        let pixels = width as usize;
        let src = &src[..pixels * self.source_bytes];
        let dst = &mut dst[..pixels * self.target_bytes];
        let src = match self.ink {
            Ink::None => src.to_vec(),
            ink => src
                .chunks_exact(4)
                .flat_map(|b| {
                    let v = f32::from_ne_bytes(b.try_into().unwrap());
                    (100.0 * if ink == Ink::Float { v } else { 1.0 - v }).to_ne_bytes()
                })
                .collect(),
        };
        // Sample bytes the transform doesn't write must survive, as they do through a slice.
        let job = Job {
            src,
            dst: dst.to_vec(),
            width,
        };
        self.jobs.send(job).expect("the lcms worker is gone");
        let job = self.done.recv().expect("the lcms worker panicked");
        dst.copy_from_slice(&job.dst);
        if self.clip {
            for pixel in dst.chunks_exact_mut(4 * self.target_channels) {
                for sample in pixel[..4 * self.target_color].chunks_exact_mut(4) {
                    let v = f32::from_ne_bytes((&*sample).try_into().unwrap());
                    sample.copy_from_slice(&v.clamp(0.0, 1.0).to_ne_bytes());
                }
            }
        }
    }
}
