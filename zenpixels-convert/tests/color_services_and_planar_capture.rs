use alloc::sync::Arc;
extern crate alloc;

use zenpixels_convert::{
    AlphaMode, AlphaPolicy, BitPacking, CaptureScratch, ChromaSiting, Cicp, CmsChoice, Color,
    ColorServices, ColorSpace, ConvertError, ConvertPolicy, DepthPolicy, DiffuseWhite, GamutPolicy,
    HdrMetadata, HlgMode, HlgRendering, Nits, PixelBuf, PixelFormat, PixelRect, PixelView,
    PixelViewMut, PlanarCaptureOptions, PlanarCapturePlan, PlanarCaptureSpec,
    PlanarReconstructionOptions, PlanarViewMut, PlaneFormat, PlaneRole, PlaneViewMut, Primaries,
    RangePolicy, RowPlan, SampleLayout, SampleType, SignalRange, Transfer, convert_to_buf, formats,
    reconstruct_planar_into, reconstruct_planar_to_buf,
};

fn make_rgb8_pattern(w: u32, h: u32) -> PixelBuf {
    let mut data = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 23 + y * 11) % 256) as u8;
            let g = ((x * 7 + y * 31 + 40) % 256) as u8;
            let b = ((x * 17 + y * 13 + 120) % 256) as u8;
            data.extend_from_slice(&[r, g, b]);
        }
    }
    PixelBuf::from_typed(data, w, h, PixelFormat::RGB8_SRGB).unwrap()
}

/// Independent exact i128 rational reference for ITU-T H.273 §8.3 full/narrow 8/10/12/16-bit
/// NCL YCbCr from 8-bit RGB, immune to IEEE-754 floating-point tie drift.
fn ref_rgb8_to_ycbcr_code_exact(
    r: u8,
    g: u8,
    b: u8,
    kr_10k: i128,
    kb_10k: i128,
    bits: u32,
    full_range: bool,
) -> (u32, u32, u32) {
    let k_den: i128 = 10_000;
    let kg_10k = k_den - kr_10k - kb_10k;
    let (ri, gi, bi) = (i128::from(r), i128::from(g), i128::from(b));
    let y_sum = kr_10k * ri + kg_10k * gi + kb_10k * bi;
    let max_in: i128 = 255;
    let max_code = (1u32 << bits) - 1;
    let shift = bits - 8;

    let round_clamp = |num: i128, den: i128, lo: u32, hi: u32| -> u32 {
        let lo_i = i128::from(lo);
        let hi_i = i128::from(hi);
        let c = num.clamp(lo_i * den, hi_i * den);
        (2 * c + den).div_euclid(2 * den).clamp(lo_i, hi_i) as u32
    };

    let (y_off, y_span, y_lo, y_hi, c_mid, c_span, c_lo, c_hi) = if full_range {
        (
            0i128,
            i128::from(max_code),
            0u32,
            max_code,
            i128::from(1u32 << (bits - 1)),
            i128::from(max_code),
            0u32,
            max_code,
        )
    } else {
        let lo = 16u32 << shift;
        let y_sp = 219u32 << shift;
        let c_sp = 224u32 << shift;
        (
            i128::from(lo),
            i128::from(y_sp),
            lo,
            lo + y_sp,
            i128::from(128u32 << shift),
            i128::from(c_sp),
            lo,
            240u32 << shift,
        )
    };

    let y_den = k_den * max_in;
    let yc = round_clamp(y_off * y_den + y_sum * y_span, y_den, y_lo, y_hi);
    let cb_den = 2 * (k_den - kb_10k) * max_in;
    let cbc = round_clamp(
        c_mid * cb_den + (k_den * bi - y_sum) * c_span,
        cb_den,
        c_lo,
        c_hi,
    );
    let cr_den = 2 * (k_den - kr_10k) * max_in;
    let crc = round_clamp(
        c_mid * cr_den + (k_den * ri - y_sum) * c_span,
        cr_den,
        c_lo,
        c_hi,
    );
    (yc, cbc, crc)
}

#[test]
fn gbr_identity_8_10_12_16_bit_exact_roundtrip_odd_dims() {
    let w = 17u32;
    let h = 13u32;
    let src = make_rgb8_pattern(w, h);
    let mut scratch = CaptureScratch::owned();

    for (sample, packing) in [
        (SampleType::U8, BitPacking::Full),
        (SampleType::U16, BitPacking::Lsb { bits: 10 }),
        (SampleType::U16, BitPacking::Msb { bits: 10 }),
        (SampleType::U16, BitPacking::Lsb { bits: 12 }),
        (SampleType::U16, BitPacking::Full),
    ] {
        let planes = [
            PlaneFormat::new(PlaneRole::Y, sample).with_packing(packing),
            PlaneFormat::new(PlaneRole::Cb, sample).with_packing(packing),
            PlaneFormat::new(PlaneRole::Cr, sample).with_packing(packing),
        ];
        let spec = PlanarCaptureSpec::full_frame(w, h, planes, ChromaSiting::Unknown, Cicp::SRGB);
        let plan =
            PlanarCapturePlan::build(src.format(), spec, &ConvertPolicy::lossless()).unwrap();
        let captured = plan.capture_to_buf(src.as_view(), &mut scratch).unwrap();
        let rebuilt = reconstruct_planar_to_buf(
            &captured.as_view(),
            SampleLayout::RGB8,
            &PlanarReconstructionOptions::nearest(),
        )
        .unwrap();
        assert_eq!(
            rebuilt.as_view().as_contiguous_bytes().unwrap(),
            src.as_view().as_contiguous_bytes().unwrap(),
            "GBR roundtrip failed for {sample:?}/{packing:?}"
        );
    }
}

#[test]
fn ycbcr_444_narrow_and_full_match_exact_rational_reference_zero_bit_exact() {
    let w = 9u32;
    let h = 7u32;
    let src = make_rgb8_pattern(w, h);
    let mut scratch = CaptureScratch::owned();

    // Includes witness pixel (3,1)=RGB8(80,92,184) -> 16-bit BT.709 narrow Y = 25222,
    // plus 8/10/12/16-bit full & narrow coverage across BT.709, BT.601, BT.2020.
    for (bits, sample, packing, full_range, mc, kr_10k, kb_10k) in [
        (
            8u32,
            SampleType::U8,
            BitPacking::Full,
            false,
            1u8,
            2126i128,
            722i128,
        ),
        (
            8u32,
            SampleType::U8,
            BitPacking::Full,
            true,
            6u8,
            2990,
            1140,
        ),
        (
            10u32,
            SampleType::U16,
            BitPacking::Lsb { bits: 10 },
            false,
            1u8,
            2126,
            722,
        ),
        (
            10u32,
            SampleType::U16,
            BitPacking::Lsb { bits: 10 },
            true,
            1u8,
            2126,
            722,
        ),
        (
            10u32,
            SampleType::U16,
            BitPacking::Msb { bits: 10 },
            false,
            9u8,
            2627,
            593,
        ),
        (
            12u32,
            SampleType::U16,
            BitPacking::Lsb { bits: 12 },
            true,
            1u8,
            2126,
            722,
        ),
        (
            12u32,
            SampleType::U16,
            BitPacking::Lsb { bits: 12 },
            false,
            1u8,
            2126,
            722,
        ),
        (
            16u32,
            SampleType::U16,
            BitPacking::Full,
            false,
            1u8,
            2126,
            722,
        ),
        (
            16u32,
            SampleType::U16,
            BitPacking::Full,
            true,
            1u8,
            2126,
            722,
        ),
    ] {
        let planes = [
            PlaneFormat::new(PlaneRole::Y, sample).with_packing(packing),
            PlaneFormat::new(PlaneRole::Cb, sample).with_packing(packing),
            PlaneFormat::new(PlaneRole::Cr, sample).with_packing(packing),
        ];
        let cicp = Cicp::new(1, 13, mc, full_range);
        let spec = PlanarCaptureSpec::full_frame(w, h, planes, ChromaSiting::Center, cicp);
        let plan =
            PlanarCapturePlan::build(src.format(), spec, &ConvertPolicy::permissive()).unwrap();
        let captured = plan.capture_to_buf(src.as_view(), &mut scratch).unwrap();
        let view = captured.as_view();
        let y_cp = view
            .plane_by_role(PlaneRole::Y)
            .unwrap()
            .as_code_plane(0)
            .unwrap();
        let cb_cp = view
            .plane_by_role(PlaneRole::Cb)
            .unwrap()
            .as_code_plane(0)
            .unwrap();
        let cr_cp = view
            .plane_by_role(PlaneRole::Cr)
            .unwrap()
            .as_code_plane(0)
            .unwrap();

        for y in 0..h {
            let srow = src.as_view().row(y);
            for x in 0..w {
                let (ry, rcb, rcr) = ref_rgb8_to_ycbcr_code_exact(
                    srow[x as usize * 3],
                    srow[x as usize * 3 + 1],
                    srow[x as usize * 3 + 2],
                    kr_10k,
                    kb_10k,
                    bits,
                    full_range,
                );
                let gy = u32::from(y_cp.code(x as usize, y as usize).unwrap());
                let gcb = u32::from(cb_cp.code(x as usize, y as usize).unwrap());
                let gcr = u32::from(cr_cp.code(x as usize, y as usize).unwrap());
                assert_eq!(
                    (gy, gcb, gcr),
                    (ry, rcb, rcr),
                    "exact 0-bit mismatch at ({x},{y}) bits={bits} full={full_range} mc={mc}"
                );
            }
        }
    }

    // Exact half-integer tie witness: BT.709 full-range RGB8(0, 14, 76) -> 16-bit Y = 3983.5 -> 3984!
    let tie_src = PixelBuf::from_typed(vec![0u8, 14, 76], 1, 1, PixelFormat::RGB8_SRGB).unwrap();
    let tie_planes = [
        PlaneFormat::new(PlaneRole::Y, SampleType::U16),
        PlaneFormat::new(PlaneRole::Cb, SampleType::U16),
        PlaneFormat::new(PlaneRole::Cr, SampleType::U16),
    ];
    let tie_spec = PlanarCaptureSpec::full_frame(
        1,
        1,
        tie_planes,
        ChromaSiting::Center,
        Cicp::new(1, 13, 1, true),
    );
    let tie_plan =
        PlanarCapturePlan::build(tie_src.format(), tie_spec, &ConvertPolicy::permissive()).unwrap();
    let tie_cap = tie_plan
        .capture_to_buf(tie_src.as_view(), &mut scratch)
        .unwrap();
    let tie_y = tie_cap
        .as_view()
        .plane_by_role(PlaneRole::Y)
        .unwrap()
        .as_code_plane(0)
        .unwrap()
        .code(0, 0)
        .unwrap();
    assert_eq!(
        tie_y, 3984,
        "exact tie RGB8(0,14,76) BT.709 full 16-bit Y = 3983.5 must round to 3984"
    );

    // Exact 16-bit subsampled chroma ring precision witness:
    // constant RGB8(0, 13, 22) 4x4 4:2:0 -> BT.709 narrow 16-bit Cr = 31213.4999975... -> 31213!
    let mut sub_bytes = Vec::with_capacity(4 * 4 * 3);
    for _ in 0..16 {
        sub_bytes.extend_from_slice(&[0u8, 13, 22]);
    }
    let sub_src = PixelBuf::from_typed(sub_bytes, 4, 4, PixelFormat::RGB8_SRGB).unwrap();
    let sub_planes = [
        PlaneFormat::new(PlaneRole::Y, SampleType::U16),
        PlaneFormat::new(PlaneRole::Cb, SampleType::U16).with_subsampling(2, 2),
        PlaneFormat::new(PlaneRole::Cr, SampleType::U16).with_subsampling(2, 2),
    ];
    let sub_spec = PlanarCaptureSpec::full_frame(
        4,
        4,
        sub_planes,
        ChromaSiting::Left,
        Cicp::new(1, 13, 1, false),
    );
    let sub_plan =
        PlanarCapturePlan::build(sub_src.format(), sub_spec, &ConvertPolicy::permissive()).unwrap();
    let sub_cap = sub_plan
        .capture_to_buf(sub_src.as_view(), &mut scratch)
        .unwrap();
    let sub_cr = sub_cap
        .as_view()
        .plane_by_role(PlaneRole::Cr)
        .unwrap()
        .as_code_plane(0)
        .unwrap()
        .code(0, 0)
        .unwrap();
    assert_eq!(
        sub_cr, 31213,
        "constant RGB8(0,13,22) 4:2:0 BT.709 narrow 16-bit Cr = 31213.4999975... must emit 31213 (no f32 ring loss)"
    );

    // Exact non-constant 16-bit subsampled chroma tie witness:
    // 2x1 RGB8 sRGB row [(0,5,5), (0,205,205)] -> BT.709 full-range 16-bit 4:2:2 Center triangle:
    // Edge clamping gives each pixel weight 1/2 -> exact Cr = 32768 - 65535 * 105/510 = 19275.5 -> 19276!
    let nc_src =
        PixelBuf::from_typed(vec![0u8, 5, 5, 0, 205, 205], 2, 1, PixelFormat::RGB8_SRGB).unwrap();
    let nc_planes = [
        PlaneFormat::new(PlaneRole::Y, SampleType::U16),
        PlaneFormat::new(PlaneRole::Cb, SampleType::U16).with_subsampling(2, 1),
        PlaneFormat::new(PlaneRole::Cr, SampleType::U16).with_subsampling(2, 1),
    ];
    let nc_spec = PlanarCaptureSpec::full_frame(
        2,
        1,
        nc_planes,
        ChromaSiting::Center,
        Cicp::new(1, 13, 1, true),
    );
    let nc_plan =
        PlanarCapturePlan::build(nc_src.format(), nc_spec, &ConvertPolicy::permissive()).unwrap();
    let nc_cap = nc_plan
        .capture_to_buf(nc_src.as_view(), &mut scratch)
        .unwrap();
    let nc_cr = nc_cap
        .as_view()
        .plane_by_role(PlaneRole::Cr)
        .unwrap()
        .as_code_plane(0)
        .unwrap()
        .code(0, 0)
        .unwrap();
    assert_eq!(
        nc_cr, 19276,
        "non-constant 2x1 [(0,5,5),(0,205,205)] BT.709 full 16-bit 4:2:2 Cr = 19275.5 must round to 19276"
    );
}

#[test]
fn ycgco_direct_code_vectors_exact() {
    // Pure RGB corners + midtone: R=(1,0,0), G=(0,1,0), B=(0,0,1), W=(1,1,1) in full 8-bit YCgCo.
    // Y = 0.25R + 0.5G + 0.25B, Cg = 0.5G - 0.25R - 0.25B, Co = 0.5(R - B).
    let src = PixelBuf::from_typed(
        vec![255u8, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255],
        4,
        1,
        PixelFormat::RGB8_SRGB,
    )
    .unwrap();
    let planes = [
        PlaneFormat::new(PlaneRole::Y, SampleType::U8),
        PlaneFormat::new(PlaneRole::Cb, SampleType::U8),
        PlaneFormat::new(PlaneRole::Cr, SampleType::U8),
    ];
    let spec = PlanarCaptureSpec::full_frame(
        4,
        1,
        planes,
        ChromaSiting::Center,
        Cicp::new(1, 13, 8, true),
    );
    let plan = PlanarCapturePlan::build(src.format(), spec, &ConvertPolicy::lossless()).unwrap();
    let mut scratch = CaptureScratch::owned();
    let buf = plan.capture_to_buf(src.as_view(), &mut scratch).unwrap();
    let view = buf.as_view();
    let y_row = view.plane_by_role(PlaneRole::Y).unwrap().row(0);
    let cg_row = view.plane_by_role(PlaneRole::Cb).unwrap().row(0);
    let co_row = view.plane_by_role(PlaneRole::Cr).unwrap().row(0);
    // R: Y=64 (0.25*255=63.75->64), Cg=128-63.75=64, Co=128+127.5=255 (clamped to 255)
    // G: Y=128 (0.5*255=127.5->128), Cg=255, Co=128
    // B: Y=64, Cg=64, Co=1 (128-127.5=0.5->1)
    // W: Y=255, Cg=128, Co=128
    assert_eq!(y_row, &[64, 128, 64, 255]);
    assert_eq!(cg_row, &[64, 255, 64, 128]);
    assert_eq!(co_row, &[255, 128, 1, 128]);
}

#[test]
fn large_coded_offset_2_pow_24_preserves_exact_chroma_phase() {
    // Non-constant 4x1 strip starting at coded x = 2^24 (16,777,216) — where naive f32
    // global coordinates lose the 0.5 center offset and omit the x+2 tap!
    let origin_x = 1u32 << 24;
    let vis = PixelRect::new(origin_x, 0, 4, 1);
    let coded = (origin_x + 4, 2);
    // Pixels [0, 0, 255, 0] in Blue (pure Cr=0, nonzero Cb only at x=2):
    // Under BT.709 full 8-bit:
    //   pixel (0,0,0) -> Cb = 0.0 -> code 128
    //   pixel (0,0,255) -> Y = 0.0722, Cb = (1 - 0.0722)/(2*(1-0.0722)) = +0.5!
    // For Center 2x chroma sample c0 at local center 0.5 (taps [-1, 0, 1, 2] with weights [1, 3, 3, 1]/8):
    //   tap x=2 has weight 1/8 -> c0 Cb = (1/8)*0.5 = 0.0625 -> 128 + 0.0625*255 = 143.9375 -> 144!
    //   (If x=2 tap were omitted by f32 precision loss, c0 Cb would be 128!)
    // For Left 2x chroma sample c0 at local center 0.0 (taps [-1, 0, 1] with weights [1, 2, 1]/4):
    //   c0 sees only pixels 0 and 1 (all black) -> Cb = 128!
    //   c1 at local center 2.0 (taps [1, 2, 3] with weights [1, 2, 1]/4):
    //   c1 sees weight 2/4 at x=2 -> c1 Cb = 0.5*0.5 = 0.25 -> 128 + 0.25*255 = 191.75 -> 192!
    let src_bytes = [0u8, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0];
    let src_view = PixelView::new(&src_bytes, 4, 1, 12, &formats::RGB8_SRGB).unwrap();
    let y_fmt = PlaneFormat::new(PlaneRole::Y, SampleType::U8);
    let cb_fmt = PlaneFormat::new(PlaneRole::Cb, SampleType::U8).with_subsampling(2, 1);
    let cr_fmt = PlaneFormat::new(PlaneRole::Cr, SampleType::U8).with_subsampling(2, 1);

    let mut scratch = CaptureScratch::owned();

    // 1. Center siting at x = 2^24:
    let spec_center = PlanarCaptureSpec::new(
        coded,
        vis,
        [y_fmt, cb_fmt, cr_fmt],
        ChromaSiting::Center,
        Cicp::new(1, 13, 1, true),
    );
    let plan_center = PlanarCapturePlan::build(
        &PixelFormat::RGB8_SRGB,
        spec_center,
        &ConvertPolicy::permissive(),
    )
    .unwrap();
    let mut y_buf = [0u8; 4];
    let mut cb_buf = [0u8; 2];
    let mut cr_buf = [0u8; 2];
    let mut dst_center = PlanarViewMut::from_array_with_crop_origin(
        4,
        1,
        (origin_x, 0),
        ChromaSiting::Center,
        [
            PlaneViewMut::new(&mut y_buf, 4, 1, 4, y_fmt).unwrap(),
            PlaneViewMut::new(&mut cb_buf, 2, 1, 2, cb_fmt).unwrap(),
            PlaneViewMut::new(&mut cr_buf, 2, 1, 2, cr_fmt).unwrap(),
        ],
    )
    .unwrap();
    plan_center
        .execute(src_view.clone(), &mut dst_center, &mut scratch)
        .unwrap();
    assert_eq!(
        cb_buf[0], 144,
        "Center 2x at x=2^24 must include x+2 tap with weight 1/8 (expected 144, got {})",
        cb_buf[0]
    );
    // c1 at local center 2.5 has taps [1, 2, 3, 4->clamped 3] weights [1, 3, 3, 1]/8 -> 3/8 * 0.5 * 255 + 128 = 175.8125 -> 176!
    assert_eq!(cb_buf[1], 176);

    // 2. Left siting at x = 2^24:
    let spec_left = PlanarCaptureSpec::new(
        coded,
        vis,
        [y_fmt, cb_fmt, cr_fmt],
        ChromaSiting::Left,
        Cicp::new(1, 13, 1, true),
    );
    let plan_left = PlanarCapturePlan::build(
        &PixelFormat::RGB8_SRGB,
        spec_left,
        &ConvertPolicy::permissive(),
    )
    .unwrap();
    let mut dst_left = PlanarViewMut::from_array_with_crop_origin(
        4,
        1,
        (origin_x, 0),
        ChromaSiting::Left,
        [
            PlaneViewMut::new(&mut y_buf, 4, 1, 4, y_fmt).unwrap(),
            PlaneViewMut::new(&mut cb_buf, 2, 1, 2, cb_fmt).unwrap(),
            PlaneViewMut::new(&mut cr_buf, 2, 1, 2, cr_fmt).unwrap(),
        ],
    )
    .unwrap();
    plan_left
        .execute(src_view, &mut dst_left, &mut scratch)
        .unwrap();
    assert_eq!(
        cb_buf,
        [128, 192],
        "Left 2x [1,2,1]/4 exact phase at x=2^24"
    );
}

#[test]
fn subsampled_420_nv12_p010_odd_crop_and_padded_strides_zero_alloc_borrowed_scratch() {
    let coded = (20u32, 16u32);
    let vis = PixelRect::new(3, 1, 11, 9);
    let src_stride = (vis.width as usize) * 3 + 5;
    let mut src_bytes = vec![0xAAu8; src_stride * vis.height as usize];
    for y in 0..vis.height as usize {
        for x in 0..vis.width as usize {
            let off = y * src_stride + x * 3;
            src_bytes[off] = 200;
            src_bytes[off + 1] = 100;
            src_bytes[off + 2] = 50;
        }
    }
    let src_view = PixelView::new(
        &src_bytes,
        vis.width,
        vis.height,
        src_stride,
        &formats::RGB8_SRGB,
    )
    .unwrap();

    let y_fmt = PlaneFormat::new(PlaneRole::Y, SampleType::U16).with_packing(BitPacking::P010);
    let uv_fmt = PlaneFormat::new(PlaneRole::CbCr, SampleType::U16)
        .with_packing(BitPacking::P010)
        .with_subsampling(2, 2);
    let spec = PlanarCaptureSpec::new(
        coded,
        vis,
        [y_fmt, uv_fmt],
        ChromaSiting::Left,
        Cicp::new(1, 13, 1, false),
    );
    let plan =
        PlanarCapturePlan::build(&PixelFormat::RGB8_SRGB, spec, &ConvertPolicy::permissive())
            .unwrap();

    let (yw, yh) = y_fmt.cropped_plane_dimensions(vis).unwrap();
    let (uvw, uvh) = uv_fmt.cropped_plane_dimensions(vis).unwrap();
    assert_eq!((yw, yh), (11, 9));
    assert_eq!((uvw, uvh), (6, 5));

    let y_stride_u16 = (yw as usize) + 3;
    let uv_stride_u16 = (uvw as usize) * 2 + 3;
    let mut y_words = vec![0u16; y_stride_u16 * yh as usize];
    let mut uv_words = vec![0u16; uv_stride_u16 * uvh as usize];
    let y_buf: &mut [u8] = bytemuck::cast_slice_mut(&mut y_words);
    let uv_buf: &mut [u8] = bytemuck::cast_slice_mut(&mut uv_words);
    let y_stride = y_stride_u16 * 2;
    let uv_stride = uv_stride_u16 * 2;

    let mut scratch_vec = vec![0.0f64; plan.required_scratch_f64s()];
    let mut borrowed_scratch = CaptureScratch::from_f64_slice(&mut scratch_vec);

    let mut dst = PlanarViewMut::from_array_with_crop_origin(
        vis.width,
        vis.height,
        (vis.x, vis.y),
        ChromaSiting::Left,
        [
            PlaneViewMut::new(y_buf, yw, yh, y_stride, y_fmt).unwrap(),
            PlaneViewMut::new(uv_buf, uvw, uvh, uv_stride, uv_fmt).unwrap(),
        ],
    )
    .unwrap();

    let alloc_info = allocation_counter::measure(|| {
        plan.execute(src_view.clone(), &mut dst, &mut borrowed_scratch)
            .unwrap();
    });
    assert_eq!(
        alloc_info.count_total, 0,
        "borrowed CaptureScratch execute must perform zero heap allocations on native route"
    );

    let tagged = dst
        .as_view()
        .with_color(ColorSpace::Rgb {
            primaries: Primaries::Bt709,
            transfer: Transfer::Srgb,
            range: SignalRange::Narrow,
            alpha: AlphaMode::None,
            white: None,
        })
        .with_matrix_coefficients(Cicp::new(1, 13, 1, false).matrix())
        .with_range_override(SignalRange::Narrow);
    let mut out_bytes = vec![0u8; (vis.width * vis.height * 3) as usize];
    let mut out_view = PixelViewMut::new(
        &mut out_bytes,
        vis.width,
        vis.height,
        (vis.width * 3) as usize,
        &formats::RGB8_SRGB,
    )
    .unwrap();
    reconstruct_planar_into(
        &tagged,
        &mut out_view,
        &PlanarReconstructionOptions::bilinear(),
    )
    .unwrap();
    for px in out_bytes.as_chunks::<3>().0 {
        assert!(
            px[0].abs_diff(200) <= 1 && px[1].abs_diff(100) <= 1 && px[2].abs_diff(50) <= 1,
            "constant P010 roundtrip drifted: got {px:?}"
        );
    }

    let mut too_small = vec![0.0f64; plan.required_scratch_f64s().saturating_sub(1)];
    let mut bad_scratch = CaptureScratch::from_f64_slice(&mut too_small);
    let err = plan
        .execute(src_view, &mut dst, &mut bad_scratch)
        .unwrap_err();
    assert_eq!(*err.error(), ConvertError::DimensionMismatch);

    // Non-constant odd-origin (x=1) 4:2:2 phase witness:
    // Coded grid x=0..5, visible rect x=1, w=4: local pixels [0:(0,0,255), 1:(0,0,0), 2:(0,0,0), 3:(0,0,0)].
    // Global chroma index 0 covers global x=0..1 (cb_ox0 = 1/2 = 0):
    //   - Center siting: center_global = 0.5 -> center_local = 0.5 - 1 = -0.5;
    //     triangle radius 2 taps k=-2,-1,0,1 -> clamped local [0,0,0,1] weights [1,3,3,1]/8 -> 7/8 blue!
    //     Cb = 128 + (7/8)*0.5*255 = 239.5625 -> 240.
    //   - Left siting: center_global = 0.0 -> center_local = -1.0;
    //     triangle radius 2 taps k=-2,-1,0 -> clamped local [0,0,0] weights [1,2,1]/4 -> 4/4 blue!
    //     Cb = 128 + 1.0*0.5*255 = 255.5 -> clamped 255.
    let odd_src_bytes = [0u8, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let odd_src = PixelView::new(&odd_src_bytes, 4, 1, 12, &formats::RGB8_SRGB).unwrap();
    let y8 = PlaneFormat::new(PlaneRole::Y, SampleType::U8);
    let cb8 = PlaneFormat::new(PlaneRole::Cb, SampleType::U8).with_subsampling(2, 1);
    let cr8 = PlaneFormat::new(PlaneRole::Cr, SampleType::U8).with_subsampling(2, 1);
    let odd_vis = PixelRect::new(1, 0, 4, 1);
    let mut own_scratch = CaptureScratch::owned();
    for (siting, expected_cb0) in [(ChromaSiting::Center, 240u8), (ChromaSiting::Left, 255u8)] {
        let odd_spec = PlanarCaptureSpec::new(
            (6, 1),
            odd_vis,
            [y8, cb8, cr8],
            siting,
            Cicp::new(1, 13, 1, true),
        );
        let odd_plan = PlanarCapturePlan::build(
            &PixelFormat::RGB8_SRGB,
            odd_spec,
            &ConvertPolicy::permissive(),
        )
        .unwrap();
        let (cw, ch) = cb8.cropped_plane_dimensions(odd_vis).unwrap();
        assert_eq!((cw, ch), (3, 1));
        let mut oy = [0u8; 4];
        let mut ocb = [0u8; 3];
        let mut ocr = [0u8; 3];
        let mut odst = PlanarViewMut::from_array_with_crop_origin(
            4,
            1,
            (1, 0),
            siting,
            [
                PlaneViewMut::new(&mut oy, 4, 1, 4, y8).unwrap(),
                PlaneViewMut::new(&mut ocb, 3, 1, 3, cb8).unwrap(),
                PlaneViewMut::new(&mut ocr, 3, 1, 3, cr8).unwrap(),
            ],
        )
        .unwrap();
        odd_plan
            .execute(odd_src.clone(), &mut odst, &mut own_scratch)
            .unwrap();
        assert_eq!(
            ocb[0], expected_cb0,
            "odd-origin x=1 phase mismatch for {siting:?}: got {}",
            ocb[0]
        );
    }
}

#[test]
fn unknown_siting_refusal_fallback_tagging_and_conflict_rejection() {
    let planes = [
        PlaneFormat::new(PlaneRole::Y, SampleType::U8),
        PlaneFormat::new(PlaneRole::Cb, SampleType::U8).with_subsampling(2, 2),
        PlaneFormat::new(PlaneRole::Cr, SampleType::U8).with_subsampling(2, 2),
    ];
    let spec = PlanarCaptureSpec::full_frame(
        4,
        4,
        planes,
        ChromaSiting::Unknown,
        Cicp::new(1, 13, 1, false),
    );
    let err = PlanarCapturePlan::build(
        &PixelFormat::RGB8_SRGB,
        spec.clone(),
        &ConvertPolicy::permissive(),
    )
    .unwrap_err();
    assert_eq!(*err.error(), ConvertError::UnsupportedRoute);

    // Explicit fallback sets resolved_siting AND tags owned PlanarBuf with Left!
    let plan_fb = PlanarCapturePlan::build_with_options(
        &PixelFormat::RGB8_SRGB,
        spec.clone(),
        &ConvertPolicy::permissive(),
        PlanarCaptureOptions::siting_aware().with_unknown_siting_fallback(ChromaSiting::Left),
        None,
    )
    .unwrap();
    assert_eq!(plan_fb.resolved_siting(), ChromaSiting::Left);

    let src = make_rgb8_pattern(4, 4);
    let mut scratch = CaptureScratch::owned();
    let owned = plan_fb.capture_to_buf(src.as_view(), &mut scratch).unwrap();
    assert_eq!(
        owned.chroma_siting(),
        ChromaSiting::Left,
        "owned PlanarBuf must be tagged with resolved_siting, not Unknown"
    );

    // Borrowed PlanarViewMut with conflicting concrete siting (Center != resolved Left) MUST fail!
    let mut y = [0u8; 16];
    let mut cb = [0u8; 4];
    let mut cr = [0u8; 4];
    let mut conflict = PlanarViewMut::from_array_with_crop_origin(
        4,
        4,
        (0, 0),
        ChromaSiting::Center,
        [
            PlaneViewMut::new(&mut y, 4, 4, 4, planes[0]).unwrap(),
            PlaneViewMut::new(&mut cb, 2, 2, 2, planes[1]).unwrap(),
            PlaneViewMut::new(&mut cr, 2, 2, 2, planes[2]).unwrap(),
        ],
    )
    .unwrap();
    let conflict_err = plan_fb
        .execute(src.as_view(), &mut conflict, &mut scratch)
        .unwrap_err();
    assert_eq!(*conflict_err.error(), ConvertError::UnsupportedRoute);

    // Borrowed PlanarViewMut with Unknown siting MUST ALSO fail when resolved_siting is Left!
    let mut unk_dst = PlanarViewMut::from_array_with_crop_origin(
        4,
        4,
        (0, 0),
        ChromaSiting::Unknown,
        [
            PlaneViewMut::new(&mut y, 4, 4, 4, planes[0]).unwrap(),
            PlaneViewMut::new(&mut cb, 2, 2, 2, planes[1]).unwrap(),
            PlaneViewMut::new(&mut cr, 2, 2, 2, planes[2]).unwrap(),
        ],
    )
    .unwrap();
    let unk_err = plan_fb
        .execute(src.as_view(), &mut unk_dst, &mut scratch)
        .unwrap_err();
    assert_eq!(*unk_err.error(), ConvertError::UnsupportedRoute);

    // Nearest with unresolved Unknown siting MUST fail (siting determines which pixel nearest picks!).
    let nearest_unk = PlanarCapturePlan::build_with_options(
        &PixelFormat::RGB8_SRGB,
        spec.clone(),
        &ConvertPolicy::permissive(),
        PlanarCaptureOptions::nearest(),
        None,
    )
    .unwrap_err();
    assert_eq!(*nearest_unk.error(), ConvertError::UnsupportedRoute);

    // Nearest with explicit Center fallback on 2x1 black+blue selects blue (Cb = 255)!
    let pair_src =
        PixelBuf::from_typed(vec![0u8, 0, 0, 0, 0, 255], 2, 1, PixelFormat::RGB8_SRGB).unwrap();
    let pair_planes = [
        PlaneFormat::new(PlaneRole::Y, SampleType::U8),
        PlaneFormat::new(PlaneRole::Cb, SampleType::U8).with_subsampling(2, 1),
        PlaneFormat::new(PlaneRole::Cr, SampleType::U8).with_subsampling(2, 1),
    ];
    let pair_spec = PlanarCaptureSpec::full_frame(
        2,
        1,
        pair_planes,
        ChromaSiting::Unknown,
        Cicp::new(1, 13, 1, true),
    );
    let nearest_center = PlanarCapturePlan::build_with_options(
        &PixelFormat::RGB8_SRGB,
        pair_spec,
        &ConvertPolicy::permissive(),
        PlanarCaptureOptions::nearest().with_unknown_siting_fallback(ChromaSiting::Center),
        None,
    )
    .unwrap();
    assert_eq!(nearest_center.resolved_siting(), ChromaSiting::Center);
    let pair_cap = nearest_center
        .capture_to_buf(pair_src.as_view(), &mut scratch)
        .unwrap();
    assert_eq!(pair_cap.chroma_siting(), ChromaSiting::Center);
    assert_eq!(
        pair_cap
            .as_view()
            .plane_by_role(PlaneRole::Cb)
            .unwrap()
            .as_code_plane(0)
            .unwrap()
            .code(0, 0)
            .unwrap(),
        255
    );

    // BoxAverage succeeds with Unknown siting, and borrowed dst MUST also be Unknown (rejects Center!).
    let box_plan = PlanarCapturePlan::build_with_options(
        &PixelFormat::RGB8_SRGB,
        spec,
        &ConvertPolicy::permissive(),
        PlanarCaptureOptions::box_average(),
        None,
    )
    .unwrap();
    assert_eq!(box_plan.resolved_siting(), ChromaSiting::Unknown);
    let mut box_bad_dst = PlanarViewMut::from_array_with_crop_origin(
        4,
        4,
        (0, 0),
        ChromaSiting::Center,
        [
            PlaneViewMut::new(&mut y, 4, 4, 4, planes[0]).unwrap(),
            PlaneViewMut::new(&mut cb, 2, 2, 2, planes[1]).unwrap(),
            PlaneViewMut::new(&mut cr, 2, 2, 2, planes[2]).unwrap(),
        ],
    )
    .unwrap();
    let box_err = box_plan
        .execute(src.as_view(), &mut box_bad_dst, &mut scratch)
        .unwrap_err();
    assert_eq!(*box_err.error(), ConvertError::UnsupportedRoute);
    let mut box_ok_dst = PlanarViewMut::from_array_with_crop_origin(
        4,
        4,
        (0, 0),
        ChromaSiting::Unknown,
        [
            PlaneViewMut::new(&mut y, 4, 4, 4, planes[0]).unwrap(),
            PlaneViewMut::new(&mut cb, 2, 2, 2, planes[1]).unwrap(),
            PlaneViewMut::new(&mut cr, 2, 2, 2, planes[2]).unwrap(),
        ],
    )
    .unwrap();
    box_plan
        .execute(src.as_view(), &mut box_ok_dst, &mut scratch)
        .unwrap();
}

#[test]
fn gamut_widening_allowed_under_forbid_and_narrow_f32_refused() {
    let planes_u8 = [
        PlaneFormat::new(PlaneRole::Y, SampleType::U8),
        PlaneFormat::new(PlaneRole::Cb, SampleType::U8),
        PlaneFormat::new(PlaneRole::Cr, SampleType::U8),
    ];
    // sRGB (BT.709 primaries) -> BT.2020 (strictly wider primaries) MUST succeed under GamutPolicy::Forbid!
    let spec_bt2020 = PlanarCaptureSpec::full_frame(
        4,
        4,
        planes_u8,
        ChromaSiting::Center,
        Cicp::new(9, 13, 9, true),
    );
    PlanarCapturePlan::build(
        &PixelFormat::RGB8_SRGB,
        spec_bt2020,
        &ConvertPolicy::permissive().with_gamut(GamutPolicy::Forbid),
    )
    .unwrap();

    // BT.2020 -> BT.709 (narrowing) MUST fail under GamutPolicy::Forbid!
    let src_bt2020 = PixelFormat::new(
        SampleLayout::RGB8,
        ColorSpace::Rgb {
            primaries: Primaries::Bt2020,
            transfer: Transfer::Srgb,
            range: SignalRange::Full,
            alpha: AlphaMode::None,
            white: None,
        },
    );
    let spec_bt709 = PlanarCaptureSpec::full_frame(
        4,
        4,
        planes_u8,
        ChromaSiting::Center,
        Cicp::new(1, 13, 1, true),
    );
    let err = PlanarCapturePlan::build(
        &src_bt2020,
        spec_bt709,
        &ConvertPolicy::permissive().with_gamut(GamutPolicy::Forbid),
    )
    .unwrap_err();
    assert_eq!(*err.error(), ConvertError::GamutLossForbidden);

    // Narrow F32 PlanarRgb MUST be refused (float planes carry normalized full range only).
    let narrow_f32_spec = PlanarCaptureSpec::full_frame(
        2,
        2,
        [
            PlaneFormat::new(PlaneRole::R, SampleType::F32),
            PlaneFormat::new(PlaneRole::G, SampleType::F32),
            PlaneFormat::new(PlaneRole::B, SampleType::F32),
        ],
        ChromaSiting::Unknown,
        Cicp::new(1, 8, 0, false),
    );
    let err_f32 = PlanarCapturePlan::build(
        &formats::RGBF32_LINEAR,
        narrow_f32_spec,
        &ConvertPolicy::permissive(),
    )
    .unwrap_err();
    assert_eq!(*err_f32.error(), ConvertError::UnsupportedRoute);
}

#[test]
fn alpha_policy_reject_matte_and_alpha_plane_preservation() {
    let rgba_fmt = PixelFormat::RGBA8_SRGB;
    let src = PixelBuf::from_typed(vec![255u8, 0, 0, 128, 0, 255, 0, 255], 2, 1, rgba_fmt).unwrap();
    let planes_no_alpha = [
        PlaneFormat::new(PlaneRole::Y, SampleType::U8),
        PlaneFormat::new(PlaneRole::Cb, SampleType::U8),
        PlaneFormat::new(PlaneRole::Cr, SampleType::U8),
    ];
    let spec_no_alpha =
        PlanarCaptureSpec::full_frame(2, 1, planes_no_alpha, ChromaSiting::Center, Cicp::SRGB);

    let err = PlanarCapturePlan::build(
        src.format(),
        spec_no_alpha.clone(),
        &ConvertPolicy::permissive().with_alpha(AlphaPolicy::Forbid),
    )
    .unwrap_err();
    assert_eq!(*err.error(), ConvertError::AlphaLossForbidden);

    let plan_matte = PlanarCapturePlan::build(
        src.format(),
        spec_no_alpha,
        &ConvertPolicy::permissive().with_alpha(AlphaPolicy::CompositeOnto(Color::BLACK)),
    )
    .unwrap();
    let mut scratch = CaptureScratch::owned();
    let buf_matte = plan_matte
        .capture_to_buf(src.as_view(), &mut scratch)
        .unwrap();
    let rec = reconstruct_planar_to_buf(
        &buf_matte.as_view(),
        SampleLayout::RGB8,
        &PlanarReconstructionOptions::nearest(),
    )
    .unwrap();
    let px = rec.as_view().as_contiguous_bytes().unwrap();
    assert!(
        (180..=195).contains(&px[0]) && px[1] == 0 && px[2] == 0,
        "linear matte expected ~188 red, got {:?}",
        &px[..3]
    );

    let planes_alpha = [
        PlaneFormat::new(PlaneRole::Y, SampleType::U8),
        PlaneFormat::new(PlaneRole::Cb, SampleType::U8),
        PlaneFormat::new(PlaneRole::Cr, SampleType::U8),
        PlaneFormat::new(PlaneRole::Alpha, SampleType::U8),
    ];
    let spec_alpha =
        PlanarCaptureSpec::full_frame(2, 1, planes_alpha, ChromaSiting::Center, Cicp::SRGB)
            .with_alpha(AlphaMode::Straight);
    let plan_alpha =
        PlanarCapturePlan::build(src.format(), spec_alpha, &ConvertPolicy::lossless()).unwrap();
    let buf_alpha = plan_alpha
        .capture_to_buf(src.as_view(), &mut scratch)
        .unwrap();
    let rec_a = reconstruct_planar_to_buf(
        &buf_alpha.as_view(),
        SampleLayout::RGBA8,
        &PlanarReconstructionOptions::nearest(),
    )
    .unwrap();
    assert_eq!(
        rec_a.as_view().as_contiguous_bytes().unwrap(),
        src.as_view().as_contiguous_bytes().unwrap()
    );
}

#[test]
fn monochrome_planar_rgb_f32_hdr_pq_hlg_and_hdr_metadata_clearing() {
    let mut scratch = CaptureScratch::owned();
    let gray =
        PixelBuf::from_typed(vec![0u8, 128, 255, 64], 2, 2, PixelFormat::GRAY8_SRGB).unwrap();
    let mono_spec = PlanarCaptureSpec::full_frame(
        2,
        2,
        [PlaneFormat::new(PlaneRole::Y, SampleType::U8)],
        ChromaSiting::Unknown,
        Cicp::new(2, 13, 0, false),
    );
    let mono_plan =
        PlanarCapturePlan::build(gray.format(), mono_spec, &ConvertPolicy::permissive()).unwrap();
    let mono_buf = mono_plan
        .capture_to_buf(gray.as_view(), &mut scratch)
        .unwrap();
    assert_eq!(
        mono_buf
            .as_view()
            .plane_by_role(PlaneRole::Y)
            .unwrap()
            .row(0)[0],
        16
    );
    assert_eq!(
        mono_buf
            .as_view()
            .plane_by_role(PlaneRole::Y)
            .unwrap()
            .row(1)[0],
        235
    );

    let lin = PixelBuf::from_typed(
        vec![0.25f32, 0.5, 0.75],
        1,
        1,
        formats::RGBF32_LINEAR.clone(),
    )
    .unwrap();
    let prgb_spec = PlanarCaptureSpec::full_frame(
        1,
        1,
        [
            PlaneFormat::new(PlaneRole::R, SampleType::F32),
            PlaneFormat::new(PlaneRole::G, SampleType::F32),
            PlaneFormat::new(PlaneRole::B, SampleType::F32),
        ],
        ChromaSiting::Unknown,
        Cicp::LINEAR_SRGB,
    );
    let prgb_plan =
        PlanarCapturePlan::build(lin.format(), prgb_spec, &ConvertPolicy::lossless()).unwrap();
    let prgb_buf = prgb_plan
        .capture_to_buf(lin.as_view(), &mut scratch)
        .unwrap();
    let rebuilt = reconstruct_planar_to_buf(
        &prgb_buf.as_view(),
        SampleLayout::RGBF32,
        &PlanarReconstructionOptions::nearest(),
    )
    .unwrap();
    let out_f: &[f32] = bytemuck::cast_slice(rebuilt.as_view().as_contiguous_bytes().unwrap());
    assert_eq!(out_f, &[0.25, 0.5, 0.75]);

    #[cfg(feature = "hdr-transfers")]
    {
        let pq_planes = [
            PlaneFormat::new(PlaneRole::Y, SampleType::U16)
                .with_packing(BitPacking::Lsb { bits: 10 }),
            PlaneFormat::new(PlaneRole::Cb, SampleType::U16)
                .with_packing(BitPacking::Lsb { bits: 10 })
                .with_subsampling(2, 2),
            PlaneFormat::new(PlaneRole::Cr, SampleType::U16)
                .with_packing(BitPacking::Lsb { bits: 10 })
                .with_subsampling(2, 2),
        ];
        let pq_spec = PlanarCaptureSpec::full_frame(
            2,
            2,
            pq_planes,
            ChromaSiting::TopLeft,
            Cicp::BT2100_PQ_YCBCR_LIMITED,
        );
        let hdr_meta = HdrMetadata {
            content_light_level: Some(zenpixels::ContentLightLevel::new(1000, 400)),
            mastering: None,
        };
        let lin_white = PixelBuf::from_typed(
            vec![1.0f32; 2 * 2 * 3],
            2,
            2,
            formats::RGBF32_LINEAR.clone(),
        )
        .unwrap()
        .with_hdr(hdr_meta);
        let pq_plan = PlanarCapturePlan::build(
            lin_white.format(),
            pq_spec,
            &ConvertPolicy::working_linear().with_sdr_white(DiffuseWhite::BT2408),
        )
        .unwrap();
        let pq_buf = pq_plan
            .capture_to_buf(lin_white.as_view(), &mut scratch)
            .unwrap();
        assert_eq!(
            pq_buf.hdr(),
            hdr_meta,
            "HDR PQ destination retains HDR metadata"
        );
        let y_code = pq_buf
            .as_view()
            .plane_by_role(PlaneRole::Y)
            .unwrap()
            .as_code_plane(0)
            .unwrap()
            .code(0, 0)
            .unwrap();
        assert!(
            (571..=575).contains(&y_code),
            "expected PQ 203-nit narrow 10-bit luma ~573, got {y_code}"
        );

        // Capturing an HDR source into SDR 8-bit YCbCr MUST clear HdrMetadata!
        let sdr_spec = PlanarCaptureSpec::full_frame(
            2,
            2,
            [
                PlaneFormat::new(PlaneRole::Y, SampleType::U8),
                PlaneFormat::new(PlaneRole::Cb, SampleType::U8),
                PlaneFormat::new(PlaneRole::Cr, SampleType::U8),
            ],
            ChromaSiting::Center,
            Cicp::new(1, 13, 1, false),
        );
        let sdr_plan =
            PlanarCapturePlan::build(lin_white.format(), sdr_spec, &ConvertPolicy::permissive())
                .unwrap();
        let sdr_buf = sdr_plan
            .capture_to_buf(lin_white.as_view(), &mut scratch)
            .unwrap();
        assert_eq!(
            sdr_buf.hdr(),
            HdrMetadata::default(),
            "SDR destination must clear source HDR metadata"
        );

        let hlg_spec = PlanarCaptureSpec::full_frame(
            2,
            2,
            pq_planes,
            ChromaSiting::TopLeft,
            Cicp::BT2100_HLG_YCBCR_LIMITED,
        );
        let hlg_plan = PlanarCapturePlan::build(
            lin_white.format(),
            hlg_spec,
            &ConvertPolicy::working_linear()
                .with_hlg_rendering(HlgRendering::REFERENCE_1000_NIT)
                .with_sdr_white(DiffuseWhite::BT2408),
        )
        .unwrap();
        let hlg_buf = hlg_plan
            .capture_to_buf(lin_white.as_view(), &mut scratch)
            .unwrap();
        let hlg_y = hlg_buf
            .as_view()
            .plane_by_role(PlaneRole::Y)
            .unwrap()
            .as_code_plane(0)
            .unwrap()
            .code(0, 0)
            .unwrap();
        assert!(
            hlg_y > 600 && hlg_y <= 940,
            "unexpected HLG diffuse white code {hlg_y}"
        );
        let _ = (HlgMode::Scene, Nits::new_positive_or_panic(100.0));
    }
}

#[test]
fn color_services_strict_no_icc_and_provider_swaps() {
    let adobe_icc = Arc::<[u8]>::from(zenpixels_convert::icc_profiles::ADOBE_RGB);
    let p3_icc = Arc::<[u8]>::from(zenpixels_convert::icc_profiles::DISPLAY_P3_V4);
    let src_icc_fmt = PixelFormat::new(
        SampleLayout::RGB8,
        ColorSpace::Icc {
            profile: Arc::clone(&adobe_icc),
            color_channels: 3,
            alpha: AlphaMode::None,
            intent: zenpixels_convert::RenderingIntent::RelativeColorimetric,
        },
    );
    let dst_icc_same_bgr = PixelFormat::new(
        SampleLayout::BGR8,
        ColorSpace::Icc {
            profile: Arc::clone(&adobe_icc),
            color_channels: 3,
            alpha: AlphaMode::None,
            intent: zenpixels_convert::RenderingIntent::RelativeColorimetric,
        },
    );
    let dst_icc_diff = PixelFormat::new(
        SampleLayout::RGB8,
        ColorSpace::Icc {
            profile: Arc::clone(&p3_icc),
            color_channels: 3,
            alpha: AlphaMode::None,
            intent: zenpixels_convert::RenderingIntent::RelativeColorimetric,
        },
    );

    let strict = ColorServices::strict_no_icc();
    assert!(strict.is_frozen());
    let policy_strict = ConvertPolicy::permissive().with_color_services(&strict);
    assert!(!policy_strict.allow_icc_substitution);
    assert!(policy_strict.cms.is_none());

    // 1a. Known-space math works under StrictNoIcc.
    let dst_p3_fmt = PixelFormat::new(
        SampleLayout::RGB8,
        ColorSpace::Rgb {
            primaries: Primaries::DisplayP3,
            transfer: Transfer::Srgb,
            range: SignalRange::Full,
            alpha: AlphaMode::None,
            white: None,
        },
    );
    let src_srgb = make_rgb8_pattern(4, 2);
    let out_p3 = convert_to_buf(&src_srgb.as_view(), &dst_p3_fmt, &policy_strict).unwrap();
    assert_eq!(out_p3.format(), &dst_p3_fmt);

    // 1b. Validated same-profile ICC carriage WITH actual RGB8 -> BGR8 swizzle works under StrictNoIcc!
    let swizzle_plan =
        RowPlan::build(&src_icc_fmt, &dst_icc_same_bgr, &policy_strict, None).unwrap();
    let mut row_out = [0u8; 6];
    swizzle_plan
        .convert_row(&[10, 20, 30, 40, 50, 60], &mut row_out, 2)
        .unwrap();
    assert_eq!(row_out, [30, 20, 10, 60, 50, 40]);

    // 1c. Malformed same-profile ICC MUST be refused even for identity under StrictNoIcc!
    let bad_icc = PixelFormat::new(
        SampleLayout::RGB8,
        ColorSpace::Icc {
            profile: Arc::from(&b"not-a-valid-icc-profile"[..]),
            color_channels: 3,
            alpha: AlphaMode::None,
            intent: zenpixels_convert::RenderingIntent::RelativeColorimetric,
        },
    );
    assert!(RowPlan::build(&bad_icc, &bad_icc, &policy_strict, None).is_err());

    // 1d. ICC evaluation AND matrix+TRC ICC substitution MUST BOTH BE REFUSED under StrictNoIcc!
    let err_sub =
        RowPlan::build(&src_icc_fmt, &PixelFormat::RGB8_SRGB, &policy_strict, None).unwrap_err();
    assert_eq!(*err_sub.error(), ConvertError::CmsRequired);
    let err_eval = RowPlan::build(&src_icc_fmt, &dst_icc_diff, &policy_strict, None).unwrap_err();
    assert_eq!(*err_eval.error(), ConvertError::CmsRequired);

    // 2. Inheritance resolution & freezing:
    let inherit = ColorServices::inherit();
    assert!(!inherit.is_frozen());
    let frozen_no_default = inherit.freeze_or_strict(None);
    assert!(frozen_no_default.is_frozen());
    assert!(matches!(
        frozen_no_default.cms_choice(),
        CmsChoice::StrictNoIcc
    ));

    // 3. Swapping two real CMS providers (MoxCms vs Lcms) via CmsBinding alone:
    #[cfg(all(feature = "cms-moxcms", feature = "cms-lcms"))]
    {
        let mox = zenpixels_convert::MoxCms::binding();
        let lcms = zenpixels_convert::Lcms::binding();
        assert_eq!(mox.id(), "org.imazen.cms.moxcms");
        assert_eq!(lcms.id(), "org.imazen.cms.lcms");
        assert!(mox.revision().contains("moxcms"));
        assert!(lcms.revision().contains("lcms"));
        assert!(!mox.quote().supports_absolute_colorimetric);
        assert!(lcms.quote().supports_absolute_colorimetric);
        // Built-in quotes are honest: arbitrary-profile build memory & stack are unknown (`None`)!
        assert_eq!(mox.quote().build_memory_bytes, None);
        assert_eq!(lcms.quote().build_memory_bytes, None);
        assert!(!mox.quote().small_stack_qualified);
        assert!(!lcms.quote().small_stack_qualified);
        assert_eq!(lcms.quote().apply_scratch_bytes, Some(0));

        let svc_mox = ColorServices::with_cms_binding(mox.clone());
        let svc_lcms = ColorServices::with_cms_binding(lcms.clone());
        let resolved = inherit.resolve_against(&svc_mox);
        assert_eq!(resolved.cms_binding().unwrap().cache_key(), mox.cache_key());
        let override_strict = strict.resolve_against(&svc_mox);
        assert!(matches!(
            override_strict.cms_choice(),
            CmsChoice::StrictNoIcc
        ));

        let p_mox = ConvertPolicy::permissive()
            .with_icc_substitution(false)
            .with_color_services(&svc_mox);
        let p_lcms = ConvertPolicy::permissive()
            .with_icc_substitution(false)
            .with_color_services(&svc_lcms);

        let plan_mox = RowPlan::build(&src_icc_fmt, &dst_icc_diff, &p_mox, None).unwrap();
        let plan_lcms = RowPlan::build(&src_icc_fmt, &dst_icc_diff, &p_lcms, None).unwrap();
        let input = [255u8, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128];
        let mut out_mox = [0u8; 12];
        let mut out_lcms = [0u8; 12];
        plan_mox.convert_row(&input, &mut out_mox, 4).unwrap();
        plan_lcms.convert_row(&input, &mut out_lcms, 4).unwrap();
        for (a, b) in out_mox.iter().zip(out_lcms.iter()) {
            assert!(
                a.abs_diff(*b) <= 2,
                "MoxCms vs Lcms disagreed: {out_mox:?} vs {out_lcms:?}"
            );
        }
    }
}

#[test]
fn depth_and_range_forbid_enforced_in_planar_capture() {
    let planes_u8 = [
        PlaneFormat::new(PlaneRole::Y, SampleType::U8),
        PlaneFormat::new(PlaneRole::Cb, SampleType::U8),
        PlaneFormat::new(PlaneRole::Cr, SampleType::U8),
    ];
    let spec_narrow = PlanarCaptureSpec::full_frame(
        4,
        4,
        planes_u8,
        ChromaSiting::Center,
        Cicp::new(1, 13, 1, false),
    );
    let err = PlanarCapturePlan::build(
        &PixelFormat::RGB8_SRGB,
        spec_narrow,
        &ConvertPolicy::permissive().with_range(RangePolicy::Forbid),
    )
    .unwrap_err();
    assert_eq!(*err.error(), ConvertError::RangeConversionForbidden);

    let spec_full =
        PlanarCaptureSpec::full_frame(4, 4, planes_u8, ChromaSiting::Center, Cicp::SRGB);
    let err_depth = PlanarCapturePlan::build(
        &formats::RGB16_SRGB,
        spec_full,
        &ConvertPolicy::permissive().with_depth(DepthPolicy::Forbid),
    )
    .unwrap_err();
    assert_eq!(*err_depth.error(), ConvertError::DepthReductionForbidden);
}
