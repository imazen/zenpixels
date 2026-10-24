// libjxl oracle for tools/jzazbz-bench: recomputes the bench's libjxl-exact
// arms with libjxl's own code and compares bit for bit.
//
// - XYB: libjxl `CubeRootAndAdd` and `kOpsinAbsorbanceMatrix` /
//   `kOpsinAbsorbanceBias` come from libjxl's headers
//   (`lib/jxl/base/fast_math-inl.h`, `lib/jxl/cms/opsin_params.h`).
//   `LinearRGBToXYB` + `ComputePremulAbsorb` (intensity_target 255, so the
//   premultiplier is exactly 1) are file-static in `lib/jxl/enc_xyb.cc`;
//   their op sequence is reproduced below line for line.
// - Butteraugli: libjxl `FastLog2f` from the header; `Gamma`,
//   `OpsinAbsorbance` and the `OpsinDynamicsImage` per-pixel body are
//   file-static in `lib/jxl/butteraugli/butteraugli.cc` and copied verbatim
//   (BSD-3-Clause, Copyright (c) the JPEG XL Project Authors).
//
// Static AVX2 target (as libjxl's AVX2 dispatch arm): MulAdd is a real FMA.
// Build: see tools/jzazbz-bench/oracle/build.sh
// Run:   libjxl_oracle DIR   (DIR from `jzazbz-bench --dump DIR` or
//        `butteraugli-opsin-dump DIR`); checks every output set present.

#include <hwy/aligned_allocator.h>
#include <hwy/highway.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "lib/jxl/base/fast_math-inl.h"
#include "lib/jxl/cms/opsin_params.h"

namespace jxl {
namespace HWY_NAMESPACE {
using namespace hwy::HWY_NAMESPACE;

// ---- butteraugli.cc (verbatim) -------------------------------------------
template <class DF, class V>
V Gamma(const DF df, V v) {
  const auto kRetMul = Set(df, 19.245013259874995f * kInvLog2e);
  const auto kRetAdd = Set(df, -23.16046239805755);
  v = ZeroIfNegative(v);
  const auto biased = Add(v, Set(df, 9.9710635769299145));
  const auto log = FastLog2f(df, biased);
  return MulAdd(kRetMul, log, kRetAdd);
}

template <bool Clamp, class DF, class V>
HWY_INLINE void OpsinAbsorbance(const DF df, const V& in0, const V& in1,
                                const V& in2, V* out0, V* out1, V* out2) {
  static const double mixi0 = 0.29956550340058319;
  static const double mixi1 = 0.63373087833825936;
  static const double mixi2 = 0.077705617820981968;
  static const double mixi3 = 1.7557483643287353;
  static const double mixi4 = 0.22158691104574774;
  static const double mixi5 = 0.69391388044116142;
  static const double mixi6 = 0.0987313588422;
  static const double mixi7 = 1.7557483643287353;
  static const double mixi8 = 0.02;
  static const double mixi9 = 0.02;
  static const double mixi10 = 0.20480129041026129;
  static const double mixi11 = 12.226454707163354;
  const V mix0 = Set(df, mixi0);
  const V mix1 = Set(df, mixi1);
  const V mix2 = Set(df, mixi2);
  const V mix3 = Set(df, mixi3);
  const V mix4 = Set(df, mixi4);
  const V mix5 = Set(df, mixi5);
  const V mix6 = Set(df, mixi6);
  const V mix7 = Set(df, mixi7);
  const V mix8 = Set(df, mixi8);
  const V mix9 = Set(df, mixi9);
  const V mix10 = Set(df, mixi10);
  const V mix11 = Set(df, mixi11);
  *out0 = MulAdd(mix0, in0, MulAdd(mix1, in1, MulAdd(mix2, in2, mix3)));
  *out1 = MulAdd(mix4, in0, MulAdd(mix5, in1, MulAdd(mix6, in2, mix7)));
  *out2 = MulAdd(mix8, in0, MulAdd(mix9, in1, MulAdd(mix10, in2, mix11)));
  if (Clamp) {
    *out0 = Max(*out0, mix3);
    *out1 = Max(*out1, mix7);
    *out2 = Max(*out2, mix11);
  }
}

void ButteraugliRows(const float* row_r, const float* row_g, const float* row_b,
                     const float* row_blurred_r, const float* row_blurred_g,
                     const float* row_blurred_b, float* row_out_x,
                     float* row_out_y, float* row_out_b, size_t xsize,
                     float intensity_target) {
  const HWY_FULL(float) df;
  const auto intensity_target_multiplier = Set(df, intensity_target);
  const auto min = Set(df, 1e-4f);
  for (size_t x = 0; x < xsize; x += Lanes(df)) {
    auto sensitivity0 = Undefined(df);
    auto sensitivity1 = Undefined(df);
    auto sensitivity2 = Undefined(df);
    {
      auto pre_mixed0 = Undefined(df);
      auto pre_mixed1 = Undefined(df);
      auto pre_mixed2 = Undefined(df);
      OpsinAbsorbance<true>(
          df, Mul(Load(df, row_blurred_r + x), intensity_target_multiplier),
          Mul(Load(df, row_blurred_g + x), intensity_target_multiplier),
          Mul(Load(df, row_blurred_b + x), intensity_target_multiplier),
          &pre_mixed0, &pre_mixed1, &pre_mixed2);
      pre_mixed0 = Max(pre_mixed0, min);
      pre_mixed1 = Max(pre_mixed1, min);
      pre_mixed2 = Max(pre_mixed2, min);
      sensitivity0 = Div(Gamma(df, pre_mixed0), pre_mixed0);
      sensitivity1 = Div(Gamma(df, pre_mixed1), pre_mixed1);
      sensitivity2 = Div(Gamma(df, pre_mixed2), pre_mixed2);
      sensitivity0 = Max(sensitivity0, min);
      sensitivity1 = Max(sensitivity1, min);
      sensitivity2 = Max(sensitivity2, min);
    }
    auto cur_mixed0 = Undefined(df);
    auto cur_mixed1 = Undefined(df);
    auto cur_mixed2 = Undefined(df);
    OpsinAbsorbance<false>(
        df, Mul(Load(df, row_r + x), intensity_target_multiplier),
        Mul(Load(df, row_g + x), intensity_target_multiplier),
        Mul(Load(df, row_b + x), intensity_target_multiplier), &cur_mixed0,
        &cur_mixed1, &cur_mixed2);
    cur_mixed0 = Mul(cur_mixed0, sensitivity0);
    cur_mixed1 = Mul(cur_mixed1, sensitivity1);
    cur_mixed2 = Mul(cur_mixed2, sensitivity2);
    const auto min01 = Set(df, 1.7557483643287353f);
    const auto min2 = Set(df, 12.226454707163354f);
    cur_mixed0 = Max(cur_mixed0, min01);
    cur_mixed1 = Max(cur_mixed1, min01);
    cur_mixed2 = Max(cur_mixed2, min2);
    Store(Sub(cur_mixed0, cur_mixed1), df, row_out_x + x);
    Store(Add(cur_mixed0, cur_mixed1), df, row_out_y + x);
    Store(cur_mixed2, df, row_out_b + x);
  }
}

// ---- enc_xyb.cc LinearRGBToXYB + ComputePremulAbsorb (op for op) ---------
void XybRows(const float* r_in, const float* g_in, const float* b_in,
             float* valx, float* valy, float* valz, size_t xsize) {
  const HWY_FULL(float) d;
  const float mul = 255.0f / 255.0f;  // intensity_target / 255
  const float* bias = jxl::cms::kOpsinAbsorbanceBias.data();
  auto m = [&](size_t j, size_t i) {
    return Set(d, jxl::cms::kOpsinAbsorbanceMatrix[j][i] * mul);
  };
  const auto nb0 = Set(d, -cbrtf(jxl::cms::kOpsinAbsorbanceBias[0]));
  const auto nb1 = Set(d, -cbrtf(jxl::cms::kOpsinAbsorbanceBias[1]));
  const auto nb2 = Set(d, -cbrtf(jxl::cms::kOpsinAbsorbanceBias[2]));
  for (size_t x = 0; x < xsize; x += Lanes(d)) {
    const auto r = Load(d, r_in + x);
    const auto g = Load(d, g_in + x);
    const auto b = Load(d, b_in + x);
    auto mixed0 = MulAdd(m(0, 0), r, MulAdd(m(0, 1), g, MulAdd(m(0, 2), b, Set(d, bias[0]))));
    auto mixed1 = MulAdd(m(1, 0), r, MulAdd(m(1, 1), g, MulAdd(m(1, 2), b, Set(d, bias[1]))));
    auto mixed2 = MulAdd(m(2, 0), r, MulAdd(m(2, 1), g, MulAdd(m(2, 2), b, Set(d, bias[2]))));
    mixed0 = ZeroIfNegative(mixed0);
    mixed1 = ZeroIfNegative(mixed1);
    mixed2 = ZeroIfNegative(mixed2);
    mixed0 = CubeRootAndAdd(mixed0, nb0);
    mixed1 = CubeRootAndAdd(mixed1, nb1);
    mixed2 = CubeRootAndAdd(mixed2, nb2);
    const auto half = Set(d, 0.5f);
    Store(Mul(half, Sub(mixed0, mixed1)), d, valx + x);
    Store(Mul(half, Add(mixed0, mixed1)), d, valy + x);
    Store(mixed2, d, valz + x);
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace jxl

namespace {

bool Exists(const std::string& p) { return std::ifstream(p).good(); }

// Highway Load/Store need vector-aligned buffers.
struct Buf {
  hwy::AlignedFreeUniquePtr<float[]> p;
  size_t len;
  explicit Buf(size_t padded) : p(hwy::AllocateAligned<float>(padded)), len(padded) {
    std::memset(p.get(), 0, padded * sizeof(float));
  }
  float* data() { return p.get(); }
  const float* data() const { return p.get(); }
  float& operator[](size_t i) { return p[i]; }
  const float& operator[](size_t i) const { return p[i]; }
};

Buf ReadF32(const std::string& p, size_t n, size_t padded) {
  Buf v(padded);
  std::ifstream f(p, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", p.c_str());
    std::exit(2);
  }
  f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float));
  if (static_cast<size_t>(f.gcount()) != n * sizeof(float)) {
    std::fprintf(stderr, "short read %s\n", p.c_str());
    std::exit(2);
  }
  return v;
}

int64_t Ordered(float f) {
  int32_t i;
  std::memcpy(&i, &f, 4);
  return i < 0 ? static_cast<int64_t>(INT32_MIN) - i : i;
}

// Returns mismatching values; prints count, max ULP and the first mismatch.
size_t Compare(const char* what, const Buf& got,
               const Buf& want, size_t n) {
  size_t bad = 0, first = SIZE_MAX;
  int64_t max_ulp = 0;
  double max_abs = 0.0, max_mag = 0.0;
  for (size_t i = 0; i < n; ++i) {
    max_mag = std::fmax(max_mag, std::fabs(static_cast<double>(want[i])));
    if (std::memcmp(&got[i], &want[i], 4) != 0) {
      max_abs = std::fmax(max_abs, std::fabs(static_cast<double>(got[i]) - want[i]));
      if (first == SIZE_MAX) first = i;
      ++bad;
      int64_t u = Ordered(got[i]) - Ordered(want[i]);
      if (u < 0) u = -u;
      if (u > max_ulp) max_ulp = u;
    }
  }
  // ULP is misleading where the output crosses zero (X = l - m); the
  // absolute error against the plane's max |value| is the readable number.
  std::printf("  %-14s mismatches %zu / %zu, max %lld ULP, max abs %.3g (plane max |v| %.4g)",
              what, bad, n, static_cast<long long>(max_ulp), max_abs, max_mag);
  if (bad) {
    std::printf("  (first i=%zu got %.9g want %.9g)", first, got[first],
                want[first]);
  }
  std::printf("\n");
  return bad;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s DIR\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[1];
  size_t n = 0;
  float intensity = 80.0f;
  {
    std::ifstream meta(dir + "/meta.txt");
    std::string line;
    while (std::getline(meta, line)) {
      if (line.rfind("n=", 0) == 0) n = std::stoull(line.substr(2));
      if (line.rfind("intensity_target=", 0) == 0)
        intensity = std::stof(line.substr(17));
    }
  }
  if (n == 0) {
    std::fprintf(stderr, "no n= in %s/meta.txt\n", dir.c_str());
    return 2;
  }
  const size_t lanes = hwy::HWY_NAMESPACE::Lanes(hwy::HWY_NAMESPACE::ScalableTag<float>());
  const size_t padded = (n + lanes - 1) / lanes * lanes;
  std::printf("libjxl oracle: %s, n=%zu, target %s, %zu lanes\n", dir.c_str(),
              n, hwy::TargetName(HWY_STATIC_TARGET), lanes);
  auto in = [&](const char* c) {
    return ReadF32(dir + "/in_" + c + ".f32", n, padded);
  };
  const auto r = in("r"), g = in("g"), b = in("b");
  size_t bad = 0;
  int checked = 0;
  if (Exists(dir + "/xyb_libjxl_0.f32")) {
    Buf o0(padded), o1(padded), o2(padded);
    jxl::HWY_NAMESPACE::XybRows(r.data(), g.data(), b.data(), o0.data(),
                                o1.data(), o2.data(), padded);
    const char* names[3] = {"xyb X", "xyb Y", "xyb B"};
    const Buf* want[3] = {&o0, &o1, &o2};
    for (int c = 0; c < 3; ++c) {
      auto got = ReadF32(dir + "/xyb_libjxl_" + std::to_string(c) + ".f32", n, padded);
      bad += Compare(names[c], got, *want[c], n);
    }
    ++checked;
  }
  if (Exists(dir + "/butteraugli_0.f32")) {
    const auto br = in("br"), bg = in("bg"), bb = in("bb");
    Buf o0(padded), o1(padded), o2(padded);
    jxl::HWY_NAMESPACE::ButteraugliRows(r.data(), g.data(), b.data(), br.data(),
                                        bg.data(), bb.data(), o0.data(),
                                        o1.data(), o2.data(), padded, intensity);
    const char* names[3] = {"butteraugli X", "butteraugli Y", "butteraugli B"};
    const Buf* want[3] = {&o0, &o1, &o2};
    for (int c = 0; c < 3; ++c) {
      auto got = ReadF32(dir + "/butteraugli_" + std::to_string(c) + ".f32", n, padded);
      bad += Compare(names[c], got, *want[c], n);
    }
    ++checked;
  }
  if (!checked) {
    std::fprintf(stderr, "no output sets found in %s\n", dir.c_str());
    return 2;
  }
  std::printf(bad ? "MISMATCH\n" : "BIT-EXACT\n");
  return bad ? 1 : 0;
}
