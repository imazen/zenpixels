#include "ultrahdr/icc.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <lcms2.h>
#include <vector>
int main(int argc, char **argv) {
  if (argc != 4)
    return 2;
  for (int i = 1; i < 4; i++) {
    std::ifstream f(argv[i], std::ios::binary);
    if (!f)
      return 3;
    std::vector<char> bytes((std::istreambuf_iterator<char>(f)), {});
    // libultrahdr's reader expects the JPEG ICC identifier/chunk header.
    const char prefix[] = {'I', 'C', 'C', '_', 'P', 'R', 'O',
                           'F', 'I', 'L', 'E', 0,   1,   1};
    std::vector<char> jpeg(prefix, prefix + 14);
    jpeg.insert(jpeg.end(), bytes.begin(), bytes.end());
    auto gamut =
        ultrahdr::IccHelper::readIccColorGamut(jpeg.data(), jpeg.size());
    std::cout << argv[i] << " gamut=" << gamut << "\n";
    if (i < 3 && gamut != UHDR_CG_DISPLAY_P3)
      return 4;
    if (i == 3 && gamut == UHDR_CG_DISPLAY_P3)
      return 5;
    if (i == 3)
      continue;
    auto input = cmsOpenProfileFromMem(bytes.data(), bytes.size());
    auto xyz = cmsCreateXYZProfile();
    if (!input || !xyz)
      return 6;
    auto transform =
        cmsCreateTransform(input, TYPE_RGB_DBL, xyz, TYPE_XYZ_DBL,
                           INTENT_RELATIVE_COLORIMETRIC, cmsFLAGS_NOOPTIMIZE);
    if (!transform)
      return 7;
    // Independent C LCMS construction from Display-P3 chromaticities,
    // with analytic transfer evaluation below; no production matrix reuse.
    cmsCIExyY white = {0.3127, 0.3290, 1.0};
    cmsCIExyYTRIPLE primaries = {
        {0.680, 0.320, 1.0}, {0.265, 0.690, 1.0}, {0.150, 0.060, 1.0}};
    auto curve = cmsBuildGamma(nullptr, 1.0);
    cmsToneCurve *curves[3] = {curve, curve, curve};
    auto reference = cmsCreateRGBProfile(&white, &primaries, curves);
    auto reference_transform =
        cmsCreateTransform(reference, TYPE_RGB_DBL, xyz, TYPE_XYZ_DBL,
                           INTENT_RELATIVE_COLORIMETRIC, cmsFLAGS_NOOPTIMIZE);
    if (!reference_transform)
      return 10;
    unsigned samples = 0;
    double max_error = 0;
    for (int r = 0; r <= 16; r++)
      for (int g = 0; g <= 16; g++)
        for (int b = 0; b <= 16; b++) {
          double v[3] = {r / 16., g / 16., b / 16.}, out[3], lin[3];
          cmsDoTransform(transform, v, out, 1);
          for (int c = 0; c < 3; c++)
            lin[c] =
                i == 1 ? (v[c] <= 0.04045 ? v[c] / 12.92
                                          : pow((v[c] + 0.055) / 1.055, 2.4))
                       : (v[c] < 0.081 ? v[c] / 4.5
                                       : pow((v[c] + 0.099) / 1.099, 1 / 0.45));
          double expected[3];
          cmsDoTransform(reference_transform, lin, expected, 1);
          samples++;
          for (int c = 0; c < 3; c++)
            max_error = std::max(max_error, std::abs(expected[c] - out[c]));
        }
    std::cout << "lcms version=" << cmsGetEncodedCMMversion()
              << " grid samples=" << samples << " max XYZ error=" << max_error
              << "\n";
    if (max_error > 0.0005)
      return 8;
    cmsDeleteTransform(reference_transform);
    cmsCloseProfile(reference);
    cmsFreeToneCurve(curve);
    cmsDeleteTransform(transform);
    cmsCloseProfile(input);
    cmsCloseProfile(xyz);
  }
  auto positive =
      ultrahdr::IccHelper::writeIccProfile(UHDR_CT_SRGB, UHDR_CG_DISPLAY_P3);
  auto gamut = ultrahdr::IccHelper::readIccColorGamut(positive->getData(),
                                                      positive->getLength());
  std::cout << "native positive gamut=" << gamut << "\n";
  return gamut == UHDR_CG_DISPLAY_P3 ? 0 : 9;
}
