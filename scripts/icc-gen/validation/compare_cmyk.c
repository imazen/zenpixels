/* Four documented reflectance samples through the C oracle, matching LcmsPort. */
#include <lcms2.h>
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    cmsHPROFILE source = cmsOpenProfileFromFile(argv[1], "r");
    cmsHPROFILE target = cmsCreate_sRGBProfile();
    if (!source || !target) return 3;
    cmsHTRANSFORM transform = cmsCreateTransform(source, TYPE_CMYK_FLT, target,
        TYPE_RGB_FLT, INTENT_RELATIVE_COLORIMETRIC,
        cmsFLAGS_BLACKPOINTCOMPENSATION | cmsFLAGS_NOOPTIMIZE | cmsFLAGS_NOCACHE);
    if (!transform) return 4;
    const float samples[][4] = {
        {0.2863f, 1.f, 0.6706f, 1.f}, {0.2347f, 0.9601f, 0.5918f, 0.9813f},
        {1.f, 1.f, 1.f, 0.f}, {1.f, 1.f, 1.f, 1.f}
    };
    for (unsigned i = 0; i < 4; ++i) {
        float ink[4], rgb[3];
        for (unsigned c = 0; c < 4; ++c) ink[c] = 100.f * (1.f - samples[i][c]);
        cmsDoTransform(transform, ink, rgb, 1);
        printf("%u\t%.9g\t%.9g\t%.9g\n", i, rgb[0], rgb[1], rgb[2]);
    }
    cmsDeleteTransform(transform);
    cmsCloseProfile(source);
    cmsCloseProfile(target);
    return 0;
}
