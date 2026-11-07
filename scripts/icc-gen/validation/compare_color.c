/* Independent Little CMS characterization; no acceptance threshold is changed. */
#include <lcms2.h>
#include <stdio.h>
#include <stdlib.h>

static int failed;

static void error_handler(cmsContext context, cmsUInt32Number code, const char *message) {
    (void)context;
    fprintf(stderr, "lcms error %u: %s\n", code, message);
    failed = 1;
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    cmsSetLogErrorHandler(error_handler);
    cmsHPROFILE old = cmsOpenProfileFromFile(argv[1], "r");
    cmsHPROFILE newer = cmsOpenProfileFromFile(argv[2], "r");
    cmsHPROFILE xyz = cmsCreateXYZProfile();
    if (!old || !newer || !xyz) return 3;
    float inputs[136][3];
    const float grid[] = {0.0f, 0.01f, 0.18f, 0.5f, 1.0f};
    unsigned count = 0;
    for (unsigned r = 0; r < 5; ++r)
        for (unsigned g = 0; g < 5; ++g)
            for (unsigned b = 0; b < 5; ++b) {
                inputs[count][0] = grid[r];
                inputs[count][1] = grid[g];
                inputs[count++][2] = grid[b];
            }
    const float extra[][3] = {
        {0.0001f, 0.0001f, 0.0001f}, {0.001f, 0.001f, 0.001f},
        {0.1f, 0.1f, 0.1f}, {0.75f, 0.75f, 0.75f},
        {-0.1f, 0.5f, 0.5f}, {1.1f, 0.5f, 0.5f},
        {0.5f, -0.1f, 0.5f}, {0.5f, 1.1f, 0.5f},
        {0.5f, 0.5f, -0.1f}, {0.5f, 0.5f, 1.1f}, {1.5f, 1.5f, 1.5f}
    };
    for (unsigned i = 0; i < sizeof(extra) / sizeof(extra[0]); ++i) {
        for (unsigned c = 0; c < 3; ++c) inputs[count][c] = extra[i][c];
        ++count;
    }
    for (unsigned pass = 0; pass < 2; ++pass) {
        unsigned intent = pass ? INTENT_ABSOLUTE_COLORIMETRIC : INTENT_RELATIVE_COLORIMETRIC;
        /* LCMS defaults to complete adaptation; use zero for native-white
           absolute colorimetry. https://www.littlecms.com/LittleCMS2.18%20API.pdf */
        cmsSetAdaptationState(pass ? 0.0 : 1.0);
        unsigned flags = cmsFLAGS_NOOPTIMIZE | cmsFLAGS_NOCACHE;
        cmsHTRANSFORM before = cmsCreateTransform(old, TYPE_RGB_FLT, xyz, TYPE_XYZ_DBL, intent, flags);
        cmsHTRANSFORM after = cmsCreateTransform(newer, TYPE_RGB_FLT, xyz, TYPE_XYZ_DBL, intent, flags);
        if (!before || !after) return 4;
        double a[136][3], b[136][3];
        cmsDoTransform(before, inputs, a, count);
        cmsDoTransform(after, inputs, b, count);
        for (unsigned i = 0; i < count; ++i) {
            printf("%u\t%u", intent, i);
            for (unsigned c = 0; c < 3; ++c) printf("\t%.9g", inputs[i][c]);
            for (unsigned c = 0; c < 3; ++c) printf("\t%.17g", a[i][c]);
            for (unsigned c = 0; c < 3; ++c) printf("\t%.17g", b[i][c]);
            putchar('\n');
        }
        cmsDeleteTransform(before);
        cmsDeleteTransform(after);
    }
    cmsCloseProfile(old);
    cmsCloseProfile(newer);
    cmsCloseProfile(xyz);
    return failed ? 5 : 0;
}
