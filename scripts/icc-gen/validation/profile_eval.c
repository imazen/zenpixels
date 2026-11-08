/* Independent test oracle only: LCMS2 float profile evaluation, no profile writing. */
#include <lcms2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned errors;
static void error(cmsContext ctx, cmsUInt32Number code, const char *message) {
    (void)ctx; ++errors; fprintf(stderr, "LCMS %u: %s\n", code, message);
}
int main(int argc, char **argv) {
    if (argc != 6) return 2;
    cmsSetLogErrorHandler(error);
    cmsHPROFILE src = cmsOpenProfileFromFile(argv[1], "r"), dst = cmsOpenProfileFromFile(argv[2], "r");
    if (!src || !dst) return 3;
    cmsUInt32Number from = cmsFormatterForColorspaceOfProfile(src, 4, TRUE), to = cmsFormatterForColorspaceOfProfile(dst, 4, TRUE);
    unsigned channels = T_CHANNELS(from), outputs = T_CHANNELS(to);
    char *tail;
    unsigned intent = (unsigned)strtoul(argv[5], &tail, 10);
    int bpc = !strcmp(tail, ":bpc");
    if (intent > 3 || (*tail && !bpc) || !channels || !outputs) return 2;
    cmsSetAdaptationState(intent == 3 ? 0.0 : 1.0);
    cmsHTRANSFORM tx = cmsCreateTransform(src, from, dst, to, intent, cmsFLAGS_NOOPTIMIZE | cmsFLAGS_NOCACHE | (bpc ? cmsFLAGS_BLACKPOINTCOMPENSATION : 0));
    if (!tx) return 4;
    FILE *input = fopen(argv[3], "rb");
    if (!input || fseek(input, 0, SEEK_END)) return 2;
    long length = ftell(input);
    if (length <= 0 || length > 64000000 || length % (channels * 4) || fseek(input, 0, SEEK_SET)) return 2;
    unsigned count = (unsigned)length / (channels * 4);
    float *in = malloc((size_t)length), *out = calloc((size_t)count * outputs, sizeof(float));
    if (!in || !out || fread(in, 1, (size_t)length, input) != (size_t)length) return 2;
    fclose(input);
    /* This probe uses normalized ICC ink; Little CMS float format uses percent. */
    if (cmsGetColorSpace(src) == cmsSigCmykData)
        for (size_t i = 0; i < (size_t)count * channels; ++i) in[i] *= 100.0f;
    cmsDoTransform(tx, in, out, count);
    if (cmsGetColorSpace(dst) == cmsSigCmykData)
        for (size_t i = 0; i < (size_t)count * outputs; ++i) out[i] /= 100.0f;
    FILE *output = strcmp(argv[4], "-") == 0 ? stdout : fopen(argv[4], "wb");
    if (!output || fwrite(out, sizeof(float) * outputs, count, output) != count || fflush(output)) return 2;
    if (output != stdout && fclose(output)) return 2;
    cmsDeleteTransform(tx); cmsCloseProfile(src); cmsCloseProfile(dst); free(in); free(out);
    return errors ? 5 : 0;
}
