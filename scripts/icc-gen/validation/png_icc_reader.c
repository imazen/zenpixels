/* Independent libpng + Little CMS profile/container roundtrip validator. */
#include <lcms2.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned warnings;
static void warning(png_structp png, png_const_charp message) {
    (void)png;
    ++warnings;
    fprintf(stderr, "libpng warning: %s\n", message);
}
int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: png_icc_reader PROFILE OUTPUT_PNG\n"); return 2; }
    FILE *file = fopen(argv[1], "rb");
    if (!file || fseek(file, 0, SEEK_END)) return 2;
    long size = ftell(file);
    if (size < 132 || size > 1000000 || fseek(file, 0, SEEK_SET)) return 2;
    unsigned char *profile = malloc((size_t)size);
    if (!profile || fread(profile, 1, (size_t)size, file) != (size_t)size) return 2;
    fclose(file);
    cmsHPROFILE cms = cmsOpenProfileFromMem(profile, (cmsUInt32Number)size);
    if (!cms) { fprintf(stderr, "Little CMS refused profile\n"); return 1; }
    cmsCloseProfile(cms);
    int color = memcmp(profile + 16, "GRAY", 4) == 0 ? PNG_COLOR_TYPE_GRAY : PNG_COLOR_TYPE_RGB;
    file = fopen(argv[2], "wb");
    if (!file) return 2;
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, warning);
    png_infop info = png ? png_create_info_struct(png) : NULL;
    if (!info || setjmp(png_jmpbuf(png))) return 1;
    png_init_io(png, file);
    png_set_IHDR(png, info, 1, 1, 8, color, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_iCCP(png, info, "CICP", PNG_COMPRESSION_TYPE_BASE, profile, (png_uint_32)size);
    png_write_info(png, info);
    unsigned char pixel[3] = {128, 128, 128};
    png_write_row(png, pixel);
    png_write_end(png, info);
    png_destroy_write_struct(&png, &info);
    fclose(file);
    file = fopen(argv[2], "rb");
    if (!file) return 2;
    png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, warning);
    info = png ? png_create_info_struct(png) : NULL;
    if (!info || setjmp(png_jmpbuf(png))) return 1;
    png_init_io(png, file);
    png_read_info(png, info);
    png_charp name;
    int method;
    png_bytep returned;
    png_uint_32 length;
    if (!png_get_iCCP(png, info, &name, &method, &returned, &length) || length != (png_uint_32)size || memcmp(returned, profile, (size_t)size)) {
        fprintf(stderr, "libpng did not retain the exact profile\n"); return 1;
    }
    png_read_row(png, pixel, NULL);
    png_read_end(png, info);
    png_destroy_read_struct(&png, &info, NULL);
    fclose(file);
    free(profile);
    return warnings ? 1 : 0;
}
