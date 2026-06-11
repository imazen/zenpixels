#!/usr/bin/env python3
"""Survey the low-byte structure of 16-bit images.

Answers: when a 16-bit buffer is "secretly 8-bit", is it bit-replicated
(v * 257, lo == hi — what zenpixels-convert's bit_replication_lossless_u16
detects) or shift-widened (v << 8, lo == 0), or something else entirely?

Classification per file (alpha channel excluded from the verdict,
reported separately):
  REPLICATED       every sample satisfies lo == hi          (v * 257)
  UNSCALED_8IN16   every sample < 256 (8-bit values stored raw, hi == 0)
  SHIFTED_8        every sample has lo == 0                 (v << 8)
  SHIFTED_kBIT     every sample is a multiple of 2^(16-k)   (k-bit content
                   shifted into the top bits: 10/12/14-bit camera/video)
  NEAR_REPLICATED  within +/-1 of the nearest v * 257 (rounded float path)
  TRUE16           none of the above — low bits carry information

Usage: u16_low_byte_survey.py <file-or-dir>...
Writes one TSV row per 16-bit image to stdout; skips non-16-bit files.
"""

import os
import sys

import numpy as np


def png_bit_depth(path):
    """Fast IHDR sniff: PNG bit depth without decoding."""
    try:
        with open(path, "rb") as f:
            head = f.read(26)
        if len(head) >= 26 and head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
            return head[24]
    except OSError:
        return None
    return None


def load_u16(path):
    """Decode to a uint16 ndarray, or None when not 16-bit / undecodable."""
    try:
        import cv2

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img if img.dtype == np.uint16 else None
    except Exception:
        pass
    if path.lower().endswith((".tif", ".tiff")):
        try:
            import tifffile

            img = tifffile.imread(path)
            return img if img.dtype == np.uint16 else None
        except Exception:
            return None
    try:
        import imageio.v3 as iio

        img = iio.imread(path)
        return img if img.dtype == np.uint16 else None
    except Exception:
        return None


def classify(samples):
    """Stats + class for a flat uint16 sample array."""
    n = int(samples.size)
    if n == 0:
        return None
    s = samples.ravel().astype(np.uint16)
    lo = s & 0xFF
    hi = s >> 8
    pct_rep = float(np.count_nonzero(lo == hi)) / n
    pct_lo0 = float(np.count_nonzero(lo == 0)) / n
    or_all = int(np.bitwise_or.reduce(s))
    tz = ((or_all & -or_all).bit_length() - 1) if or_all else 16
    v8 = np.clip(np.rint(s / 257.0), 0, 255).astype(np.int32)
    max_dev_rep = int(np.abs(s.astype(np.int32) - v8 * 257).max())
    if pct_rep == 1.0:
        cls = "REPLICATED"
    elif or_all < 256:
        cls = "UNSCALED_8IN16"
    elif tz >= 8:
        cls = "SHIFTED_8"
    elif tz >= 1:
        cls = f"SHIFTED_{16 - tz}BIT"
    elif max_dev_rep <= 1:
        cls = "NEAR_REPLICATED"
    else:
        cls = "TRUE16"
    return {
        "n": n,
        "pct_rep": pct_rep,
        "pct_lo0": pct_lo0,
        "or_all": or_all,
        "tz": tz,
        "max_dev_rep": max_dev_rep,
        "cls": cls,
    }


def survey(path):
    img = load_u16(path)
    if img is None:
        return None
    if img.ndim == 2:
        color, alpha = img, None
        shape = f"{img.shape[1]}x{img.shape[0]}x1"
    else:
        ch = img.shape[2]
        shape = f"{img.shape[1]}x{img.shape[0]}x{ch}"
        if ch in (2, 4):  # GA / BGRA: last channel is alpha
            color, alpha = img[..., : ch - 1], img[..., ch - 1]
        else:
            color, alpha = img, None
    c = classify(color)
    if c is None:
        return None
    a = classify(alpha) if alpha is not None else None
    return (
        f"{c['cls']}\t{shape}\t{c['n']}\t{c['pct_rep']:.4f}\t{c['pct_lo0']:.4f}\t"
        f"{c['or_all']:#06x}\t{c['tz']}\t{c['max_dev_rep']}\t"
        f"{a['cls'] if a else '-'}\t{path}"
    )


def gather(args):
    for arg in args:
        if os.path.isfile(arg):
            yield arg
            continue
        for root, _dirs, files in os.walk(arg):
            for name in sorted(files):
                p = os.path.join(root, name)
                low = name.lower()
                if low.endswith(".png"):
                    if png_bit_depth(p) == 16:
                        yield p
                elif low.endswith((".tif", ".tiff")):
                    yield p


def main():
    print(
        "class\tshape\tn_color_samples\tpct_lo_eq_hi\tpct_lo_eq_0\t"
        "or_all\ttrailing_zero_bits\tmax_dev_from_replicated\talpha_class\tpath"
    )
    for path in gather(sys.argv[1:]):
        try:
            row = survey(path)
        except Exception as e:  # noqa: BLE001 — survey tool, log and move on
            print(f"ERROR\t-\t-\t-\t-\t-\t-\t-\t-\t{path}: {e}", file=sys.stderr)
            continue
        if row:
            print(row)


if __name__ == "__main__":
    main()
