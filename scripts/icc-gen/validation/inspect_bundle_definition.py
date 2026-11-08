#!/usr/bin/env python3
"""Inspect committed ICC definitions without regenerating or rewriting profiles.

Uses the independent system liblz4 decoder. The JSON reports disagreements
with display-white requirements; it does not accept replacement pixels or blobs.
An optional git revision permits inspection of the original bundle history.
"""
import argparse
from collections import Counter
import ctypes
import ctypes.util
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess


ROOT = Path(__file__).resolve().parents[3]
INDEX = "zenpixels-convert/src/icc_profiles/cicp_bundle_index.rs"
BLOB = "zenpixels-convert/src/profiles/cicp_bundle.lz4"
PCS_D50 = (0.9642, 1.0, 0.8249)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision", nargs="?")
    args = parser.parse_args()

    def read(path):
        if args.revision:
            return subprocess.check_output(
                ["git", "show", f"{args.revision}:{path}"], cwd=ROOT)
        return (ROOT / path).read_bytes()

    index, blob = read(INDEX).decode(), read(BLOB)
    library = ctypes.util.find_library("lz4")
    if library is None:
        raise RuntimeError("system liblz4 is required for independent read-back")
    decoder = ctypes.CDLL(library).LZ4_decompress_safe
    decoder.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    decoder.restype = ctypes.c_int
    groups = re.findall(
        r"BundleGroup \{ transfer: (\d+), blob_offset: (\d+), "
        r"compressed_len: (\d+), decompressed_len: (\d+) \}", index)
    assert groups, "no bundle groups found"
    raw = []
    for _, offset, size, decoded in (map(int, row) for row in groups):
        source = ctypes.create_string_buffer(blob[offset:offset + size])
        target = ctypes.create_string_buffer(decoded)
        assert decoder(source, target, size, decoded) == decoded
        raw.append(target.raw)

    entries = re.findall(
        r"BundleProfile \{ primaries: (\d+), transfer: (\d+), "
        r"group_index: (\d+), offset_in_group: (\d+), len: (\d+) \}", index)
    assert entries, "no profile locators found"
    whites, mismatches, mappings, curves = Counter(), Counter(), Counter(), {}
    examples = {}
    missing_rgb_white = 0
    for p, tc, group, offset, size in (map(int, row) for row in entries):
        data = raw[group][offset:offset + size]
        assert len(data) == int.from_bytes(data[:4], "big")
        space = data[16:20].decode().strip()
        mappings[space] += 1
        tags = {}
        for i in range(int.from_bytes(data[128:132], "big")):
            record = data[132 + 12 * i:144 + 12 * i]
            start, length = struct.unpack(">II", record[4:])
            assert start + length <= len(data)
            tags[record[:4]] = data[start:start + length]
        if space == "RGB" and b"wtpt" not in tags:
            missing_rgb_white += 1
        if space == "GRAY":
            white = struct.unpack(">iii", tags[b"wtpt"][8:20])
            whites[white] += 1
            if data[12:16] == b"mntr" and data[8] >= 4 and any(
                    abs(value / 65536 - target) > 0.00005
                    for value, target in zip(white, PCS_D50)):
                mismatches[space] += 1
        signature = b"kTRC" if space == "GRAY" else b"rTRC"
        curve = tags[signature]
        shape = [curve[:4].decode()]
        if curve[:4] == b"curv":
            count = int.from_bytes(curve[8:12], "big")
            shape.append(count)
            if count == 1:
                shape.append(int.from_bytes(curve[12:14], "big") / 256)
        else:
            shape.append(int.from_bytes(curve[8:10], "big"))
        curves.setdefault(tc, set()).add(tuple(shape))
        if (space, p, tc) == ("GRAY", 1, 8):
            examples["gray-1-8"] = dict(
                sha256=hashlib.sha256(data).hexdigest(),
                profile_class=data[12:16].decode(), version=data[8:12].hex(),
                wtpt=[v / 65536 for v in struct.unpack(">iii", tags[b"wtpt"][8:20])])
    print(json.dumps(dict(
        source=args.revision or "working tree",
        blob_sha256=hashlib.sha256(blob).hexdigest(), blob_bytes=len(blob),
        mappings=dict(mappings), missing_rgb_wtpt=missing_rgb_white,
        gray_display_wtpt_mismatches=sum(mismatches.values()),
        gray_whites=[dict(xyz=[v / 65536 for v in key], mappings=count)
                     for key, count in sorted(whites.items())],
        transfer_encodings={tc: sorted(values) for tc, values in sorted(curves.items())},
        examples=examples), indent=2))


if __name__ == "__main__":
    main()
