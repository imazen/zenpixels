#!/usr/bin/env python3
"""Export and compare two generated ICC bundles without replacing either.

Usage: compare_bundles.py OLD_CONVERT_DIR NEW_CONVERT_DIR NEW_OUTPUT_DIR
Requires the system LZ4 library only for this independent validation tool.
"""

import collections
import ctypes
import ctypes.util
import hashlib
import json
from pathlib import Path
import re
import sys


def unpack(crate, expected=334):
    source = (crate / "src/icc_profiles/cicp_bundle_index.rs").read_text()
    blob = (crate / "src/profiles/cicp_bundle.lz4").read_bytes()
    library = ctypes.CDLL(ctypes.util.find_library("lz4"))
    decode = library.LZ4_decompress_safe
    decode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    decode.restype = ctypes.c_int
    groups = []
    for fields in re.findall(r"BundleGroup \{ ([^\n}]+) \}", source):
        values = dict((key, int(value)) for key, value in re.findall(r"(\w+): (\d+)", fields))
        start = values["blob_offset"]
        encoded = blob[start:start + values["compressed_len"]]
        destination = ctypes.create_string_buffer(values["decompressed_len"])
        size = decode(encoded, destination, len(encoded), len(destination))
        assert size == len(destination), (size, values)
        groups.append(destination.raw)
    profiles = {}
    for label, table in [("rgb", "BUNDLE_PROFILES"), ("gray", "GRAY_BUNDLE_PROFILES")]:
        match = re.search(r"static " + table + r":.*?= \[(.*?)\n\];", source, re.S)
        assert match, table
        for fields in re.findall(r"BundleProfile \{ ([^\n}]+) \}", match[1]):
            values = dict((key, int(value)) for key, value in re.findall(r"(\w+): (\d+)", fields))
            start = values["offset_in_group"]
            data = groups[values["group_index"]][start:start + values["len"]]
            assert len(data) == values["len"]
            profiles[f'{label}-{values["primaries"]}-{values["transfer"]}'] = data
    assert len(profiles) == expected, len(profiles)
    return blob, profiles


def tags(data):
    count = int.from_bytes(data[128:132], "big")
    result = {}
    for index in range(count):
        entry = 132 + 12 * index
        name = data[entry:entry + 4].decode("ascii")
        start = int.from_bytes(data[entry + 4:entry + 8], "big")
        size = int.from_bytes(data[entry + 8:entry + 12], "big")
        result[name] = data[start:start + size]
    return result


def main():
    old_root, new_root, output = map(Path, sys.argv[1:4])
    expected = int(sys.argv[4]) if len(sys.argv) > 4 else 334
    old_blob, old = unpack(old_root)
    new_blob, new = unpack(new_root, expected)
    output.mkdir(parents=True, exist_ok=False)
    removed = sorted(old.keys() - new.keys())
    assert not new.keys() - old.keys(), "unexpected new assigned combination"
    if expected == 313:
        assert len(removed) == 21 and all(name.endswith("-18") for name in removed), removed
    else:
        assert not removed, removed
    changed = []
    counts = collections.Counter()
    for name, data in new.items():
        (output / (name + ".icc")).write_bytes(data)
        before, after = tags(old[name]), tags(data)
        differences = [key for key in sorted(before.keys() | after.keys())
                       if before.get(key) != after.get(key)]
        if old[name] != data:
            changed.append({"profile": name, "old_bytes": len(old[name]),
                            "new_bytes": len(data), "changed_tags": differences})
            counts.update(differences)
    result = {"old_blob_bytes": len(old_blob), "new_blob_bytes": len(new_blob),
              "old_sha256": hashlib.sha256(old_blob).hexdigest(),
              "new_sha256": hashlib.sha256(new_blob).hexdigest(),
              "profiles": len(new), "removed_targetless_hlg": removed, "changed_profiles": len(changed),
              "changed_rgb": sum(row["profile"].startswith("rgb-") for row in changed),
              "changed_gray": sum(row["profile"].startswith("gray-") for row in changed),
              "changed_tag_counts": dict(counts), "changes": changed}
    (output / "bundle-comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "changes"}, indent=2))


if __name__ == "__main__":
    main()
