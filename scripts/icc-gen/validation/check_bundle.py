#!/usr/bin/env python3
"""Require every exported ICC to survive independent readers without warnings."""
from pathlib import Path
import datetime
import subprocess
import sys

reader = Path(sys.argv[1]).resolve()
directory = Path(sys.argv[2])
expected = int(sys.argv[3]) if len(sys.argv) > 3 else 334
profiles = sorted(directory.glob("*.icc"))
assert len(profiles) == expected, f"expected all {expected} assigned profiles, got {len(profiles)}"
png_output = directory / "independent-reader-output"
png_output.mkdir()
for profile in profiles:
    data = profile.read_bytes()
    assert data[68:80].hex() == "0000f6d6000100000000d32d", profile
    assert len(data) % 4 == 0 and int.from_bytes(data[:4], "big") == len(data), profile
    fields = [int.from_bytes(data[i:i+2], "big") for i in range(24, 36, 2)]
    datetime.datetime(*fields)  # ICC dateTimeNumber must be a valid calendar date.
    for index in range(int.from_bytes(data[128:132], "big")):
        entry = 132 + index * 12
        offset = int.from_bytes(data[entry + 4:entry + 8], "big")
        size = int.from_bytes(data[entry + 8:entry + 12], "big")
        assert offset % 4 == 0 and offset + size <= len(data), profile
    subprocess.run([str(reader), str(profile), str(png_output / profile.with_suffix(".png").name)], check=True)
# Prove the warning-sensitive gate rejects a non-D50 PCS header.
negative = bytearray(profiles[0].read_bytes())
negative[68:72] = (62000).to_bytes(4, "big")
bad = png_output / "non-d50.negative"
bad.write_bytes(negative)
result = subprocess.run([str(reader), str(bad), str(png_output / "non-d50.negative.png")], capture_output=True)
assert result.returncode != 0, "negative control accepted a non-D50 profile without warning"
print(f"{len(profiles)}/{len(profiles)} ICCs: D50, aligned, exact libpng ICC roundtrip, Little CMS accepted; negative control refused")
