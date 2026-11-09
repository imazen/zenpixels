#!/usr/bin/env python3
"""Compare complete decoded-color receipts without accepting a tolerance."""
import csv
import json
from pathlib import Path
import struct
import sys


def load(path):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    result = {}
    for row in rows:
        key = (row["profile"], row["intent"], row["sample"])
        assert key not in result, key
        result[key] = row
    return result


def main():
    left, right = map(load, sys.argv[1:3])
    assert left.keys() == right.keys(), "different input cells"
    columns = [f"{side}_{channel}" for side in ["old", "new"] for channel in "xyz"]
    mismatches = []
    channels = 0
    maximum = 0.0
    for key, a in left.items():
        b = right[key]
        for c in "rgb":
            assert struct.pack("f", float(a[c])) == struct.pack("f", float(b[c])), (key, c)
        for c in columns:
            x, y = float(a[c]), float(b[c])
            if struct.pack("d", x) != struct.pack("d", y):
                channels += 1
                maximum = max(maximum, abs(x - y))
                if len(mismatches) < 8:
                    mismatches.append(dict(cell=key, channel=c, left=x, right=y))
    result = dict(cells=len(left), channels=len(left) * 6,
                  mismatched_channels=channels, max_abs_difference=maximum,
                  first_mismatches=mismatches)
    Path(sys.argv[3]).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
