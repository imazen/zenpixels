#!/usr/bin/env python3
"""Characterize old/new ICC decoded color through independent Little CMS.

Usage: compare_bundle_color.py COMPARATOR OLD_PROFILES NEW_PROFILES NEW_OUTPUT_DIR
Input whites: ITU-T H.273 Table 2, https://www.itu.int/rec/T-REC-H.273
This writes observations, not replacement fixture values or acceptance limits.
"""

import csv
import json
import math
from pathlib import Path
import subprocess
import sys


def source_white(primaries):
    x, y = {4: (0.310, 0.316), 8: (0.310, 0.316),
            11: (0.314, 0.351)}.get(primaries, (0.3127, 0.3290))
    return [x / y, 1.0, (1.0 - x - y) / y]


def main():
    comparator, old, new, output = map(Path, sys.argv[1:5])
    expected = int(sys.argv[5]) if len(sys.argv) > 5 else 159
    profiles = sorted(new.glob("rgb-*.icc"))
    assert len(profiles) == expected
    output.mkdir(parents=True, exist_ok=False)
    summary = {}
    with (output / "lcms-color.tsv").open("w") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["profile", "intent", "sample", "r", "g", "b",
                         "old_x", "old_y", "old_z", "new_x", "new_y", "new_z"])
        for profile in profiles:
            process = subprocess.run([str(comparator.resolve()), str(old / profile.name),
                                      str(profile)], text=True, capture_output=True, check=True)
            for line in process.stdout.splitlines():
                fields = line.split("\t")
                assert len(fields) == 11
                writer.writerow([profile.name] + fields)
                intent, sample = map(int, fields[:2])
                inputs = list(map(float, fields[2:5]))
                before, after = list(map(float, fields[5:8])), list(map(float, fields[8:11]))
                assert all(math.isfinite(value) for value in before + after)
                bucket = summary.setdefault(str(intent), {
                    "adaptation_state": 0.0 if intent == 3 else 1.0,
                    "cells": 0, "changed_cells": 0, "changed_channels": 0,
                    "max_abs_xyz_difference": 0.0, "neutral_max_difference": 0.0,
                    "extended_max_difference": 0.0, "pq_max_difference": 0.0,
                    "hlg_max_difference": 0.0, "old_white_max_xyz_error": 0.0,
                    "new_white_max_xyz_error": 0.0})
                delta = max(abs(a - b) for a, b in zip(before, after))
                bucket["cells"] += 1
                bucket["changed_cells"] += delta != 0
                bucket["changed_channels"] += sum(a != b for a, b in zip(before, after))
                if delta > bucket["max_abs_xyz_difference"]:
                    bucket["max_abs_xyz_difference"] = delta
                    bucket["worst"] = {"profile": profile.name, "sample": sample,
                                       "input": inputs, "old": before, "new": after}
                if inputs[0] == inputs[1] == inputs[2]:
                    bucket["neutral_max_difference"] = max(bucket["neutral_max_difference"], delta)
                if any(value < 0 or value > 1 for value in inputs):
                    bucket["extended_max_difference"] = max(bucket["extended_max_difference"], delta)
                primaries, transfer = map(int, profile.stem.split("-")[1:])
                for code, key in [(16, "pq_max_difference"), (18, "hlg_max_difference")]:
                    if transfer == code:
                        bucket[key] = max(bucket[key], delta)
                if inputs == [1.0, 1.0, 1.0]:
                    expected = source_white(primaries) if intent == 3 else [0.9642, 1.0, 0.8249]
                    for values, key in [(before, "old_white_max_xyz_error"),
                                        (after, "new_white_max_xyz_error")]:
                        error = max(abs(a - b) for a, b in zip(values, expected))
                        bucket[key] = max(bucket[key], error)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
