#!/usr/bin/env python3
"""Score candidate RGB ICC colorimetry against H.273 and PCS D50.

Usage: triage_profiles.py REFERENCE_TSV NEW_OUTPUT_DIR NAME=PROFILE_DIR ...
Records errors without changing existing pixel expectations or choosing a binary.
Transfer/CICP payload equality is checked separately to isolate colorimetry.
"""
import csv
import hashlib
import json
from pathlib import Path
import struct
import sys

D50 = [0.9642, 1.0, 0.8249]


def tags(data):
    count = int.from_bytes(data[128:132], "big")
    result = {}
    for i in range(count):
        record = data[132 + i * 12:144 + i * 12]
        name = record[:4].decode("ascii")
        offset, size = struct.unpack(">II", record[4:])
        assert offset + size <= len(data)
        result[name] = data[offset:offset + size]
    return result


def fixed(payload, count):
    return [v / 65536 for v in struct.unpack(">" + "i" * count, payload[8:8 + count * 4])]


def main():
    reference, output = map(Path, sys.argv[1:3])
    with reference.open() as stream:
        refs = {int(r["primaries"]): r for r in csv.DictReader(stream, delimiter="\t")}
    output.mkdir(parents=True, exist_ok=False)
    summary, records, curves = {}, [], {}
    for candidate in sys.argv[3:]:
        label, folder = candidate.split("=", 1)
        paths = sorted(Path(folder).glob("rgb-*.icc"))
        assert len(paths) == 159
        maxima = {"colorant_error": 0.0, "pcs_white_sum_error": 0.0,
                  "chad_white_error": 0.0, "wtpt_error": 0.0}
        missing_white, mismatched_tags = [], []
        for path in paths:
            p, tc = map(int, path.stem.split("-")[1:])
            ref, data = refs[p], path.read_bytes()
            actual = tags(data)
            # The first candidate supplies transfer/tag controls; equivalence is byte-exact.
            control = curves.setdefault(path.name, {k: actual[k] for k in ["rTRC", "gTRC", "bTRC", "cicp"]})
            for name, payload in control.items():
                if actual[name] != payload: mismatched_tags.append([path.name, name])
            colorants = [fixed(actual[name], 3) for name in ["rXYZ", "gXYZ", "bXYZ"]]
            target = [[float(ref[f"{c}_{xyz}"]) for xyz in "xyz"] for c in "rgb"]
            native_white = [float(ref[f"white_{xyz}"]) for xyz in "xyz"]
            chad = fixed(actual["chad"], 9)
            errors = dict(
                colorant_error=max(abs(a - b) for col, refcol in zip(colorants, target) for a, b in zip(col, refcol)),
                pcs_white_sum_error=max(abs(sum(col[i] for col in colorants) - D50[i]) for i in range(3)),
                chad_white_error=max(abs(sum(chad[i * 3 + j] * native_white[j] for j in range(3)) - D50[i]) for i in range(3)),
            )
            if "wtpt" in actual:
                errors["wtpt_error"] = max(abs(a - b) for a, b in zip(fixed(actual["wtpt"], 3), D50))
            else:
                missing_white.append(path.name)
            for key, value in errors.items(): maxima[key] = max(maxima[key], value)
            records.append(dict(candidate=label, profile=path.name, primaries=p, transfer=tc,
                                sha256=hashlib.sha256(data).hexdigest(), **errors))
        summary[label] = dict(profiles=len(paths), max_abs_errors=maxima,
                              missing_wtpt=len(missing_white), changed_transfer_or_cicp_tags=mismatched_tags)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "profiles.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
