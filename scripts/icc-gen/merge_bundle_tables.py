#!/usr/bin/env python3
"""Add regenerated bundle hashes while retaining recognition of existing profiles."""
from pathlib import Path
import re
import sys

source, destination = map(Path, sys.argv[1:])
def retained_row(row):
    # Curve closeness and a CICP hint do not qualify the profile's complete
    # matrix, native-white adaptation or rendering-intent behavior. Historical
    # and fresh bundle fingerprints remain useful identification metadata.
    if "cicp bundle:" in row:
        row = re.sub(r"Safe::[A-Za-z]+", "Safe::IdOnly", row)
    return row


pattern = re.compile(r"^    (?:// )?\(0x([0-9a-f_]+),", re.MULTILINE)
for kind in ("rgb", "gray"):
    target = destination / f"icc_table_{kind}.inc"
    original = target.read_text()
    fresh = (source / target.name).read_text()
    rows = {}
    for match in pattern.finditer(original):
        end = original.find("\n", match.start())
        rows[int(match[1].replace("_", ""), 16)] = retained_row(original[match.start():end])
    added = 0
    for match in pattern.finditer(fresh):
        end = fresh.find("\n", match.start())
        row = fresh[match.start():end]
        if "cicp bundle:" not in row:
            continue
        key = int(match[1].replace("_", ""), 16)
        if key not in rows:
            rows[key] = retained_row(row)
            added += 1
    start = pattern.search(original).start()
    output = original[:start] + "\n".join(rows[key] for key in sorted(rows)) + "\n]\n"
    target.write_text(output)
    print(f"{kind}: added {added} generated bundle hashes; retained existing recognition")
