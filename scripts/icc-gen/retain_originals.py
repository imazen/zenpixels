#!/usr/bin/env python3
"""Verify original ICC bytes from a receipt; optionally retain a portable archive.

No destination means verification only. Fingerprint/qualification evidence is
carried from the receipt, not recomputed or upgraded by this byte-retention tool.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re


def write_new_or_identical(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"refusing to replace different retained data: {path}")
    else:
        with path.open("xb") as output:
            output.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("destination", type=Path, nargs="?")
    args = parser.parse_args()
    receipt_bytes = args.receipt.read_bytes()
    receipt = json.loads(receipt_bytes)
    rows = receipt["rows"]
    if not rows:
        raise ValueError("receipt contains no retained rows")
    objects = {}
    archived_rows = []
    # Validate every source before writing any new archive object.
    for row in rows:
        sha = row["sha256"]
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ValueError(f"invalid SHA-256: {sha!r}")
        source = Path(row["file"])
        if not source.is_absolute():
            source = args.receipt.parent / source
        if source.stat().st_size != row["size"]:
            raise ValueError(f"original size mismatch: {source}")
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError(f"original SHA-256 mismatch: {source}")
        objects[sha] = data
        entry = dict(row)
        # Keep source provenance, including any links, exactly as recorded.
        entry.setdefault("source_file", row["file"])
        entry["file"] = f"profiles/{sha}.icc"
        archived_rows.append(entry)
    if args.destination is not None:
        root = args.destination
        (root / "profiles").mkdir(parents=True, exist_ok=True)
        archive = dict(receipt)
        archive.update(
            archive_format="zenpixels-retained-icc-v1",
            source_receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
            rows=archived_rows,
        )
        manifest = (json.dumps(archive, indent=2, sort_keys=True) + "\n").encode()
        # Refuse a conflicting manifest before adding objects to that archive.
        target = root / "manifest.json"
        if target.exists() and target.read_bytes() != manifest:
            raise FileExistsError(f"preserve the existing archive manifest: {target}")
        for sha, data in objects.items():
            write_new_or_identical(root / "profiles" / f"{sha}.icc", data)
        write_new_or_identical(target, manifest)
    print(json.dumps({
        "verified_rows": len(rows), "retained_objects": len(objects),
        "object_bytes": sum(map(len, objects.values())),
        "scope": "full-file size and SHA-256; copied fingerprint evidence unchanged",
        "destination": str(args.destination) if args.destination else None,
    }, indent=2))


if __name__ == "__main__":
    main()
