#!/usr/bin/env python3
"""Reconstruct the six frozen experimental dependencies; never overwrite work."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parent
REV = "841044439077ec1ec67a71b2db979ab51d304fcf"


def digest(directory):
    hashes = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and (path.name == "Cargo.toml" or "src" in path.relative_to(directory).parts):
            hashes[path.relative_to(directory).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def main():
    output = ROOT / "variants"
    if output.exists():
        raise SystemExit("variants already exists; preserve it before preparing a new copy")
    url = f"https://api.github.com/repos/lilith/moxcms/tarball/{REV}"
    request = urllib.request.Request(url, headers={"User-Agent": "moxcms-gamma-harness"})
    with urllib.request.urlopen(request, timeout=60) as response:
        archive = response.read()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as source:
        files = {}
        for member in source.getmembers():
            parts = Path(member.name).parts[1:]
            if member.isfile() and parts and ".." not in parts:
                name = Path(*parts)
                if name.as_posix() == "Cargo.toml" or parts[0] == "src" or name.name.startswith(("LICENSE", "README")):
                    files[name] = source.extractfile(member).read()
    expected = json.loads((ROOT / "variant-sha256.json").read_text())
    for variant in expected:
        directory = output / variant
        directory.mkdir(parents=True)
        for name, data in files.items():
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(ROOT / "patches" / f"{variant}.patch")], cwd=directory, check=True)
        actual = digest(directory)
        if actual != expected[variant]:
            raise SystemExit(f"{variant}: source digest mismatch: {actual}")
        print(f"{variant}: verified {actual}")


if __name__ == "__main__":
    main()
