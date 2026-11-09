#!/usr/bin/env python3
"""Prepare a manifest-only descendant in an explicitly owned isolated repo.

Usage: prepare-stable-cap.py REPO BASE OWNER MANIFEST [MANIFEST ...]
Does not push, move an existing bookmark, or modify implementation/test files.
"""

import json
from pathlib import Path
import subprocess
import sys


def main():
    repo, base, owner, *manifests = sys.argv[1:]
    root = Path(repo).resolve()
    marker = root / ".workongoing"
    assert marker.exists() and owner in marker.read_text(), "caller must hold the repository lock"

    def run(*args):
        return subprocess.check_output(args, cwd=root, text=True).strip()

    assert not run("jj", "diff", "--name-only"), "commit current changes before starting"
    original = run("git", "rev-parse", base)
    run("jj", "new", original, "-m", "build: admit stable SIMD without changing consumer source")
    changed = []
    replacements = [(b"<=0.9.31-beta.1", b"<0.10.0"),
                    (b"<=0.9.31-beta.2", b"<0.10.0"),
                    (b"<0.9.31", b"<0.10.0")]
    for name in manifests:
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts
        assert relative.name == "Cargo.toml"
        path = root / relative
        before = path.read_bytes()
        lines = before.splitlines(keepends=True)
        for index, line in enumerate(lines):
            if (b"archmage" in line or b"magetypes" in line) and not line.lstrip().startswith(b"#"):
                for old, new in replacements:
                    line = line.replace(old, new)
                lines[index] = line
        after = b"".join(lines)
        if before != after:
            path.write_bytes(after)
            changed.append(name)
    assert changed, "no recognized SIMD ceiling found"
    run("jj", "describe", "-m", "build: admit stable SIMD without changing consumer source\n\n"
        "Replace only temporary SIMD ceilings while retaining floors and features.\n"
        "Implementation, model data, fixtures and numerical expectations remain\n"
        "unchanged. Consumer-graph qualification and owner integration are separate.")
    candidate = run("jj", "log", "-r", "@", "--no-graph", "--template", "commit_id")
    actual = run("git", "diff", "--name-only", original, candidate).splitlines()
    assert sorted(actual) == sorted(changed), actual
    run("jj", "new", "-m", "investigate: qualify stable consumer dependency graph")
    print(json.dumps({"base": original, "candidate": candidate, "files": changed}))


if __name__ == "__main__":
    main()
