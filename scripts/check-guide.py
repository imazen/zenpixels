#!/usr/bin/env python3
"""Compile and run the Markdown examples against this checkout's real crates."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, help="Additional browser-generated Rust Markdown")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    scratch = Path(env.setdefault("TMPDIR", str(Path.home() / "tmp")))
    scratch.mkdir(parents=True, exist_ok=True)
    build = subprocess.run(
        ["cargo", "build", "--locked", "-p", "zenpixels-convert", "--features", "rgb,imgref",
         "--message-format=json-render-diagnostics"],
        cwd=root, env=env, text=True, stdout=subprocess.PIPE, check=True,
    )
    libraries = {}
    for line in build.stdout.splitlines():
        message = json.loads(line)
        if message.get("reason") == "compiler-message":
            print(message["message"].get("rendered", ""), end="", flush=True)
        if message.get("reason") == "compiler-artifact":
            for filename in message["filenames"]:
                if filename.endswith(".rlib"):
                    libraries[message["target"]["name"]] = Path(filename)
    names = ("zenpixels", "zenpixels_convert", "rgb")
    missing = set(names) - libraries.keys()
    if missing:
        raise RuntimeError(f"Cargo did not report required libraries: {sorted(missing)}")
    externs = []
    for name in names:
        externs.extend(["--extern", f"{name}={libraries[name]}"])
    documents = [root / "docs/guide.md", root / "README.md", root / "zenpixels-convert/README.md"]
    if args.examples:
        documents.append(args.examples.resolve(strict=True))
    for document in documents:
        print(f"Checking Rust examples: {document}", flush=True)
        subprocess.run(
            ["rustdoc", "--edition=2024", "--test", str(document),
             "--test-args=--test-threads=2", "-L", f"dependency={libraries['zenpixels'].parent}",
             *externs],
            cwd=root, env=env, check=True,
        )


if __name__ == "__main__":
    main()
