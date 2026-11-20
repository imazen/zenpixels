#!/usr/bin/env python3
"""Run corrected API contract regressions with default/CMS and minimal features."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="zenpixels-contract-cases-") as directory:
    project = Path(directory)
    (project / "src").mkdir()
    manifest = f"""
[package]
name = "zenpixels-contract-cases"
version = "0.0.0"
edition = "2024"
[features]
default = ["std", "cms"]
std = ["zenpixels/std", "zenpixels-convert/std"]
cms = ["std", "zenpixels-convert/cms-moxcms"]
[dependencies]
zenpixels = {{ path = {json.dumps(str(root / 'zenpixels'))}, default-features = false, features = ["imgref", "icc"] }}
zenpixels-convert = {{ path = {json.dumps(str(root / 'zenpixels-convert'))}, default-features = false }}
bytemuck = "1"
rgb = "0.8"
imgref = "1.12"
whereat = "0.1.5"
[patch.crates-io]
zenpixels = {{ path = {json.dumps(str(root / 'zenpixels'))} }}
archmage = {{ git = "https://github.com/imazen/archmage", rev = "303341570757936368fca1ee7254a19075cbff08" }}
magetypes = {{ git = "https://github.com/imazen/archmage", rev = "303341570757936368fca1ee7254a19075cbff08" }}
garb = {{ git = "https://github.com/imazen/garb", rev = "f22a8c5412de72255de80b5597a484c796f4e40d" }}
linear-srgb = {{ git = "https://github.com/imazen/linear-srgb", rev = "360fb968cda351b2e161d10e66b36414f4d3b9e7" }}
"""
    (project / "Cargo.toml").write_text(manifest)
    modules = []
    for name in ("storage", "conversion", "output_cms"):
        path = root / "docs/contract-cases" / (name + ".rs")
        modules.append(f'#[path = {json.dumps(str(path))}] mod {name};')
    (project / "src/lib.rs").write_text("#![cfg(test)]\n" + "\n".join(modules))
    env = dict(os.environ, CARGO_TARGET_DIR=str(Path(os.environ.get("CARGO_TARGET_DIR", root / "target")) / "contract-cases"))
    for features in ([], ["--no-default-features"]):
        subprocess.run(["cargo", "test", "--quiet", *features],
                       cwd=project, env=env, check=True)
