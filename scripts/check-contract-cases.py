#!/usr/bin/env python3
"""Run current-behavior review evidence, NOT correctness/release acceptance tests.

Most assertions deliberately recognize existing bugs. A failing assertion after
an implementation fix means its review case should be updated, not the fix undone.
"""
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
"""
    (project / "Cargo.toml").write_text(manifest)
    modules = []
    for name in ("storage", "conversion", "output_cms"):
        path = root / "docs/contract-cases" / (name + ".rs")
        modules.append(f'#[path = {json.dumps(str(path))}] mod {name};')
    (project / "src/lib.rs").write_text("#![cfg(test)]\n" + "\n".join(modules))
    env = dict(os.environ, CARGO_TARGET_DIR=str(root / "target/contract-cases"))
    for features in ([], ["--no-default-features"]):
        subprocess.run(["cargo", "test", "--offline", "--quiet", *features],
                       cwd=project, env=env, check=True)
    # The breaking release removes no_std Clone rather than silently losing
    # owned CMS state. Check both halves of that feature-dependent contract.
    (project / "src/bin").mkdir()
    (project / "src/bin/clone.rs").write_text("""
use zenpixels_convert::RowConverter;
fn assert_clone<T: Clone>() {}
fn main() { assert_clone::<RowConverter>(); }
""")
    command = ["cargo", "check", "--offline", "--quiet", "--bin", "clone"]
    subprocess.run(command, cwd=project, env=env, check=True)
    negative = subprocess.run(command + ["--no-default-features"],
                              cwd=project, env=env, text=True, capture_output=True)
    if negative.returncode == 0 or "RowConverter: Clone" not in negative.stderr:
        raise RuntimeError("Expected no_std RowConverter Clone bound rejection:\n" + negative.stderr)
    print("std Clone accepted; no_std Clone rejected without erasing CMS state")
