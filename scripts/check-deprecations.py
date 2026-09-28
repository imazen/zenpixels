#!/usr/bin/env python3
"""Check warnings at a real downstream boundary, including inferred receivers.

Run after fetching workspace dependencies: python3 scripts/check-deprecations.py
No extra Rust test dependency; each probe denies deprecation warnings locally.
"""

import json
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PLAN = """
use zenpixels_convert::{ConvertPlan, PixelDescriptor};
fn plan() -> ConvertPlan {
    ConvertPlan::new(PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBA8_SRGB).unwrap()
}
"""
PROBES = {
    "predicate": (
        "fn main() { let d = zenpixels_convert::PixelDescriptor::RGB8_SRGB; "
        "let _ = zenpixels_convert::requires_cms(&d, &d); }",
        "requires_cms",
        False,
    ),
    "estimate": (PLAN + "fn main() { let _ = plan().estimate(1, 1); }", "estimate", True),
    "estimate_in": (
        PLAN + """
#[allow(deprecated)]
fn inputs() -> (zenpixels_convert::ImageCharacteristics, zenpixels_convert::ComputeEnvironment) {
    (zenpixels_convert::ImageCharacteristics::new(1, 1, PixelDescriptor::RGB8_SRGB),
     zenpixels_convert::ComputeEnvironment::new())
}
fn main() { let (image, compute) = inputs(); let _ = plan().estimate_in(&image, &compute); }
""",
        "estimate_in",
        True,
    ),
    "estimate_inferred": (
        """
#[allow(deprecated)]
fn legacy_provider() -> zenpixels_convert::ResourceEstimate {
    zenpixels_convert::ResourceEstimate::unknown()
}
fn main() { let _ = legacy_provider().wall_ms(); }
""",
        "wall_ms",
        True,
    ),
    "adapted_inferred": (
        """
#[allow(deprecated)]
fn legacy_provider() -> zenpixels_convert::adapt::Adapted<'static> {
    zenpixels_convert::adapt::Adapted {
        data: std::borrow::Cow::Borrowed(&[0, 0, 0]),
        descriptor: zenpixels_convert::PixelDescriptor::RGB8_SRGB, width: 1, rows: 1,
    }
}
fn main() { let _ = legacy_provider().as_pixel_slice(); }
""",
        "as_pixel_slice",
        False,
    ),
    "cow": (
        """
fn main() {
    let d = zenpixels_convert::PixelDescriptor::RGB8_SRGB;
    let result = zenpixels_convert::adapt::adapt_for_encode_cow(&[0, 0, 0], d, 1, 1, 3, &[d]).unwrap();
    let _ = result.as_slice();
}
""",
        None,
        False,
    ),
}
for name, expression in {
    "ComputeEnvironment": "ComputeEnvironment::new()",
    "ImageCharacteristics": "ImageCharacteristics::new(1, 1, zenpixels_convert::PixelDescriptor::RGB8_SRGB)",
    "ResourceEstimate": "ResourceEstimate::unknown()",
    "SimdTier": "SimdTier::Unknown",
}.items():
    PROBES[name.lower()] = (f"fn main() {{ let _ = zenpixels_convert::{expression}; }}", name, True)
PROBES["estimate_module"] = (
    "fn main() { let _ = zenpixels_convert::estimate::ResourceEstimate::unknown(); }",
    "ResourceEstimate",
    True,
)



def main():
    with tempfile.TemporaryDirectory(prefix="zenpixels-deprecations-") as directory:
        project = Path(directory)
        (project / "src/bin").mkdir(parents=True)
        # JSON strings are valid TOML strings for these absolute paths.
        (project / "Cargo.toml").write_text(f"""
[package]
name = "zenpixels-deprecation-probe"
version = "0.0.0"
edition = "2021"
[features]
default = ["zenpixels-convert/default"]
estimation-experimental = ["zenpixels-convert/estimation-experimental"]
[dependencies]
zenpixels = {{ path = {json.dumps(str(ROOT / "zenpixels"))}, default-features = false }}
zenpixels-convert = {{ path = {json.dumps(str(ROOT / 'zenpixels-convert'))}, default-features = false }}
[patch.crates-io]
zenpixels = {{ path = {json.dumps(str(ROOT / 'zenpixels'))} }}
""")
        for name, (source, _, _) in PROBES.items():
            (project / f"src/bin/{name}.rs").write_text("#![deny(deprecated)]\n" + source)
        env = dict(os.environ, CARGO_TARGET_DIR=str(ROOT / "target/deprecation-ui"))
        count = 0
        for defaults in (True, False):
            for opted_in in (False, True):
                for name, (_, expected, estimation) in PROBES.items():
                    command = ["cargo", "check", "--offline", "--message-format=json", "--bin", name]
                    if not defaults:
                        command.append("--no-default-features")
                    if opted_in:
                        command.extend(["--features", "estimation-experimental"])
                    result = subprocess.run(command, cwd=project, env=env, capture_output=True, text=True)
                    errors = []
                    for line in result.stdout.splitlines():
                        item = json.loads(line)
                        if item.get("reason") == "compiler-message" and item["message"]["level"] == "error":
                            errors.append(item["message"])
                    should_warn = expected is not None and not (estimation and opted_in)
                    if should_warn:
                        ok = result.returncode != 0 and bool(errors) and all(
                            (error.get("code") or {}).get("code") == "deprecated" for error in errors
                        ) and any(expected in error["message"] for error in errors)
                    else:
                        ok = result.returncode == 0
                    if not ok:
                        raise AssertionError(
                            f"{name}: default={defaults}, opt_in={opted_in}, expected_warning={should_warn}\n"
                            + result.stderr + "\n" + "\n".join(e.get("rendered", e["message"]) for e in errors)
                        )
                    count += 1
        print(f"Passed {count} downstream deprecation checks (default/no-default, opt-in/off).")


if __name__ == "__main__":
    main()
