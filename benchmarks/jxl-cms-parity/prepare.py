#!/usr/bin/env python3
"""Build the standalone adapter harness against supplied experiment sources.

No input checkout is edited. The decoder must already admit the selected SIMD
cohort. This helper does not relax production dependency requirements.
"""
import argparse
import hashlib
import json
from pathlib import Path


def source_digest(root):
    files = sorted(p for p in root.rglob('*') if p.is_file()
                   and (p.suffix == '.rs' or p.name == 'Cargo.toml')
                   and not any(part in {'.git', '.jj', 'target'} for part in p.relative_to(root).parts))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode() + b'\0')
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return {'files': len(files), 'sha256': digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pixels', type=Path)
    parser.add_argument('decoder', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    pixels, decoder, out = [p.resolve() for p in (args.pixels, args.decoder, args.output)]
    source = Path(__file__).resolve().parent / 'src/main.rs'
    q = json.dumps
    manifest = f'''[package]
name = "jxl-cms-parity-review"
version = "0.0.0"
edition = "2024"
publish = false
[workspace]
[dependencies]
jxl = {{ package = "zenjxl-decoder", path = {q(str(decoder / 'zenjxl-decoder'))}, features = ["cms"] }}
jxl_cli = {{ package = "zenjxl-decoder-cli", path = {q(str(decoder / 'zenjxl-decoder-cli'))}, default-features = false }}
zenpixels = {{ path = {q(str(pixels / 'zenpixels'))} }}
zenpixels-convert = {{ path = {q(str(pixels / 'zenpixels-convert'))}, features = ["cms-moxcms"] }}
moxcms = {{ version = "=0.9.1", features = ["options", "extended_range"] }}
whereat = "0.1.5"
bytemuck = "1.25.2"
[patch.crates-io]
garb = {{ git = "https://github.com/imazen/garb", rev = "605276d717ea0b31a1cd2a36ae05301383c7b8b5" }}
linear-srgb = {{ git = "https://github.com/imazen/linear-srgb", rev = "52896f4b24664315586bd60e5f048253b1f69052" }}
[[bin]]
name = "jxl-cms-parity-review"
path = {q(str(source))}
'''
    out.mkdir(parents=True, exist_ok=False)
    (out / 'Cargo.toml').write_text(manifest)
    (out / 'provenance.json').write_text(json.dumps({
        'adapter_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'pixels_sources': source_digest(pixels),
        'decoder_sources': source_digest(decoder),
    }, indent=2) + '\n')
    print(out / 'Cargo.toml')


if __name__ == '__main__':
    main()
