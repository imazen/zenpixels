#!/usr/bin/env python3
"""Prepare an isolated native-codec consumer graph with a frozen core/pixel cohort.

The inputs are local checkouts or git-archive exports. Source trees are read-only;
Cargo's lock and target belong to the new output directory. This is resolution
qualification, not a codec correctness test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tomllib


def literal(value):
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return '[' + ', '.join(map(literal, value)) + ']'
    if isinstance(value, dict):
        return '{ ' + ', '.join(f'{json.dumps(k)} = {literal(v)}' for k, v in value.items()) + ' }'
    raise TypeError(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('codec', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('core', type=Path)
    parser.add_argument('pixels', type=Path)
    args = parser.parse_args()
    codec, out, core, pixels = [p.resolve() for p in (args.codec, args.output, args.core, args.pixels)]
    root = tomllib.loads((codec / 'Cargo.toml').read_text())
    crate = codec if 'package' in root else codec / codec.name
    package = tomllib.loads((crate / 'Cargo.toml').read_text())
    name = package['package']['name']
    features = [f for f in ['zencodec', 'encode', 'decode'] if f in package.get('features', {})]
    patches = root.get('patch', {})
    for table in patches.values():
        for entry in table.values():
            if 'path' in entry:
                entry['path'] = str((codec / entry['path']).resolve())
    registry = patches.setdefault('crates-io', {})
    for key, directory in [('zenpixels', 'zenpixels'), ('zenpixels-convert', 'zenpixels-convert')]:
        registry[key] = {'path': str(pixels / directory)}
    core_entries = {}
    for key in ['zencodec', 'zenmeta', 'zencodec-testkit', 'zencodec-io', 'zencodec-coroutine', 'zencodec-media', 'zencodec-media-testkit']:
        directory = core if key == 'zencodec' else core / key
        if (directory / 'Cargo.toml').exists():
            core_entries[key] = {'path': str(directory)}
    registry.update(core_entries)
    patches.setdefault('https://github.com/imazen/zencodec', {}).update(core_entries)
    registry['garb'] = {'git': 'https://github.com/imazen/garb', 'rev': '58ce9cad7b337418a7abe9b1f6e3e4bd1928b385'}
    registry['linear-srgb'] = {'git': 'https://github.com/imazen/linear-srgb', 'rev': '52896f4b24664315586bd60e5f048253b1f69052'}
    lines = ['[package]', f'name = "{name}-cohort-resolution"', 'version = "0.0.0"', 'edition = "2024"', 'publish = false', '', '[workspace]', '', '[dependencies]', f'{name} = {literal({"path": str(crate), "default-features": False, "features": features})}']
    for source, table in patches.items():
        lines += ['', f'[patch.{json.dumps(source)}]']
        lines += [f'{json.dumps(key)} = {literal(value)}' for key, value in table.items()]
    out.mkdir(parents=True, exist_ok=False)
    (out / 'src').mkdir()
    (out / 'src/lib.rs').write_text('#![forbid(unsafe_code)]\n')
    (out / 'Cargo.toml').write_text('\n'.join(lines) + '\n')
    revision = subprocess.check_output(['git', '-C', str(codec), 'rev-parse', 'HEAD'], text=True).strip()
    (out / 'provenance.json').write_text(json.dumps({'codec': name, 'git_head': revision, 'root_manifest_sha256': hashlib.sha256((codec / 'Cargo.toml').read_bytes()).hexdigest(), 'crate_manifest_sha256': hashlib.sha256((crate / 'Cargo.toml').read_bytes()).hexdigest(), 'working_tree_status': subprocess.check_output(['git', '-C', str(codec), 'status', '--porcelain'], text=True), 'core': (core / '.source-revision').read_text().strip(), 'pixels': (pixels / '.source-revision').read_text().strip(), 'features': features}, indent=2) + '\n')
    print(out / 'Cargo.toml')


if __name__ == '__main__':
    main()
