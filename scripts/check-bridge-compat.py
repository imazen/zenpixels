#!/usr/bin/env python3
"""Test identical consumer source against all packaged core/convert pairings.

Requires verified 0.2.17 and 0.3.1 .crate archives in --packages.
Optional --siblings adds real local zencodec/zenpipe public buffer boundaries.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packages', type=Path, required=True)
    parser.add_argument('--siblings', type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='zenpixels-compat-', dir='/tmp') as directory:
        base = Path(directory)
        for name in ['zenpixels', 'zenpixels-convert']:
            for version in ['0.2.17', '0.3.1']:
                with tarfile.open(args.packages / f'{name}-{version}.crate') as archive:
                    archive.extractall(base, filter='data')
        count = 0
        for core in ['0.2.17', '0.3.1']:
            for convert in ['0.2.17', '0.3.1']:
                project = base / f'consumer-{core}-{convert}'
                (project / 'src').mkdir(parents=True)
                shutil.copyfile(ROOT / 'tests/compat/lib.rs', project / 'src/lib.rs')
                siblings = ''
                sibling_features = '[]'
                sibling_patch = ''
                if args.siblings:
                    siblings = f'\nzencodec = {{ path = {json.dumps(str(args.siblings / "zencodec"))}, optional = true }}\nzenpipe = {{ path = {json.dumps(str(args.siblings / "zenpipe"))}, default-features = false, optional = true }}\n'
                    sibling_features = '["dep:zencodec", "dep:zenpipe"]'
                    sibling_patch = '\n'.join(f'{name} = {{ path = {json.dumps(str(args.siblings / name))} }}' for name in ['zencodec', 'zenresize'])
                (project / 'Cargo.toml').write_text(f'''
[package]
name = "zenpixels-bridge-consumer"
version = "0.0.0"
edition = "2024"
[features]
default = ["std", "zenpixels-convert/icc-db"]
std = ["zenpixels/std", "zenpixels-convert/std"]
interop = ["zenpixels/rgb", "zenpixels/imgref", "zenpixels-convert/rgb", "zenpixels-convert/imgref"]
experimental = ["zenpixels-convert/hdr-experimental", "zenpixels-convert/estimation-experimental"]
legacy-features = ["zenpixels/planar", "zenpixels-convert/planar", "zenpixels-convert/serde"]
siblings = {sibling_features}
[dependencies]
zenpixels = {{ version = "={core}", default-features = false }}
zenpixels-convert = {{ version = "={convert}", default-features = false }}
rgb = "0.8"
{siblings}
[patch.crates-io]
zenpixels = {{ path = {json.dumps(str(base / f"zenpixels-{core}"))} }}
zenpixels-convert = {{ path = {json.dumps(str(base / f"zenpixels-convert-{convert}"))} }}
{sibling_patch}
''')
                env = dict(os.environ, CARGO_TARGET_DIR=str(Path(os.environ.get('CARGO_TARGET_DIR', Path(tempfile.gettempdir()) / 'zenpixels-validation')) / 'bridge-compat'))
                cases = [('--no-default-features',), (), ('--features', 'interop,experimental,legacy-features')]
                if args.siblings:
                    cases.append(('--features', 'siblings,interop'))
                for flags in cases:
                    subprocess.run(['cargo', 'test', '--quiet', *flags], cwd=project, env=env, check=True)
                    metadata = json.loads(subprocess.check_output(['cargo', 'metadata', '--format-version', '1', *flags], cwd=project, env=env, text=True))
                    # cargo metadata includes optional packages, so count only
                    # nodes selected in this feature graph.
                    selected = {node['id'] for node in metadata['resolve']['nodes']}
                    for name in ['zenpixels', 'zenpixels-convert']:
                        versions = [p['version'] for p in metadata['packages'] if p['name'] == name and p['id'] in selected]
                        assert len(versions) == 1, (name, versions)
                    count += 1
                print(f'PASS core {core}, converter {convert}', flush=True)
        # Upgrade a real existing bridge lockfile without changing consumer source.
        project = base / 'consumer-0.2.17-0.2.17'
        manifest = project / 'Cargo.toml'
        manifest.write_text(manifest.read_text().replace('0.2.17', '0.3.1'))
        subprocess.run(['cargo', 'update', '--quiet'], cwd=project, env=env, check=True)
        flags = ['--features', 'siblings,interop'] if args.siblings else []
        subprocess.run(['cargo', 'test', '--quiet', '--locked', *flags], cwd=project, env=env, check=True)
        metadata = json.loads(subprocess.check_output(['cargo', 'metadata', '--format-version', '1', '--locked', *flags], cwd=project, env=env, text=True))
        selected = {node['id'] for node in metadata['resolve']['nodes']}
        for name in ['zenpixels', 'zenpixels-convert']:
            versions = [p['version'] for p in metadata['packages'] if p['name'] == name and p['id'] in selected]
            assert versions == ['0.3.1'], (name, versions)
        print(f'Passed {count} packaged same-source feature/pairing cases and upgraded-lockfile case.', flush=True)


if __name__ == '__main__':
    main()
