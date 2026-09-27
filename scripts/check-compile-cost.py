#!/usr/bin/env python3
"""Compare identical cold/warm/edited checks from two git archives, without worktrees.

Usage: python3 scripts/check-compile-cost.py BASELINE CURRENT --out report.json
Build caches are fresh for each revision; dependencies must already be fetched.
Times are a local single-run acceptance observation, not a statistical benchmark.
"""
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline')
    parser.add_argument('current')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = {'rustc': subprocess.check_output(['rustc', '-Vv'], text=True), 'jobs': 4, 'revisions': {}}
    with tempfile.TemporaryDirectory(prefix='zenpixels-compile-', dir='/tmp') as directory:
        for name, revision in [('baseline', args.baseline), ('current', args.current)]:
            revision = subprocess.check_output(['git', 'rev-parse', revision], cwd=root, text=True).strip()
            project = Path(directory) / name
            project.mkdir()
            archive = subprocess.check_output(['git', 'archive', revision], cwd=root)
            with tarfile.open(fileobj=io.BytesIO(archive)) as source:
                source.extractall(project, filter='data')
            env = dict(os.environ, CARGO_TARGET_DIR=str(project / 'target'), CARGO_TERM_COLOR='never')
            measurements = {}
            for package, flags in [('zenpixels', ['--no-default-features']), ('zenpixels-convert', [])]:
                command = ['cargo', 'check', '--offline', '--locked', '-j4', '-p', package, *flags]
                for mode in ['cold', 'warm', 'edited']:
                    if mode == 'edited':
                        with (project / package / 'src/lib.rs').open('a') as source:
                            source.write('\n// Compile acceptance: source invalidation.\n')
                    start = time.monotonic()
                    result = subprocess.run(command, cwd=project, env=env, text=True, capture_output=True)
                    elapsed = time.monotonic() - start
                    if result.returncode:
                        raise RuntimeError(result.stderr)
                    measurements[f'{package}/{mode}'] = round(elapsed, 4)
                    print(f'{name} {package} {mode}: {elapsed:.3f}s', flush=True)
            report['revisions'][name] = {'revision': revision, 'seconds': measurements}
    args.out.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
