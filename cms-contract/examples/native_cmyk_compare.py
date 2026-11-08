#!/usr/bin/env python3
"""Compare the existing native CMYK input/reference with production backends.
Only scalar summaries and worst samples are written; RGB outputs stay in memory.
"""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
for name in ['evaluator', 'source', 'input', 'reference', 'rgb-profile', 'output']:
    p.add_argument('--'+name, type=Path, required=True)
a = p.parse_args()
inputs = a.input.read_bytes()
reference_bytes = a.reference.read_bytes()
reference = array.array('f', reference_bytes)
assert len(inputs) % 16 == 0
assert len(inputs) // 16 * 3 == len(reference)
assert all(math.isfinite(x) for x in reference)
report = {'input_sha256': hashlib.sha256(inputs).hexdigest(),
          'source_sha256': hashlib.sha256(a.source.read_bytes()).hexdigest(),
          'reference_sha256': hashlib.sha256(reference_bytes).hexdigest(),
          'bound': 0.0013, 'components': len(reference), 'backends': {}}
clip = lambda x: min(1.0, max(0.0, x))
for name, mode in [('moxcms', 'mox:named-srgb:inverted'), ('rust-lcms', 'production:named-srgb:inverted')]:
    out = subprocess.run([str(a.evaluator.resolve()), str(a.source), str(a.rgb_profile), str(a.input), '-', mode],
                         capture_output=True, check=True)
    values = array.array('f', out.stdout)
    assert len(values) == len(reference)
    assert all(math.isfinite(x) for x in values)
    errors = [abs(clip(x)-clip(y)) for x,y in zip(values, reference)]
    i = max(range(len(errors)), key=errors.__getitem__)
    source = array.array('f', inputs)
    report['backends'][name] = {'output_sha256': hashlib.sha256(out.stdout).hexdigest(),
        'maximum_error': errors[i], 'violations': sum(x > 0.0013 for x in errors),
        'worst': {'pixel': i//3, 'channel': i%3,
                  'inverted_cmyk': list(source[i//3*4:i//3*4+4]),
                  'actual': values[i], 'reference': reference[i]}}
    a.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
