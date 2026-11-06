#!/usr/bin/env python3
"""Reconstruct one explicitly named optimized experiment; never overwrite it."""
import argparse,hashlib,io,json,subprocess,tarfile,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REV='4ab190f0e76f95b332a04ce059b51c893e2f5d79'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('name',choices=['opt-or','opt-wide','opt-pixel','opt-cold'])
parser.add_argument("--record-new",action="store_true",help="freeze a newly introduced source manifest; never replace an existing record")
args=parser.parse_args()
out=ROOT/'variants'/args.name
if out.exists(): raise SystemExit(f'preserve existing experiment first: {out}')
request=urllib.request.Request(f'https://api.github.com/repos/lilith/moxcms/tarball/{REV}',headers={'User-Agent':'gamma-experiment'})
with urllib.request.urlopen(request,timeout=60) as response: data=response.read()
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
 for member in archive.getmembers():
  parts=Path(member.name).parts[1:]
  if member.isfile() and parts and '..' not in parts and (parts[0]=='src' or parts[-1] in ['Cargo.toml','README.md','LICENSE.md','LICENSE-APACHE.md']):
   target=out.joinpath(*parts);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.extractfile(member).read())
subprocess.run(['patch','--batch','--fuzz=0','-p1','-i',str(ROOT/'patches'/(args.name+'.patch'))],cwd=out,check=True)
manifest=out/'Cargo.toml'
s=manifest.read_text().replace('workspace = { members = ["app", "fuzz"] }','').replace('version = "0.9.1"','version = "0.9.1-'+args.name+'"',1)
manifest.write_text(s+'\n[workspace]\n')
hashes={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file() and (p.name=='Cargo.toml' or 'src' in p.relative_to(out).parts)}
record={'base':REV,'name':args.name,'patch_sha256':hashlib.sha256((ROOT/'patches'/(args.name+'.patch')).read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()}
expected_path=ROOT/'expected-optimized.json'
expected_all=json.loads(expected_path.read_text())
if args.record_new:
 assert args.name not in expected_all
 expected_all[args.name]=record
 expected_path.write_text(json.dumps(expected_all,indent=2)+'\n')
expected=expected_all[args.name]
assert record == expected, (record,expected)
(out/'experiment-source.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
