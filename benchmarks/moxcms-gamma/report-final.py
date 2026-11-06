#!/usr/bin/env python3
"""Render the frozen final study's selected rows without using rejected cells."""
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('study',type=Path);p.add_argument('comparisons',type=Path);p.add_argument('output',type=Path)
a=p.parse_args()
study=json.loads((a.study/'study.json').read_text())
rows=list(csv.DictReader(a.comparisons.open()))
selected=[]
for direction in ['encode','decode']:
 for zero,negative in [(0,0),(0,50),(10,0),(100,0)]:
  selected.append(f'eval/{direction}/zero{zero}/neg{negative}/w1024')
for transform in ['cicp_gamma22_to_bt2020','cicp_smpte428_to_bt2020','cicp_gamma28_to_bt2020','adobe_to_prophoto_extended']:
 selected.append(f'transform/{transform}/w1024')
for zero in [0,10,100]:selected.append(f'transform/prophoto_to_srgb/zero{zero}/w1024')
lines=['| 1024 RGB pixels | Original µs | Sign only µs | First range µs | Final µs | Final vs declared baseline |',
       '|---|---:|---:|---:|---:|---|']
record=[]
for name in selected:
 cell=next(c for c in study['cells'] if c['group']==name)
 if cell['status']!='complete':
  lines.append(f'| {name} | — | — | — | — | Resource checks rejected cell |')
  record.append({'group':name,'status':cell['status']});continue
 group=[r for r in rows if r['group']==name]
 assert group
 means={r['candidate']:float(r['candidate_mean_us']) for r in group}
 means[group[0]['baseline']]=float(group[0]['baseline_mean_us'])
 r=next(r for r in group if r['candidate']=='opt_cold')
 def value(key):return f'{means[key]:.3f}' if key in means else '—'
 verdict=f"{float(r['estimate_pct']):+.2f}% [{float(r['lower_pct']):+.2f}%, {float(r['upper_pct']):+.2f}%], {r['verdict']}"
 lines.append('| '+name+' | '+' | '.join(value(k) for k in ['before','sign_fix','guarded','opt_cold'])+' | '+verdict+' |')
 record.append({'group':name,'status':'complete','means_us':means,'comparison':r})
a.output.write_text('\n'.join(lines)+'\n')
a.output.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
print(a.output)
