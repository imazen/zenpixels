import json
d=json.load(open(__import__('sys').argv[1]))
rows={}
for g in d['comparisons']:
    n=g['throughput']['Elements']
    rows[g['group_name']]={b['name']:(b['summary']['median']/n,b['summary']['mad']/n) for b in g['benchmarks']}
methods=['clip','soft-table','soft-table+gate','solve-1','solve-2','solve-3','solve-4','line-rgb']
print('ns/pixel, median of 30 rounds, kernel only, r7900x (Zen 4), v3 dispatch; not measured on Zen 5')
print('set'.ljust(44),*[m[:10].rjust(10) for m in methods])
for g,v in rows.items():
    if g.startswith('kernel'): print(g[7:].ljust(44)[:44],*[f"{v.get(m,(float('nan'),0))[0]:10.2f}" for m in methods])
print()
print('whole P3 u8 -> sRGB u8 row, ns/pixel')
for g,v in rows.items():
    if g.startswith('u8'): print(g[7:].ljust(44)[:44],*[f"{k} {x[0]:6.2f}" for k,x in v.items()])
