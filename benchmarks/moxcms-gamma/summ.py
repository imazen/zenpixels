import re,sys
arms=sys.argv[2].split(',')
g=None; d={}
for line in open(sys.argv[1],errors='replace'):
    m=re.match(r'\s+([a-z_0-9/]+/w\d+)\s+(\d+) rounds',line)
    if m: g=m.group(1); d[g]={'rounds':m.group(2)}; continue
    m=re.search(r'[├╰]─ (\w+)\s+([\d.]+)\s*(ns|µs|us|ms)?\s*±[\d.]+\s*(ns|µs|us|ms)\s+\[([^\]]+)\]',line)
    if m and g:
        u=m.group(3) or m.group(4)
        v=float(m.group(2))*{'ns':1,'µs':1000,'us':1000,'ms':1e6}[u]
        d[g][m.group(1)]=(v,m.group(5))
for g,r in d.items():
    if arms[0] not in r: print('??',g,r); continue
    print(f"{g:42} {arms[0]} {r[arms[0]][0]:>9.0f}ns | "+" | ".join(f"{a} {r[a][0]:>8.0f} {r[a][1]:>17}" for a in arms[1:] if a in r))
print(len(d),'groups')
