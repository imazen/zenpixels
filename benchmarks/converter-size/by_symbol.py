import re,collections,os
d=os.path.expanduser('~/tmp/csa/')
def syms(f):
    r={}
    for l in open(d+f,errors='ignore'):
        p=l.split(None,3)
        if len(p)==4 and p[2] in 'tTwWrRdDbB': r.setdefault(p[3].strip(),[]).append((int(p[1]),p[2],int(p[0])))
    return r
B=syms('base.nm');C=syms('conv.nm')
new={k:sum(s for s,_,_ in v)-sum(s for s,_,_ in B.get(k,[])) for k,v in C.items()}
newt={k:v for k,v in new.items() if v>0 and C[k][0][1] in 'tTwW'}
newd={k:v for k,v in new.items() if v>0 and C[k][0][1] not in 'tTwW'}
print('new code symbols',len(newt),'bytes',sum(newt.values()),'| new data',len(newd),sum(newd.values()))
def crate(n):
    m=re.search(r'(zenpixels_convert|zenpixels|linear_srgb|garb|magetypes|archmage|whereat|once_cell|core|alloc|std|gimli|addr2line|miniz_oxide|hashbrown|libm|rgb|bytemuck)\b',n)
    return m.group(1) if m else 'other'
c=collections.Counter()
for k,v in newt.items(): c[crate(k)]+=v
print('-- new code by crate'); [print(f'{v:>8} {k}') for k,v in c.most_common()]
# tiers
t=collections.Counter();tn=collections.Counter()
for k,v in newt.items():
    if crate(k) not in('zenpixels_convert','linear_srgb','garb','magetypes'): continue
    m=re.search(r'_(v4x|v4|v3|v2|neon|wasm128|scalar)(\b|::|<|$)',k)
    x=m.group(1) if m else 'not tier-suffixed'
    t[x]+=v;tn[x]+=1
print('-- zen code by SIMD tier in symbol name'); [print(f'{v:>8} {tn[k]:>4} {k}') for k,v in t.most_common()]
# modules
m=collections.Counter()
for k,v in newt.items():
    if crate(k)!='zenpixels_convert': continue
    x=re.search(r'zenpixels_convert::(\w+)(::(\w+))?',k)
    m[(x.group(1)+('::'+x.group(3) if x and x.group(3) and not x.group(3).startswith('__') and x.group(1) in('convert',) else '')) if x else 'impl']+=v
print('-- zenpixels_convert by module'); [print(f'{v:>8} {k}') for k,v in m.most_common(12)]
print('-- new non-zen code, largest'); 
for k,v in sorted(((k,v) for k,v in newt.items() if crate(k) not in('zenpixels_convert','linear_srgb','garb','magetypes')),key=lambda x:-x[1])[:14]: print(f'{v:>8} {k[:84]}')
print('-- new data, largest')
for k,v in sorted(newd.items(),key=lambda x:-x[1])[:12]: print(f'{v:>8} {k[:84]}')
print('-- largest zen functions')
for k,v in sorted(((k,v) for k,v in newt.items() if crate(k) in('zenpixels_convert','linear_srgb','garb','magetypes')),key=lambda x:-x[1])[:24]: print(f'{v:>8} {k[:84]}')
