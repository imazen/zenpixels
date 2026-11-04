import re,collections,os,sys
d=os.path.expanduser('~/tmp/csa/')
# ---- line tables: bytes per source file, inlining included
def lines(fn):
    per=collections.Counter(); perline=collections.Counter(); ranges=[]
    dirs={};files={}
    cur=None;prev=None
    hdr=re.compile(r'^0x[0-9a-f]+\s+(\d+)\s+\d+\s+(\d+)\s')
    st=None
    with open(d+fn,errors='ignore') as f:
        for l in f:
            if l.startswith('debug_line['):
                dirs={};files={};prev=None;continue
            m=re.match(r'include_directories\[\s*(\d+)\] = "(.*)"',l)
            if m: dirs[int(m.group(1))]=m.group(2);continue
            m=re.match(r'file_names\[\s*(\d+)\]:',l)
            if m: st=int(m.group(1));continue
            m=re.match(r'\s+name: "(.*)"',l)
            if m and st is not None: files[st]=[m.group(1),0];continue
            m=re.match(r'\s+dir_index: (\d+)',l)
            if m and st is not None: files[st][1]=int(m.group(1));st=None;continue
            m=re.match(r'^(0x[0-9a-f]+)\s+(\d+)\s+\d+\s+(\d+)\s',l)
            if m:
                a=int(m.group(1),16);ln=int(m.group(2));fi=int(m.group(3))
                if prev is not None and a>prev[0]:
                    n,di=files.get(prev[2],('?',0))
                    p=dirs.get(di,'')+'/'+n
                    per[p]+=a-prev[0];perline[(p,prev[1])]+=a-prev[0]
                    ranges.append((prev[0],a,p,prev[1]))
                prev=None if 'end_sequence' in l else (a,ln,fi)
    return per,perline,ranges
pc,plc,rc=lines('conv.lines');pb,_,_=lines('base.lines')
def group(p):
    for k,v in [('/magetypes','magetypes'),('/archmage','archmage'),('/linear-srgb','linear-srgb'),('/garb','garb'),('zenpixels-convert/src','zenpixels-convert src'),('zenpixels/src','zenpixels src'),('library/core','rust core'),('library/alloc','rust alloc'),('library/std','rust std'),('library/stdarch','std::arch intrinsics'),('core_arch','std::arch intrinsics'),('/gimli','gimli'),('/addr2line','addr2line')]:
        if k in p: return v
    return 'other'
g=collections.Counter()
for p,v in pc.items(): g[group(p)]+=v
for p,v in pb.items(): g[group(p)]-=v
print('== bytes added, by source of the inlined code (conv minus base)')
for k,v in g.most_common(14):
    if abs(v)>200: print(f'{v:>8} {k}')
print('== largest source files inside convert code')
diff=collections.Counter({p:pc[p]-pb.get(p,0) for p in pc})
for p,v in diff.most_common(16):
    print(f'{v:>8} {p[-62:]}')
print('== top lines')
for (p,l),v in plc.most_common(18):
    if group(p) in('rust std','gimli','addr2line'):continue
    print(f'{v:>7} {p[-46:]}:{l}')
# ---- call graph and dominators
sz={}
for l in open(d+'conv.nm',errors='ignore'):
    p=l.split(None,3)
    if len(p)==4 and p[2] in 'tTwW': sz[p[3].strip()]=sz.get(p[3].strip(),0)+int(p[1])
edges=collections.defaultdict(set);cur=None
rx=re.compile(r'<([^>+]*(?:<[^>]*>[^>+]*)*?)(?:\+0x[0-9a-f]+)?>\s*$')
for l in open(d+'conv.dis',errors='ignore'):
    m=re.match(r'^[0-9a-f]+ <(.*)>:$',l)
    if m: cur=m.group(1);continue
    if cur is None: continue
    if re.search(r'\b(call|jmp|j[a-z]+|lea)\b',l):
        m=re.search(r'<(.+?)(\+0x[0-9a-f]+)?>\s*$',l)
        if m:
            t=m.group(1)
            if t!=cur and t in sz: edges[cur].add(t)
root='conv::main'
print('== root',root,root in sz,'nodes',len(sz))
reach={root};st=[root];order=[]
seen=set()
def dfs(r):
    stack=[(r,iter(sorted(edges[r])))];seen.add(r);post=[]
    while stack:
        n,it=stack[-1]
        for c in it:
            if c not in seen:
                seen.add(c);stack.append((c,iter(sorted(edges[c]))));break
        else:
            post.append(n);stack.pop()
    return post
post=dfs(root);rpo=list(reversed(post));idx={n:i for i,n in enumerate(rpo)}
preds=collections.defaultdict(list)
for a in rpo:
    for b in edges[a]:
        if b in idx: preds[b].append(a)
idom={root:root}
def inter(a,b):
    while a!=b:
        while idx[a]>idx[b]: a=idom[a]
        while idx[b]>idx[a]: b=idom[b]
    return a
ch=True
while ch:
    ch=False
    for n in rpo[1:]:
        ps=[p for p in preds[n] if p in idom]
        if not ps: continue
        new=ps[0]
        for p in ps[1:]: new=inter(p,new)
        if idom.get(n)!=new: idom[n]=new;ch=True
ret=collections.Counter()
for n in reversed(rpo):
    ret[n]+=sz.get(n,0)
    if n!=root: ret[idom[n]]+=ret[n]
print('reachable from main via direct edges:',len(rpo),'symbols',sum(sz.get(n,0) for n in rpo),'bytes of',sum(sz.values()))
print('== retained size: bytes that vanish if this function goes')
for n,v in ret.most_common(40):
    print(f'{v:>8} {sz.get(n,0):>6} {n[:78]}')
print('== children of the dispatchers')
for key in ['apply_step_u8','apply_step_f32','apply_step_u16','convert_row_buffered','apply_step']:
    for n in rpo:
        if key in n:
            kids=[(ret[c],c) for c in rpo if idom.get(c)==n and c!=n]
            if kids:
                print('--',n[:80],'retained',ret[n],'own',sz.get(n,0))
                for v,c in sorted(kids,reverse=True)[:18]: print(f'{v:>8} {c[-70:]}')

# ---- every dispatched kernel, by family, using exclusive retained size
print('== FAMILY')
A=[n for n in rpo if n.endswith('apply_step_u8')][0]
kids=[(ret[c],c) for c in rpo if idom.get(c)==A and c!=A]
fam=[('matte compositing',r'matte'),('gray',r'gray'),('HLG',r'hlg'),('PQ',r'(^|_|:)pq(_|$)'),('gamut matrix and LUT',r'gamut|fast_gamut|matlut|lutdec|hybrids|lin_to_u8|u8_rgb|u8_rgba'),('premultiply',r'premul'),('oklab',r'oklab'),('bt709',r'bt709'),('gamma22',r'gamma22|adobe'),('sRGB and linear transfer',r'srgb|linear'),('add or drop alpha, swizzle',r'alpha|bgra|rgb_to_rgba|swizzle|rgba_to_rgb'),('f16',r'f16'),('u16 and u8 depth',r'u16|u8|depth|naive')]
c=collections.Counter();k=collections.Counter()
un=[]
for v,n in kids:
    low=n.split('::')[-1].lower()
    for f,r in fam:
        if re.search(r,low): c[f]+=v;k[f]+=1;break
    else: c['other']+=v;k['other']+=1;un.append((v,n))
print('sum of children',sum(v for v,_ in kids),'kernels',len(kids),'apply_step own',sz.get(A,0))
for f,v in c.most_common(): print(f'{v:>8} {k[f]:>3} {f}')
print('-- unclassified');[print(f'{v:>8} {n[-60:]}') for v,n in sorted(un,reverse=True)[:8]]
print('== other roots under main')
for v,n in sorted([(ret[c],c) for c in rpo if idom.get(c)==root and c!=root],reverse=True)[:8]: print(f'{v:>8} {n[:76]}')
