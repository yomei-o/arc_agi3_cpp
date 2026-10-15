import re,sys,collections
# aggregate FOLDRR lines: [name k] FOLDRR <set> <nq> <rrModel> <rrHand>
agg=collections.defaultdict(lambda: collections.defaultdict(lambda:[0,0.,0.])); its=collections.defaultdict(list)
for f in sys.argv[1:]:
  for l in open(f):
    m=re.match(r'\[(\S+) (\d)\] FOLDRR (\S+) (\d+) ([\d.]+) ([\d.]+)',l)
    if m:
      a=agg[m.group(1)][m.group(3).replace('f_x','').replace('f_y','').replace('.txt','')]
      a[0]+=int(m.group(4)); a[1]+=float(m.group(5)); a[2]+=float(m.group(6))
    m=re.match(r'\[(\S+) (\d)\] best iteration (\d+)',l)
    if m: its[m.group(1)].append(int(m.group(3)))
sets=['c','crm','t','trm','s','o']
print('%-16s %-22s'%('run','iters')+' '.join('%-15s'%s for s in sets)+'  nonlib(t+trm)/2  inlib(s+o)/2')
for r in agg:
  d=agg[r]; row=[]
  for s in sets:
    n,m,h=d[s]; row.append('%.3f(%+.3f)'%(m/n,(m-h)/n) if n else '-')
  def mm(ss): return sum(d[s][1]/d[s][0] for s in ss)/len(ss), sum((d[s][1]-d[s][2])/d[s][0] for s in ss)/len(ss)
  a=mm(['t','trm']); b=mm(['s','o'])
  print('%-16s %-22s'%(r,','.join(map(str,its[r])))+' '.join('%-15s'%x for x in row)+'  %.3f(%+.3f)  %.3f(%+.3f)'%(a+b))
h=next(iter(agg.values())) if agg else None
if h: print('hand            ', ' '.join('%s=%.3f'%(s,h[s][2]/h[s][0]) for s in sets if h[s][0]))
