# build_genpool.py <dir> <pool_stem> <qry_mols.txt> <gc.txt> <out_stem> <removeTruth 0/1>
import sys, struct
d, ps, qm, gc, out, rem = sys.argv[1:7]
rem = rem == '1'
hb = open(f'{d}/{ps}_head.bin', 'rb').read()
recs = [struct.unpack_from('<fii', hb, 12 * i) for i in range(len(hb) // 12)]
mols = [l.rstrip('\r\n') for l in open(f'{d}/{ps}_mols.txt', encoding='utf8')]
truth = set(l.split('\t')[0] for l in open(qm, encoding='utf8') if l.strip())
names = [m.split('\t')[0] for m in mols]
removed = 0
if rem:
    for i, n in enumerate(names):
        if n in truth:
            mols[i] = 'REMOVED' + n[7:] + '\t' + mols[i].split('\t', 1)[1]
            removed += 1
    recs = [r for r in recs if names[r[1]] not in truth]
have = set(names) - (truth if rem else set())
added = 0
for l in open(gc, encoding='utf8'):
    if l.startswith('G '):
        continue
    p = l.rstrip('\n').split('\t')
    if p[0] in have:
        continue
    have.add(p[0])
    mols.append(p[0] + '\t' + p[1])
    recs.append((float(p[2]), len(mols) - 1, 0))
    added += 1
recs.sort()
with open(f'{d}/{out}_head.bin', 'wb') as f:
    for r in recs:
        f.write(struct.pack('<fii', *r))
with open(f'{d}/{out}_mols.txt', 'w', encoding='utf8', newline='\n') as f:
    f.write('\n'.join(mols) + '\n')
print(out, 'removed', removed, 'added', added, 'records', len(recs))
