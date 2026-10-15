# gen2_prep.py: post-process a phys_gen feature dump for fitting experiments
# without re-running phys_gen:
#   python gen2_prep.py <features_in> <det_in> <prior.txt|-> <cap> <out> [no2=0]
# - fills the op-prior column (last) from the details file + prior table
#   (a chain's prior = sum of its ops' priors, candidate = max over its chains)
# - keeps only the first <cap> generated candidates of each query (the dump
#   lists them in phys_gen's order: source sim desc, steps, rank)
# - no2=1 drops generated candidates whose fewest steps is 2
import sys
fin, din, prior_p, cap, outp = sys.argv[1:6]
cap = int(cap)
no2 = len(sys.argv) > 6 and sys.argv[6] == '1'
prior, dflt = {}, 0.0
if prior_p != '-':
    for l in open(prior_p):
        k, v = l.split()
        if k == 'default':
            dflt = float(v)
        else:
            prior[k] = float(v)
det = {}
for l in open(din, encoding='utf8'):
    p = l.rstrip('\n').split('\t')
    det.setdefault(p[0], []).append(p)
def cp(chs):
    return max(sum(prior.get(o, dflt) for o in ch.split('/')) for ch in chs.split(';'))
f = open(fin)
out = open(outp, 'w')
nq = kept = tot = 0
for line in f:
    h = line.split()
    name, t, n = h[1], int(h[2]), int(h[3])
    rows = [f.readline().split() for _ in range(n)]
    d = det.get(name, [])
    keep = []
    k = 0
    newt = -1
    for i, r in enumerate(rows):
        if float(r[7]) > 0:  # generated
            dd = d[k] if k < len(d) else None
            k += 1
            if k > cap:
                continue
            if no2 and dd is not None and dd[5] == '2':
                continue
            if prior_p != '-' and dd is not None:
                r[12] = f'{cp(dd[8]):.4f}'
        if i == t:
            newt = len(keep)
        keep.append(r)
    out.write(f'Q {name} {newt} {len(keep)}\n')
    for r in keep:
        out.write(' '.join(r) + '\n')
    nq += 1; kept += newt >= 0
print(fin, '->', outp, 'queries', nq, 'truth present', kept)
