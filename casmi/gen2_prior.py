# gen2_prior.py: per-op log prior for phys_gen's op-prior feature, from the
# generated-candidate details (det=...) of a TRAINING holdout run:
#   python gen2_prior.py <det1.txt>[,det2.txt...] <out_prior.txt> [alpha=20]
# prior(op) = log((hits + alpha*p0) / (n + alpha)) over one-step candidates
# made by op (p0 = overall one-step hit rate); 'default' = log(p0).
# A two-step chain's prior inside phys_gen is the sum over its ops.
import sys, math
from collections import Counter
dets = sys.argv[1].split(',')
out = sys.argv[2]
alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
n, h = Counter(), Counter()
for d in dets:
    for line in open(d, encoding='utf8'):
        p = line.rstrip('\n').split('\t')
        if len(p) < 9:
            continue
        t = int(p[2])
        for ch in p[8].split(';'):
            if '/' in ch or not ch:
                continue
            n[ch] += 1
            h[ch] += t
N, H = sum(n.values()), sum(h.values())
p0 = (H + 1) / (N + 1)
with open(out, 'w') as f:
    f.write(f'default {math.log(p0):.4f}\n')
    for op in sorted(n, key=lambda o: -n[o]):
        v = math.log((h[op] + alpha * p0) / (n[op] + alpha))
        f.write(f'{op} {v:.4f}\n')
        print(f'{op:8s} n={n[op]:7d} hits={h[op]:4d} logprior={v:.3f}')
print('p0', p0, 'N', N, 'H', H)
