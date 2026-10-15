# gen2_hand.py: v20 hand fusion + a small grid over generation-feature weights
# on phys_gen feature dumps (same arithmetic as phys_gen's hand fusion).
#   python gen2_hand.py sel1:w,sel2:w,... rep1,rep2,...
# selects the grid point maximising the weighted mean MRR over the sel files,
# prints MRR of every file for the best few points.
import sys, itertools
import numpy as np

def load(p):
    Q = []
    f = open(p)
    for line in f:
        h = line.split()
        t, n = int(h[2]), int(h[3])
        rows = [f.readline().split() for _ in range(n)]
        X = np.array([[float(v) for v in r[:6]] + [float(v) for v in (r[7:13] if len(r) >= 13 else [0] * 6)]
                      for r in rows], dtype=np.float64).reshape(-1, 12)
        Q.append((X, t))
    return Q

def z(v):
    return (v - v.mean()) / (v.std() + 1e-6)

def prep(Q):
    out = []
    for X, t in Q:
        if len(X) == 0:
            out.append((None, None, t)); continue
        base = z(X[:, 0]) + 0.25 * z(X[:, 1]) + 1.0 * z(X[:, 2]) + 8 * np.maximum(X[:, 3], 0) ** 32 + 24 * X[:, 4]
        out.append((base, X[:, 6:12], t))
    return out

def mrr(P, gw):
    tot = 0.0
    for base, G, t in P:
        if t is None or t < 0 or base is None:
            continue
        s = base + G @ gw
        better = int((s >= s[t]).sum()) - 1
        if better < 25:
            tot += 1.0 / (better + 1)
    return tot / len(P)

sel = []
for it in sys.argv[1].split(','):
    p, w = (it.split(':') + ['1'])[:2]
    sel.append((p, float(w), prep(load(p))))
rep = [(p, prep(load(p))) for p in sys.argv[2].split(',')] if len(sys.argv) > 2 else []
res = []
for g0, g1, g2, g3, g5 in itertools.product([-4, -2, -1, 0, 1, 2], [-24, -16, -8, 0, 8], [0, -1], [0, 1, 2, 3, 4], [0, 0.5, 1, 1.5, 2, 3]):
    gw = np.array([g0, g1, g2, g3, 0, g5], dtype=np.float64)
    s = sum(w * mrr(P, gw) for _, w, P in sel) / sum(w for _, w, _ in sel)
    res.append((s, gw))
res.sort(key=lambda r: -r[0])
print('zero gen weights:', ' '.join(f'{p} {mrr(P, np.zeros(6)):.4f}' for p, _, P in sel), ' '.join(f'{p} {mrr(P, np.zeros(6)):.4f}' for p, P in rep))
# coordinate ascent from the best grid point (all 6 weights)
best_s, best = res[0][0], res[0][1].copy()
for step in (1.0, 0.5, 0.25):
    improved = True
    while improved:
        improved = False
        for k in range(6):
            for d in (step, -step):
                g = best.copy(); g[k] += d
                s2 = sum(w * mrr(P, g) for _, w, P in sel) / sum(w for _, w, _ in sel)
                if s2 > best_s + 1e-5:
                    best_s, best, improved = s2, g, True
res.insert(0, (best_s, best))
for s, gw in res[:8]:
    print(f'sel {s:.4f} gw {gw.tolist()} |', ' '.join(f'{p} {mrr(P, gw):.4f}' for p, _, P in sel), '|',
          ' '.join(f'{p} {mrr(P, gw):.4f}' for p, P in rep))
