# Finer score-mode grid + cross-holdout selection (pick on cqry, report tqry and vice versa).
import sys
sys.path.insert(0, r"C:\prog\casmi\pcslot")
import pcslot_grid as G
d = sys.argv[1]
tags = [("c", "c2"), ("c_rm", "c2_rm"), ("t", "t2"), ("t_rm", "t2_rm"), ("s", "s2"), ("o", "o2")]
runs = [(lab, *G.load(d, tag)) for lab, tag in tags]
rk = G.ranker(w_pop=2)
res = []
for k in (1, 2, 3):
    for g in (0.8, 0.9, 0.95, 2.0):
        gate = (lambda g: lambda q: q["lib"] < g)(g)
        for sw in (0.5, 0.75, 1, 1.25, 1.5, 2):
            for bias in (-4, -5, -6, -7, -8, -9, -10, -12, -14):
                for pmin in (1, 2):
                    v = [G.mrr_score(p, Q, rk, gate, k, sw, bias, pmin) for _, p, Q in runs]
                    res.append((f"k{k} lib<{g} sw{sw} bias{bias} min{pmin}", v))
base = [G.mrr(p, Q, lambda q: [], lambda q: False, [], 0) for _, p, Q in runs]
def mix(v, idx_in, idx_rm, f):
    return (1 - f) * sum(v[i] for i in idx_in) / len(idx_in) + f * sum(v[i] for i in idx_rm) / len(idx_rm)
print("base", " ".join("%.3f" % x for x in base))
for f in (0.1, 0.2, 0.3):
    for sel, ev in (("c", "t"), ("t", "c")):
        si = {"c": ([0], [1]), "t": ([2], [3])}[sel]
        ei = {"c": ([0], [1]), "t": ([2], [3])}[ev]
        # selection objective also charges the library holdouts (in-pool: half natural, half library)
        obj = lambda v: mix(v, si[0] + [4, 5], si[1], f) if False else (1 - f) * (0.5 * v[si[0][0]] + 0.25 * v[4] + 0.25 * v[5]) + f * v[si[1][0]]
        best = max(res, key=lambda r: obj(r[1]))
        eb = (1 - f) * (0.5 * base[ei[0][0]] + 0.25 * base[4] + 0.25 * base[5]) + f * base[ei[1][0]]
        ev_ = (1 - f) * (0.5 * best[1][ei[0][0]] + 0.25 * best[1][4] + 0.25 * best[1][5]) + f * best[1][ei[1][0]]
        print(f"f={f} select on {sel}: {best[0]} -> eval on {ev}: {eb:.4f} -> {ev_:.4f} (+{ev_-eb:.4f})  all: " + " ".join("%.3f" % x for x in best[1]))
print("--- full table (min1/min2, lib<0.9, k2) ---")
for n, v in res:
    if n.startswith("k2 lib<0.9 ") or n.startswith("k2 lib<2.0 sw1 "):
        print("%-36s" % n + " ".join("%.4f" % x for x in v))
