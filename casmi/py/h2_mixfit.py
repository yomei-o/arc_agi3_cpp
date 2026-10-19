"""Fit LB = sum_c pi_c * MRR_c(version) over holdout components (S = sibling, D = denovo, R = denovo truth removed).

    python h2_mixfit.py <results.tsv>     (tsv: tag <TAB> mrr, tags like haqry_v24, haqry_v24_rm)
LB targets have FPNet's +0.008 (v20->v22) taken out of v23/v24/v25 (v26 ran without FPNet),
because every local run is FPNet-free. pi on a 0.02 simplex grid, least squares over versions.
"""
import itertools, sys

VERS = ["v20", "v21", "v23", "v24", "v25", "v26"]
LB = {"v20": 0.291, "v21": 0.250, "v23": 0.313 - 0.008, "v24": 0.322 - 0.008, "v25": 0.321 - 0.008, "v26": 0.286}
R = {}
for l in open(sys.argv[1]):
    t, v = l.split()
    R[t] = float(v)


def comp(name):
    if name.endswith("_rm"):
        q = name[:-3]
        return {v: (R.get(f"{q}_{v}_rm", 0.0) if v in ("v24", "v25", "v26") else 0.0) for v in VERS}
    return {v: R.get(f"{name}_{v}") for v in VERS}


S = ["hasqry", "hbsqry", "hxqry"]
D = ["haqry", "hcqry", "hbqry"]
RM = ["haqry_rm", "hcqry_rm", "hbqry_rm"]
res = []
grid = [i / 50 for i in range(51)]
for s, d, r in itertools.product(S, D, RM):
    cs, cd, cr = comp(s), comp(d), comp(r)
    if any(x is None for c in (cs, cd, cr) for x in c.values()):
        continue
    for a in grid:
        for b in grid:
            if a + b > 1 + 1e-9:
                continue
            c = 1 - a - b
            pred = {v: a * cs[v] + b * cd[v] + c * cr[v] for v in VERS}
            err = sum((pred[v] - LB[v]) ** 2 for v in VERS)
            # delta-only error (absolute level free): compare differences to v20
            derr = sum(((pred[v] - pred["v20"]) - (LB[v] - LB["v20"])) ** 2 for v in VERS)
            k = sum(pred[v] * LB[v] for v in VERS) / max(1e-12, sum(pred[v] ** 2 for v in VERS))
            serr = sum((k * pred[v] - LB[v]) ** 2 for v in VERS)
            res.append((err, derr, s, d, r, a, b, c, pred, serr, k))
print("LB adj: " + " ".join(f"{v} {LB[v]:.3f}" for v in VERS))
for lab, key in (("absolute", 0), ("deltas only (level free)", 1), ("scaled: LB = k * mix", 9)):
    res.sort(key=lambda x: x[key])
    print(f"--- best by {lab}")
    seen = set()
    for x in res:
        err, derr, s, d, r, a, b, c, pred, serr, k = x
        if (s, d, r) in seen:
            continue
        seen.add((s, d, r))
        if len(seen) > 6:
            break
        kk = k if key == 9 else 1.0
        print(f"rmse {(err/6)**.5:.4f} drmse {(derr/6)**.5:.4f} srmse {(serr/6)**.5:.4f} k {k:.2f}  S={s}:{a:.2f} D={d}:{b:.2f} R={r}:{c:.2f} | "
              + " ".join(f"{v} {kk*pred[v]:.3f}" for v in VERS))
