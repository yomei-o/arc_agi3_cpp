"""Aryl-C: which barrier tracks the observed fragments, direct heterolysis or the ipso route?

    python ts_ipso_vs_obs.py <ts_ipso.csv> <huckel_fqry2.json>
AUC of (-barrier) for "a fragment of this cut is observed" vs not (0.5 = no information).
"""
import csv, json, sys
import numpy as np

lab = {}
for m in json.load(open(sys.argv[2])):
    for c in m["cuts"]:
        b = m["bonds"][c["b"]]
        lab[(m["name"], min(b["i"], b["j"]), max(b["i"], b["j"]))] = int(max(c["oA"], c["oB"]) > 0)
rows = []
for r in csv.DictReader(open(sys.argv[1])):
    i, j = int(r["i"]), int(r["j"])
    y = lab.get((r["name"], min(i, j), max(i, j)))
    if y is not None:
        rows.append((float(r["direct"]), float(r["ipso_total"]), float(r["pt_cost"]), float(r["ipso_scan"]), y))


def auc(score, y):
    pos = [s for s, t in zip(score, y) if t]; neg = [s for s, t in zip(score, y) if not t]
    return np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]), len(pos), len(neg)


def boot(score, y, n=2000, seed=0):
    rng = np.random.default_rng(seed); s = np.array(score); t = np.array(y); out = []
    for _ in range(n):
        k = rng.integers(0, len(s), len(s))
        if t[k].min() == t[k].max():
            continue
        out.append(auc(s[k], t[k])[0])
    return np.percentile(out, [2.5, 97.5])


y = [r[4] for r in rows]
print(f"labelled aryl-C bonds {len(rows)} (observed {sum(y)})")
for name, k in (("direct heterolysis", 0), ("ipso route total", 1), ("ipso: proton move only", 2), ("ipso: cleavage only", 3)):
    s = [-r[k] for r in rows]
    a, p, n = auc(s, y)
    lo, hi = boot(s, y)
    print(f"  {name:24s} AUC {a:.3f}  95% CI [{lo:.3f}, {hi:.3f}]   median barrier obs {np.median([r[k] for r in rows if r[4]]):6.1f} / not {np.median([r[k] for r in rows if not r[4]]):6.1f}")
