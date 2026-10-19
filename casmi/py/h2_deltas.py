"""Per-holdout version deltas with paired standard errors (per-molecule reciprocal rank, same scoring as mrr.py).

    python h2_deltas.py <dir with h2o csvs and *_mols.txt>
Prints for every holdout the MRR of each version and the deltas of the LB steps, with the
paired SE at n=400 and the SE the same per-molecule spread would give on a ~130-molecule public LB.
"""
import csv, sys
from pathlib import Path
import numpy as np

D = Path(sys.argv[1])
STEPS = [("v20", "v21", "learned fusion", -0.041), ("v20", "v23", "REV2 (LB: v22->v23)", +0.014),
         ("v23", "v24", "generation", +0.009), ("v24", "v25", "PubChem", -0.001),
         ("v25", "v26", "LambdaMART (LB also drops FPNet)", -0.035)]


def rr(path, truth):
    c = {}
    with open(path, encoding="utf-8") as f:
        r = csv.reader(f); next(r)
        for row in r:
            c[row[0]] = row[1].split(";")[:25] if len(row) > 1 else []
    return np.array([1.0 / (c.get(k, []).index(k) + 1) if k in c.get(k, []) else 0.0 for k in truth])


for q, suf in [("haqry", ""), ("hcqry", ""), ("hbqry", ""), ("hasqry", ""), ("hbsqry", ""),
               ("haqry", "_rm"), ("hcqry", "_rm"), ("hbqry", "_rm"),
               ("hasqry", "_prod"), ("hcqry", "_prod"), ("haqry", "_rm_prod")]:
    truth = [l.split("\t")[0] for l in open(D / f"{q}_mols.txt", encoding="utf-8") if l.strip()]
    R = {}
    for v in ("v20", "v21", "v23", "v24", "v25", "v26"):
        p = D / (f"{q}_{v}_rm.csv" if suf == "_rm" else f"{q}_{v}_rm_prod.csv" if suf == "_rm_prod"
                 else f"{q}_{v}{suf}.csv")
        if p.exists():
            R[v] = rr(p, truth)
        elif suf.startswith("_rm") and v in ("v20", "v21", "v23"):
            R[v] = np.zeros(len(truth))
    print(f"== {q}{suf}: " + "  ".join(f"{v} {R[v].mean():.3f}" for v in R))
    for a, b, lab, lb in STEPS:
        if a in R and b in R:
            d = R[b] - R[a]
            se = d.std(ddof=1) / np.sqrt(len(d))
            print(f"   {a}->{b} {lab:34s} {d.mean():+.3f}  (SE {se:.3f}; at n=130 {d.std(ddof=1)/np.sqrt(130):.3f})  LB {lb:+.3f}")
