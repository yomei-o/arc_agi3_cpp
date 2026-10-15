"""Do the xTB scan barriers (ts_scan.py) agree with the spectra?

1. class means vs the barrier order phys_rank fitted to spectra
   (glycosidic < acyl < C-het < C-C ~ aryl-C)
2. within a bond class: are bonds with lower barriers the ones whose fragment
   is observed? (labels from huckel_fqry2.json: an even-electron cation of
   either side within 0.01 Da in the molecule's [M+H]+ spectra)

    python ts_analyze.py <ts_scan.csv> <huckel_fqry2.json>
"""
import csv, json, sys
from collections import defaultdict
import numpy as np

rows = list(csv.DictReader(open(sys.argv[1])))
lab = {}
for m in json.load(open(sys.argv[2])):
    bonds = m["bonds"]
    for c in m["cuts"]:
        b = bonds[c["b"]]
        lab[(m["name"], min(b["i"], b["j"]), max(b["i"], b["j"]))] = int(max(c["oA"], c["oB"]) > 0)

by = defaultdict(list)
for r in rows:
    i, j = int(r["i"]), int(r["j"])
    y = lab.get((r["name"], min(i, j), max(i, j)))
    by[r["cls"]].append((float(r["barrier"]), float(r["reaction"]), int(r["plateau"]), y))

print(f"{len(rows)} scans\n")
print(f"{'class':11s} {'n':>4s} {'barrier mean':>13s} {'median':>7s} {'reaction':>9s} {'plateau':>8s} {'observed':>9s}")
order = sorted(by, key=lambda k: np.median([x[0] for x in by[k]]))
for k in order:
    v = by[k]
    b = np.array([x[0] for x in v]); rc = np.array([x[1] for x in v]); pl = np.mean([x[2] for x in v])
    ys = [x[3] for x in v if x[3] is not None]
    print(f"{k:11s} {len(v):4d} {b.mean():13.1f} {np.median(b):7.1f} {rc.mean():9.1f} {100*pl:7.0f}% {100*np.mean(ys) if ys else float('nan'):8.0f}%")


def auc(pos, neg):
    if not pos or not neg:
        return None
    return np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])


print("\nlower barrier -> observed?  AUC of (-barrier) for observed vs not (0.5 = no information)")
allp, alln = [], []
for k in order:
    pos = [-x[0] for x in by[k] if x[3] == 1]
    neg = [-x[0] for x in by[k] if x[3] == 0]
    a = auc(pos, neg)
    allp += pos; alln += neg
    print(f"  within {k:11s} obs {len(pos):3d} / not {len(neg):3d}  AUC {a if a is None else round(a, 3)}")
print(f"  pooled (class differences included)  AUC {auc(allp, alln):.3f}")
