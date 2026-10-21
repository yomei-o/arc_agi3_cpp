"""cmp_scores.py <a.json> <b.json>: agreement of two ice_score.py outputs on their common (mol, smiles, spectrum)."""
import json, sys
import numpy as np
A, B = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
d, n, same_top = [], 0, 0
for m in A:
    if m not in B:
        continue
    best = []
    for s, rows in A[m].items():
        rb = {r[3]: r for r in B[m].get(s, [])}
        for r in rows:
            if r[3] in rb:
                d.append(abs(r[0] - rb[r[3]][0]))
    a = {s: np.mean([r[0] for r in v]) for s, v in A[m].items() if v}
    b = {s: np.mean([r[0] for r in v]) for s, v in B[m].items() if v and s in a}
    if a and b:
        n += 1
        same_top += max(a, key=a.get) == max(b, key=b.get)
d = np.array(d)
print(f"{len(d)} (mol, cand, spectrum) pairs: |d entropy sim| mean {d.mean():.5f} max {d.max():.5f}; "
      f"same top candidate in {same_top}/{n} molecules")
