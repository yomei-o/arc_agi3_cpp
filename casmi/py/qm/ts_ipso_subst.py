"""Law test: donors ortho/para to the ipso carbon should make the proton move cheaper.

In the arenium ion (proton on the ipso carbon) the positive charge sits ortho
and para to the ipso carbon. Lone-pair donors (O, N attached to the ring) at
those positions stabilise it; at meta they should not (textbook electrophilic
aromatic substitution). Prediction: proton-move cost decreases with the number
of ortho/para donors, and not with meta donors.

    python ts_ipso_subst.py <ts_ipso.csv> <ts_scan.csv>
"""
import csv, sys
import numpy as np
from rdkit import Chem

smi = {r["name"]: r["smiles"] for r in csv.DictReader(open(sys.argv[2]))}
rows = []
for r in csv.DictReader(open(sys.argv[1])):
    m = Chem.MolFromSmiles(smi[r["name"]])
    i, j = int(r["i"]), int(r["j"])
    ipso, sub = (i, j) if m.GetAtomWithIdx(i).GetIsAromatic() else (j, i)
    ring = next((list(rg) for rg in m.GetRingInfo().AtomRings() if ipso in rg and len(rg) == 6), None)
    if ring is None:
        continue
    k0 = ring.index(ipso)
    pos = {ring[(k0 + d) % 6]: ("ortho" if d in (1, 5) else "meta" if d in (2, 4) else "para") for d in range(1, 6)}
    cnt = {"ortho": 0, "meta": 0, "para": 0}
    for a, p in pos.items():
        for nb in m.GetAtomWithIdx(a).GetNeighbors():
            if nb.GetIdx() in ring or nb.GetIdx() == sub:
                continue
            if nb.GetAtomicNum() in (7, 8) and not nb.GetIsAromatic() and \
               not any(b.GetBondTypeAsDouble() == 2 for b in nb.GetBonds()):
                cnt[p] += 1
    rows.append((cnt["ortho"] + cnt["para"], cnt["meta"], float(r["pt_cost"]), float(r["ipso_total"])))

op = np.array([x[0] for x in rows]); me = np.array([x[1] for x in rows]); pt = np.array([x[2] for x in rows])
tot = np.array([x[3] for x in rows])
print(f"{len(rows)} aryl-C bonds on 6-rings")
for k in sorted(set(op)):
    s = op == k
    print(f"  ortho/para donors {k}: n {s.sum():3d}  proton-move cost median {np.median(pt[s]):6.1f}  total {np.median(tot[s]):6.1f}")
for k in sorted(set(me)):
    s = me == k
    print(f"  meta donors       {k}: n {s.sum():3d}  proton-move cost median {np.median(pt[s]):6.1f}  total {np.median(tot[s]):6.1f}")
A = np.column_stack([np.ones(len(pt)), op, me])
c, *_ = np.linalg.lstsq(A, pt, rcond=None)
res = pt - A @ c
se = np.sqrt(np.diag(np.linalg.inv(A.T @ A)) * res.var(ddof=3))
print(f"linear fit: proton-move cost = {c[0]:.1f} {c[1]:+.1f}(+-{se[1]:.1f}) * n_ortho/para {c[2]:+.1f}(+-{se[2]:.1f}) * n_meta  [kcal/mol]")
