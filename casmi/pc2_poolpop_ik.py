"""pool730 -> PubChem popularity by InChIKey first block (the metric's identity).

pool730_mols.txt names ARE InChIKey14s, so the lookup is a join, no structure matching:
  CID-InChI-Key.gz (CID, InChI, InChIKey)  ->  CIDs per pool IK14
  pop(CID) = log1p(#SID) + log1p(#PMID), as pop_build.cpp: read from the pc table (pc_cid.bin / pc_pop.bin)
  when the CID is there, else counted from CID-SID.tsv / CID-PMID.tsv
  pool pop = max over the IK14's CIDs; 0 when the IK14 is not in PubChem.
Writes <out>_pop.bin (float32 per pool line), <out>_cid.bin (int32 CID of the max, -1) and prints the found rate.

usage: python pc2_poolpop_ik.py <casmi dir> <pubchem dir> <out prefix>
"""
import gzip
import math
import sys

import numpy as np

D, P, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
names = [l.split("\t", 1)[0].strip() for l in open(f"{D}/pool730_mols.txt", encoding="utf-8")]
idx = {}
for i, n in enumerate(names):
    idx.setdefault(n, []).append(i)
print("pool", len(names), "distinct IK14", len(idx), flush=True)

cids = {}  # IK14 -> list of CIDs
nl = 0
with gzip.open(f"{P}/CID-InChI-Key.gz", "rt", encoding="utf-8", errors="replace") as f:
    for l in f:
        nl += 1
        t = l.rfind("\t")
        k = l[t + 1:t + 15]
        if k in idx:
            cids.setdefault(k, []).append(int(l[:l.find("\t")]))
        if nl % 20000000 == 0:
            print("  inchikey rows", nl, "matched IK14", len(cids), flush=True)
print("CID-InChI-Key rows", nl, "pool IK14 found", len(cids), flush=True)

allc = np.array(sorted({c for v in cids.values() for c in v}), np.int64)
pc_cid = np.fromfile(f"{D}/pc_cid.bin", np.int32).astype(np.int64)
pc_pop = np.fromfile(f"{D}/pc_pop.bin", np.float32)
o = np.argsort(pc_cid)
sc = pc_cid[o]
pos = np.searchsorted(sc, allc)
pos[pos >= len(sc)] = 0
intab = sc[pos] == allc
popc = {int(c): float(pc_pop[o[p]]) for c, p, ok in zip(allc, pos, intab) if ok}
miss = set(int(c) for c, ok in zip(allc, intab) if not ok)
print("matched CIDs", len(allc), "in pc table", int(intab.sum()), "counted from SID/PMID", len(miss), flush=True)
if miss:
    cnt = {}
    for fn in ("CID-SID.tsv", "CID-PMID.tsv"):
        c = {}
        with open(f"{P}/{fn}", "rb") as f:
            for l in f:
                t = l.find(b"\t")
                v = int(l[:t] if t >= 0 else l)
                if v in miss:
                    c[v] = c.get(v, 0) + 1
        cnt[fn] = c
        print("  counted", fn, len(c), flush=True)
    for v in miss:
        popc[v] = math.log1p(cnt["CID-SID.tsv"].get(v, 0)) + math.log1p(cnt["CID-PMID.tsv"].get(v, 0))

pop = np.zeros(len(names), np.float32)
best = np.full(len(names), -1, np.int32)
for k, v in cids.items():
    c = max(v, key=lambda x: popc.get(x, 0.0))
    for i in idx[k]:
        pop[i] = popc.get(c, 0.0)
        best[i] = c
pop.tofile(f"{OUT}_pop.bin")
best.tofile(f"{OUT}_cid.bin")
found = best >= 0
print(f"pool_pop_ik: {len(names)} molecules, found {found.sum()} ({found.mean():.4f}); mean pop all {pop.mean():.3f}, "
      f"found {pop[found].mean():.3f}; pop>=5 {np.mean(pop >= 5):.4f}")
h = np.fromfile(f"{D}/pool730_head.bin", np.int32).reshape(-1, 3)
tr = np.zeros(len(names), bool)
ok = (h[:, 1] >= 0) & (h[:, 1] < len(names))
tr[h[ok, 1]] = h[ok, 2] != 0
for nm, m in (("train", tr), ("non-train (COCONUT only)", ~tr)):
    print(f"  {nm}: {m.sum()} molecules, found {found[m].mean():.4f}, mean pop {pop[m].mean():.3f} "
          f"(found only {pop[m & found].mean():.3f})")
