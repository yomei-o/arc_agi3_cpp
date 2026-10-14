# FPNet pool scores aligned to comb_<q>.txt candidate order -> fpnet_<q>.txt ; prints FPNet-alone MRR@25
# usage: score.py <casmi_dir> <q> [<q> ...]
import sys, numpy as np, time
D = sys.argv[1]
NB = 10226; NBY = 1279
keys = [l.split("\t", 1)[0] for l in open(f"{D}/pool730_mols.txt", encoding="utf-8")]
k2r = {}
for i, k in enumerate(keys): k2r.setdefault(k, i)
FP = np.memmap(f"{D}/pool730_fp10226.bin", dtype=np.uint8, mode="r").reshape(-1, NBY)
OK = np.fromfile(f"{D}/pool730_fp10226.bin.ok", dtype=np.uint8)
assert FP.shape[0] == len(keys) == len(OK)
for q in sys.argv[2:]:
    t0 = time.time()
    h = np.fromfile(f"{D}/fpnet/gg_{q}_mol.bin", dtype="<i4", count=2)
    M = np.fromfile(f"{D}/fpnet/gg_{q}_mol.bin", dtype="<f4", offset=8).reshape(h[0], h[1] + 1)
    Z = {int(m): M[i, 1:] for i, m in enumerate(M[:, 0].view("<i4"))}
    qk = {l.split("\t", 1)[0]: i for i, l in enumerate(open(f"{D}/{q}_mols.txt", encoding="utf-8"))}
    lines = open(f"{D}/comb_{q}.txt").read().split("\n")
    out = []; rr = []; nq = 0; miss = 0; i = 0; hit = 0
    while i < len(lines):
        L = lines[i]
        if not L.startswith("Q "): i += 1; continue
        _, qn, tr, n = L.split(); tr = int(tr); n = int(n)
        names = [lines[i + 1 + j].split()[6] for j in range(n)]; i += 1 + n
        z = Z.get(qk.get(qn, -1))
        rows = np.array([k2r.get(c, -1) for c in names]); miss += int((rows < 0).sum())
        sc = np.full(n, -1e9, np.float64)
        if z is not None and n:
            good = rows >= 0
            F = np.unpackbits(np.asarray(FP[rows[good]]), axis=1)[:, :NB].astype(np.float32)
            sc[good] = F @ z.astype(np.float32)
            sc[good & (OK[np.maximum(rows, 0)] == 0)] = -1e9
        out.append(L); out += [f"{c} {s:.4f}" for c, s in zip(names, sc)]
        nq += 1
        if tr >= 0:
            rk = 1 + int((sc > sc[tr]).sum()); hit += rk == 1
            rr.append(1.0 / rk if rk <= 25 else 0.0)
        else: rr.append(0.0)
    open(f"{D}/fpnet_{q}.txt", "w").write("\n".join(out) + "\n")
    rr = np.array(rr); inc = np.array([1])
    print(f"{q}: {nq} queries, MRR@25 {rr.mean():.4f}, top1 {hit/nq:.3f}, truth-in-list {np.mean([1 if x>0 else 0 for x in rr]):.3f} (rr>0), missing cand names {miss}, {time.time()-t0:.0f}s", flush=True)
