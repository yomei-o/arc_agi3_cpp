"""pc2_grid: offline grid over pool popularity (ppop) and the PubChem channel v2 (phys_v28pc pcv=2).

Re-simulates phys_v28pc's final list from its dumps, so one C++ run per (holdout, rm) covers the grid:
  <tag>_pp.txt   per query the top max(ppn,25) of the final order BEFORE the pool-popularity step
                 (name rank score isPool pop candname)
  <tag>_det.txt  pcdet: "Q name libMax ..." line, "V2 mean sd relz S pop_top r_topNP r_pool havePool" line,
                 then the listed PubChem candidates (cid truth rev pop phys mlp ana fused zp zm zr smiles)
Scoring = mrr.py on a CSV: 1/rank of the query's own name in the top 25, 0 otherwise.

usage: python pc2_grid.py <mols.txt> <dir> <tag> [<mols.txt> <dir> <tag> ...]   (prints a table per tag)
       python pc2_grid.py --pair <mols.txt> <dir> <tag_rm0> <tag_rm1> ...         (normal + answer-removed pairs, breakeven)
"""
import sys
from collections import defaultdict


ASIS = False  # --asis: keep the dump's PubChem order (pcrank=fused runs)
POP = None  # --pop <pool_mols.txt> <pop.bin>: pool popularity by candidate name (overrides the dump's pop column)


def load(mols, d, tag):
    truth = [l.split("\t")[0] for l in open(mols, encoding="utf-8") if l.strip()]
    pp = defaultdict(list)
    for l in open(f"{d}/{tag}_pp.txt", encoding="utf-8"):
        p = l.rstrip("\n").split("\t")
        if len(p) < 6:
            continue
        pop = float(p[4]) if POP is None or p[3] != "1" else POP.get(p[5], 0.0)
        pp[p[0]].append((int(p[1]), float(p[2]), int(p[3]), pop, p[5]))
    det = {}
    cur = None
    for l in open(f"{d}/{tag}_det.txt", encoding="utf-8"):
        p = l.rstrip("\n").split("\t")
        if p[0] == "Q":
            cur = {"lib": float(p[2]), "pc": [], "v2": None}
            det[p[1]] = cur
        elif p[0] == "V2":
            cur["v2"] = dict(mean=float(p[1]), sd=float(p[2]), relz=float(p[3]), S=float(p[4]), pop=float(p[5]))
        else:
            cur["pc"].append(dict(cid=p[0], truth=p[1] == "1", rev=float(p[2]), pop=float(p[3]), fused=float(p[7])))
    return truth, pp, det


def base_list(q, pp, ppw, ppn):
    L = sorted(pp.get(q, []))
    names = [x[4] for x in L]
    if ppw and len(L) > 1:
        idx = [i for i, x in enumerate(L) if i < ppn and x[2] == 1]
        if len(idx) > 1:
            s = [L[i][1] for i in idx]
            m = sum(s) / len(s)
            sd = (sum((v - m) ** 2 for v in s) / len(s)) ** 0.5 + 1e-6
            key = sorted(((L[i][1] - m) / sd + ppw * L[i][3], -k, i) for k, i in enumerate(idx))[::-1]
            for j, i in enumerate(idx):
                names[i] = L[key[j][2]][4]
    return names[:25]


def simulate(truth, pp, det, ppw=0.0, ppn=60, pc=False, pcw=0.15, gate=0.9, relhi=1e9, relp=1e9, sp=1e9, popp=5.0,
             slo=(4, 8, 12, 16, 20), shi=(2, 4, 6, 8, 10), stats=None, asis=False, ppgate=9.0):
    tot = 0.0
    for q in truth:
        D = det.get(q)
        L = base_list(q, pp, ppw if (D is None or D["lib"] < ppgate) else 0.0, ppn)
        if pc and D and D["v2"] and D["pc"] and D["lib"] < gate:
            v = D["v2"]
            c = list(D["pc"]) if (asis or ASIS) else sorted(D["pc"], key=lambda e: -((e["rev"] - v["mean"]) / v["sd"] + pcw * e["pop"]))
            hi = v["relz"] > relhi
            sl = set(shi if hi else slo)
            k = len(sl)
            picks = [q if e["truth"] else "P" + e["cid"] for e in c[:k]]
            S = (c[0]["rev"] - v["mean"]) / v["sd"] + pcw * c[0]["pop"]
            out, a, b = [], 0, 0
            for pos in range(1, 26):
                if pos in sl and b < len(picks):
                    out.append(picks[b]); b += 1
                elif a < len(L):
                    out.append(L[a]); a += 1
                elif b < len(picks):
                    out.append(picks[b]); b += 1
                else:
                    break
            prom = v["relz"] > relp or (S > sp and c[0]["pop"] >= popp)
            if prom and picks:
                out = [picks[0]] + [x for x in out if x != picks[0]]
                out = out[:25]
            if stats is not None:
                stats["open"] += 1; stats["hi"] += hi; stats["prom"] += prom
                stats["promT"] += prom and c[0]["truth"]
            L = out
        if q in L[:25]:
            tot += 1.0 / (L.index(q) + 1)
    return tot / len(truth)


CONFIGS = [
    ("plain (v27a)", dict()),
    ("ppop 0.10", dict(ppw=0.10)),
    ("ppop 0.15", dict(ppw=0.15)),
    ("ppop 0.25", dict(ppw=0.25)),
    ("ppop 0.50", dict(ppw=0.50)),
    ("ppop 0.15 gate lib<0.9", dict(ppw=0.15, ppgate=0.9)),
    ("ppop 0.15 gate lib<0.7", dict(ppw=0.15, ppgate=0.7)),
    ("pc slots lo only", dict(pc=True)),
    ("pc lo, pcw .25", dict(pc=True, pcw=0.25)),
    ("pc lo, pcw 0", dict(pc=True, pcw=0.0)),
]
for rh in (4, 6, 8):
    CONFIGS.append((f"pc relhi {rh}", dict(pc=True, relhi=rh)))
for rp in (1, 2, 3, 4, 6):
    CONFIGS.append((f"pc prom relz>{rp}", dict(pc=True, relp=rp)))
for s in (4, 5, 6, 8):
    CONFIGS.append((f"pc prom S>{s}&pop>=5", dict(pc=True, sp=s)))
for rp in (2, 3, 4):
    for s in (6,):
        CONFIGS.append((f"pc relhi 6 prom relz>{rp}|S>{s}&pop>=5", dict(pc=True, relhi=6, relp=rp, sp=s)))
        CONFIGS.append((f"  + ppop .15 (relz>{rp})", dict(pc=True, relhi=6, relp=rp, sp=s, ppw=0.15)))


CONFIGS.append(("CHOSEN relhi6 relz>3|S>6&pop>=5 +ppop.15 gate.9", dict(pc=True, relhi=6, relp=3, sp=6, ppw=0.15, ppgate=0.9)))


def main():
    global POP, ASIS
    a = sys.argv[1:]
    if a and a[0] == "--asis":
        ASIS = True
        a = a[1:]
    if a and a[0] == "--pop":
        import numpy as np
        nm = [l.split("\t", 1)[0].strip() for l in open(a[1], encoding="utf-8")]
        pv = np.fromfile(a[2], np.float32)
        POP = dict(zip(nm, pv.tolist()))
        a = a[3:]
    if a and a[0] == "--pair":
        a = a[1:]
        groups = [a[i:i + 4] for i in range(0, len(a), 4)]
        res = {}
        for mols, d, t0, t1 in groups:
            D0, D1 = load(mols, d, t0), load(mols, d, t1)
            for name, cfg in CONFIGS:
                st0 = defaultdict(int); st1 = defaultdict(int)
                r0 = simulate(*D0, stats=st0, **cfg); r1 = simulate(*D1, stats=st1, **cfg)
                res[(t0, name)] = (r0, r1, st0, st1)
        print("config".ljust(40) + "".join(f"{g[2]:>22}" for g in groups) + "   mean d0   mean d1  breakeven f")
        for name, _ in CONFIGS:
            row, d0s, d1s = name.ljust(40), [], []
            for g in groups:
                r0, r1, s0, s1 = res[(g[2], name)]
                b0, b1, _, _ = res[(g[2], CONFIGS[0][0])]
                d0s.append(r0 - b0); d1s.append(r1 - b1)
                row += f"  {r0:.3f}/{r1:.3f} p{s0['prom']:>3}"
            m0, m1 = sum(d0s) / len(d0s), sum(d1s) / len(d1s)
            f = (-m0) / (m1 - m0) if (m1 - m0) > 1e-9 and m0 < 0 else (0.0 if m0 >= 0 and m1 >= 0 else float("nan"))
            print(row + f"  {m0:+.4f}  {m1:+.4f}  {f:.3f}")
        return
    for i in range(0, len(a), 3):
        D = load(*a[i:i + 3])
        print(a[i + 2])
        for name, cfg in CONFIGS:
            st = defaultdict(int)
            r = simulate(*D, stats=st, **cfg)
            print(f"  {name:40s} {r:.4f}  open {st['open']} hi {st['hi']} prom {st['prom']} (truth {st['promT']})")


if __name__ == "__main__":
    main()
