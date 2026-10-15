# PubChem-slot merge policy grid (phys_pc.cpp's pcdet dumps + plain CSVs).
#
#   python pcslot_grid.py <dir> <tag:label> ...   (e.g. c:cqry c_rm:cqry_rm ...)
#
# For each holdout run <tag>, reads <dir>/<tag>.csv (the plain final list,
# eval names) and <dir>/<tag>_det.txt (per query: libMax, top1/top2 of the
# fused main list, analog top sim; every fully scored PubChem candidate with
# truth flag, rev, pop, phys, mlp, ana, fused, zp, zm, zr, smiles).
# Simulates "insert the top-k PubChem candidates at fixed slots when the gate
# is open" and prints MRR@25 per run, plus the mixtures.
#
# Truth for a PubChem candidate: the C++ stereo-blind canonical-SMILES match;
# with --ik the picked candidates are also compared by InChIKey14 (RDKit).
import csv, itertools, sys
from collections import defaultdict


def load(d, tag):
    plain = {}
    with open(f"{d}/{tag}.csv", encoding="utf-8") as f:
        r = csv.reader(f); next(r)
        for row in r:
            plain[row[0]] = row[1].split(";") if len(row) > 1 and row[1] else []
    Q = {}
    cur = None
    for line in open(f"{d}/{tag}_det.txt", encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if p[0] == "Q":
            cur = dict(name=p[1], lib=float(p[2]), top1=float(p[3]), top2=float(p[4]), ana=float(p[5]),
                       ncand=int(p[6]), win=int(p[8]), pc=[], topS=[float(x) for x in p[12:]])
            Q[p[1]] = cur
        else:
            cur["pc"].append(dict(cid=p[0], truth=int(p[1]), rev=float(p[2]), pop=float(p[3]), phys=float(p[4]),
                                  mlp=float(p[5]), ana=float(p[6]), fused=float(p[7]), zp=float(p[8]),
                                  zm=float(p[9]), zr=float(p[10]), smi=p[11]))
    return plain, Q


def mrr(plain, Q, pcs, gate, slots, k):
    """pcs(q) -> ordered list of PubChem candidates; gate(q) -> bool."""
    tot = 0.0
    for name, L in plain.items():
        q = Q.get(name)
        pick = []
        if q is not None and gate(q):
            pick = ["__T__" if c["truth"] else "__P__" for c in pcs(q)[:k]]
        out, a, b = [], 0, 0
        sl = set(slots[:k])
        for pos in range(1, 26):
            if pos in sl and b < len(pick): out.append(pick[b]); b += 1
            elif a < len(L): out.append(L[a]); a += 1
            elif b < len(pick): out.append(pick[b]); b += 1
        for i, x in enumerate(out):
            if x == name or x == "__T__":
                tot += 1.0 / (i + 1); break
    return tot / max(1, len(plain))


def mrr_score(plain, Q, pcs, gate, k, sw, bias, pmin):
    """score-mode merge: a pick enters where zp+0.25zm+zr+24ana+sw*pop+bias beats the next main score (pos >= pmin)."""
    tot = 0.0
    for name, L in plain.items():
        q = Q.get(name)
        pick, ps = [], []
        if q is not None and gate(q):
            for c in pcs(q)[:k]:
                pick.append("__T__" if c["truth"] else "__P__")
                ps.append(c["zp"] + 0.25 * c["zm"] + c["zr"] + 24 * c["ana"] + sw * c["pop"] + bias)
        S = q["topS"] if q is not None else []
        out, a, b = [], 0, 0
        for pos in range(1, 26):
            if b < len(pick) and pos >= pmin and (a >= len(S) or ps[b] > S[a]): out.append(pick[b]); b += 1
            elif a < len(L): out.append(L[a]); a += 1
            else: break
        for i, x in enumerate(out):
            if x == name or x == "__T__":
                tot += 1.0 / (i + 1); break
    return tot / max(1, len(plain))


def ranker(w_m=0.25, w_r=1.0, w_a=24.0, w_pop=0.5, w_p=1.0):
    def f(q):
        key = lambda c: -(w_p * c["zp"] + w_m * c["zm"] + w_r * c["zr"] + w_a * c["ana"] + w_pop * c["pop"])
        if "_s" not in q: q["_s"] = {}
        k = (w_m, w_r, w_a, w_pop, w_p)
        if k not in q["_s"]: q["_s"][k] = sorted(q["pc"], key=key)
        return q["_s"][k]
    return f


def main():
    d = sys.argv[1]
    runs = []
    for a in sys.argv[2:]:
        if a.startswith("--"): continue
        tag, label = a.split(":")
        runs.append((label, *load(d, tag)))
    labels = [r[0] for r in runs]
    # PubChem-only ranking quality: MRR of the truth within the PubChem list, per ranker
    print("== PubChem list alone (truth rank within the fully scored PubChem candidates) ==")
    rankers = {}
    for wp in (0, 0.5, 1, 2, 5, 10, 20, 50):
        rankers[f"fused+{wp}pop"] = ranker(w_pop=wp)
    for wp in (0, 1, 5, 20):
        rankers[f"rev+{wp}pop"] = ranker(w_m=0, w_r=1, w_a=0, w_pop=wp, w_p=0)
    rankers["pop+0.01rev"] = ranker(w_m=0, w_r=0.01, w_a=0, w_pop=1, w_p=0)
    for wa in (0, 8, 48):
        rankers[f"fused(wa{wa})+10pop"] = ranker(w_a=wa, w_pop=10)
    print("%-22s" % "ranker" + "".join("%12s" % l for l in labels))
    for rn, rk in rankers.items():
        row = []
        for lab, plain, Q in runs:
            t, top = 0.0, 0
            for q in Q.values():
                for i, c in enumerate(rk(q)[:25]):
                    if c["truth"]:
                        t += 1 / (i + 1); top += i < 5; break
            row.append("%6.3f/%3d" % (t / max(1, len(plain)), top))
        print("%-22s" % rn + "".join("%12s" % x for x in row))
    print("\n== merge grid: MRR@25 per run (base = no insertion) ==")
    base = [mrr(p, Q, lambda q: [], lambda q: False, [], 0) for _, p, Q in runs]
    print("%-48s" % "setting" + "".join("%9s" % l for l in labels))
    print("%-48s" % "base" + "".join("%9.4f" % x for x in base))
    SLOTS = {"2,4,6,8,10": [2, 4, 6, 8, 10], "4,8,12,16,20": [4, 8, 12, 16, 20], "11-15": [11, 12, 13, 14, 15],
             "21-25": [21, 22, 23, 24, 25], "16,18,20,22,24": [16, 18, 20, 22, 24], "6,10,14,18,22": [6, 10, 14, 18, 22]}
    res = []
    RANKERS = [a[4:] for a in sys.argv if a.startswith('--r=')] or ["fused+0.5pop", "fused+5pop", "fused+20pop", "rev+20pop"]
    for rn in RANKERS:
        rk = rankers[rn]
        for sn, sl in SLOTS.items():
            for k in (1, 2, 3, 5):
                for g in (0.3, 0.5, 0.7, 0.8, 0.9, 2.0):
                    gate = (lambda g: lambda q: q["lib"] < g)(g)
                    v = [mrr(p, Q, rk, gate, sl, k) for _, p, Q in runs]
                    name = f"{rn} slots[{sn}] k{k} lib<{g}"
                    res.append((name, v))
                    print("%-48s" % name + "".join("%9.4f" % x for x in v), flush=True)
                for mg in (0.5, 1.0, 2.0):
                    gate = (lambda mg: lambda q: q["lib"] < 0.9 and q["top1"] - q["top2"] < mg)(mg)
                    v = [mrr(p, Q, rk, gate, sl, k) for _, p, Q in runs]
                    name = f"{rn} slots[{sn}] k{k} lib<0.9&marg<{mg}"
                    res.append((name, v))
                    print("%-48s" % name + "".join("%9.4f" % x for x in v), flush=True)
    if "--score" in sys.argv:
        for rn in RANKERS:
            rk = rankers[rn]
            for k in (1, 2, 3, 5):
                for g in (0.9, 2.0):
                    gate = (lambda g: lambda q: q["lib"] < g)(g)
                    for sw in (0, 1, 2, 5):
                        for bias in (-8, -6, -4, -3, -2, -1, 0, 1, 2):
                            for pmin in (1, 2, 4):
                                v = [mrr_score(p, Q, rk, gate, k, sw, bias, pmin) for _, p, Q in runs]
                                name = f"SCORE {rn} k{k} lib<{g} sw{sw} bias{bias} min{pmin}"
                                res.append((name, v))
                                print("%-48s" % name + "".join("%9.4f" % x for x in v), flush=True)
    import pickle
    pickle.dump((labels, base, res), open(f"{d}/grid.pkl", "wb"))


if __name__ == "__main__":
    main()
