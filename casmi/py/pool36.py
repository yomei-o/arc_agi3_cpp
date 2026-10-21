"""pool36: pool-ranking value of extra terms on top of v35 (pool36_STATUS.md).

Offline re-scoring of phys_v36pc every-candidate dumps (cfdump=<tag>_cf.txt: name idx score isPool pop candname cnn
att fp3, score = the v35 main fusion incl. attpool=2). For a term set {cnn: w, att: a, fp3: f, phys...}:
  score' = score + w z(cnn) + (a - 2) z(att) + f z(fp3)        (z over the query's candidates, as the C++ zscore)
then the v35 tail is reproduced exactly with py/boltz_merge.py: top max(60,25) -> pool popularity re-fill (ppop 0.15,
gate 0.9) -> Boltzmann merge with the PubChem list (<tag>_det.txt) at the v35 setting T1 2 T2 0.5 pi 0.2 relz a 0.5.
Two lists per query: 'pool' = the main list top 25 after the re-fill (no PubChem), 'final' = after the merge.
The w=0 case is checked against the C++ CSVs (<tag>.csv plain = pool list, <tag>_ins.csv = final).

usage: python pool36.py <dir with dumps + <qry>_mols.txt> <outdir> tag:qry[,tag:qry...] [shuffle_tag_map]
"""
import math
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import boltz_merge as bm  # noqa: E402

TIMS = None
V35 = dict(T1=2, T2=0.5, pi=0.2, pimode="relz", a=0.5)
COL = {"cnn": 5, "att": 6, "fp3": 7}  # load_cf tuple: idx score isPool pop name cnn att fp3


def zs(v):
    n = len(v)
    m = sum(v) / max(1, n)
    s = math.sqrt(sum((x - m) ** 2 for x in v) / max(1, n)) + 1e-6
    return [(x - m) / s for x in v]


def load_cf(path):
    Q = defaultdict(list)
    for l in open(path, encoding="utf-8"):
        p = l.rstrip("\n").split("\t")
        if len(p) < 9:
            continue
        Q[p[0]].append((int(p[1]), float(p[2]), int(p[3]), float(p[4]), p[5], float(p[6]), float(p[7]), float(p[8])))
    return Q


def write_pp(cf, terms, path, perm_seed=None):
    rnd = random.Random(perm_seed) if perm_seed is not None else None
    with open(path, "w", encoding="utf-8") as f:
        for q, rows in cf.items():
            sc = [r[1] for r in rows]
            for k, w in terms.items():
                if w == 0:
                    continue
                v = [r[COL[k]] for r in rows]
                if rnd is not None and k == "cnn":
                    rnd.shuffle(v)
                z = zs(v)
                sc = [a + w * b for a, b in zip(sc, z)]
            o = sorted(range(len(rows)), key=lambda i: (-sc[i], rows[i][0]))
            for r, i in enumerate(o[:60]):
                x = rows[i]
                f.write(f"{q}\t{r}\t{sc[i]:.6g}\t{x[2]}\t{x[3]:.6g}\t{x[4]}\n")


def lists(d, mols, tag):
    Q = bm.load(d, mols, tag)
    return Q, [q["main"][:25] for q in Q], [bm.boltz_list(q, **V35) for q in Q]


def rrs(Q, L):
    return [bm.rr(q, l) for q, l in zip(Q, L)]


def csv_lists(path):
    c = {}
    for l in open(path, encoding="utf-8").read().splitlines()[1:]:
        k, _, v = l.partition(",")
        c[k] = v.split(";") if v else []
    return c


def boot(base, new, B=2000, seed=1):
    rnd = random.Random(seed)
    d = [a - b for a, b in zip(new, base)]
    n = len(d)
    ms = sorted(sum(d[rnd.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return sum(d) / n, ms[int(0.025 * B)], ms[int(0.975 * B) - 1]


def main():
    D, OUT = sys.argv[1], sys.argv[2]
    os.makedirs(OUT, exist_ok=True)
    sets = [x.split(":") for x in sys.argv[3].split(",")]
    grid = [("v35 (w=0)", {})]
    for w in (0.5, 1, 2, 3, 4):
        grid.append((f"cnn {w}", {"cnn": w}))
    for w in (1, 2):
        grid.append((f"cnn {w} PERM-control", {"cnn": w, "_perm": 1}))
    for a in (0, 1, 3, 4):
        grid.append((f"attpool {a}", {"att": a - 2}))
    for a, w in ((1, 1), (3, 1), (1, 2), (3, 2)):
        grid.append((f"attpool {a} + cnn {w}", {"att": a - 2, "cnn": w}))
    for f in (0.5, 1, 2):
        grid.append((f"fp3 {f}", {"fp3": f}))
    grid.append(("cnn 1 + fp3 1", {"cnn": 1, "fp3": 1}))
    for tag, mols in sets:
        cf = load_cf(f"{D}/{tag}_cf.txt")
        global TIMS
        TIMS = None
        hb = f"{D}/{mols}_head.bin"
        if os.environ.get("POOL36_TIMS") == "1" and os.path.exists(hb):
            import struct
            raw = open(hb, "rb").read()
            TIMS = set()
            for k in range(len(raw) // 20):
                _, mol, _, _, tg = struct.unpack_from("<fiiii", raw, 20 * k)
                if mol >= 0 and tg >= 0 and tg % 16 == 0:
                    TIMS.add(mol)
            print(f"   (subset column = queries with a timsTOF spectrum: {len(TIMS)})")
        g = list(grid)
        shp = f"{D}/{tag}_shuf_cf.txt"  # same run with cnnshuf=7: query q scored with query q+7's CNN p
        if os.path.exists(shp):
            sh = load_cf(shp)
            ok = all(len(sh.get(q, [])) == len(r) and all(a[0] == b[0] and a[4] == b[4] for a, b in zip(r, sh[q]))
                     for q, r in cf.items())
            if ok:
                cf2 = {q: [a[:5] + (b[5],) + a[6:] for a, b in zip(r, sh[q])] for q, r in cf.items()}
                for w in (1, 2, 3):
                    g.append((f"cnn {w} SHUF-query control", {"cnn": w, "_cf": cf2}))
            else:
                print("shuffle dump does not align, skipped")
        # exactness check of the w=0 reconstruction against the C++ dump and CSVs
        Q0, P0, F0 = lists(D, mols, tag)
        cpool, cins = csv_lists(f"{D}/{tag}.csv"), csv_lists(f"{D}/{tag}_ins.csv")
        dp = sum(1 for q, l in zip(Q0, P0) if cpool.get(q["name"], [])[:25] != l)
        di = sum(1 for q, l in zip(Q0, F0) if cins.get(q["name"], [])[:25] != l)
        base_p, base_f = rrs(Q0, P0), rrs(Q0, F0)
        print(f"== {tag} ({mols}, {len(Q0)} queries): C++ dump sim vs CSV: pool-list diffs {dp}, final diffs {di}")
        print(f"   {'config':28s} {'pool MRR':>9s} {'d pool [95% CI]':>28s} {'final MRR':>9s} {'d final [95% CI]':>28s}")
        for name, terms in g:
            t = {k: v for k, v in terms.items() if not k.startswith("_")}
            ptag = f"{tag}__g"
            if not t:
                # re-sorted from the every-candidate dump (checks the reconstruction itself)
                write_pp(cf, {}, f"{OUT}/{ptag}_pp.txt")
            else:
                write_pp(terms.get("_cf", cf), t, f"{OUT}/{ptag}_pp.txt", perm_seed=7 if terms.get("_perm") else None)
            for ext in ("_det.txt",):
                src, dst = f"{D}/{tag}{ext}", f"{OUT}/{ptag}{ext}"
                if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
                    open(dst, "wb").write(open(src, "rb").read())
            mp = f"{OUT}/{mols}_mols.txt"
            if not os.path.exists(mp):
                open(mp, "wb").write(open(f"{D}/{mols}_mols.txt", "rb").read())
            Q, P, F = lists(OUT, mols, ptag)
            rp, rf = rrs(Q, P), rrs(Q, F)
            bp, bf = boot(base_p, rp), boot(base_f, rf)
            # hard subset: no near-identical library spectrum (libMax < 0.9) = where the pool fusion decides
            hi = [i for i, q in enumerate(Q0) if q["lib"] < 0.9]
            if TIMS is not None:  # optional: queries with a timsTOF spectrum (tag % 16 == 0) instead
                hi = [i for i in range(len(Q0)) if i in TIMS]
            bh = boot([base_f[i] for i in hi], [rf[i] for i in hi])
            print(f"   {name:28s} {sum(rp)/len(rp):9.4f} {bp[0]:+.4f} [{bp[1]:+.4f},{bp[2]:+.4f}]      "
                  f"{sum(rf)/len(rf):9.4f} {bf[0]:+.4f} [{bf[1]:+.4f},{bf[2]:+.4f}]   lib<0.9 n={len(hi)} "
                  f"{sum(rf[i] for i in hi)/max(1,len(hi)):.4f} {bh[0]:+.4f} [{bh[1]:+.4f},{bh[2]:+.4f}]", flush=True)


if __name__ == "__main__":
    main()
