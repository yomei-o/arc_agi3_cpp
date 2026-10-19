"""Rank-level fusion of the main (v28a) list with a second, differently built list, on holdout2.

rf_eval.py <rfdir> <casmi dir> [out.tsv]
  lists per component (rf_STATUS.md):
    main      rf/<tag>.csv            phys_v29rf v28a config, names (InChIKey14 / G<k>), 40 deep
    rev/phys/mlp/ana  rf/<tag>_<eng>.csv  same candidates ranked by one engine
    lib       rf/lib_<qry>.csv        match.cpp library search (SMILES -> InChIKey14 via <lib>_mols.txt), 40 deep
  identity = the InChIKey14 name (what mrr.py scores); the library list's SMILES are mapped through the
  library's mols file, so a molecule found by both lists gets one vote.
Rules (final 25 always filled by: fused order, then main, then the library list):
  base        main then lib (the current notebook merge)
  rrf k w     1/(k+r1) + w/(k+r2) over the union of both lists' top D (ranks 1-based), ties -> main rank
  e2r2        main with engine 2's top-1 put at rank 2 (unless already at rank 1/2)
  rrf+e2r2    rrf then engine 2's top-1 at rank 2
Scores: MRR@25 per component, paired bootstrap (2000) CI of the difference to base,
mixture S=0.5 hasqry (or hxqry) + D=0.42 hcqry + R=0.08 hcqry-rm.
"""
import csv, random, sys
import numpy as np
from pathlib import Path

RF = Path(sys.argv[1]); CAS = Path(sys.argv[2])
OUT = sys.argv[3] if len(sys.argv) > 3 else None
COMP = {  # component -> (tag, qry, lib)
    "has": ("has0", "hasqry", "ylib"), "hc": ("hc0", "hcqry", "ylib"), "hcrm": ("hc1", "hcqry", "ylib"),
    "hx": ("hx0", "hxqry", "zlib"), "ha": ("ha0", "haqry", "ylib"),
    "hb": ("hb0", "hbqry", "ylib"), "hbs": ("hbs0", "hbsqry", "ylib")}
ENG2 = ["lib", "rev", "ana", "phys", "mlp"]


def read_lists(p):
    d = {}
    with open(p, encoding="utf-8") as f:
        r = csv.reader(f); next(r)
        for row in r:
            if row:
                d[row[0]] = [s for s in (row[1].split(";") if len(row) > 1 else []) if s]
    return d


_smi2name = {}
def smi2name(lib):
    if lib not in _smi2name:
        m = {}
        with open(CAS / f"{lib}_mols.txt", encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 2 and p[1] and p[1] not in m:
                    m[p[1]] = p[0]
        _smi2name[lib] = m
    return _smi2name[lib]


def dedup(seq):
    out, seen = [], set()
    for s in seq:
        if s not in seen:
            seen.add(s); out.append(s)
    return out


def fill(head, main, lib, K=25):
    return dedup(head + main + lib)[:K]


def rrf(a, b, k, w, D=40):
    sc = {}
    for r, s in enumerate(a[:D]):
        sc[s] = sc.get(s, 0.0) + 1.0 / (k + r + 1)
    for r, s in enumerate(b[:D]):
        sc[s] = sc.get(s, 0.0) + w / (k + r + 1)
    ra = {s: r for r, s in enumerate(a)}
    return sorted(sc, key=lambda s: (-sc[s], ra.get(s, 1e9)))


def e2r2(a, b):
    if not b:
        return list(a)
    t = b[0]
    a = [s for s in a if s != t] if t not in a[:1] else list(a)
    if t in a[:2]:
        return a
    return a[:1] + [t] + a[1:]


def rr(lst, truth):
    l = lst[:25]
    return 1.0 / (l.index(truth) + 1) if truth in l else 0.0


def load_comp(c):
    tag, q, lib = COMP[c]
    if not (RF / f"{tag}.csv").exists() or not (RF / f"lib_{q}.csv").exists():
        return None
    truth = [l.split("\t")[0] for l in open(CAS / f"{q}_mols.txt", encoding="utf-8") if l.strip()]
    main = read_lists(RF / f"{tag}.csv")
    m = smi2name(lib)
    libl = {k: dedup([m.get(s, "S:" + s) for s in v]) for k, v in read_lists(RF / f"lib_{q}.csv").items()}
    eng = {"lib": libl}
    best = {}
    bp = RF / f"lib_{q}_best.csv"
    if bp.exists():
        with open(bp, encoding="utf-8") as f:
            r = csv.reader(f); next(r)
            for row in r:
                best[row[0]] = float(row[1])
    eng["_best"] = best
    for e in ENG2[1:]:
        p = RF / f"{tag}_{e}.csv"
        if p.exists():
            eng[e] = read_lists(p)
    return truth, main, libl, eng


def rules():
    R = [("base", None)]
    for e in ENG2:
        R.append((f"{e}:e2r2", (e, "e2r2")))
        for k in (1, 3, 10):
            for w in (0.3, 0.6, 1.0):
                R.append((f"{e}:rrf k{k} w{w}", (e, "rrf", k, w)))
        R.append((f"{e}:rrf k3 w0.6+e2r2", (e, "rrfe", 3, 0.6)))
        R.append((f"{e}:rrf k3 w0.3+e2r2", (e, "rrfe", 3, 0.3)))
    for th in (0.9, 1.0, 1.1, 1.2):
        for k, w in ((3, 0.6), (3, 1.0), (1, 1.0)):
            R.append((f"lib:rrf k{k} w{w} if best>={th}", ("lib", "rrfg", k, w, th)))
        R.append((f"lib:e2r2 if best>={th}", ("lib", "e2r2g", 0, 0, th)))
        R.append((f"lib:e2r1 if best>={th}", ("lib", "e2r1g", 0, 0, th)))
    R.append(("main25:base", ("depth25",)))
    # production depth: pool_candidates.csv and direct_submission.csv are 25 deep in v28a
    for k, w in ((1, 0.6), (1, 1.0), (3, 0.6), (3, 1.0)):
        R.append((f"D25 lib:rrf k{k} w{w}", ("lib", "rrf", k, w, None, 25)))
    R.append(("D25 lib:e2r2", ("lib", "e2r2", 0, 0, None, 25)))
    R.append(("D25 lib:rrf k3 w0.3+e2r2", ("lib", "rrfe", 3, 0.3, None, 25)))
    R.append(("D25 lib:rrf k3 w0.6+e2r2", ("lib", "rrfe", 3, 0.6, None, 25)))
    return R


def apply(rule, truth, main, libl, eng):
    out = []
    for t in truth:
        a = main.get(t, []); L = libl.get(t, [])
        D = rule[5] if rule is not None and len(rule) > 5 else 40
        a = a[:D]; L = L[:D]
        if rule is None:
            f = fill([], a, L)
        elif rule[0] == "depth25":
            f = fill([], a[:25], L)
        else:
            e = rule[0]
            if e not in eng:
                return None
            b = eng[e].get(t, [])[:D]
            g = eng["_best"].get(t, -1e9) >= rule[4] if rule[1] in ("rrfg", "e2r2g", "e2r1g") else True
            if not g:
                f = fill([], a, L)
            elif rule[1] == "e2r1g":
                f = fill(b[:1], a, L)
            elif rule[1] in ("e2r2", "e2r2g"):
                f = fill(e2r2(a, b), a, L)
            elif rule[1] in ("rrf", "rrfg"):
                f = fill(rrf(a, b, rule[2], rule[3]), a, L)
            else:
                f = fill(e2r2(rrf(a, b, rule[2], rule[3]), b), a, L)
        out.append(rr(f, t))
    return out


_IDX = {}
def bidx(N, n=2000):
    if N not in _IDX:
        _IDX[N] = np.random.default_rng(N).integers(0, N, size=(n, N))
    return _IDX[N]


def bmeans(diffs):
    d = np.asarray(diffs, dtype=float)
    return d[bidx(len(d))].mean(axis=1)


def boot(diffs):
    v = bmeans(diffs)
    return np.quantile(v, 0.025), np.quantile(v, 0.975)


def main():
    data = {c: load_comp(c) for c in COMP}
    data = {c: d for c, d in data.items() if d}
    res = {}
    for name, rule in rules():
        res[name] = {}
        for c, (truth, mn, libl, eng) in data.items():
            v = apply(rule, truth, mn, libl, eng)
            if v is not None:
                res[name][c] = v
    base = res["base"]
    lines = []
    hdr = "rule\t" + "\t".join(f"{c}\td{c}\tCI{c}" for c in data) + "\tmixS\tdmixS\tCImixS\tmixX\tdmixX"
    lines.append(hdr)
    rnd = random.Random(7)
    for name, per in res.items():
        row = [name]
        for c in data:
            if c not in per:
                row += ["", "", ""]; continue
            m = sum(per[c]) / len(per[c])
            d = [x - y for x, y in zip(per[c], base[c])]
            lo, hi = boot(d) if name != "base" else (0, 0)
            row += [f"{m:.4f}", f"{sum(d)/len(d):+.4f}", f"[{lo:+.3f},{hi:+.3f}]"]
        for S, tag in (("has", "S"), ("hx", "X")):
            if all(c in per for c in (S, "hc", "hcrm")):
                mix = 0.5 * sum(per[S]) / len(per[S]) + 0.42 * sum(per["hc"]) / len(per["hc"]) + 0.08 * sum(per["hcrm"]) / len(per["hcrm"])
                dd = {c: [x - y for x, y in zip(per[c], base[c])] for c in (S, "hc", "hcrm")}
                dm = 0.5 * sum(dd[S]) / len(dd[S]) + 0.42 * sum(dd["hc"]) / len(dd["hc"]) + 0.08 * sum(dd["hcrm"]) / len(dd["hcrm"])
                if tag == "S":
                    # independent resamples per component (hc and hc-rm share molecules: same index draw)
                    bs = 0.5 * bmeans(dd[S]) + 0.42 * bmeans(dd["hc"]) + 0.08 * bmeans(dd["hcrm"])
                    row += [f"{mix:.4f}", f"{dm:+.4f}", f"[{np.quantile(bs,0.025):+.4f},{np.quantile(bs,0.975):+.4f}]"]
                else:
                    row += [f"{mix:.4f}", f"{dm:+.4f}"]
            else:
                row += ["", "", ""] if tag == "S" else ["", ""]
        lines.append("\t".join(row))
    txt = "\n".join(lines)
    print(txt)
    if OUT:
        Path(OUT).write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
