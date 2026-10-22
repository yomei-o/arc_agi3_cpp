"""pool38: pool-ranking value of model variants (FPNet epochs, CNN seeds / fp3c16k CNN, FT epochs / seeds) on the timsTOF
holdouts, offline on the phys_v36gs every-candidate dumps (models38_STATUS.md). Generalises py/pool37ft.py.

Base dump <D>/<tag>_cf.txt (name idx score isPool pop candname cnn att fp3; score = v35 main fusion incl. attpool 2, cnn =
cnn_a_c Tanimoto) + <tag>_det.txt + <qry>_mols.txt. Extra per-candidate columns:
  name=cf:<path>               the cnn column of another phys_v36gs dump of the same run (other cnnmodel=), rows matched
                               by (query, candidate name)
  name=tan:<mol.bin>#<bits>    continuous Tanimoto sum_bits p / (n + sum p - sum_bits p), p = sigmoid(mol logit)
  name=dot:<mol.bin>#<bits>    logit dot product sum_bits logit (the fpnet_pool score form)
     <bits> = "<bits.bin>::<mols.txt>[;<bits.bin>::<mols.txt>...]" sparse fingerprint files (u32 n, u32 nbits, per row
     u16 count + u16 bits), names from the mols file (pool candidates by name, generated ones "<query>:G<k>").
  Paths may contain {q} (query set, e.g. hxqry) and {s} (short set name from the sets argument, e.g. x / b / bs).
  name=avg:<c1>+<c2>[+...]     mean of already defined columns (raw values)
  name=gmean:<col>             <col> with the generated candidates (isPool 0) set to the query pool mean (production FPNet fill)
For tan/dot columns a shuffled-query version "<name>~" (query q scored with query q+7's prediction) is made too.
A candidate without bits gets the query's mean (should not happen: every candidate has bits; counts are printed).

Configs: "cnn:2,ft:2" = score + 2 z(cnn) + 2 z(ft) (z over the query's candidates), "@perm=ft" permutes ft within the query.
Then the pool36 tail (top 60 -> pool popularity re-fill -> v35 Boltzmann merge), final-list MRR@25, paired bootstrap (2000)
vs v35 and vs the reference config (first config named in ref=, default "cnn:2").

usage: python pool38.py <D> <OUT> <tag:qry:s[,...]> <configs separated by ';'> col=... [col=...] [ref=<config>]
"""
import math
import os
import random
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pool36 as p36  # noqa: E402

_BITS = {}


def load_bits(spec):
    if spec in _BITS:
        return _BITS[spec]
    B = {}
    for part in spec.split(";"):
        bp, mp = part.split("::")
        raw = open(bp, "rb").read()
        n, nb = struct.unpack_from("<II", raw, 0)
        names = [l.split("\t")[0] for l in open(mp, encoding="utf-8").read().splitlines()]
        assert len(names) == n, (bp, len(names), n)
        a = np.frombuffer(raw, dtype="<u2", offset=8)
        off = 0
        for i in range(n):
            c = int(a[off])
            B[names[i]] = a[off + 1:off + 1 + c]
            off += 1 + c
    _BITS[spec] = B
    return B


def load_mol(path):
    raw = open(path, "rb").read()
    nm, nb = struct.unpack_from("<ii", raw, 0)
    off = 8
    P = {}
    for _ in range(nm):
        (mol,) = struct.unpack_from("<i", raw, off)
        off += 4
        P[mol] = np.frombuffer(raw, dtype="<f4", count=nb, offset=off).astype(np.float64)
        off += 4 * nb
    return P


def model_col(cf, order, qnames, kind, molp, bitspec, shift):
    Pm = load_mol(molp)
    Z = {qnames[m]: v for m, v in Pm.items() if 0 <= m < len(qnames)}
    B = load_bits(bitspec)
    idx = {q: i for i, q in enumerate(order)}
    out, miss = {}, 0
    for q, rows in cf.items():
        src = order[(idx[q] + shift) % len(order)] if shift else q
        z = Z.get(src)
        if z is None:
            out[q] = [0.0] * len(rows)
            continue
        if kind == "tan":
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            sp = p.sum()
        vals = []
        for r in rows:
            b = B.get(r[4])
            if b is None:
                b = B.get(f"{q}:{r[4]}")
            if b is None:
                vals.append(None)
                miss += 1
                continue
            if kind == "tan":
                s = p[b].sum()
                vals.append(float(s / (len(b) + sp - s + 1e-9)))
            else:
                vals.append(float(z[b].sum()))
        have = [v for v in vals if v is not None]
        m0 = sum(have) / max(1, len(have))
        out[q] = [m0 if v is None else v for v in vals]
    return out, miss


def main():
    D, OUT, sets_arg, cfg_arg = sys.argv[1:5]
    cols, ref = [], "cnn:2"
    for a in sys.argv[5:]:
        if a.startswith("col="):
            cols.append(a[4:])
        elif a.startswith("ref="):
            ref = a[4:]
    os.makedirs(OUT, exist_ok=True)
    configs = [c.strip() for c in cfg_arg.split(";") if c.strip()]
    for st in sets_arg.split(","):
        tag, qry, s = st.split(":")
        cf = p36.load_cf(f"{D}/{tag}_cf.txt")
        qnames = [l.split("\t")[0] for l in open(f"{D}/{qry}_mols.txt", encoding="utf-8").read().splitlines()]
        order = [q for q in qnames if q in cf]
        C = {"cnn": {q: [r[5] for r in rows] for q, rows in cf.items()},
             "att": {q: [r[6] for r in rows] for q, rows in cf.items()},
             "fp3": {q: [r[7] for r in rows] for q, rows in cf.items()}}
        info = []
        for c in cols:
            name, spec = c.split("=", 1)
            kind, rest = spec.split(":", 1)
            rest = rest.replace("{q}", qry).replace("{s}", s)
            if kind == "cf":
                o = p36.load_cf(rest)
                col, miss = {}, 0
                for q, rows in cf.items():
                    m = {r[4]: r[5] for r in o.get(q, [])}
                    v = [m.get(r[4]) for r in rows]
                    miss += sum(x is None for x in v)
                    mu = sum(x for x in v if x is not None) / max(1, sum(x is not None for x in v))
                    col[q] = [mu if x is None else x for x in v]
                C[name] = col
                info.append(f"{name}: cf missing {miss}")
            elif kind in ("tan", "dot"):
                molp, bitspec = rest.split("#", 1)
                if not os.path.exists(molp):
                    info.append(f"{name}: MISSING {molp}")
                    continue
                C[name], miss = model_col(cf, order, qnames, kind, molp, bitspec, 0)
                C[name + "~"], _ = model_col(cf, order, qnames, kind, molp, bitspec, 7)
                info.append(f"{name}: {kind} missing bits {miss}")
            elif kind == "gmean":  # name=gmean:<col> : generated candidates (isPool 0) get the query's pool mean (= production phys for FPNet)
                if rest not in C:
                    info.append(f"{name}: MISSING column {rest}")
                    continue
                src = C[rest]
                col = {}
                for q, rows in cf.items():
                    pv = [v for r, v in zip(rows, src[q]) if r[2] == 1]
                    mu = sum(pv) / max(1, len(pv))
                    col[q] = [v if r[2] == 1 else mu for r, v in zip(rows, src[q])]
                C[name] = col
            elif kind == "avg":
                parts = rest.split("+")
                C[name] = {q: [sum(C[p][q][i] for p in parts) / len(parts) for i in range(len(cf[q]))] for q in cf}
        nall = sum(len(r) for r in cf.values())
        Q0, P0, F0 = p36.lists(D, qry, tag)
        base_f = p36.rrs(Q0, F0)
        print(f"== {tag} ({qry}, {len(Q0)} queries, {nall} candidates) " + "; ".join(info))
        print(f"   {'config':40s} {'pool MRR':>9s} {'final MRR':>9s} {'d final vs v35 [95% CI]':>28s} "
              f"{'d final vs ' + ref + ' [95% CI]':>34s}")
        ref_f = None
        res = {}
        for cfg in [ref] + configs:
            if cfg in res:
                continue
            need = [t.split(":")[0].strip() for t in cfg.split(",") if ":" in t and not t.strip().startswith("@")]
            if any(k not in C for k in need):
                print(f"   {cfg:40s} skipped (column missing)")
                continue
            terms, perm = {}, set()
            for t in cfg.split(","):
                t = t.strip()
                if not t or t == "v35":
                    continue
                if t.startswith("@perm="):
                    perm.add(t[6:])
                    continue
                k, w = t.split(":")
                terms[k] = float(w)
            rnd = random.Random(7)
            ptag = f"{tag}__m38"
            with open(f"{OUT}/{ptag}_pp.txt", "w", encoding="utf-8") as f:
                for q, rows in cf.items():
                    sc = [r[1] for r in rows]
                    for k, w in terms.items():
                        if w == 0:
                            continue
                        v = list(C[k][q])
                        if k in perm:
                            rnd.shuffle(v)
                        z = p36.zs(v)
                        sc = [a + w * b for a, b in zip(sc, z)]
                    o = sorted(range(len(rows)), key=lambda i: (-sc[i], rows[i][0]))
                    for rk, i in enumerate(o[:60]):
                        x = rows[i]
                        f.write(f"{q}\t{rk}\t{sc[i]:.6g}\t{x[2]}\t{x[3]:.6g}\t{x[4]}\n")
            for src, dst in ((f"{D}/{tag}_det.txt", f"{OUT}/{ptag}_det.txt"), (f"{D}/{qry}_mols.txt", f"{OUT}/{qry}_mols.txt")):
                if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
                    open(dst, "wb").write(open(src, "rb").read())
            Q, Pl, F = p36.lists(OUT, qry, ptag)
            rp, rf = p36.rrs(Q, Pl), p36.rrs(Q, F)
            res[cfg] = rf
            if ref_f is None:
                ref_f = rf
            if cfg == ref and cfg not in configs:
                continue
            b1 = p36.boot(base_f, rf)
            b2 = p36.boot(ref_f, rf)
            print(f"   {cfg:40s} {sum(rp)/len(rp):9.4f} {sum(rf)/len(rf):9.4f} {b1[0]:+.4f} [{b1[1]:+.4f},{b1[2]:+.4f}]"
                  f"       {b2[0]:+.4f} [{b2[1]:+.4f},{b2[2]:+.4f}]", flush=True)


if __name__ == "__main__":
    main()
