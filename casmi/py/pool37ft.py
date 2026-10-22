"""pool37ft: pool-ranking value of the pretrained-transformer fingerprint model (pretrain_STATUS.md section 8) on top of
v35 / pool36, offline on the phys_v36pc every-candidate dumps (cfdump: name idx score isPool pop candname cnn att fp3).

New column ft = continuous Tanimoto sum_bits p / (n + sum p - sum_bits p) of the candidate's fp3c16k bits with the
fine-tuned model's per-molecule p = sigmoid(mean logit) (pt_train mollogits -> <x>_mol.bin), the same form as the CNN
column. Pool candidates get bits from fp3_prep apply on their pool730 SMILES (<bits.bin> + <mols.txt>, same order).
Generated candidates (G<k>, isPool 0) have no SMILES in the dump -> imputed per query:
  impute=reg  : linear fit of ft on the CNN column over the query's pool candidates (default)
  impute=mean : the query's mean pool ft  (z ~ 0: the term neither helps nor hurts them)
  impute=min  : the query's minimum pool ft (pessimistic for generated candidates)
Then score' = score + w z(cnn) + u z(ft) (z over the query's candidates) and pool36's tail (top 60 -> pool popularity
re-fill -> v35 Boltzmann merge), MRR@25, paired bootstrap vs v35 and vs v35 + 2 z(cnn) (the pool36 choice).
Controls: ft from query q+7 (SHUF-query), ft permuted within the query (PERM).

usage: python pool37ft.py <dumpdir> <outdir> <bits.bin> <mols.txt> tag:qry:molfile[,...] [impute=reg|mean|min]
       [extra=<gbits.bin>::<gmols.txt>]   (bits of the generated candidates from phys_v36gs gsdump, names <query>:G<k>;
                                           with it nothing is imputed)
"""
import math
import os
import random
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pool36 as p36  # noqa: E402


def load_bits(path, mols):
    raw = open(path, "rb").read()
    n, nb = struct.unpack_from("<II", raw, 0)
    off = 8
    names = [l.split("\t")[0] for l in open(mols, encoding="utf-8").read().splitlines()]
    assert len(names) == n, (len(names), n)
    B = {}
    for i in range(n):
        (c,) = struct.unpack_from("<H", raw, off)
        off += 2
        B[names[i]] = struct.unpack_from(f"<{c}H", raw, off)
        off += 2 * c
    return B, nb


def load_mol(path):
    raw = open(path, "rb").read()
    nm, nb = struct.unpack_from("<ii", raw, 0)
    off = 8
    P = {}
    for _ in range(nm):
        (mol,) = struct.unpack_from("<i", raw, off)
        off += 4
        z = struct.unpack_from(f"<{nb}f", raw, off)
        off += 4 * nb
        P[mol] = [1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x)))) for x in z]
    return P


def tani(p, sp, bits):
    s = 0.0
    for b in bits:
        s += p[b]
    return s / (len(bits) + sp - s + 1e-9)


def add_ft(cf, qnames, P, B, impute, shift=0):
    order = [q for q in qnames if q in cf]
    idx = {q: i for i, q in enumerate(order)}
    out = {}
    for q, rows in cf.items():
        src = order[(idx[q] + shift) % len(order)] if shift else q
        p = P.get(src)
        if p is None:
            out[q] = [r + (0.0,) for r in rows]
            continue
        sp = sum(p)
        ft = []
        for r in rows:
            k = r[4] if r[4] in B else f"{q}:{r[4]}"  # generated candidates: "<query>:G<k>" (from gsdump SMILES)
            ft.append(tani(p, sp, B[k]) if k in B else None)
        have = [(r[5], f) for r, f in zip(rows, ft) if f is not None]
        if impute == "reg" and len(have) > 2:
            mx = sum(a for a, _ in have) / len(have)
            my = sum(b for _, b in have) / len(have)
            sxx = sum((a - mx) ** 2 for a, _ in have) + 1e-12
            k = sum((a - mx) * (b - my) for a, b in have) / sxx
            fill = lambda r: my + k * (r[5] - mx)  # noqa: E731
        elif impute == "min" and have:
            m0 = min(b for _, b in have)
            fill = lambda r: m0  # noqa: E731
        else:
            m0 = sum(b for _, b in have) / max(1, len(have))
            fill = lambda r: m0  # noqa: E731
        out[q] = [r + (f if f is not None else fill(r),) for r, f in zip(rows, ft)]
    return out


def main():
    D, OUT, bitsp, molsp = sys.argv[1:5]
    sets = [x.split(":") for x in sys.argv[5].split(",")]
    impute = "reg"
    for a in sys.argv[6:]:
        if a.startswith("impute="):
            impute = a.split("=", 1)[1]
    os.makedirs(OUT, exist_ok=True)
    B, nb = load_bits(bitsp, molsp)
    for a in sys.argv[6:]:  # extra=<bits.bin>:<mols.txt> (generated candidates' bits, names "<query>:G<k>")
        if a.startswith("extra="):
            eb, em = a.split("=", 1)[1].split("::")
            B.update(load_bits(eb, em)[0])
    p36.COL["ft"] = 8
    grid = [("v35 (w=0)", {})]
    for w in (0.5, 1, 2, 3):
        grid.append((f"ft {w}", {"ft": w}))
    grid.append(("cnn 2 (pool36)", {"cnn": 2}))
    for w in (0.5, 1, 2, 3):
        grid.append((f"cnn 2 + ft {w}", {"cnn": 2, "ft": w}))
    for c, w in ((1, 1), (1, 2), (3, 1)):
        grid.append((f"cnn {c} + ft {w}", {"cnn": c, "ft": w}))
    for w in (1, 2):
        grid.append((f"cnn 2 + ft {w} SHUF-query", {"cnn": 2, "ft": w, "_shuf": 1}))
        grid.append((f"cnn 2 + ft {w} PERM", {"cnn": 2, "ft": w, "_perm": 1}))
    for tag, mols, molfile in sets:
        cf = p36.load_cf(f"{D}/{tag}_cf.txt")
        qnames = [l.split("\t")[0] for l in open(f"{D}/{mols}_mols.txt", encoding="utf-8").read().splitlines()]
        Pm = load_mol(molfile)
        P = {qnames[m]: v for m, v in Pm.items() if 0 <= m < len(qnames)}
        cft = add_ft(cf, qnames, P, B, impute)
        csh = add_ft(cf, qnames, P, B, impute, shift=7)
        npool = sum(1 for q, rows in cf.items() for r in rows if r[4] in B or f"{q}:{r[4]}" in B)
        nall = sum(len(rows) for rows in cf.values())
        Q0, P0, F0 = p36.lists(D, mols, tag)
        base_p, base_f = p36.rrs(Q0, P0), p36.rrs(Q0, F0)
        ref_f = None
        print(f"== {tag} ({mols}, {len(Q0)} queries, impute={impute}): candidates with bits {npool}/{nall}, "
              f"queries with ft p {len(P)}")
        print(f"   {'config':30s} {'pool MRR':>9s} {'final MRR':>9s} {'d final vs v35 [95% CI]':>28s} "
              f"{'d final vs cnn2 [95% CI]':>28s}")
        for name, terms in grid:
            t = {k: v for k, v in terms.items() if not k.startswith("_")}
            ptag = f"{tag}__ft"
            src = csh if terms.get("_shuf") else cft
            if terms.get("_perm"):
                rnd = random.Random(7)
                src2 = {}
                for q, rows in src.items():
                    v = [r[8] for r in rows]
                    rnd.shuffle(v)
                    src2[q] = [r[:8] + (x,) for r, x in zip(rows, v)]
                src = src2
            p36.write_pp(src, t, f"{OUT}/{ptag}_pp.txt")
            dst = f"{OUT}/{ptag}_det.txt"
            if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(f"{D}/{tag}_det.txt"):
                open(dst, "wb").write(open(f"{D}/{tag}_det.txt", "rb").read())
            mp = f"{OUT}/{mols}_mols.txt"
            if not os.path.exists(mp) or os.path.getsize(mp) != os.path.getsize(f"{D}/{mols}_mols.txt"):
                open(mp, "wb").write(open(f"{D}/{mols}_mols.txt", "rb").read())
            Q, Pl, F = p36.lists(OUT, mols, ptag)
            rp, rf = p36.rrs(Q, Pl), p36.rrs(Q, F)
            if name.startswith("cnn 2 (pool36)"):
                ref_f = rf
            b1 = p36.boot(base_f, rf)
            b2 = p36.boot(ref_f, rf) if ref_f is not None else (0, 0, 0)
            print(f"   {name:30s} {sum(rp)/len(rp):9.4f} {sum(rf)/len(rf):9.4f} {b1[0]:+.4f} [{b1[1]:+.4f},{b1[2]:+.4f}]"
                  f"     {b2[0]:+.4f} [{b2[1]:+.4f},{b2[2]:+.4f}]", flush=True)


if __name__ == "__main__":
    main()
