"""v37_check: phys_v37pc C++ check against pool37ft.py (v37_STATUS.md).
usage: python v37_check.py <mols.txt> <ftdump.bin> <mollogits _mol.bin> <csv>...
 - per-query mean logits of the ranker (ftdump, id = query index, names in <ftdump>.txt) vs pt_train mollogits
   (id = molecule index in mols.txt): max / mean |diff|, and max |diff| of sigmoid p
 - MRR@25 (4 decimals, as mrr.py: names) of each csv
"""
import csv
import math
import struct
import sys


def load(path):
    raw = open(path, "rb").read()
    n, nb = struct.unpack_from("<ii", raw, 0)
    off, out = 8, {}
    for _ in range(n):
        (i,) = struct.unpack_from("<i", raw, off)
        off += 4
        out[i] = struct.unpack_from(f"<{nb}f", raw, off)
        off += 4 * nb
    return out, nb


mols = [l.split("\t")[0] for l in open(sys.argv[1], encoding="utf-8").read().splitlines() if l.strip()]
A, nb = load(sys.argv[2])
qn = {int(l.split("\t")[0]): l.split("\t")[1] for l in open(sys.argv[2] + ".txt", encoding="utf-8").read().splitlines()}
B, nb2 = load(sys.argv[3])
Bn = {mols[i]: v for i, v in B.items()}
mx = mp = s = 0.0
cnt = 0
missing = 0
sig = lambda x: 1 / (1 + math.exp(-max(-30, min(30, x))))  # noqa: E731
for i, v in A.items():
    w = Bn.get(qn[i])
    if w is None:
        missing += 1
        continue
    for a, b in zip(v, w):
        d = abs(a - b)
        mx = max(mx, d); s += d; cnt += 1
        mp = max(mp, abs(sig(a) - sig(b)))
print(f"ftdump {len(A)} queries x {nb} bits vs mollogits {len(B)} x {nb2}: missing {missing}, "
      f"max|dlogit| {mx:.3g}, mean {s / max(1, cnt):.3g}, max|dp| {mp:.3g}")
for p in sys.argv[4:]:
    c = {}
    with open(p, encoding="utf-8") as f:
        r = csv.reader(f); next(r)
        for row in r:
            c[row[0]] = row[1].split(";") if len(row) > 1 else []
    t = 0.0
    for k in mols:
        l = c.get(k, [])[:25]
        if k in l:
            t += 1 / (l.index(k) + 1)
    print(f"{p}: MRR@25 {t / len(mols):.4f}")
