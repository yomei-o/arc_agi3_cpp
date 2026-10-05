r"""Where do the missing 58% go?

The search finds the answer inside its top 25 for 167 of 400 molecules. Before
tuning the similarity it is worth knowing which of three things happened to the
other 233, because the three have nothing to do with each other:

  1. the answer is not in the library at all      - nothing to tune
  2. it is in the library but outside the mass window or adduct filter
                                                  - widen the gate
  3. it is a candidate but ranks below 25          - fix the scoring

Tuning the scoring when the problem is the gate is the kind of mistake that
costs days, so this counts all three first.
"""
from __future__ import annotations

import struct, sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\prog\casmi")
HEAD = np.dtype([("precursor", "<f4"), ("mol", "<i4"), ("off", "<i4"),
                 ("n", "<i4"), ("tag", "<i4")])


def load_heads(stem: str):
    h = np.fromfile(ROOT / f"{stem}_head.bin", dtype=HEAD)
    names, smiles = [], []
    for line in (ROOT / f"{stem}_mols.txt").read_text(encoding="utf-8").splitlines():
        a, _, b = line.partition("\t")
        names.append(a)
        smiles.append(b)
    return h, np.array(names), smiles


lib, lib_name, lib_smi = load_heads("lib")
qry, qry_name, qry_smi = load_heads("qry")
print(f"library {len(lib)} spectra / {len(lib_name)} molecules")
print(f"queries {len(qry)} spectra / {len(qry_name)} molecules")

lib_key_of_mol = lib_name
in_library = {k: i for i, k in enumerate(lib_name)}

# 1. is the answer in the library at all?
present = np.array([q in in_library for q in qry_name])
print(f"\nanswer present in library:      {present.sum():3d} / {len(qry_name)}")

# 2. does it survive the mass window and the adduct filter?
order = np.argsort(lib["precursor"])
mz = lib["precursor"][order]
mol = lib["mol"][order]
tag = lib["tag"][order]

for ppm in (5, 10, 20, 50):
    reachable = 0
    cand_counts = []
    for q in qry:
        truth = qry_name[q["mol"]]
        if truth not in in_library:
            continue
        want = in_library[truth]
        w = q["precursor"] * ppm * 1e-6
        i, j = np.searchsorted(mz, [q["precursor"] - w, q["precursor"] + w])
        sl = slice(i, j)
        same_adduct = (tag[sl] < 0) | (q["tag"] < 0) | (tag[sl] // 16 == q["tag"] // 16)
        hits = mol[sl][same_adduct]
        cand_counts.append(len(np.unique(hits)))
        if want in hits:
            reachable += 1
    cc = np.array(cand_counts) if cand_counts else np.array([0])
    print(f"  {ppm:3d} ppm + adduct: reachable {reachable:3d}"
          f"   candidates median {np.median(cc):5.0f}  mean {cc.mean():6.0f}")

# and without the adduct filter, to see what it costs
for ppm in (10, 20):
    reachable = 0
    for q in qry:
        truth = qry_name[q["mol"]]
        if truth not in in_library:
            continue
        w = q["precursor"] * ppm * 1e-6
        i, j = np.searchsorted(mz, [q["precursor"] - w, q["precursor"] + w])
        if in_library[truth] in mol[i:j]:
            reachable += 1
    print(f"  {ppm:3d} ppm, any adduct: reachable {reachable:3d}")
