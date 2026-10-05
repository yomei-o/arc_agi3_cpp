r"""Choose which 25 of the candidates to actually submit.

The metric is the best Tanimoto among the 25 guesses, so the order of the list
is worth nothing and only the set counts. That changes what a good list looks
like: 25 candidates taken straight off the top of a cosine ranking are mostly
isomers of each other - same mass, same window, often the same skeleton with a
methyl moved - so they stand or fall together. Twenty-five guesses spread over
the chemical space the spectrum allows cover more ground.

Maximal marginal relevance, the usual way to trade the two off:

    pick argmax_i  (1-L) * relevance_i  -  L * max_sim(i, already picked)

L = 0 keeps the cosine ranking exactly. L = 1 ignores the spectrum and spreads
out as far as possible. Relevance comes from the rank, since the submission
file does not carry the raw cosine.

    python select_mmr.py <pool.csv> <truth.txt> <out.csv> [lambda] [K]
"""
from __future__ import annotations

import csv, sys
from pathlib import Path

import numpy as np
from rdkit import DataStructs, RDLogger
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

pool_path, truth_path, out_path = map(Path, sys.argv[1:4])
LAM = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
K = int(sys.argv[5]) if len(sys.argv) > 5 else 25

fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2)

rows = list(csv.reader(pool_path.open(encoding="utf-8")))
header, rows = rows[0], rows[1:]

out_rows = []
for mol_id, smis in rows:
    cands = [s for s in smis.split(";") if s]
    fps, keep = [], []
    for s in cands:
        m = MolFromSmiles(s)
        if m is None:
            continue
        fps.append(fp_gen.GetSparseFingerprint(m))
        keep.append(s)
    n = len(keep)
    if n <= K:
        out_rows.append((mol_id, ";".join(keep) if keep else "CCO"))
        continue

    rel = np.linspace(1.0, 0.0, n)          # rank 1 is the most relevant
    chosen = [0]                            # the cosine's own top pick, always
    # running max similarity of every candidate to the chosen set
    msim = np.array(DataStructs.BulkTanimotoSimilarity(fps[0], fps))
    taken = np.zeros(n, dtype=bool)
    taken[0] = True
    while len(chosen) < K:
        val = (1.0 - LAM) * rel - LAM * msim
        val[taken] = -1e9
        j = int(np.argmax(val))
        chosen.append(j)
        taken[j] = True
        msim = np.maximum(msim, DataStructs.BulkTanimotoSimilarity(fps[j], fps))
    out_rows.append((mol_id, ";".join(keep[j] for j in chosen)))

with out_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(out_rows)
print(f"lambda {LAM}  wrote {len(out_rows)} rows to {out_path}")
