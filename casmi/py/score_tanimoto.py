r"""Score a holdout submission the way the competition scores it.

The competition is de novo identification: the 400 test molecules were made by
Enveda and their spectra have never been published, so an exact-match metric
would read zero for everyone. The official tutorial checkpoints on
`val_tanimoto`, builds its fingerprints with
`rdFingerprintGenerator.GetMorganGenerator(radius=2).GetSparseFingerprint`,
and deduplicates candidates on the first InChIKey block because "the metric
ignores stereochemistry". So the score is Tanimoto similarity between the
answer and the guesses, and the only open question is how the 25 guesses
combine.

This prints every plausible combination, because the right strategy differs:

  max over 25   -> ranking is worth nothing and diversity is worth everything
  mean over 25  -> a bad 25th guess costs as much as a good one is worth
  rank weighted -> ranking matters, and the list should be ordered by belief

    python score_tanimoto.py <submission.csv> <truth_mols.txt>
"""
from __future__ import annotations

import csv, sys
from pathlib import Path

import numpy as np
from rdkit import DataStructs, RDLogger
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

sub_path = Path(sys.argv[1])
truth_path = Path(sys.argv[2])
K = int(sys.argv[3]) if len(sys.argv) > 3 else 25

fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
_cache: dict[str, object] = {}


def fp(smiles: str):
    if smiles in _cache:
        return _cache[smiles]
    m = MolFromSmiles(smiles)
    f = fp_gen.GetSparseFingerprint(m) if m is not None else None
    _cache[smiles] = f
    return f


truth: dict[str, str] = {}
for line in truth_path.read_text(encoding="utf-8").splitlines():
    a, _, b = line.partition("\t")
    if b:
        truth[a] = b

rows = list(csv.reader(sub_path.open(encoding="utf-8")))[1:]
best, first, mean, rr = [], [], [], []
n_cand, n_bad, n_dup = [], 0, []
for mol_id, smis in rows:
    t = truth.get(mol_id)
    if t is None:
        continue
    tf = fp(t)
    if tf is None:
        continue
    cands = [s for s in smis.split(";") if s][:K]
    sims, seen = [], set()
    for s in cands:
        cf = fp(s)
        if cf is None:
            n_bad += 1
            continue
        seen.add(s)
        sims.append(DataStructs.TanimotoSimilarity(tf, cf))
    if not sims:
        sims = [0.0]
    n_cand.append(len(cands))
    n_dup.append(len(cands) - len(seen))
    best.append(max(sims))
    first.append(sims[0])
    mean.append(float(np.mean(sims)))
    # rank-weighted: 1/rank normalised, the usual shape when order counts
    w = np.array([1.0 / (i + 1) for i in range(len(sims))])
    rr.append(float(np.dot(w, sims) / w.sum()))

n = len(best)
b = np.array(best)
print(f"scored {n} molecules, {n_bad} unparseable guesses, "
      f"{np.mean(n_cand):.1f} guesses each, {np.mean(n_dup):.1f} exact dups")
print(f"  max  over {K:2d}   {b.mean():.4f}")
print(f"  rank 1        {np.mean(first):.4f}")
print(f"  mean over {K:2d}   {np.mean(mean):.4f}")
print(f"  rank weighted {np.mean(rr):.4f}")
print(f"  exact (T=1)   {100.0 * (b >= 0.999).mean():.1f}%"
      f"   T>=0.7 {100.0 * (b >= 0.7).mean():.1f}%"
      f"   T>=0.5 {100.0 * (b >= 0.5).mean():.1f}%")

# How much of the max comes from how deep in the list? If almost all of it is
# in the first few, the tail is dead weight and should be spent on diversity.
for k in (1, 3, 5, 10, 25):
    if k > K:
        break
    vals = []
    for mol_id, smis in rows:
        t = truth.get(mol_id)
        if t is None or fp(t) is None:
            continue
        s = [DataStructs.TanimotoSimilarity(fp(t), fp(x))
             for x in smis.split(";")[:k] if x and fp(x) is not None]
        vals.append(max(s) if s else 0.0)
    print(f"  max over top-{k:<2d} {np.mean(vals):.4f}")
