"""Duplicate rate of a submission-format CSV under the competition's identity.

Identity (as the competition states it: stereo and tautomers ignored, InChIKey first block):
  RemoveStereochemistry -> RDKit TautomerEnumerator canonical tautomer -> InChIKey[:14]
Also reports the weaker identities to see where the duplicates come from:
  ik14   = InChIKey[:14] of the molecule as written (stereo already outside the first block)
  metric = ik14 after stereo strip + canonical tautomer

usage: pp_dupcheck.py <submission.csv> [more.csv ...]
CSV cells must be SMILES (a Kaggle submission, or a ranker run with `smiles`).
"""
import csv
import sys
from functools import lru_cache

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
_te = rdMolStandardize.TautomerEnumerator()


@lru_cache(maxsize=None)
def keys(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return (None, None)
    try:
        ik = Chem.MolToInchiKey(m)[:14]
    except Exception:
        ik = None
    try:
        m2 = Chem.Mol(m)
        Chem.RemoveStereochemistry(m2)
        mk = Chem.MolToInchiKey(_te.Canonicalize(m2))[:14]
    except Exception:
        mk = None
    return (ik or None, mk or None)


def main():
    for p in sys.argv[1:]:
        rows = []
        with open(p, encoding="utf-8") as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                rows.append((row[0], [s for s in row[1].split(";") if s] if len(row) > 1 else []))
        n_slots = n_bad = 0
        wasted_ik = wasted_m = 0
        rows_dup_ik = rows_dup_m = 0
        first_pos = []  # position (1-based) of the first wasted slot
        for mid, lst in rows:
            lst = lst[:25]
            n_slots += len(lst)
            seen_ik, seen_m = set(), set()
            w_ik = w_m = 0
            for i, s in enumerate(lst):
                ik, mk = keys(s)
                if mk is None:
                    n_bad += 1
                    continue
                if ik in seen_ik:
                    w_ik += 1
                seen_ik.add(ik)
                if mk in seen_m:
                    w_m += 1
                    if w_m == 1:
                        first_pos.append(i + 1)
                seen_m.add(mk)
            wasted_ik += w_ik
            wasted_m += w_m
            rows_dup_ik += w_ik > 0
            rows_dup_m += w_m > 0
        n = max(1, len(rows))
        fp = sorted(first_pos)
        print(f"{p}: rows {len(rows)} slots {n_slots} unparsable {n_bad}")
        print(f"  ik14 duplicates:   rows with any {rows_dup_ik} ({rows_dup_ik / n:.1%}), wasted slots {wasted_ik} ({wasted_ik / max(1, n_slots):.2%})")
        print(f"  metric duplicates: rows with any {rows_dup_m} ({rows_dup_m / n:.1%}), wasted slots {wasted_m} ({wasted_m / max(1, n_slots):.2%})")
        if fp:
            print(f"  first wasted slot position: median {fp[len(fp) // 2]}, <=5: {sum(x <= 5 for x in fp)}, <=10: {sum(x <= 10 for x in fp)}")


if __name__ == "__main__":
    main()
