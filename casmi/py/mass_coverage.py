r"""Does COCONUT's mass range even cover the test set's?

The denovo holdout's 0/400 InChIKey14 hit rate against COCONUT is not
necessarily a pool-expansion verdict - it may just mean held-out TRAIN
molecules (drug-like, Enveda-180-flavoured synthetic chemistry) are not
natural products, while the real test (Enveda is a natural-product drug
discovery company; CASMI itself is historically a natural-product/metabolite
identification challenge) may be a different population entirely. Checking
structure identity against the real test is impossible without its answers,
but checking mass coverage is not: if COCONUT has next to nothing near the
real test's precursor masses, it cannot help regardless of domain.
"""
from __future__ import annotations

import csv, sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

root = Path(sys.argv[1])
coconut_path = Path(sys.argv[2])

te = pq.read_table(root / "test.parquet", columns=["precursor_mz", "molecule_id"]).to_pandas()
q = te.drop_duplicates("molecule_id").precursor_mz.to_numpy(dtype=np.float64)
print(f"test: {len(q)} molecules, precursor range {q.min():.1f}-{q.max():.1f}")

csv.field_size_limit(10_000_000)
masses = []
with coconut_path.open(encoding="utf-8", errors="replace") as f:
    r = csv.DictReader(f)
    for row in r:
        m = row.get("exact_molecular_weight", "")
        if m:
            try:
                masses.append(float(m))
            except ValueError:
                pass
masses = np.sort(np.array(masses))
print(f"COCONUT: {len(masses)} masses, range {masses.min():.1f}-{masses.max():.1f}")

# A precursor is [M+H]+ / [M-H]- / etc - try the common adduct mass shifts to
# get the neutral mass, then see how many COCONUT entries sit within a loose
# window of it (loose on purpose: this checks coverage, not a candidate list).
SHIFTS = {"[M+H]+": 1.00728, "[M-H]-": -1.00728, "[M+Na]+": 22.98922,
          "[M+NH4]+": 18.03437}
for ppm in (10, 50):
    any_hit = np.zeros(len(q), dtype=bool)
    for name, shift in SHIFTS.items():
        neutral = q - shift
        w = neutral * ppm * 1e-6 + 0.01
        lo = np.searchsorted(masses, neutral - w)
        hi = np.searchsorted(masses, neutral + w)
        any_hit |= (hi > lo)
    print(f"  +-{ppm} ppm (any of {list(SHIFTS)}): "
          f"{any_hit.sum()} / {len(q)} molecules have >=1 COCONUT mass nearby "
          f"({100.0*any_hit.mean():.1f}%)")
