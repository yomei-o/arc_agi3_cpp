r"""Is train.parquet's precursor_mz a measurement or a calculation?

This decides the single most load-bearing constant in the matcher. Tuning the
mass window on a holdout carved out of train gave 0.05 ppm, two hundred times
tighter than the 10 ppm the field uses. A window that tight only makes sense if
the numbers being compared came off the same calculator rather than out of two
different mass spectrometers - and if that is what happened, the tuning is an
artefact and the window deletes the answer on the real test.

So: take molecules that have several spectra of the same adduct in train and
look at how far apart their precursor values are. Identical to the last digit
means calculated. A few ppm of scatter means measured.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\prog\casmi")

f = pq.ParquetFile(ROOT / "train.parquet")
vals: dict[tuple, list] = defaultdict(list)
n = 0
for b in f.iter_batches(batch_size=200000,
                        columns=["inchikey14", "adduct", "precursor_mz",
                                 "instrument_type"]):
    d = b.to_pandas()
    d = d[(d.instrument_type == "timsTOF") & d.inchikey14.notna()]
    for k, a, m in zip(d.inchikey14, d.adduct, d.precursor_mz):
        if m == m:
            vals[(k, a)].append(float(m))
    n += len(b)
    if n > 1200000:
        break

spreads = []
for v in vals.values():
    if len(v) >= 2:
        v = np.array(v)
        spreads.append((v.max() - v.min()) / v.mean() * 1e6)
s = np.array(spreads)
print(f"{len(s)} (molecule, adduct) groups with >=2 timsTOF spectra")
for q in (50, 75, 90, 95, 99, 100):
    print(f"  spread p{q:<3d} {np.percentile(s, q):8.3f} ppm")
print(f"  exactly identical: {100.0 * (s == 0).mean():.1f}%")

# And the test set: how does its precursor compare to the nearest train value?
t = pq.read_table(ROOT / "test.parquet", columns=["precursor_mz"]).to_pandas()
tr = pq.read_table(ROOT / "train_head_probe.parquet").to_pandas() if False else None
print(f"\ntest precursor count {len(t)}; "
      f"decimal digits look like {t.precursor_mz.iloc[:3].tolist()}")
