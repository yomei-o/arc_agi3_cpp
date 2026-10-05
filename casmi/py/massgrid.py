r"""How exactly do test precursors line up with train precursors?

The mass window is the one constant that can silently delete the answer, and
two incompatible stories about it are both defensible:

  * train's precursor_mz is calculated, not measured (63.5% of
    (molecule, adduct) groups are bit-identical), so if test's is calculated
    the same way the two agree to the rounding and a sub-ppm window is right;
  * a real instrument's precursor error runs to several ppm, so anything under
    10 ppm throws answers away.

Which one holds is visible without labels. If test values come off the same
calculator, the nearest train precursor sits implausibly close - far closer
than chance would put it given how dense the training masses are.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\prog\casmi")

tr = pq.read_table(ROOT / "train.parquet", columns=["precursor_mz"]).to_pandas()
m = np.sort(tr.precursor_mz.dropna().to_numpy(dtype=np.float64))
te = pq.read_table(ROOT / "test.parquet", columns=["precursor_mz", "molecule_id"]).to_pandas()
q = te.drop_duplicates("molecule_id").precursor_mz.to_numpy(dtype=np.float64)
print(f"train {len(m)} precursors, test {len(q)} molecules")


def decimals(x):
    s = np.abs(x - np.round(x, 4))
    return float((s < 1e-9).mean())


print(f"fraction that are exactly 4-decimal: train {decimals(m[:200000]):.3f}  "
      f"test {decimals(q):.3f}")

i = np.searchsorted(m, q)
best = np.full(len(q), np.inf)
for off in (-1, 0):
    j = np.clip(i + off, 0, len(m) - 1)
    best = np.minimum(best, np.abs(m[j] - q) / q * 1e6)
print("\nnearest train precursor, in ppm:")
for p in (10, 25, 50, 75, 90, 99):
    print(f"  p{p:<3d} {np.percentile(best, p):9.4f}")
print(f"  within 0.3 ppm: {100.0 * (best < 0.3).mean():5.1f}%"
      f"   1 ppm: {100.0 * (best < 1).mean():5.1f}%"
      f"   5 ppm: {100.0 * (best < 5).mean():5.1f}%")

# How dense are the training masses around a test precursor? If a 1 ppm window
# already holds dozens of distinct spectra, "the nearest one is within 1 ppm"
# says nothing - the comparison below is what makes the first number mean
# something.
for ppm in (0.3, 1.0, 5.0, 10.0):
    w = q * ppm * 1e-6
    lo = np.searchsorted(m, q - w)
    hi = np.searchsorted(m, q + w)
    n = hi - lo
    print(f"  +-{ppm:5.1f} ppm: train spectra in window, median {np.median(n):6.0f}")
