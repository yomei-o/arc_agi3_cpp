r"""One representative spectrum per train molecule, for analog propagation.

Scoring a pool candidate against every donor spectrum within a mass-shift
window is only tractable if the donor set is small. It is not, as measured: a
+-200 Da window around a typical test precursor already catches ~1.9M of
train's 2.5M spectra - almost the whole library, because the library is dense
across its whole mass range rather than clustered. Deduplicating to the
most-informative spectrum per molecule (most peaks, as a cheap proxy for
"most spectral information") cuts the donor set from 2.5M to 276k, a ~9x
reduction - the first lever before the window itself has to shrink too.

head.bin's off field tracks the LOGICAL peak count (cumulative sum of n), not
a raw byte offset - export_bin.py increments it by k, while the raw file holds
2k floats (mz block then it block) per spectrum. Since rows are written in the
same order in both files, row i's raw float position is exactly 2*off[i],
which is what makes slicing an arbitrary subset of rows out of the raw file
possible without replaying every prior row.

    python dedup_donors.py <root>
"""
from __future__ import annotations

import sys
import numpy as np

HEAD = np.dtype([("precursor", "<f4"), ("mol", "<i4"), ("off", "<i4"),
                 ("n", "<i4"), ("tag", "<i4")])

root = sys.argv[1]
h = np.fromfile(root + "/train_head.bin", dtype=HEAD)
raw = np.fromfile(root + "/train_peaks.bin", dtype="<f4")

order = np.lexsort((-h["n"], h["mol"]))     # within a molecule, most peaks first
_, first = np.unique(h["mol"][order], return_index=True)
keep = order[first]
keep = keep[np.argsort(h["mol"][keep])]     # donor rows in molecule-id order
print(f"train spectra {len(h)} -> donors {len(keep)} (one per molecule)")

new_off = 0
new_head = np.zeros(len(keep), dtype=HEAD)
peak_chunks = []
for j, i in enumerate(keep):
    n = int(h["n"][i])
    raw_start = 2 * int(h["off"][i])
    peak_chunks.append(raw[raw_start:raw_start + 2 * n])
    new_head[j] = (h["precursor"][i], h["mol"][i], new_off, n, h["tag"][i])
    new_off += n

with open(root + "/donor_head.bin", "wb") as f:
    f.write(new_head.tobytes())
with open(root + "/donor_peaks.bin", "wb") as f:
    for chunk in peak_chunks:
        f.write(chunk.tobytes())
print(f"wrote donor_head.bin ({len(keep)} records) and donor_peaks.bin "
      f"({new_off} peaks)")
