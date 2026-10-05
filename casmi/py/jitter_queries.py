r"""Put the instrument's precursor error back into a holdout's queries.

train.parquet's precursor_mz is calculated, not measured: 63.5% of
(molecule, adduct) groups carry bit-identical values and the 99th percentile
spread inside a group is under 1 ppm. So a query carved out of train matches
its own library rows to the last digit, a 0.05 ppm mass window scores far
better than a 1 ppm one, and the tuning that follows is an artefact of the
file rather than a fact about mass spectrometry. On the real test the
precursor comes from a different instrument; the published window-recall curve
for this competition reads 0.972 at 5 ppm and 1.000 at 10 ppm.

This rewrites only the query header, leaving the peaks and the answers alone,
so the sweep can be re-run against a holdout that cannot be won by matching
rounding.

    python jitter_queries.py <root> <in_stem> <out_stem> <ppm_sigma> [seed]
"""
from __future__ import annotations

import shutil, sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
SRC, DST = sys.argv[2], sys.argv[3]
SIGMA = float(sys.argv[4])
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 7

HEAD = np.dtype([("precursor", "<f4"), ("mol", "<i4"), ("off", "<i4"),
                 ("n", "<i4"), ("tag", "<i4")])

h = np.fromfile(ROOT / f"{SRC}_head.bin", dtype=HEAD)
rng = np.random.default_rng(SEED)
h["precursor"] = (h["precursor"]
                  * (1.0 + rng.normal(0.0, SIGMA * 1e-6, size=len(h)))).astype("<f4")
h.tofile(ROOT / f"{DST}_head.bin")
for ext in ("_peaks.bin", "_mols.txt"):
    shutil.copyfile(ROOT / f"{SRC}{ext}", ROOT / f"{DST}{ext}")
print(f"{len(h)} queries, sigma {SIGMA} ppm -> {DST}_*")
