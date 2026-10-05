import sys
import numpy as np
HEAD = np.dtype([("precursor","<f4"),("mol","<i4"),("off","<i4"),("n","<i4"),("tag","<i4")])
root = sys.argv[1]
stem = sys.argv[2]
h = np.fromfile(f"{root}\{stem}_head.bin", dtype=HEAD)
print(stem, "spectra", len(h), "peaks/spectrum: mean %.1f median %.0f max %d min %d" % (
    h["n"].mean(), np.median(h["n"]), h["n"].max(), h["n"].min()))
