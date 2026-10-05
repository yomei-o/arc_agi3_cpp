import sys
import numpy as np
import pyarrow.parquet as pq
HEAD = np.dtype([("precursor","<f4"),("mol","<i4"),("off","<i4"),("n","<i4"),("tag","<i4")])
root = sys.argv[1]
h = np.fromfile(root + "/donor_head.bin", dtype=HEAD)
mz = np.sort(h["precursor"])
te = pq.read_table(root + "/test.parquet", columns=["precursor_mz","molecule_id"]).to_pandas()
q = te.drop_duplicates("molecule_id").precursor_mz.to_numpy()
for w in (10, 20, 50, 100, 200):
    counts = []
    for m in q:
        lo = np.searchsorted(mz, m - w)
        hi = np.searchsorted(mz, m + w)
        counts.append(hi - lo)
    counts = np.array(counts)
    print(f"+-{w:3d} Da: mean {counts.mean():7.0f} median {np.median(counts):7.0f} max {counts.max()}  (n={len(q)} queries)")
