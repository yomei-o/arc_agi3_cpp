import sys
import numpy as np
HEAD = np.dtype([("precursor","<f4"),("mol","<i4"),("off","<i4"),("n","<i4"),("tag","<i4")])
root = sys.argv[1]
h = np.fromfile(root + "/train_head.bin", dtype=HEAD)
mz = np.sort(h["precursor"])
import pyarrow.parquet as pq
te = pq.read_table(root + "/test.parquet", columns=["precursor_mz","molecule_id"]).to_pandas()
q = te.drop_duplicates("molecule_id").precursor_mz.to_numpy()[:20]
for w in (50, 100, 200):
    counts = []
    for m in q:
        lo = np.searchsorted(mz, m - w)
        hi = np.searchsorted(mz, m + w)
        counts.append(hi - lo)
    counts = np.array(counts)
    print(f"+-{w} Da: mean {counts.mean():.0f} median {np.median(counts):.0f} max {counts.max()}")
