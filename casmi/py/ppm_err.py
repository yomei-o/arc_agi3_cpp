"""Distribution of train.parquet precursor_error_ppm by instrument (how far measured precursors sit from the formula mass)."""
import numpy as np
import pyarrow.parquet as pq

f = pq.ParquetFile(r"C:\prog\casmi\train.parquet")
by = {}
for b in f.iter_batches(batch_size=500000, columns=["instrument_type", "precursor_error_ppm", "ingest_lib"]):
    d = b.to_pandas()
    for (ins, lib), g in d.groupby(["instrument_type", "ingest_lib"]):
        by.setdefault((ins, lib), []).append(g.precursor_error_ppm.dropna().to_numpy())
for k, v in sorted(by.items()):
    a = np.concatenate(v)
    if len(a) < 1000:
        continue
    q = np.quantile(np.abs(a), [0.5, 0.9, 0.99])
    print(f"{k[0]:12s} {k[1]:20s} n {len(a):8d}  |err| median {q[0]:.2f} p90 {q[1]:.2f} p99 {q[2]:.2f}  exact0 {np.mean(a == 0):.2f}  sd {a.std():.2f}")
