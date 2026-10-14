import sys, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1])
from casmi import chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
d = sys.argv[2]
meta = pd.read_parquet(d + r"\pool_meta.parquet"); print(meta.columns.tolist(), len(meta))
fp = np.load(d + r"\pool_fp.npy", mmap_mode="r"); print(fp.shape)
bits = np.load(d + r"\fp_bits.npy")
scol = [c for c in meta.columns if "smi" in c.lower()][0]
rng = np.random.default_rng(0); idx = rng.choice(len(meta), 300, replace=False)
eq = diff = 0; nd = []
for i in idx:
    r = chem.raw_fingerprint(meta[scol].iloc[i])
    if r is None: continue
    a = np.packbits(r[bits]); b = np.asarray(fp[i])
    if np.array_equal(a, b): eq += 1
    else: diff += 1; nd.append(int((np.unpackbits(a) != np.unpackbits(b)).sum()))
print("equal", eq, "diff", diff, nd[:20])
