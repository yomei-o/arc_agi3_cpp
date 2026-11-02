"""Reference angle words (brot 12) requantized to fewer bits: fraction of zero words and of zero 5-word deltas.
    python chem_angle_coarse.py submission.npz"""
import sys
import numpy as np
w = np.load(sys.argv[1])["angle_words"].astype(np.int64)
for b in (12, 10, 8, 6):
    q = ((w + (1 << (12 - b - 1) if b < 12 else 0)) >> (12 - b)) % (1 << b)
    J = 5
    K = (75 + J - 1) // J
    pad = np.zeros((w.shape[0], K * J), dtype=np.int64); pad[:, :75] = q
    nzrows = [np.any(pad[:, k*J:(k+1)*J] != (pad[:, (k-1)*J:k*J] if k else 0), axis=1).mean() for k in range(K)]
    print(f"brot {b}: zero words {100*np.mean(q == 0):.1f}%, nonzero 5-word delta rows {100*np.mean(nzrows):.1f}%")
