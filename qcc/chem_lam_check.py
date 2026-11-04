"""lambda_H2 of a factorization npz in the verifier's convention (H2_I stores w_B = W + sum_b w)."""
import sys
import numpy as np
for f in sys.argv[1:]:
    z = np.load(f)
    w, wB = z["W"].astype(np.float64), z["H2_I"].astype(np.float64)
    lam = 0.25 * np.sum((np.sum(np.abs(w), axis=1) + np.abs(wB)) ** 2)
    print(f, "lambda_H2 %.3f" % lam, "sum|w| %.3f" % np.abs(w).sum(), "W shape", w.shape)
