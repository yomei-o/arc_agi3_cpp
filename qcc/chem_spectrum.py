"""Benchmark III lower bound: how many rank-one terms vec(X) vec(X)^T does the shifted two-electron target need?

error_2 = || sum_(rho c) X_rc (x) X_rc - tei - (S (x) I + I (x) S)/2 ||_F (verifier, dfthc_math). The fitted part is a
PSD supermatrix of rank <= R*C, so the best possible error with T terms is the eigenvalue tail of the target
supermatrix (Eckart-Young; negative eigenvalues can never be fitted). Prints the minimal T for several error levels.
Run in C:\\prog\\qcc\\verifiers on the build machine.
"""
import numpy as np
from dfthcsos.scoreinfra.reference import load_reference, load_fixed_shifts, PACKAGED_FCIDUMP
from dfthcsos.scoreinfra.dfthc_math import symmetric_from_lower

ref = load_reference(PACKAGED_FCIDUMP)
tei = np.asarray(ref["tei"], dtype=np.float64)
N = tei.shape[0]
S = symmetric_from_lower(load_fixed_shifts()["Hsym_lower"], N).astype(np.float64)
tgt = tei.copy()
d = np.arange(N)
tgt[:, :, d, d] += 0.5 * S[:, :, None]
tgt[d, d, :, :] += 0.5 * S[None, :, :]
M = tgt.reshape(N * N, N * N)
M = 0.5 * (M + M.T)
ev = np.linalg.eigvalsh(M)[::-1]
neg = np.sqrt(np.sum(ev[ev < 0] ** 2))
pos = ev[ev > 0]
tail = np.sqrt(np.cumsum((pos ** 2)[::-1])[::-1])  # tail[t] = error using the first t terms
tail = np.append(tail, 0.0)
print(f"N {N}, positive eigenvalues {len(pos)}, negative part floor {neg * 1000:.3f} mHa, ecore {ref['ecore']:.4f}")
for e in (0.050, 0.030, 0.020, 0.015, 0.010, 0.005):
    t = int(np.argmax(np.sqrt(tail ** 2 + neg ** 2) <= e)) if np.any(np.sqrt(tail ** 2 + neg ** 2) <= e) else None
    print(f"error_2 <= {e * 1000:.0f} mHa needs at least {t} rank-one terms (R*C)")
print("top eigenvalues", np.round(pos[:10], 4))
np.save("chem_eigs.npy", ev)
