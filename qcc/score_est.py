"""Benchmark I score estimate from inv_model (exact over every x) and a static Toffoli count, mirroring the
verifier's cost: eps = pi * min_x(tol(x) - |err(x)|) / n_rot, T per rotation 0.56 log2(1/eps) + 5.3,
C = Toffoli + T/4, score = Q * C.  Phase bits j < DROP are not rotated (their weight is in err).
    python score_est.py toffolis qubits [A=.. D=.. WY=.. W1=.. W2=..] [DROP=k]
"""
import math, sys
import numpy as np
from inv_model import P, N_P, MAX_X, model

tof, q = int(sys.argv[1]), int(sys.argv[2])
kw = {k: int(v) for k, v in (a.split("=") for a in sys.argv[3:])}
drop = kw.pop("DROP", 0)
p = P(**kw)
xs = np.arange(1, MAX_X + 1, dtype=np.int64)
Y, s = model(p, xs)
F = p.WY + N_P
Yp = (Y >> drop) << drop
phase = (Yp % (1 << F)) / 2.0**F               # turns, mod 1
target = (1 / np.sqrt(xs)) % 1.0
err = (phase - target + 0.5) % 1.0 - 0.5
tol = 2.0**-24 / np.sqrt(xs)
margin = (tol - np.abs(err)).min()
nbits = F - drop                                 # rotated bits j = drop .. F-1
n_rot = max(0, nbits - 3)                        # angles 1/2, 1/4 Clifford, 1/8 = T
t_base = 1 if nbits >= 3 else 0
eps = math.pi * margin / n_rot
T = t_base + n_rot * (0.56 * math.log2(1 / eps) + 5.3)
C = tof + T / 4
print(f"{p} drop {drop}: margin {margin:.3e} turns, n_rot {n_rot}, eps {eps:.2e}, T {T:.0f}, C {C:.0f}, Q {q}, "
      f"score {q * C:.4g}  (reference 2.08e6)")
