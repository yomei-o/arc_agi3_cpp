"""Benchmark III cost model explorer: T_total and Q_total from (R, B, C, brot, J) with the verifier's heuristic
formulas (costs_prepare.SpinFreeData_V1, chi = 10) and plain-QROM ANGLES circuits (L - 2 Toffolis per batch per
pass, measured on the sample: 3974 at L = 3976), K = ceil(75 / J) batches, forward + reverse.
    python chem_cost.py R B C brot J [lambda_eff]
"""
import math, sys
from costs_prepare import SpinFreeData_V1

N = 76


def costs(R, B, C, brot, J, angles_per_batch=None):
    m = SpinFreeData_V1(R=R, B=B, C=C, N=N, chi=10, beta=brot)
    L = N + R * B
    K = math.ceil((N - 1) / J)
    per = (L - 2) if angles_per_batch is None else angles_per_batch
    ang_T = 2 * K * per
    oq, iq = m.outer_state_prep_qubit_cost(), m.inner_state_prep_qubit_cost()
    sel = math.ceil(math.log2(L))
    ang_temp = sel - 1                     # unary iteration temporaries (sample: 11 at 12 selection bits)
    T = ang_T + m.outer_state_prep() + m.inner_state_prep() + m.givens_rotations()
    persistent = oq["persistent"] + iq["persistent"] + J * brot
    temp = max(oq["temporary"], iq["temporary"], ang_temp)
    return dict(L=L, K=K, ang_T=ang_T, outer=m.outer_state_prep(), inner=m.inner_state_prep(),
                givens=m.givens_rotations(), T=T, persistent=persistent, outer_temp=oq["temporary"],
                inner_temp=iq["temporary"], Q=persistent + temp, cap=sel + J * brot + ang_temp)


if __name__ == "__main__":
    a = [int(v) for v in sys.argv[1:6]]
    lam = float(sys.argv[6]) if len(sys.argv) > 6 else None
    c = costs(*a)
    print(c)
    if lam:
        print("cost %.4g ratio %.3f" % (lam * c["T"] * c["Q"], 3355599588.3155365 / (lam * c["T"] * c["Q"])))
