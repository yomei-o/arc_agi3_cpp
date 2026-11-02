"""Benchmark III factorization probe: train the starter kit's DF-THC objective (host tooling, JAX) at a chosen
(R, B, C), seeded from the provided 65/60/60 baseline (truncated), and report error / lambda_eff.
Run from C:\\prog\\qcc\\starter\\starter_kit on the build machine.
    python chem_train.py R B C epochs [lr] [eps_dfthc] [seed_file]
"""
import os, sys, time
sys.path.insert(0, os.getcwd())
from starter.task1_factorization import train_df_thc_qaggle as T

R, B, C, ep = (int(v) for v in sys.argv[1:5])
lr = float(sys.argv[5]) if len(sys.argv) > 5 else 0.001
eps = float(sys.argv[6]) if len(sys.argv) > 6 else 0.010
seed = sys.argv[7] if len(sys.argv) > 7 else T.PACKAGED_BASELINE
out = os.path.join("runs_own", f"FeMoCo_{R}_{B}_{C}_ep{ep}_lr{lr}_eps{eps}_{int(time.time())}.npz")
os.makedirs("runs_own", exist_ok=True)
t0 = time.time()
T.train_system("FeMoCo76o113e", epochs=ep, lr=lr, print_interval=max(1, ep // 20), rank=R, bases=B, copies=C,
               pad=0, epsilon_dfthc=eps, seed_solution_file=seed, output_file=out)
print("done in %.0fs ->" % (time.time() - t0), out, flush=True)
