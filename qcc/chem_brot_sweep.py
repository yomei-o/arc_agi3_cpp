"""Angle-quantization cost of a factorization: verifier metrics after encoding at several brot, on the path and
log-depth trees (starter task2 brot_sweep). Run in C:\prog\qcc\starter\starter_kit.
    python chem_brot_sweep.py factorization.npz [brot ...]"""
import os, sys
sys.path.insert(0, os.getcwd())
from starter.task2_angles.encode import brot_sweep, TREES

f = sys.argv[1]
brots = [int(b) for b in sys.argv[2:]] or [6, 7, 8, 10, 12]
for tree in ("log_depth", "path"):
    for r in brot_sweep("FeMoCo76o113e", f, TREES[tree](76), brots):
        print(f"{tree:9s} brot {str(r['brot']):4s} error_1 {1000 * r['error_1']:7.2f} mHa  error_2 {1000 * r['error_2']:7.2f} mHa  "
              f"Lambda {r['Lambda']:8.2f}  E_gap {r['E_gap']:.4f}  lambda_eff {r['lambda_eff']:.3f}", flush=True)
