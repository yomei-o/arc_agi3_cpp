"""pc_sweep: calibrate the PubChem-channel strength of the Boltzmann merge (W["pi"], W["a"], W["T2"]) offline on the
phys_v38td dumps: hcqry with the truth removed from the pool (t38c_rm1: the answer can only come from PubChem /
generation = the hidden test's out-of-pool part) against the in-pool sets (t38c, t38x, t38b, t38bs: every PubChem
pick that wins rank 1 is a loss). Prints per-set MRR@25 and the mixture f*rm1 + (1-f)*mean(in-pool) for f = 0.1 / 0.2.

    python pc_sweep.py <dumpdir> [f=0.15]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tune38_core as tc

D = sys.argv[1]
SETS = [("hcqry_rm1", "t38c_rm1", "hcqry"), ("hcqry", "t38c", "hcqry"), ("hxqry", "t38x", "hxqry"),
        ("hbqry", "t38b", "hbqry"), ("hbsqry", "t38bs", "hbsqry")]
S = {}
for name, tag, mols in SETS:
    if os.path.exists(f"{D}/{tag}_td.txt"):
        S[name] = tc.load(D, tag, f"{D}/{mols}_mols.txt")
print("sets:", {k: len(v) for k, v in S.items()})
base = dict(tc.V37)


def ev(W):
    return {k: sum(r) / len(r) for k, r in ((k, tc.eval_set(Q, W)) for k, Q in S.items())}


b = ev(base)
print("v37 (pi %.2f a %.2f T2 %.2f): " % (base["pi"], base["a"], base["T2"]) + "  ".join(f"{k} {v:.4f}" for k, v in b.items()))
inpool = [k for k in S if k != "hcqry_rm1"]
rows = []
for pi in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
    for a in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        for T2 in (0.5, 1.0):
            W = dict(base, pi=pi, a=a, T2=T2)
            m = ev(W)
            ip = sum(m[k] for k in inpool) / len(inpool) if inpool else 0.0
            rm = m.get("hcqry_rm1", 0.0)
            rows.append((pi, a, T2, rm, ip, m))
print("pi    a    T2   rm1     inpool   mix0.1   mix0.2   " + "  ".join(inpool))
for pi, a, T2, rm, ip, m in rows:
    print(f"{pi:<5} {a:<4} {T2:<4} {rm:.4f}  {ip:.4f}  {0.1 * rm + 0.9 * ip:.4f}  {0.2 * rm + 0.8 * ip:.4f}   "
          + "  ".join(f"{m[k]:.4f}" for k in inpool))
best = {}
for f in (0.1, 0.15, 0.2, 0.3):
    best[f] = max(rows, key=lambda r: f * r[3] + (1 - f) * r[4])
    r = best[f]
    print(f"best for out-of-pool fraction {f}: pi {r[0]} a {r[1]} T2 {r[2]}  rm1 {r[3]:.4f} inpool {r[4]:.4f}  "
          f"mix {f * r[3] + (1 - f) * r[4]:.4f} (v37 mix {f * b.get('hcqry_rm1', 0) + (1 - f) * sum(b[k] for k in inpool) / len(inpool):.4f})")
