"""tune38_variants: v37 vs the tuned weights and partial variants (which change carries the gain?).

    python tune38_variants.py <dump dir> <tuned.json>
Per set: final MRR and paired-bootstrap delta vs v37 (tune38_core exact re-scoring).
"""
import json
import sys

import tune38_core as tc

D, TJ = sys.argv[1], sys.argv[2]
SETS = [("hxqry", "t38x"), ("hbqry", "t38b"), ("hbsqry", "t38bs"), ("hasqry", "t38s"), ("hcqry", "t38c")]
S = {q: tc.load(D, t, f"{D}/{q}_mols.txt") for q, t in SETS}
V37 = dict(tc.V37)
T = dict(V37)
T.update(json.load(open(TJ))["tuned"])


def var(**kw):
    w = dict(T)
    w.update(kw)
    return w


VARIANTS = [
    ("tuned", T),
    ("tuned + lib back (lib32 8, ana 24)", var(lib32=8.0, ana=24.0)),
    ("tuned + half lib (lib32 4, ana 12)", var(lib32=4.0, ana=12.0)),
    ("tuned + lib only (lib32 8, ana 0)", var(lib32=8.0)),
    ("tuned + ana only (lib32 0, ana 24)", var(ana=24.0)),
    ("v37 with lib32 0, ana 0 only", dict(V37, lib32=0.0, ana=0.0)),
]
base = {q: tc.eval_set(S[q], V37) for q in S}
print("v37: " + "  ".join(f"{q} {sum(r)/len(r):.4f}" for q, r in base.items()))
for name, W in VARIANTS:
    parts = []
    tims = 0.0
    for q in S:
        r = tc.eval_set(S[q], W)
        m, lo, hi = tc.boot(base[q], r)
        parts.append(f"{q} {sum(r)/len(r):.4f} ({m:+.4f} [{lo:+.3f},{hi:+.3f}])")
        if q in ("hxqry", "hbqry", "hbsqry"):
            tims += (sum(r) / len(r)) / 3
    print(f"== {name}: timsTOF mean {tims:.4f}")
    print("   " + "\n   ".join(parts))
