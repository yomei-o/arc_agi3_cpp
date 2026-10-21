r"""ice_mix.py <dir> <S tag> <D tag> <R tag> [variants...]

holdout2 yardstick: 0.5*S + 0.42*D + 0.08*R (S = hasqry or hxqry, D = hcqry, R = hcqry rm=1) of the per-query
MRR deltas vs `base`, from ice_eval.py's rr_<tag>.tsv; 95% CI by bootstrapping each set independently.
Also the same on the answers that are NOT MassSpecGym molecules (in_msg = 0), the leak-free part."""
import sys
from pathlib import Path

import numpy as np

d = Path(sys.argv[1])
tags = sys.argv[2:5]
W = [0.5, 0.42, 0.08]


def load(tag):
    lines = (d / f"rr_{tag}.tsv").read_text().splitlines()
    head = lines[0].split("\t")
    rows = [l.split("\t") for l in lines[1:]]
    cols = {h: np.array([float(r[i]) for r in rows]) for i, h in enumerate(head) if i >= 1}
    return cols


T = [load(t) for t in tags]
variants = sys.argv[5:] or [h for h in T[0] if h not in ("in_msg", "base")]
rng = np.random.default_rng(11)
B = 4000
print(f"mixture {W} of {tags}; per-set delta shown as S / D / R")
print(f"{'variant':16s} {'mix d':>7s} {'95% CI':>17s}   {'S':>7s} {'D':>7s} {'R':>7s} | nonMSG mix d [95% CI]")
for v in variants:
    out = []
    for sub in (False, True):
        ms, bs = [], np.zeros(B)
        for w, c in zip(W, T):
            keep = c["in_msg"] == 0 if sub else np.ones(len(c["base"]), bool)
            dd = (c[v] - c["base"])[keep]
            ms.append(dd.mean() if len(dd) else 0.0)
            if len(dd):
                bs += w * dd[rng.integers(0, len(dd), (B, len(dd)))].mean(1)
        out.append((sum(w * m for w, m in zip(W, ms)), np.percentile(bs, 2.5), np.percentile(bs, 97.5), ms))
    a, b = out
    print(f"{v:16s} {a[0]:+.4f} [{a[1]:+.4f},{a[2]:+.4f}]   {a[3][0]:+.4f} {a[3][1]:+.4f} {a[3][2]:+.4f} |"
          f" {b[0]:+.4f} [{b[1]:+.4f},{b[2]:+.4f}]")
