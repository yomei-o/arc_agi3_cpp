"""pc_promo: PubChem rank-1 promotion on top of the Boltzmann merge, calibrated offline in OUR score units.

The public lineage's biggest single LB step was promoting the PubChem channel's top pick to rank 1 when its evidence
is strong ((S > 6 and pop >= 5) or rel > 150 in their units; +0.024..0.037). Our ranker's v2 promotion
(relz > relp or (S > sp and pop_top >= popp)) is switched off by alloc=boltz. Here: final list = tune38_core's exact
Boltzmann re-scoring, then promotion of the PubChem top pick (pcn[0]) to rank 1 if the rule fires (gate libSim < 0.9
as the channel). Gain on hcqry rm1 (answer out of the pool) vs loss on the in-pool sets.

    python pc_promo.py <dumpdir>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tune38_core as tc  # noqa: E402

D = sys.argv[1]
SETS = [("hcqry_rm1", "t38c_rm1", "hcqry"), ("hcqry", "t38c", "hcqry"), ("hxqry", "t38x", "hxqry"),
        ("hbqry", "t38b", "hbqry"), ("hbsqry", "t38bs", "hbsqry")]


def v2(path):
    """query -> (relz, S, pop_top) from the V2 lines."""
    out, q = {}, None
    for l in open(path, encoding="utf-8"):
        p = l.rstrip("\n").split("\t")
        if p[0] == "Q":
            q = p[1]
        elif p[0] == "V2" and q is not None:
            out[q] = (float(p[3]), float(p[4]), float(p[5]))
    return out


S, V = {}, {}
for name, tag, mols in SETS:
    if os.path.exists(f"{D}/{tag}_td.txt"):
        S[name] = tc.load(D, tag, f"{D}/{mols}_mols.txt")
        V[name] = v2(f"{D}/{tag}_det.txt")
W = dict(tc.V37)
base_lists = {k: [tc.final_list(q, W)[1] for q in Q] for k, Q in S.items()}


def mrr(k, rule):
    tot, nprom, good = 0.0, 0, 0
    for q, L in zip(S[k], base_lists[k]):
        L = list(L)
        st = V[k].get(q["name"])
        if st and q["pcn"] and q["lib"] < tc.GATE and rule(*st):
            top = q["pcn"][0]
            if top in L:
                L.remove(top)
            L = [top] + L
            nprom += 1
            good += top == q["name"]
        tot += tc.rr(q, L)
    return tot / len(S[k]), nprom, good


print("base: " + "  ".join(f"{k} {mrr(k, lambda *a: False)[0]:.4f}" for k in S))
# distribution of the statistics, for choosing grids
import numpy as np  # noqa: E402

for k in S:
    a = np.array(list(V[k].values()))
    if len(a):
        print(f"{k}: relz p50 {np.median(a[:, 0]):.2f} p90 {np.quantile(a[:, 0], .9):.2f} p99 {np.quantile(a[:, 0], .99):.2f}"
              f" | S p50 {np.median(a[:, 1]):.2f} p90 {np.quantile(a[:, 1], .9):.2f} | pop p50 {np.median(a[:, 2]):.2f} "
              f"p90 {np.quantile(a[:, 2], .9):.2f}")
inpool = [k for k in S if k != "hcqry_rm1"]
rows = []
for relp in (1e9, 2, 3, 4, 5, 6, 8):
    for sp in (1e9, 3, 4, 5, 6):
        for popp in (3, 5, 7):
            if sp == 1e9 and popp != 3:
                continue
            rule = (lambda rp, s_, pp: (lambda relz, S_, pop: relz > rp or (S_ > s_ and pop >= pp)))(relp, sp, popp)
            r = {k: mrr(k, rule) for k in S}
            ip = sum(r[k][0] for k in inpool) / len(inpool)
            rows.append((relp, sp, popp, r["hcqry_rm1"][0], ip, r))
b = rows[0]
print(f"{'relp':>5} {'sp':>5} {'popp':>4}  rm1     inpool   mix.1   mix.2  | promoted (correct) rm1 / hcqry / hx / hb / hbs")
for relp, sp, popp, rm, ip, r in rows:
    pr = " / ".join(f"{r[k][1]}({r[k][2]})" for k in S)
    print(f"{relp:>5.0f} {sp:>5.0f} {popp:>4}  {rm:.4f}  {ip:.4f}  {0.1 * rm + 0.9 * ip:.4f}  {0.2 * rm + 0.8 * ip:.4f}  | {pr}")
for f in (0.1, 0.15, 0.2):
    best = max(rows, key=lambda x: f * x[3] + (1 - f) * x[4])
    print(f"best at out-of-pool fraction {f}: relp {best[0]} sp {best[1]} popp {best[2]}  "
          f"gain {f * (best[3] - b[3]) + (1 - f) * (best[4] - b[4]):+.4f}")
