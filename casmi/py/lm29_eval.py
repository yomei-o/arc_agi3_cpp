"""lm29_eval.py - offline evaluation of the v29 ranker from lmart_v29 OOF score files.

usage: lm29_eval.py <oof files (glob ok)> [--gate g,...] [--blend b,...] [--noppop] [--rows]

Each OOF file: "S <set> <name> <truth> <n> <libmax>" then n lines "<model> <hand> <isGen> <pop>".
Per query the final score is phys_v29lm's:
    out = mean(hand) + sd(hand) * ((1-blend) z(hand) + blend z(model + 1e-4 z(hand)))   if libmax < gate
    out = hand                                                                       otherwise
then the v28a pool-popularity step (ppop=0.15, ppn=60, ppgate=0.9): the pool candidates' places in the
top 60 are re-filled by z(out over them) + 0.15 pop. FPNet is off (it leaks on holdouts).
MRR@25 = 1/position of the truth in the final list (ties: averaged over the tied block before ppop -
negligible with the 1e-4 tie-break). Hand arm = blend 0 (= v28a without FPNet).
Reports per set, paired SE vs hand, and the mixture yardstick 0.5 hasqry + 0.42 hcqry + 0.08 hcqry rm,
plus an all-sets mixture 0.5 mean(S) + 0.42 mean(D) + 0.08 mean(R).
"""
import glob
import math
import sys

import numpy as np

S_SETS = ["has0", "hbs0", "hx0"]
D_SETS = ["ha0", "hc0", "hb0"]
R_SETS = ["ha1", "hc1", "hb1"]


def load(paths):
    Q = {}
    for p in paths:
        with open(p) as f:
            lines = f.read().split("\n")
        i = 0
        while i < len(lines):
            ln = lines[i]
            if not ln.startswith("S "):
                i += 1
                continue
            _, path, name, t, n, lmax = ln.split()
            n = int(n)
            arr = np.array([[float(x) for x in lines[i + 1 + k].split()] for k in range(n)], dtype=np.float64)
            sset = path.replace("\\", "/").split("/")[-1].replace("_f.txt", "")
            Q[(sset, name)] = (int(t), float(lmax), arr)
            i += 1 + n
    return Q


def z(v):
    m = v.mean()
    s = math.sqrt(((v - m) ** 2).mean()) + 1e-6
    return (v - m) / s


def final_rr(t, lmax, arr, gate, blend, ppop=True):
    if t < 0:
        return 0.0
    model, hand, isgen, pop = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    if lmax < gate and blend > 0:
        zh = z(hand)
        zr = z(model + 1e-4 * zh)
        out = hand.mean() + (math.sqrt(((hand - hand.mean()) ** 2).mean()) + 1e-6) * ((1 - blend) * zh + blend * zr)
    else:
        out = hand
    n = len(out)
    order = sorted(range(n), key=lambda i: -out[i])
    if ppop and lmax < 0.9:
        pos = [r for r in range(min(60, n)) if isgen[order[r]] < 0.5]
        idx = [order[r] for r in pos]
        if len(idx) > 1:
            v = out[idx]
            m = v.mean()
            s = math.sqrt(((v - m) ** 2).mean()) + 1e-6
            key = [((out[i] - m) / s + 0.15 * pop[i], i) for i in idx]
            key = sorted(key, key=lambda x: -x[0])  # stable
            for j, r in enumerate(pos):
                order[r] = key[j][1]
        r = order.index(t)
        return 1.0 / (r + 1) if r < 25 else 0.0
    better = int((out > out[t]).sum())
    eq = int((out == out[t]).sum()) - 1
    return 1.0 / (better + 1 + 0.5 * eq) if better < 25 else 0.0


def evaluate(Q, gate, blend, ppop=True):
    per = {}
    for (s, name), (t, lmax, arr) in Q.items():
        per.setdefault(s, {})[name] = final_rr(t, lmax, arr, gate, blend, ppop)
    return per


def main():
    args = sys.argv[1:]
    gates, blends, ppop = [1e9], [1.0], True
    files = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--gate":
            gates = [float(x) for x in args[i + 1].split(",")]; i += 2; continue
        if a == "--blend":
            blends = [float(x) for x in args[i + 1].split(",")]; i += 2; continue
        if a == "--noppop":
            ppop = False; i += 1; continue
        files += glob.glob(a)
        i += 1
    Q = load(sorted(files))
    hand = evaluate(Q, 1e9, 0.0, ppop)
    sets = [s for s in S_SETS + D_SETS + R_SETS if s in hand]
    print("queries:", {s: len(hand[s]) for s in sets})
    print("hand (v28a, FPNet off):", " ".join(f"{s} {np.mean(list(hand[s].values())):.4f}" for s in sets))

    def mix(per):
        m = lambda s: np.mean(list(per[s].values())) if s in per else float("nan")
        y = 0.5 * m("has0") + 0.42 * m("hc0") + 0.08 * m("hc1")
        a = 0.5 * np.mean([m(s) for s in S_SETS]) + 0.42 * np.mean([m(s) for s in D_SETS]) + 0.08 * np.mean([m(s) for s in R_SETS])
        return y, a

    hy, ha = mix(hand)
    print(f"hand mixture: yardstick {hy:.4f}  all-sets {ha:.4f}")
    for g in gates:
        for b in blends:
            per = evaluate(Q, g, b, ppop)
            parts = []
            worstS = 1e9
            for s in sets:
                d = np.array([per[s][k] - hand[s][k] for k in hand[s]])
                se = d.std(ddof=1) / math.sqrt(len(d))
                parts.append(f"{s} {np.mean(list(per[s].values())):.4f} ({d.mean():+.4f}±{se:.4f})")
                if s in S_SETS:
                    worstS = min(worstS, d.mean() / max(se, 1e-9))
            y, a = mix(per)
            ok = "PASS" if worstS > -1.0 and y > hy else "fail"
            print(f"gate {g:<5g} blend {b:<4g} | " + "  ".join(parts) +
                  f" | mix {y:.4f} ({y - hy:+.4f}) all {a:.4f} ({a - ha:+.4f}) | worst S t {worstS:+.2f} {ok}")


if __name__ == "__main__":
    main()
