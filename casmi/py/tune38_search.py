"""tune38_search: joint re-tuning of the v37 hand weights on the timsTOF yardstick (tune38_STATUS.md).

Yardstick = mean of the final-list MRR@25 of hxqry, hbqry, hbsqry (all timsTOF); hasqry / hcqry reported only.
Exact offline re-scoring from the phys_v38td term dumps (tune38_core; checked against the C++ CSVs by tune38_check).

Search = coordinate descent over coarse grids (multiples of 0.25 or coarser), one weight at a time, from the v37 weights.
2-fold split by molecule (md5 of the molecule id):
  main ("both halves"): a value is accepted only if it improves fold A AND fold B (each > 0, mean >= MIN_GAIN);
  cross-fit: CD fitted on fold A only -> reported on fold B, and B -> A (honest out-of-fold estimate of the procedure);
  random-label control: every query's label replaced by a random candidate of its own list (seed 7); the same
  both-halves CD from v37 -> how much 'gain' the procedure finds on noise; plus the tuned config's delta on those labels.
usage: python tune38_search.py <dumpdir> <outdir> [sweeps=3]
"""
import hashlib
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tune38_core as tc  # noqa: E402

TIMS = [("hxqry", "t38x"), ("hbqry", "t38b"), ("hbsqry", "t38bs")]
REPORT = [("hasqry", "t38s"), ("hcqry", "t38c")]
GRID = {
    "fp": [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3],
    "cnn": [0, 0.5, 1, 1.5, 2, 2.5, 3, 4],
    "ft": [0, 0.5, 1, 1.5, 2, 2.5, 3, 4],
    "att": [0, 0.5, 1, 1.5, 2, 2.5, 3, 4],
    "phys": [0.5, 0.75, 1, 1.25, 1.5, 2],
    "mlp": [0, 0.25, 0.5, 0.75, 1],
    "rev": [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2],
    "lib32": [0, 4, 6, 8, 10, 12, 16],
    "ana": [0, 8, 12, 16, 20, 24, 28, 32, 40],
    "ppop": [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4],
    "g0": [-4, -3, -2.5, -2, -1.5, -1, 0],
    "g1": [-1, 0, 1, 2],
    "g2": [-2, -1.5, -1, -0.5, 0],
    "g3": [2, 3, 3.5, 4, 4.5, 5, 6],
    "g4": [-1, -0.5, 0, 0.5, 1],
    "g5": [0, 0.5, 1, 1.5, 2, 2.5, 3],
}
MIN_GAIN = 0.0005


def fold_of(name):
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16) % 2


class Data:
    def __init__(self, D, sets, labels=None):
        self.S = {}
        for qry, tag in sets:
            self.S[qry] = tc.load(D, tag, f"{D}/{qry}_mols.txt")
        self.fold = {k: [fold_of(q["name"]) for q in Q] for k, Q in self.S.items()}
        self.labels = labels  # {set: [name per query]} or None

    def rrs(self, W, k):
        Q = self.S[k]
        lab = self.labels[k] if self.labels else None
        return tc.eval_set(Q, W, label=lab)

    def score(self, W):
        """-> (yardstick on fold 0, fold 1, all) + per-set rr lists"""
        per = {k: self.rrs(W, k) for k in self.S}
        f = []
        for fo in (0, 1, None):
            v = 0.0
            for k, r in per.items():
                ii = [i for i in range(len(r)) if fo is None or self.fold[k][i] == fo]
                v += sum(r[i] for i in ii) / max(1, len(ii))
            f.append(v / len(per))
        return f, per


def cd(data, W0, rule, sweeps, log, tag):
    """rule: 'both' (fold 0 and fold 1 each improve), 'f0' (fit fold 0), 'f1' (fit fold 1)."""
    W = dict(W0)
    cur, _ = data.score(W)
    log(f"[{tag}] start: f0 {cur[0]:.4f} f1 {cur[1]:.4f} all {cur[2]:.4f}")
    for sw in range(sweeps):
        changed = False
        for k, vals in GRID.items():
            best, bestv = None, None
            for v in vals:
                if v == W[k]:
                    continue
                W2 = dict(W)
                W2[k] = v
                s, _ = data.score(W2)
                d0, d1 = s[0] - cur[0], s[1] - cur[1]
                if rule == "both":
                    ok, gain = d0 > 0 and d1 > 0 and (d0 + d1) / 2 >= MIN_GAIN, (d0 + d1) / 2
                elif rule == "f0":
                    ok, gain = d0 >= MIN_GAIN, d0
                else:
                    ok, gain = d1 >= MIN_GAIN, d1
                if ok and (best is None or gain > best[0]):
                    best, bestv = (gain, s), v
            if best is not None:
                log(f"[{tag}] sweep {sw} {k}: {W[k]} -> {bestv}  f0 {best[1][0]:.4f} ({best[1][0]-cur[0]:+.4f}) "
                    f"f1 {best[1][1]:.4f} ({best[1][1]-cur[1]:+.4f}) all {best[1][2]:.4f}")
                W[k] = bestv
                cur = best[1]
                changed = True
        if not changed:
            break
    return W, cur


def opt_string(W):
    """phys_v37pc arguments for W (phys weight normalised to 1: every score weight / wp, bT1 / wp; ppop is scale-free)."""
    p = W["phys"]
    f = lambda x: ("%g" % round(x / p, 4))  # noqa: E731
    gw = ":".join(f(W[f"g{k}"]) for k in range(6))
    return dict(mlp=f(W["mlp"]), rev=f(W["rev"]), lib=f(W["lib32"]), ana=f(W["ana"]), wf=f(W["fp"]),
                opts=f"gw={gw},attpool={f(W['att'])},cnnpool={f(W['cnn'])},ftpool={f(W['ft'])},ppop={W['ppop']:g},"
                     f"bT1={f(W['T1'])}")


def main():
    D, OUT = sys.argv[1], sys.argv[2]
    sweeps = 3
    global TIMS, REPORT
    for a in sys.argv[3:]:
        if a.startswith("sweeps="):
            sweeps = int(a.split("=")[1])
        # tims=hnqry:n_base_nopop,... / report=... : other yardstick sets (hn_STATUS.md: the natural-product x timsTOF dumps)
        if a.startswith("tims="):
            TIMS = [tuple(x.split(":")) for x in a.split("=", 1)[1].split(",") if x]
        if a.startswith("report="):
            REPORT = [tuple(x.split(":")) for x in a.split("=", 1)[1].split(",") if x]
        if a.startswith("skip="):   # weights left out of the search (e.g. ft when the dumps have no FT term)
            for k in a.split("=", 1)[1].split(","):
                GRID.pop(k, None)
    os.makedirs(OUT, exist_ok=True)
    lf = open(f"{OUT}/search.log", "w", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        lf.write(s + "\n")
        lf.flush()

    t0 = time.time()
    data = Data(D, TIMS)
    rep = Data(D, [x for x in REPORT if os.path.exists(f"{D}/{x[1]}_td.txt")])
    log(f"loaded {', '.join(f'{k} {len(v)}' for k, v in data.S.items())}; report sets {list(rep.S)}; "
        f"fold sizes {[sum(1 for k in data.S for f in data.fold[k] if f == fo) for fo in (0, 1)]}")
    base, base_per = data.score(tc.V37)
    log(f"v37 weights: yardstick {base[2]:.4f} (f0 {base[0]:.4f} f1 {base[1]:.4f}); "
        + " ".join(f"{k} {sum(v)/len(v):.4f}" for k, v in base_per.items()))

    # main: both halves
    Wb, sb = cd(data, tc.V37, "both", sweeps, log, "both")
    # cross-fit
    Wa, _ = cd(data, tc.V37, "f0", sweeps, log, "fit f0")
    Wc, _ = cd(data, tc.V37, "f1", sweeps, log, "fit f1")
    sa, _ = data.score(Wa)
    sc, _ = data.score(Wc)
    log(f"cross-fit: fit f0 -> f1 {sa[1]:.4f} ({sa[1]-base[1]:+.4f}) | fit f1 -> f0 {sc[0]:.4f} ({sc[0]-base[0]:+.4f}) | "
        f"out-of-fold mean gain {((sa[1]-base[1]) + (sc[0]-base[0]))/2:+.4f}")

    # random-label control
    rnd = random.Random(7)
    lab = {k: [rnd.choice(q["cname"]) if q["n"] else "" for q in Q] for k, Q in data.S.items()}
    ctrl = Data.__new__(Data)
    ctrl.S, ctrl.fold, ctrl.labels = data.S, data.fold, lab
    cb, _ = ctrl.score(tc.V37)
    Wr, sr = cd(ctrl, tc.V37, "both", sweeps, log, "random-label")
    ct, _ = ctrl.score(Wb)
    log(f"random-label control: v37 {cb[2]:.4f}; CD on random labels finds {sr[2]-cb[2]:+.4f} "
        f"(f0 {sr[0]-cb[0]:+.4f} f1 {sr[1]-cb[1]:+.4f}); the tuned config on random labels {ct[2]-cb[2]:+.4f}")

    # final report per set
    log("\nfinal: tuned (both-halves) vs v37, paired bootstrap 2000")
    res = {}
    for name, dd in (("timsTOF", data), ("report", rep)):
        _, p0 = dd.score(tc.V37)
        _, p1 = dd.score(Wb)
        for k in dd.S:
            b = tc.boot(p0[k], p1[k])
            fo = dd.fold[k]
            per_f = []
            for f in (0, 1):
                ii = [i for i in range(len(fo)) if fo[i] == f]
                per_f.append(sum(p1[k][i] - p0[k][i] for i in ii) / len(ii))
            nb = sum(1 for a, c in zip(p0[k], p1[k]) if c > a)
            nw = sum(1 for a, c in zip(p0[k], p1[k]) if c < a)
            log(f"  {k:7s} v37 {sum(p0[k])/len(p0[k]):.4f} tuned {sum(p1[k])/len(p1[k]):.4f} "
                f"d {b[0]:+.4f} [{b[1]:+.4f}, {b[2]:+.4f}]  f0 {per_f[0]:+.4f} f1 {per_f[1]:+.4f}  better {nb} worse {nw}")
            res[k] = dict(v37=sum(p0[k]) / len(p0[k]), tuned=sum(p1[k]) / len(p1[k]), d=b, folds=per_f)
    log(f"tuned weights: {json.dumps(Wb)}")
    log(f"changed: {', '.join(f'{k} {tc.V37[k]} -> {Wb[k]}' for k in Wb if Wb[k] != tc.V37[k])}")
    log(f"phys_v37pc arguments: {opt_string(Wb)}")
    log(f"cross-fit weights f0: {', '.join(f'{k} {Wa[k]}' for k in Wa if Wa[k] != tc.V37[k])}")
    log(f"cross-fit weights f1: {', '.join(f'{k} {Wc[k]}' for k in Wc if Wc[k] != tc.V37[k])}")
    log(f"random-label weights: {', '.join(f'{k} {Wr[k]}' for k in Wr if Wr[k] != tc.V37[k])}")
    json.dump(dict(tuned=Wb, fitf0=Wa, fitf1=Wc, random=Wr, res=res), open(f"{OUT}/tuned.json", "w"), indent=1)
    log(f"time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
