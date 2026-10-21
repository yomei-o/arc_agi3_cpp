"""v35_tune: re-tune the Boltzmann merge (v33, alloc=boltz) on the attention score scale (v34 attn=1, pcrank=rev,
attpool=w) and compare v35 with v33 / v34 (v35_STATUS.md). Uses the exact simulator of py/boltz_merge.py on the
phys_v35pc dumps (<tag>_pp.txt main top 60, <tag>_det.txt PubChem list); the C++ output of the chosen config is
checked against the simulation separately.

usage: python v35_tune.py <dir> [attpool tags...]
  <dir> holds <qry>_mols.txt, the dumps <tag>_pp.txt/_det.txt/_ins.csv of the v34-config runs
  (tags hcap<w>_0, hcap<w>_1, hasap<w>_0, haap2_1, hbsap2_0, hxap2_0; w in 15, 2, 25),
  and the v33 reference CSVs v33_<set>.csv (hc0 hc1 has0 hx0; optional ha1 hbs0).
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import boltz_merge as bm  # noqa: E402

D = sys.argv[1]
MOLS = {"hc0": "hcqry", "hc1": "hcqry", "ha1": "haqry", "has0": "hasqry", "hbs0": "hbsqry", "hx0": "hxqry"}
TAGS = {"hc0": "hcap{w}_0", "hc1": "hcap{w}_1", "has0": "hasap{w}_0", "ha1": "haap2_1", "hbs0": "hbsap2_0",
        "hx0": "hxap2_0"}
MIX = bm.MIX  # 0.5 has + 0.42 hc + 0.08 hc rm=1
SIDE = ("has0", "hbs0", "hx0")


def truth(m):
    return [l.split("\t")[0] for l in open(f"{D}/{m}_mols.txt", encoding="utf-8") if l.strip()]


def csv_rr(path, m):
    c = {}
    for l in open(path, encoding="utf-8").read().splitlines()[1:]:
        k, _, v = l.partition(",")
        c[k] = v.split(";") if v else []
    out = []
    for k in truth(m):
        L = c.get(k, [])[:25]
        out.append(1.0 / (L.index(k) + 1) if k in L else 0.0)
    return out


def mean(v):
    return sum(v) / len(v)


def mixv(R, idx=None):
    t = 0.0
    for k, w in MIX.items():
        ii = idx[k] if idx else range(len(R[k]))
        t += w * sum(R[k][i] for i in ii) / len(ii)
    return t


def configs():
    C = []
    for T1 in (1, 1.5, 2, 3, 4):
        for T2 in (0.5, 0.75, 1, 1.5, 2, 3):
            for pi in (0.05, 0.1, 0.2, 0.3, 0.5):
                for a in (0, 0.25, 0.5, 1.0):
                    C.append((f"T1={T1} T2={T2} pi={pi} a={a}", dict(T1=T1, T2=T2, pi=pi, pimode="relz", a=a)))
    return C


def sim(S, cfg):
    return {k: [bm.rr(q, bm.boltz_list(q, **cfg)) for q in Q] for k, Q in S.items()}


def boot(base, new, keys, B=2000, seed=1):
    rnd = random.Random(seed)
    res = {}
    mixs = []
    samp = {k: [] for k in keys}
    for _ in range(B):
        mv = 0.0
        for k in keys:
            d = [a - b for a, b in zip(new[k], base[k])]
            m = mean([d[rnd.randrange(len(d))] for _ in d])
            samp[k].append(m)
            mv += MIX.get(k, 0) * m
        mixs.append(mv)

    def ci(a):
        a = sorted(a)
        return a[int(0.025 * len(a))], a[int(0.975 * len(a)) - 1]
    for k in keys:
        res[k] = (mean(new[k]) - mean(base[k]),) + ci(samp[k])
    if all(k in keys for k in MIX):
        res["mix"] = (sum(w * (mean(new[k]) - mean(base[k])) for k, w in MIX.items()),) + ci(mixs)
    return res


def fmt(r):
    return f"{r[0]:+.4f} [{r[1]:+.4f}, {r[2]:+.4f}]"


def main():
    ws = sys.argv[2:] or ["2"]
    C = configs()
    for w in ws:
        S, V34 = {}, {}
        for k, t in TAGS.items():
            tag = t.format(w=w)
            if os.path.exists(f"{D}/{tag}_pp.txt"):
                S[k] = bm.load(D, MOLS[k], tag)
                V34[k] = csv_rr(f"{D}/{tag}_ins.csv", MOLS[k])
        V33 = {k: csv_rr(f"{D}/v33_{k}.csv", MOLS[k]) for k in S if os.path.exists(f"{D}/v33_{k}.csv")}
        print(f"\n######## attpool {w}: sets {list(S)}")
        print("v34 (slots):", " ".join(f"{k} {mean(v):.4f}" for k, v in V34.items()), f" mix {mixv(V34):.4f}" if all(k in V34 for k in MIX) else "")
        if V33:
            print("v33        :", " ".join(f"{k} {mean(v):.4f}" for k, v in V33.items()))
        R = {n: sim(S, cfg) for n, cfg in C}
        R["v33set"] = sim(S, dict(T1=2, T2=1, pi=0.2, pimode="relz", a=0.25))

        def ok_side(n, idx):
            for k in SIDE:
                if k in S:
                    ii = idx[k]
                    if sum(R[n][k][i] - V34[k][i] for i in ii) / len(ii) < -0.0005:
                        return False
            return True
        allidx = {k: list(range(len(Q))) for k, Q in S.items()}
        good = sorted((n for n in R if ok_side(n, allidx)), key=lambda n: -mixv(R[n]))
        print("config".ljust(30), " mix   ", " ".join(f"{k:>7}" for k in S))
        print("v34 slots".ljust(30), f"{mixv(V34):.4f}", " ".join(f"{mean(V34[k]):.4f} " for k in S))
        for n in ["v33set"] + good[:25]:
            print(n.ljust(30), f"{mixv(R[n]):.4f}", " ".join(f"{mean(R[n][k]):.4f} " for k in S),
                  "" if ok_side(n, allidx) else "(side-gate FAIL)")
        print("best per pi (side gate, full data):")
        for pi in (0.05, 0.1, 0.2, 0.3, 0.5):
            cc = [n for n in good if f"pi={pi} " in n]
            if cc:
                n = cc[0]
                print("  " + n.ljust(30), f"{mixv(R[n]):.4f}", " ".join(f"{mean(R[n][k]):.4f} " for k in S))
        # 2-fold (seeded split by query; fit by mixture with the side gate on the fit half; test on the other)
        rnd = random.Random(2610)
        halves = {k: [rnd.random() < 0.5 for _ in Q] for k, Q in S.items()}
        print("2-fold:")
        for h in (True, False):
            fit = {k: [i for i, x in enumerate(halves[k]) if x == h] for k in S}
            tst = {k: [i for i, x in enumerate(halves[k]) if x != h] for k in S}
            cands = [n for n, _ in C if ok_side(n, fit)]
            best = max(cands, key=lambda n: mixv(R[n], fit))
            row = f"  half {int(h)} best {best:30s} fit d {mixv(R[best], fit) - mixv(V34, fit):+.4f} test d {mixv(R[best], tst) - mixv(V34, tst):+.4f} |"
            for k in S:
                ii = tst[k]
                row += f" {k} {sum(R[best][k][i] - V34[k][i] for i in ii) / len(ii):+.4f}"
            print(row)
        pick = os.environ.get("V35PICK")
        for n in ([pick] if pick else []) + ["v33set"] + good[:3]:
            if n not in R:
                continue
            print(f"bootstrap {n}: vs v34:", "; ".join(f"{k} {fmt(v)}" for k, v in boot(V34, R[n], list(S)).items()))
            ks = [k for k in S if k in V33]
            if ks:
                print(f"bootstrap {n}: vs v33:", "; ".join(f"{k} {fmt(v)}" for k, v in boot({k: V33[k] for k in ks}, {k: R[n][k] for k in ks}, ks).items()))
        if pick and pick in R:
            od = os.path.join(D, "sim")
            os.makedirs(od, exist_ok=True)
            cfg = dict(C)[pick] if pick in dict(C) else None
            if cfg:
                for k, Q in S.items():
                    bm.write_csv(f"{od}/ap{w}_{k}.csv", Q, [bm.boltz_list(q, **cfg) for q in Q])


if __name__ == "__main__":
    main()
