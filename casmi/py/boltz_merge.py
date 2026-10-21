"""boltz_merge: two-population Boltzmann merge of the main list and the fused PubChem list (pc5).

Offline prototype on phys_v30pc/v32pc dumps (run with ppdet=<tag>_pp.txt, pcdet=<tag>_det.txt, pcrank=fused):
  main list   = the v30 final order (pool popularity 0.15 re-fill, ppgate 0.9, reproduced as pc2_grid.base_list);
                energies s_i = the final score at that rank (the dump's score column sorted descending, so the
                ppop re-fill keeps a monotone profile), top 60.
  PubChem     = the fused list (det rows in order); energies t_j = 'fused' (z over the 50 + 0.15 pop) [tform=z]
                or the main-scale hand fusion zp + 0.4 zm(...) recovered as fused-0.15pop un-normalised [not dumped
                exactly -> tform=z only for fixed-pi; the joint variant uses rev z].
  p_i = (1-pi) softmax(s/T1)_i,  q_j = pi softmax(t/T2)_j ; rank the union by probability; top 25.
  gate: libMax >= 0.9 or empty PubChem list -> the v30 list unchanged (= plain, as in v30).

The v30 inserted list (gated fixed slots + promotion) is reproduced for the baseline (checked against <tag>_ins.csv).

usage: python boltz_merge.py <dumpdir> <outdir>      (runs the whole study, prints tables)
"""
import math
import os
import random
import sys
from collections import defaultdict

SETS = [  # (label, mols, tag)  -- dumps under <dumpdir>
    ("hc0", "hcqry", "hcT0"),
    ("hc1", "hcqry", "hcT1"),
    ("ha1", "haqry", "ha30_1"),
    ("has0", "hasqry", "has30_0"),
    ("hbs0", "hbsqry", "hbs30_0"),
    ("hx0", "hxqry", "hx30_0"),
]
PPW, PPN, PPGATE, GATE = 0.15, 60, 0.9, 0.9


def load(d, mols, tag):
    truth = [l.split("\t")[0] for l in open(f"{d}/{mols}_mols.txt", encoding="utf-8") if l.strip()]
    pp = defaultdict(list)
    for l in open(f"{d}/{tag}_pp.txt", encoding="utf-8"):
        p = l.rstrip("\n").split("\t")
        if len(p) < 6:
            continue
        pp[p[0]].append((int(p[1]), float(p[2]), int(p[3]), float(p[4]), p[5]))
    det, cur = {}, None
    for l in open(f"{d}/{tag}_det.txt", encoding="utf-8"):
        p = l.rstrip("\n").split("\t")
        if p[0] == "Q":
            cur = {"lib": float(p[2]), "pc": [], "v2": None}
            det[p[1]] = cur
        elif p[0] == "V2":
            cur["v2"] = dict(mean=float(p[1]), sd=float(p[2]), relz=float(p[3]), S=float(p[4]), pop=float(p[5]))
        else:
            cur["pc"].append(dict(cid=p[0], truth=p[1] == "1", rev=float(p[2]), pop=float(p[3]), fused=float(p[7]),
                                  zp=float(p[8]), zm=float(p[9]), zr=float(p[10]), ana=float(p[6])))
    Q = []
    for q in truth:
        L = sorted(pp.get(q, []))
        names = [x[4] for x in L]
        D = det.get(q)
        lib = D["lib"] if D else 0.0
        if lib < PPGATE and len(L) > 1:
            idx = [i for i, x in enumerate(L) if i < PPN and x[2] == 1]
            if len(idx) > 1:
                s = [L[i][1] for i in idx]
                m = sum(s) / len(s)
                sd = (sum((v - m) ** 2 for v in s) / len(s)) ** 0.5 + 1e-6
                key = sorted(((L[i][1] - m) / sd + PPW * L[i][3], -k, i) for k, i in enumerate(idx))[::-1]
                for j, i in enumerate(idx):
                    names[i] = L[key[j][2]][4]
        sc = sorted((x[1] for x in L), reverse=True)
        pcl = D["pc"] if D else []
        Q.append(dict(name=q, main=names, s=sc, lib=lib, v2=D["v2"] if D else None,
                      pcn=[q if e["truth"] else "P" + e["cid"] for e in pcl],
                      t=[e["fused"] for e in pcl], rev=[e["rev"] for e in pcl], pop=[e["pop"] for e in pcl]))
    return Q


def v30_list(q):
    L = q["main"][:25]
    if not (q["lib"] < GATE) or not q["pcn"] or q["v2"] is None:
        return L
    v = q["v2"]
    hi = v["relz"] > 6
    sl = set((2, 4, 6, 8, 10) if hi else (4, 8, 12, 16, 20))
    picks = q["pcn"][:len(sl)]
    out, a, b = [], 0, 0
    for pos in range(1, 26):
        if pos in sl and b < len(picks):
            out.append(picks[b]); b += 1
        elif a < len(L):
            out.append(L[a]); a += 1
        elif b < len(picks):
            out.append(picks[b]); b += 1
        else:
            break
    prom = v["relz"] > 3 or (v["S"] > 6 and q["pop"][0] >= 5)
    if prom and picks:
        out = [picks[0]] + [x for x in out if x != picks[0]]
        out = out[:25]
    return out


def lse(v):
    m = max(v)
    return m + math.log(sum(math.exp(x - m) for x in v))


def boltz_list(q, T1, T2, pi, pimode="fixed", a=0.0, b=0.0, smode="raw", K1=60):
    """pimode fixed: prior pi. pimode='lib': logit(pi) = logit(pi0) + a*(0.5 - libMax)*... see below."""
    L = q["main"]
    if not (q["lib"] < GATE) or not q["pcn"] or not L:
        return L[:25]
    s = q["s"][:min(K1, len(L))]
    names = L[:len(s)]
    if smode == "z":  # per-query standardised main scores (spread-free: robust to an extra FPNet term)
        m = sum(s) / len(s)
        sd = (sum((x - m) ** 2 for x in s) / len(s)) ** 0.5 + 1e-6
        s = [(x - s[0]) / sd for x in s]
    t = q["t"]
    e1 = [x / T1 for x in s]
    e2 = [x / T2 for x in t]
    l1, l2 = lse(e1), lse(e2)
    if pimode == "fixed":
        lp = math.log(pi / (1 - pi))
    elif pimode == "gap":  # pi falls with the main list's top margin (best pool candidate confident -> answer in pool)
        g = (q["s"][0] - q["s"][1]) if len(q["s"]) > 1 else 0.0
        lp = math.log(pi / (1 - pi)) - a * g
    elif pimode == "lib":  # pi falls with the best library similarity
        lp = math.log(pi / (1 - pi)) - a * q["lib"]
    elif pimode == "relz":  # pi rises when the window's best non-pool REV2 score beats the best pool candidate's
        r = q["v2"]["relz"] if q["v2"] and abs(q["v2"]["relz"]) < 1e8 else 0.0
        lp = math.log(pi / (1 - pi)) + a * r
    else:
        raise ValueError(pimode)
    # log prob: main  log(1-pi) + e1 - l1 ; pubchem log(pi) + e2 - l2 ; compare with logit
    items = [(x - l1, n) for x, n in zip(e1, names)] + [(x - l2 + lp, n) for x, n in zip(e2, q["pcn"])]
    items.sort(key=lambda z: -z[0])
    out = []
    for _, n in items:
        out.append(n)
        if len(out) == 25:
            break
    return out


def rr(q, L):
    return 1.0 / (L.index(q["name"]) + 1) if q["name"] in L[:25] else 0.0


def write_csv(path, Q, lists):
    with open(path, "w", encoding="utf-8") as f:
        f.write("molecule_id,smiles\n")
        for q, L in zip(Q, lists):
            f.write(q["name"] + "," + ";".join(L) + "\n")


# ---------------------------------------------------------------- study
IN_W = {"has0": 0.5, "hc0": 0.42}  # yardstick mixture: 0.5 hasqry + 0.42 hcqry + 0.08 hcqry-rm
MIX = {"has0": 0.5, "hc0": 0.42, "hc1": 0.08}
SIDE = ("has0", "hbs0", "hx0")  # sibling / xinstr: gate = no loss


def per_query(Q, f):
    return [rr(q, f(q)) for q in Q]


def grid_configs():
    C = [("v30", None)]
    for smode in ("raw", "z"):
        for T1 in (1, 1.5, 2, 3, 4, 6):
            for T2 in (0.5, 0.75, 1, 1.5, 2):
                for pi in (0.05, 0.1, 0.2, 0.3, 0.5):
                    C.append((f"fixed {smode} T1={T1} T2={T2} pi={pi}", dict(T1=T1, T2=T2, pi=pi, smode=smode)))
    for mode, As in (("gap", (0.25, 0.5, 1.0)), ("lib", (1.0, 2.0, 4.0)), ("relz", (0.25, 0.5, 1.0))):
        for T1 in (1.5, 2, 3):
            for T2 in (0.75, 1, 1.5):
                for pi in (0.1, 0.2, 0.3, 0.5):
                    for a in As:
                        C.append((f"{mode} raw T1={T1} T2={T2} pi={pi} a={a}", dict(T1=T1, T2=T2, pi=pi, pimode=mode, a=a)))
    return C


def func(cfg):
    if cfg is None:
        return v30_list
    return lambda q: boltz_list(q, **cfg)


def mixval(rows, idx=None):
    t = 0.0
    for k, w in MIX.items():
        v = rows[k]
        ii = idx[k] if idx else range(len(v))
        t += w * sum(v[i] for i in ii) / len(ii)
    return t


def boot(dlt, B=2000, seed=1):
    """dlt: {set: per-query delta list}; returns (mean, lo, hi) of the mixture delta and per set."""
    rnd = random.Random(seed)
    out = {}
    keys = list(dlt)
    samp = {k: [] for k in keys}
    mixs = []
    for _ in range(B):
        mv = 0.0
        for k in keys:
            v = dlt[k]
            m = sum(v[rnd.randrange(len(v))] for _ in range(len(v))) / len(v)
            samp[k].append(m)
            mv += MIX.get(k, 0) * m
        mixs.append(mv)
    def ci(a):
        a = sorted(a)
        return a[int(0.025 * len(a))], a[int(0.975 * len(a)) - 1]
    for k in keys:
        out[k] = (sum(dlt[k]) / len(dlt[k]),) + ci(samp[k])
    out["mix"] = (sum(MIX.get(k, 0) * sum(dlt[k]) / len(dlt[k]) for k in keys),) + ci(mixs)
    return out


def main():
    d, od = sys.argv[1], sys.argv[2]
    os.makedirs(od, exist_ok=True)
    S = {}
    for lab, m, t in SETS:
        if os.path.exists(f"{d}/{t}_pp.txt"):
            S[lab] = load(d, m, t)
    print("sets:", {k: len(v) for k, v in S.items()})
    C = grid_configs()
    R = {}  # name -> {set: per-query rr}
    for name, cfg in C:
        f = func(cfg)
        R[name] = {k: per_query(Q, f) for k, Q in S.items()}
    base = R["v30"]
    def mean(v): return sum(v) / len(v)
    # 2-fold split by query (seeded), fit on one half by the mixture, test on the other; both directions
    rnd = random.Random(2610)
    halves = {k: [rnd.random() < 0.5 for _ in Q] for k, Q in S.items()}
    def ok_side(name, idx):  # sibling/xinstr gate on the fitting half: no loss vs v30 (>= -0.0005)
        for k in SIDE:
            if k in S:
                ii = idx[k]
                if sum(R[name][k][i] - base[k][i] for i in ii) / len(ii) < -0.0005:
                    return False
        return True
    print("\n== 2-fold: fit on half (mixture, side gate), test on the other half ==")
    for h in (True, False):
        fit = {k: [i for i, x in enumerate(halves[k]) if x == h] for k in S}
        tst = {k: [i for i, x in enumerate(halves[k]) if x != h] for k in S}
        cands = [n for n, _ in C if n != "v30" and ok_side(n, fit)]
        for fam in ("fixed raw", "fixed z", "gap", "lib", "relz"):
            cc = [n for n in cands if n.startswith(fam)]
            if not cc:
                continue
            best = max(cc, key=lambda n: mixval(R[n], fit))
            row = f"  half {int(h)} {fam:9s} best {best:42s} fit d {mixval(R[best], fit) - mixval(base, fit):+.4f}  test d {mixval(R[best], tst) - mixval(base, tst):+.4f}  |"
            for k in S:
                ii = tst[k]
                row += f" {k} {sum(R[best][k][i] - base[k][i] for i in ii) / len(ii):+.4f}"
            print(row)
    # full-data ranking
    print("\n== full data: top configs by mixture (side gate on all queries) ==")
    allidx = {k: list(range(len(Q))) for k, Q in S.items()}
    good = [n for n, _ in C if n != "v30" and ok_side(n, allidx)]
    good.sort(key=lambda n: -mixval(R[n]))
    hdr = "config".ljust(44) + "  mix   " + " ".join(f"{k:>7}" for k in S)
    print(hdr)
    print("v30".ljust(44) + f" {mixval(base):.4f} " + " ".join(f"{mean(base[k]):.4f} " for k in S))
    for n in good[:30]:
        print(n.ljust(44) + f" {mixval(R[n]):.4f} " + " ".join(f"{mean(R[n][k]):.4f} " for k in S))
    print("\n== best per pi (fixed raw, side gate) ==")
    for pi in (0.05, 0.1, 0.2, 0.3, 0.5):
        cc = [n for n in good if n.startswith("fixed raw") and n.endswith(f"pi={pi}")]
        if cc:
            n = max(cc, key=lambda n: mixval(R[n]))
            print(n.ljust(44) + f" {mixval(R[n]):.4f} " + " ".join(f"{mean(R[n][k]):.4f} " for k in S))
    return S, C, R, good


if __name__ == "__main__":
    main()
