"""tune38_core: exact offline re-scoring of the phys_v37pc final list from phys_v38td every-candidate term dumps.

tdump (<tag>_td.txt) columns, one line per candidate (x = term absent for this query):
  name idx isPool pop candname phys mlp rev lib ana g0..g5 att cnn ft fp final smiles
The C++ fusion (phys_v37pc `fused` + the FPNet stage), reproduced term by term:
  f = wp z(phys) + wm z(mlp) + wr z(rev) + wa z(att) + wc z(cnn) + wt z(ft) + wl lib^32 + wn ana + sum_k g_k gF_k
      [+ wf z(fp) when the query has FPNet scores]                     (z over the query's candidates, as C++ zscore)
then the tail as py/boltz_merge.py: top 60 (desc score, idx) -> pool-popularity re-fill of the pool candidates' places
(z(score over them) + ppop * pop, only when libMax < 0.9) -> Boltzmann merge with the PubChem list of <tag>_det.txt
(p_i = (1-pi) softmax(s/T1), q_j = pi softmax(t/T2), logit(pi) += a relz; libMax >= 0.9 -> main list) -> top 25.
The PubChem list (det) does not depend on the hand weights (pcrank=rev, relz from REV2), so it is kept fixed.
"""
import math
import random
from collections import defaultdict

import numpy as np

TERMS = ["phys", "mlp", "rev", "att", "cnn", "ft", "lib32", "ana", "g0", "g1", "g2", "g3", "g4", "g5", "fp"]
# v37nofp production weights (notebook cell 8 + phys_v37pc defaults)
V37 = dict(phys=1.0, mlp=0.25, rev=1.0, att=2.0, cnn=2.0, ft=2.0, lib32=8.0, ana=24.0,
           g0=-2.0, g1=0.0, g2=-1.0, g3=4.0, g4=0.0, g5=1.5, fp=1.0,
           ppop=0.15, T1=3.0, T2=0.5, pi=0.1, a=0.5)
PPN, PPGATE, GATE, K1 = 60, 0.9, 0.9, 60


def zs(v):
    v = np.asarray(v, dtype=np.float64)
    m = v.mean()
    s = math.sqrt(((v - m) ** 2).mean()) + 1e-6
    return (v - m) / s


def load_det(path):
    det, cur = {}, None
    for l in open(path, encoding="utf-8"):
        p = l.rstrip("\n").split("\t")
        if p[0] == "Q":
            cur = {"lib": float(p[2]), "pc": [], "v2": None}
            det[p[1]] = cur
        elif p[0] == "V2":
            cur["v2"] = dict(relz=float(p[3]))
        else:
            cur["pc"].append((p[0], p[1] == "1", float(p[7])))
    return det


def load(dirp, tag, molsfile):
    """-> list of query dicts (order of <qry>_mols.txt); queries without a dump row keep an empty main list."""
    truth = [l.split("\t")[0] for l in open(molsfile, encoding="utf-8") if l.strip()]
    rows = defaultdict(list)
    for l in open(f"{dirp}/{tag}_td.txt", encoding="utf-8"):
        p = l.rstrip("\n").split("\t")
        if len(p) < 22:
            continue
        rows[p[0]].append(p)
    det = load_det(f"{dirp}/{tag}_det.txt")
    Q = []
    for q in truth:
        R = rows.get(q, [])
        D = det.get(q)
        d = dict(name=q, n=len(R), lib=D["lib"] if D else 0.0)
        pcl = D["pc"] if D else []
        d["pcn"] = [q if tr else "P" + cid for cid, tr, _ in pcl]
        d["t"] = np.array([t for _, _, t in pcl], dtype=np.float64)
        r = D["v2"]["relz"] if D and D["v2"] and abs(D["v2"]["relz"]) < 1e8 else 0.0
        d["relz"] = r
        if R:
            col = lambda k: [x[k] for x in R]  # noqa: E731
            d["idx"] = np.array([int(x) for x in col(1)])
            d["isPool"] = np.array([int(x) for x in col(2)])
            d["pop"] = np.array([float(x) for x in col(3)])
            d["cname"] = col(4)
            d["smiles"] = col(21)
            d["final_cpp"] = np.array([float(x) for x in col(20)])
            Z = np.zeros((len(R), len(TERMS)))
            for j, (k, c) in enumerate((("phys", 5), ("mlp", 6), ("rev", 7))):
                if R[0][c] != "x":
                    Z[:, j] = zs([float(x) for x in col(c)])
            lib = np.array([float(x) if x != "x" else 0.0 for x in col(8)])
            ana = np.array([float(x) if x != "x" else 0.0 for x in col(9)])
            Z[:, TERMS.index("lib32")] = lib ** 32
            Z[:, TERMS.index("ana")] = ana
            for k in range(6):
                Z[:, TERMS.index(f"g{k}")] = [float(x) for x in col(10 + k)]
            for k, c in (("att", 16), ("cnn", 17), ("ft", 18), ("fp", 19)):
                if R[0][c] != "x":
                    Z[:, TERMS.index(k)] = zs([float(x) for x in col(c)])
            d["Z"] = Z
            d["has"] = {k: R[0][c] != "x" for k, c in (("att", 16), ("cnn", 17), ("ft", 18), ("fp", 19))}
            d["lib_raw"] = lib
        Q.append(d)
    return Q


def wvec(W):
    return np.array([W[k] for k in TERMS], dtype=np.float64)


def scores(q, W):
    return q["Z"] @ wvec(W)


def main_list(q, sc, ppop):
    """top 60 by (desc score, idx) + pool popularity re-fill; returns (names, sorted scores)."""
    o = np.lexsort((q["idx"], -sc))[:max(PPN, 25)]
    names = [q["cname"][i] for i in o]
    s = sc[o]
    if q["lib"] < PPGATE and len(o) > 1 and ppop is not None:
        ip = [j for j, i in enumerate(o) if j < PPN and q["isPool"][i] == 1]
        if len(ip) > 1:
            v = np.array([s[j] for j in ip])
            m = v.mean()
            sd = math.sqrt(((v - m) ** 2).mean()) + 1e-6
            key = sorted((((s[j] - m) / sd + ppop * q["pop"][o[j]]), -k, j) for k, j in enumerate(ip))[::-1]
            nn = list(names)
            for jj, j in enumerate(ip):
                nn[j] = names[key[jj][2]]
            names = nn
    return names, s


def lse(v):
    m = v.max()
    return m + math.log(np.exp(v - m).sum())


def final_list(q, W, sc=None):
    if q["n"] == 0:
        return [], []
    if sc is None:
        sc = scores(q, W)
    names, s = main_list(q, sc, W["ppop"])
    if not (q["lib"] < GATE) or not q["pcn"]:
        return names[:25], names[:25]
    s = s[:min(K1, len(names))]
    nm = names[:len(s)]
    e1 = s / W["T1"]
    e2 = q["t"] / W["T2"]
    l1, l2 = lse(e1), lse(e2)
    lp = math.log(W["pi"] / (1 - W["pi"])) + W["a"] * q["relz"]
    items = [(x - l1, n) for x, n in zip(e1, nm)] + [(x - l2 + lp, n) for x, n in zip(e2, q["pcn"])]
    items.sort(key=lambda z: -z[0])
    return names[:25], [n for _, n in items[:25]]


def rr(q, L):
    return 1.0 / (L.index(q["name"]) + 1) if q["name"] in L[:25] else 0.0


def eval_set(Q, W, idx=None, label=None):
    """per-query rr of the final list (label: optional replacement truth names)."""
    out = []
    for k, q in enumerate(Q):
        if idx is not None and k not in idx:
            continue
        _, F = final_list(q, W)
        nm = label[k] if label is not None else q["name"]
        out.append(1.0 / (F.index(nm) + 1) if nm in F[:25] else 0.0)
    return out


def boot(base, new, B=2000, seed=1):
    rnd = random.Random(seed)
    d = [a - b for a, b in zip(new, base)]
    n = len(d)
    ms = sorted(sum(d[rnd.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return sum(d) / n, ms[int(0.025 * B)], ms[int(0.975 * B) - 1]


def csv_lists(path):
    c = {}
    for l in open(path, encoding="utf-8").read().splitlines()[1:]:
        k, _, v = l.partition(",")
        c[k] = v.split(";") if v else []
    return c
