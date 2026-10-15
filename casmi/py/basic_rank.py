"""Does "basic molecules fragment more" help rank candidates?

approx_test.py found that the mobile-proton features carry between-molecule
information (molecules with a strong basic site show more fragment peaks) but
nothing about which bond breaks. Ranking candidates is a between-molecule
question, so test: candidate basicity b (proton-affinity class of its most
basic site) and cleavable-bond count k, matched against the query's observed
fragment richness r (mean peaks per spectrum below precursor - 1.5).

Score = base (v20 hand fusion + gen2 weights, as gentfm_eval.py) + w * feature,
features: z(b), z(k), z(b)*z(r), z(k)*z(r) (z within the query's candidates;
r z-scored across queries). w picked on cqry, reported on tqry/sqry/oqry, with
the reversed sign as control. MRR@25 with ties averaged.

    python basic_rank.py
"""
import array, struct, sys
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
D = r"C:\prog\casmi"
G = D + r"\gtfo"
GW = np.array([-2, 0, -1, 4, 0, 1.5])

PATS = [(225, "[NX3;H2,H1,H0;!$(N-C=[O,S,N]);!$(N-a);!$(N-[SX4]);!$(N#*);!$(N=*);!$([N]-[N,O])]"),
        (225, "[NX2;!a]=[CX3]"),               # imine / amidine
        (222, "[nX2;r6]"),                     # pyridine-type
        (215, "[nX2;r5]"),                     # imidazole N3
        (210, "[NX3;H2,H1,H0;$(N-a);!$(N-C=O)]"),  # aniline
        (212, "[OX1]=C-[NX3]"),                # amide O
        (198, "[OX1]=C-[OX2]"),                # ester O
        (196, "[OX1]=[CX3]"),                  # ketone/aldehyde O
        (190, "[OX2]")]
PATS = [(v, Chem.MolFromSmarts(s)) for v, s in PATS]


def mol_feats(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return np.nan, np.nan
    b = 180.0
    for v, p in PATS:
        if v > b and m.HasSubstructMatch(p):
            b = v
    ri = m.GetRingInfo()
    k = sum(1 for bd in m.GetBonds() if bd.GetBondType() == Chem.BondType.SINGLE and not ri.NumBondRings(bd.GetIdx())
            and bd.GetBeginAtom().GetAtomicNum() > 1 and bd.GetEndAtom().GetAtomicNum() > 1)
    return b, k


def richness(stem):
    names = [l.split("\t")[0] for l in open(fr"{D}\{stem}_mols.txt", encoding="utf-8")]
    hb = open(fr"{D}\{stem}_head.bin", "rb").read()
    acc = {}
    for i in range(len(hb) // 20):
        p, mol, off, n, tag = struct.unpack_from("<fiiii", hb, 20 * i)
        acc.setdefault(names[mol], []).append(n)
    return {k: np.log1p(np.mean(v)) for k, v in acc.items()}


def load(p, rich):
    Q = []
    f = open(p)
    for line in f:
        h = line.split()
        name, t, n = h[1], int(h[2]), int(h[3])
        rows = [f.readline().split() for _ in range(n)]
        if n == 0:
            Q.append(None); continue
        X = np.array([[float(v) for v in r[:6]] for r in rows])
        Gm = np.array([[float(v) for v in r[7:13]] for r in rows])
        F = np.array([mol_feats(r[-1]) for r in rows])
        Q.append(dict(X=X, G=Gm, F=F, t=t, r=rich.get(name, np.nan)))
    return Q


def z(v):
    v = np.where(np.isnan(v), np.nanmean(v) if np.isfinite(np.nanmean(v)) else 0, v)
    return (v - v.mean()) / (v.std() + 1e-6)


def base(q):
    X = q["X"]
    return z(X[:, 0]) + 0.25 * z(X[:, 1]) + z(X[:, 2]) + 8 * np.clip(X[:, 3], 0, None) ** 32 + 24 * X[:, 4] + q["G"] @ GW


def rr(s, t):
    if t < 0:
        return 0.0
    st = s[t]; better = int((s > st).sum()); ties = int((s == st).sum()) - 1
    return float(np.mean([1.0 / (better + 1 + j) if better + j < 25 else 0.0 for j in range(ties + 1)]))


def feat(q, kind, rmu, rsd):
    b, k = z(q["F"][:, 0]), z(q["F"][:, 1])
    r = 0.0 if np.isnan(q["r"]) else (q["r"] - rmu) / rsd
    return {"b": b, "k": k, "b*r": b * r, "k*r": k * r}[kind]


def mrr(Q, kind, w, rmu, rsd):
    return np.mean([0.0 if q is None else rr(base(q) + (w * feat(q, kind, rmu, rsd) if w else 0), q["t"]) for q in Q])


sets = {}
for lab, f, stem in (("cqry", "f_c_A.txt", "cqry"), ("tqry", "f_t_A.txt", "tqry"),
                     ("sqry", "f_s_A.txt", "sqry"), ("oqry", "f_o_A.txt", "oqry")):
    try:
        sets[lab] = load(fr"{G}\{f}", richness(stem))
        print(lab, "loaded", len(sets[lab]), flush=True)
    except FileNotFoundError as e:
        print(lab, "skipped", e)
allr = np.array([q["r"] for Q in sets.values() for q in Q if q is not None and not np.isnan(q["r"])])
rmu, rsd = allr.mean(), allr.std()
print("base:", {k: round(mrr(Q, "b", 0, rmu, rsd), 4) for k, Q in sets.items()})
W = [-0.6, -0.3, -0.15, 0.15, 0.3, 0.6]
for kind in ("b", "k", "b*r", "k*r"):
    gc = {w: mrr(sets["cqry"], kind, w, rmu, rsd) for w in W}
    wb = max(gc, key=gc.get)
    res = {k: round(mrr(Q, kind, wb, rmu, rsd), 4) for k, Q in sets.items()}
    ctl = {k: round(mrr(Q, kind, -wb, rmu, rsd), 4) for k, Q in sets.items()}
    print(f"{kind:4s} cqry grid {', '.join(f'{w:+.2f}:{v:.4f}' for w, v in gc.items())}")
    print(f"     best w {wb:+.2f}: {res}   reversed: {ctl}", flush=True)
