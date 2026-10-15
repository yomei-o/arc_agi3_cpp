"""Battery of cheap physical approximations for CID cleavage, one yardstick.

Data: huckel_fqry2.json (fqry [M+H]+, acyclic single-bond cuts, observed
intensity of each side's even-electron cation, +0.3 Da chance control).

Baseline features per cut: bond class (element pair x aromatic/carbonyl
attachment) + size bin of the smaller piece (heavy atoms). Each approximation
adds a few numeric features; we report 5-fold (by molecule) held-out
log-likelihood gain per cut and within-molecule AUC, for the real labels and
for the chance-control labels (any "gain" there is an artifact).

Approximations (all on the heavy-atom graph, implicit H):
  pa     : proton affinity table -> most basic site s; graph distance d(s,bond),
           exp(-d/lambda) (mobile proton as a first-order IIR in distance)
  resist : Kirchhoff resistance distance from s to the bond (conductance =
           bond order), and the current through the bond when 1 unit is
           injected at s and drained uniformly (proton spreading)
  eem    : electronegativity equalization (capacitor network), total charge +1;
           charge of each side, |q_a - q_b| bond polarity
  estate : Kier-Hall E-state, field-like 1/(r+1)^2 propagation; S_a, S_b, |diff|
Charge-side test: cuts where exactly one side is observed; which side keeps the
charge: larger piece / side with the most basic site / side with larger EEM charge.

    python approx_test.py huckel_fqry2.json
"""
import json, sys
import numpy as np

data = json.load(open(sys.argv[1]))
PQN = {"C": 2, "N": 2, "O": 2, "F": 2, "S": 3, "P": 3, "Cl": 3, "Br": 4, "I": 5, "B": 2, "Si": 3, "Se": 4}
ZV = {"C": 4, "N": 5, "O": 6, "F": 7, "S": 6, "P": 5, "Cl": 7, "Br": 7, "I": 7, "B": 3, "Si": 4, "Se": 6}
# EEM (eV): chi, hardness; crude, Mortier/Gasteiger-like magnitudes
CHI = {"C": 5.3, "N": 7.0, "O": 8.5, "S": 6.9, "P": 5.6, "F": 10.4, "Cl": 8.3, "Br": 7.6, "I": 6.8}
ETA = {"C": 5.0, "N": 6.0, "O": 6.5, "S": 4.2, "P": 4.9, "F": 7.0, "Cl": 4.7, "Br": 4.2, "I": 3.7}


def graph(mol):
    n = len(mol["atoms"])
    nb = [[] for _ in range(n)]
    for b in mol["bonds"]:
        o = 1.5 if b["ar"] else b["o"]
        nb[b["i"]].append((b["j"], o)); nb[b["j"]].append((b["i"], o))
    D = np.full((n, n), 99, int)
    for s in range(n):
        D[s, s] = 0; q = [s]
        for u in q:
            for v, _ in nb[u]:
                if D[s, v] == 99:
                    D[s, v] = D[s, u] + 1; q.append(v)
    return n, nb, D


def pa_site(mol, nb):
    """proton affinity estimate (kcal/mol) per atom; ~ literature values of model compounds"""
    at = mol["atoms"]
    pa = np.full(len(at), 150.0)
    for i, a in enumerate(at):
        el = a["el"]
        dbo = any(o == 2 for _, o in nb[i])
        nbr_carbonyl = any(any(o2 == 2 and at[k]["el"] == "O" for k, o2 in nb[j]) for j, _ in nb[i])
        nbr_arom = any(at[j]["ar"] for j, _ in nb[i])
        if el == "N":
            if a["ar"]:
                pa[i] = 222 if a["h"] == 0 else 200          # pyridine / pyrrole
            elif dbo:
                pa[i] = 225                                   # imine / amidine
            elif nbr_carbonyl:
                pa[i] = 195                                   # amide N (O protonates)
            elif nbr_arom:
                pa[i] = 210                                   # aniline
            else:
                pa[i] = 225                                   # aliphatic amine
        elif el == "O":
            if dbo:
                j = [j for j, o in nb[i] if o == 2][0]
                amide = any(at[k]["el"] == "N" for k, _ in nb[j])
                ester = sum(at[k]["el"] == "O" for k, _ in nb[j]) >= 2
                pa[i] = 212 if amide else (198 if ester else 196)
            elif a["ar"]:
                pa[i] = 190                                   # furan O
            else:
                pa[i] = 192 if a["h"] == 0 else 188           # ether / alcohol
        elif el == "S":
            pa[i] = 200
        elif el == "C" and a["ar"]:
            pa[i] = 180
    return pa


def laplacian_pinv(n, nb):
    L = np.zeros((n, n))
    for u in range(n):
        for v, o in nb[u]:
            L[u, v] -= o; L[u, u] += o
    return np.linalg.pinv(L)


def eem(mol, D, total=1.0):
    at = mol["atoms"]; n = len(at)
    A = np.zeros((n + 1, n + 1)); rhs = np.zeros(n + 1)
    r = 1.45 * np.maximum(D, 1).astype(float)
    for i, a in enumerate(at):
        el = a["el"]
        chi = CHI.get(el, 6.0) - 0.15 * a["h"] * 0  # H implicit, ignored
        if a["ar"] or a["hyb"] == "SP2":
            chi += 0.3
        A[i, :n] = 14.4 / r[i]
        A[i, i] = 2 * ETA.get(el, 5.0)
        A[i, n] = -1.0
        rhs[i] = -chi
    A[n, :n] = 1.0; rhs[n] = total
    return np.linalg.solve(A, rhs)[:n]


def estate(mol, nb, D):
    at = mol["atoms"]; n = len(at)
    I = np.zeros(n)
    for i, a in enumerate(at):
        d = len(nb[i]); dv = ZV.get(a["el"], 4) - a["h"]
        N = PQN.get(a["el"], 2)
        I[i] = ((2.0 / N) ** 2 * dv + 1) / max(d, 1)
    S = I.copy()
    for i in range(n):
        for j in range(n):
            if i != j and D[i, j] < 99:
                S[i] += (I[i] - I[j]) / (D[i, j] + 1) ** 2
    return S


def bond_class(mol, nb, a, b):
    at = mol["atoms"]
    def tag(i):
        e = at[i]["el"]
        carb = any(o == 2 and at[j]["el"] == "O" for j, o in nb[i])
        return e + ("ar" if at[i]["ar"] else "") + ("CO" if carb else "")
    return "|".join(sorted([tag(a), tag(b)]))


def size_bin(k):
    return 0 if k <= 1 else 1 if k == 2 else 2 if k <= 4 else 3 if k <= 8 else 4 if k <= 15 else 5


rows = []   # (mol, class, sizebin, feats..., yreal, yctrl)
side = []   # (mol, larger_is_A, basic_in_A, eemA_gt_B, obsA)
FEAT = ["pa_exp1", "pa_exp2", "pa_dist", "R_s", "I_flow", "eem_side_max", "eem_pol", "es_a", "es_b", "es_diff", "pa_max_side"]
for mi, mol in enumerate(data):
    n, nb, D = graph(mol)
    if n < 3:
        continue
    pa = pa_site(mol, nb)
    s = int(np.argmax(pa))
    Lp = laplacian_pinv(n, nb)
    R = np.diag(Lp)[:, None] + np.diag(Lp)[None, :] - 2 * Lp
    inj = -np.ones(n) / n; inj[s] += 1.0
    V = Lp @ inj
    q = eem(mol, D)
    S = estate(mol, nb, D)
    for c in mol["cuts"]:
        bd = mol["bonds"][c["b"]]
        a, b = bd["i"], bd["j"]
        A, B = set(c["A"]), set(c["B"])
        if a not in A:
            a, b = b, a
        d = min(D[s, a], D[s, b])
        qA, qB = sum(q[i] for i in A), sum(q[i] for i in B)
        paA, paB = max(pa[i] for i in A), max(pa[i] for i in B)
        f = [np.exp(-d / 1.0), np.exp(-d / 2.0), min(d, 12), 0.5 * (R[s, a] + R[s, b]),
             abs(V[a] - V[b]), max(qA, qB), abs(q[a] - q[b]), max(S[a], S[b]), min(S[a], S[b]),
             abs(S[a] - S[b]), (max(paA, paB) - 200) / 10]
        yr = int(max(c["oA"], c["oB"]) > 0); yc = int(max(c["cA"], c["cB"]) > 0)
        rows.append((mi, bond_class(mol, nb, a, b), size_bin(min(len(A), len(B))), f, yr, yc))
        oa, ob = c["oA"] > 0, c["oB"] > 0
        if oa != ob:
            side.append((len(A) > len(B), paA > paB if paA != paB else None, qA > qB, oa, paA, paB))

print(f"molecules {len(data)}  cuts {len(rows)}  observed {np.mean([r[4] for r in rows]):.3f}  chance {np.mean([r[5] for r in rows]):.3f}")

# ---------- charge side
sd = side
print(f"\ncharge side (n={len(sd)} cuts with exactly one side observed)")
print(f"  larger piece          {100*np.mean([l == o for l, _, _, o, *_ in sd]):.1f}%")
dec = [(p, o) for _, p, _, o, *_ in sd if p is not None]
print(f"  most basic site side  {100*np.mean([p == o for p, o in dec]):.1f}%  (decided on {len(dec)})")
print(f"  larger EEM charge     {100*np.mean([e == o for _, _, e, o, *_ in sd]):.1f}%")
comb = [(p if p is not None and abs(pa - pb) >= 10 else l) == o for l, p, _, o, pa, pb in sd]
print(f"  basic site if dPA>=10 else larger piece  {100*np.mean(comb):.1f}%")

# ---------- which bonds break: logistic regression with per-feature gain
classes = sorted({r[1] for r in rows})
cid = {c: i for i, c in enumerate(classes)}
mols = np.array([r[0] for r in rows])
Xb = np.zeros((len(rows), len(classes) + 6))
for k, r in enumerate(rows):
    Xb[k, cid[r[1]]] = 1; Xb[k, len(classes) + r[2]] = 1
F = np.array([r[3] for r in rows], float)
F = (F - F.mean(0)) / (F.std(0) + 1e-9)
Yr = np.array([r[4] for r in rows], float); Yc = np.array([r[5] for r in rows], float)


def fit(X, y, l2=1.0, it=30):
    w = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return w


def cv(X, y, folds=5):
    um = np.unique(mols); rng = np.random.default_rng(0); rng.shuffle(um)
    fold = {m: i % folds for i, m in enumerate(um)}
    fo = np.array([fold[m] for m in mols])
    ll = 0.0; pred = np.zeros(len(y))
    for k in range(folds):
        tr, te = fo != k, fo == k
        w = fit(X[tr], y[tr])
        p = np.clip(1 / (1 + np.exp(-X[te] @ w)), 1e-6, 1 - 1e-6)
        ll += np.sum(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p)); pred[te] = p
    return ll / len(y), pred


def mol_auc(score, y):
    out = []
    for m in np.unique(mols):
        idx = mols == m
        P, N = score[idx & (y == 1)], score[idx & (y == 0)]
        if len(P) and len(N):
            out.append(np.mean((P[:, None] > N[None, :]) + 0.5 * (P[:, None] == N[None, :])))
    return np.mean(out)


ones = np.ones((len(rows), 1))
for lab, Y in (("REAL", Yr), ("CHANCE +0.3Da", Yc)):
    base_ll, base_p = cv(np.hstack([ones, Xb]), Y)
    print(f"\n[{lab}] baseline (bond class + size bin): ll/cut {base_ll:.4f}  within-mol AUC {mol_auc(base_p, Y):.3f}")
    groups = {"pa (exp d/1, d/2, d, PA max)": [0, 1, 2, 10], "resistance + current": [3, 4],
              "EEM charges": [5, 6], "E-state": [7, 8, 9], "ALL": list(range(len(FEAT)))}
    for name, cols in groups.items():
        ll, p = cv(np.hstack([ones, Xb, F[:, cols]]), Y)
        print(f"   + {name:28s} gain {1000*(ll-base_ll):+7.2f} mnat/cut   AUC {mol_auc(p, Y):.3f}")
    if lab == "REAL":
        for j, nm in enumerate(FEAT):
            print(f"      raw single {nm:14s} within-mol AUC {mol_auc(F[:, j], Y):.3f}")

# ---------- within-molecule only: centre features per molecule (removes between-molecule effects)
Fc = F.copy()
for m in np.unique(mols):
    idx = mols == m
    Fc[idx] -= Fc[idx].mean(0)
Fc /= Fc.std(0) + 1e-9
# continuous size controls: log heavy atoms of smaller / larger piece, centred too
print("\n[REAL, features centred per molecule]")
base_ll, base_p = cv(np.hstack([ones, Xb]), Yr)
for name, cols in {"pa": [0, 1, 2, 10], "resistance": [3], "current": [4], "EEM": [5, 6], "E-state": [7, 8, 9], "ALL": list(range(len(FEAT)))}.items():
    ll, p = cv(np.hstack([ones, Xb, Fc[:, cols]]), Yr)
    print(f"   + {name:12s} gain {1000*(ll-base_ll):+7.2f} mnat/cut   AUC {mol_auc(p, Yr):.3f}")
