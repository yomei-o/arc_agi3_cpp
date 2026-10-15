"""Regress xTB cleavage energies and proton affinities on cheap graph descriptors.

    python qm_fit.py <prefix> [prefix2 ...]

Reads <prefix>_{mol,sites,cuts}.csv from qm2.py. Every descriptor here is a
plain graph rule (element, aromatic flag, H count, neighbours, ring flag), so
the same formula is computed in C++ (phys_qm.cpp) on rdkit-min.
"""
import csv, math, os, sys, zlib
from collections import defaultdict

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
H = 627.509
SINGLE, DOUBLE = Chem.BondType.SINGLE, Chem.BondType.DOUBLE


def has_double_o(m, c):
    return any(b.GetBondType() == DOUBLE and b.GetOtherAtom(c).GetAtomicNum() == 8 for b in c.GetBonds())


def anomeric(m, c):
    if c.GetAtomicNum() != 6 or c.GetIsAromatic():
        return False
    n_o = sum(1 for b in c.GetBonds() if b.GetOtherAtom(c).GetAtomicNum() == 8 and b.GetBondType() == SINGLE)
    return n_o >= 2 and m.GetRingInfo().NumAtomRings(c.GetIdx()) > 0


def c_anomeric(m, c):
    ri = m.GetRingInfo()
    if c.GetAtomicNum() != 6 or c.GetIsAromatic() or ri.NumAtomRings(c.GetIdx()) == 0:
        return False
    n_o = n_r = 0
    for b in c.GetBonds():
        if b.GetOtherAtom(c).GetAtomicNum() == 8 and b.GetBondType() == SINGLE:
            n_o += 1
            n_r += ri.NumBondRings(b.GetIdx()) > 0
    return n_o == 1 and n_r == 1


CLS = ["intact", "acyl", "glycosidic", "C-het", "aryl-C", "C-C", "ring", "ring-ar", "other", "C-glyc"]


def bond_class(m, b):
    x, y = b.GetBeginAtom(), b.GetEndAtom()
    if b.GetBondType() != SINGLE:
        return 8
    zx, zy = x.GetAtomicNum(), y.GetAtomicNum()
    if zx == 6 and zy == 6:
        if (c_anomeric(m, x) and y.GetIsAromatic()) or (c_anomeric(m, y) and x.GetIsAromatic()):
            return 9
        return 4 if (x.GetIsAromatic() or y.GetIsAromatic()) else 5
    c = x if zx == 6 else (y if zy == 6 else None)
    if c is None:
        return 8
    h = y if c is x else x
    zh = h.GetAtomicNum()
    if zh in (7, 8) and has_double_o(m, c):
        return 1
    if zh == 8 and anomeric(m, c):
        return 2
    if zh in (7, 8, 16):
        return 3
    return 8


# ---- directional cut descriptors: A (cut atom a) keeps the charge, B (cut atom b) leaves with an H on b

def lone_pair_nb(m, a, skip):
    """single-bonded O/N neighbours of a (other than skip): oxocarbenium / iminium donors"""
    nO = nN = 0
    for bd in a.GetBonds():
        o = bd.GetOtherAtom(a)
        if o.GetIdx() == skip or bd.GetBondType() != SINGLE or o.GetIsAromatic():
            continue
        if o.GetAtomicNum() == 8:
            nO += 1
        elif o.GetAtomicNum() == 7 and not has_double_o(m, o):
            nN += 1
    return nO, nN


def pi_nb(m, a, skip):
    """aromatic neighbour (benzylic), C=C neighbour (allylic), C=O neighbour (alpha-carbonyl)"""
    ar = al = co = 0
    for bd in a.GetBonds():
        o = bd.GetOtherAtom(a)
        if o.GetIdx() == skip or o.GetAtomicNum() == 1:
            continue
        if o.GetIsAromatic():
            ar = 1
        elif o.GetAtomicNum() == 6:
            for b2 in o.GetBonds():
                if b2.GetBondType() == DOUBLE:
                    if b2.GetOtherAtom(o).GetAtomicNum() == 6:
                        al = 1
                    elif b2.GetOtherAtom(o).GetAtomicNum() == 8:
                        co = 1
    return ar, al, co


FEAT = ["1", "a_N", "a_O", "a_S", "a_ar", "a_acyl", "a_deg0", "a_deg2", "a_deg3", "a_alphaO", "a_alphaN",
        "a_benzyl", "a_allyl", "a_alphaCO", "a_ring", "a_dblC",
        "b_N", "b_O", "b_S", "b_ar", "b_CO", "paB", "paMax", "lnA", "lnB"]


PA_CONST = None  # set once the PA table is fitted


def dfeat(m, a_idx, b_idx, nA, nB, pamax=0.0):
    a, b = m.GetAtomWithIdx(a_idx), m.GetAtomWithIdx(b_idx)
    za, zb = a.GetAtomicNum(), b.GetAtomicNum()
    deg = sum(1 for o in a.GetNeighbors() if o.GetIdx() != b_idx)  # heavy neighbours left on the cation
    nO, nN = lone_pair_nb(m, a, b_idx)
    ar, al, co = pi_nb(m, a, b_idx)
    dblC = any(bd.GetBondType() == DOUBLE and bd.GetOtherAtom(a).GetAtomicNum() == 6 for bd in a.GetBonds())
    f = [1, za == 7, za == 8, za == 16, a.GetIsAromatic(), za == 6 and has_double_o(m, a),
         deg == 0, deg == 2, deg >= 3, min(nO, 2), min(nN, 1),
         ar and not a.GetIsAromatic(), al, co, m.GetRingInfo().NumAtomRings(a_idx) > 0, dblC,
         zb == 7, zb == 8, zb == 16, b.GetIsAromatic(), zb == 6 and has_double_o(m, b) or
         (zb == 8 and any(o.GetAtomicNum() == 6 and has_double_o(m, o) for o in b.GetNeighbors() if o.GetIdx() != a_idx)),
         (PA_CONST[PA_TYPES.index(pa_type(m, b))] if (PA_CONST and is_site(b)) else 0.0),
         pamax, math.log(nA), math.log(nB)]
    return [float(x) for x in f]


# ---- proton affinity site types

PA_TYPES = ["N_pyridine", "N_pyrrole", "N_amide", "N_aniline", "N_imine", "N_amine1", "N_amine2", "N_amine3",
            "N_other", "O_amideCO", "O_esterCO", "O_acidCO", "O_ketone", "O_ketoneConj", "O_alcohol", "O_ether",
            "O_ringEther", "O_phenol", "O_arylEther", "O_esterSingle", "O_arom", "O_other", "S", "C_arom"]


def is_site(a):
    z = a.GetAtomicNum()
    if z == 7 and a.GetFormalCharge() > 0 and a.GetTotalNumHs() == 0 and not any(
            o.GetFormalCharge() < 0 for o in a.GetNeighbors()):
        return False
    return z in (7, 8, 16) or (z == 6 and a.GetIsAromatic() and a.GetTotalNumHs() > 0)


def pa_type(m, a):
    z = a.GetAtomicNum()
    if z == 7:
        if a.GetFormalCharge() != 0:
            return "N_other"
        if a.GetIsAromatic():
            return "N_pyrrole" if (a.GetTotalNumHs() > 0 or a.GetDegree() == 3) else "N_pyridine"
        amide = aryl = imine = sulf = False
        nC = 0
        for bd in a.GetBonds():
            o = bd.GetOtherAtom(a)
            if bd.GetBondType() == DOUBLE:
                imine = True
            if o.GetAtomicNum() == 6:
                nC += 1
                amide |= has_double_o(m, o)
                aryl |= o.GetIsAromatic()
            if o.GetAtomicNum() in (15, 16):
                sulf = True
        if sulf:
            return "N_other"
        if amide:
            return "N_amide"
        if imine:
            return "N_imine"
        if aryl:
            return "N_aniline"
        return ["N_amine1", "N_amine1", "N_amine2", "N_amine3"][min(nC, 3)]
    if z == 8:
        if a.GetFormalCharge() != 0:
            return "O_other"
        if a.GetIsAromatic():
            return "O_arom"
        for bd in a.GetBonds():
            if bd.GetBondType() == DOUBLE:
                c = bd.GetOtherAtom(a)
                if c.GetAtomicNum() != 6:
                    return "O_other"
                nN = oN = arN = cc = False
                for b2 in c.GetBonds():
                    x = b2.GetOtherAtom(c)
                    if x.GetIdx() == a.GetIdx():
                        continue
                    if x.GetAtomicNum() == 7:
                        nN = True
                    if x.GetAtomicNum() == 8:
                        oN = x.GetTotalNumHs() == 0 and True or "acid"
                    if x.GetIsAromatic():
                        arN = True
                    if x.GetAtomicNum() == 6 and any(b3.GetBondType() == DOUBLE for b3 in x.GetBonds()):
                        cc = True
                if nN:
                    return "O_amideCO"
                if oN == "acid":
                    return "O_acidCO"
                if oN:
                    return "O_esterCO"
                return "O_ketoneConj" if (arN or cc) else "O_ketone"
        nbs = [o for o in a.GetNeighbors()]
        if any(o.GetAtomicNum() not in (6, 1) for o in nbs):
            return "O_other"
        if any(o.GetAtomicNum() == 6 and has_double_o(m, o) for o in nbs):
            return "O_esterSingle"
        if any(o.GetIsAromatic() for o in nbs):
            return "O_phenol" if a.GetTotalNumHs() > 0 else "O_arylEther"
        if a.GetTotalNumHs() > 0:
            return "O_alcohol"
        return "O_ringEther" if m.GetRingInfo().NumAtomRings(a.GetIdx()) > 0 else "O_ether"
    if z == 16:
        return "S"
    return "C_arom"


def arom_donors(m, a, dm):
    """aromatic C-H site: exocyclic O/N donors at ortho (graph distance 1) / para (3) ring positions"""
    n = 0
    for o in m.GetAtoms():
        if not o.GetIsAromatic() or o.GetIdx() == a.GetIdx():
            continue
        if dm[a.GetIdx()][o.GetIdx()] not in (1, 3):
            continue
        for x in o.GetNeighbors():
            if not x.GetIsAromatic() and x.GetAtomicNum() in (7, 8) and m.GetBondBetweenAtoms(o.GetIdx(), x.GetIdx()).GetBondType() == SINGLE:
                n += 1
    return n


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def ridge(X, y, lam=1e-3, wts=None):
    W = np.ones(len(y)) if wts is None else wts
    A = (X * W[:, None]).T @ X + lam * np.eye(X.shape[1])
    A[0, 0] -= lam
    return np.linalg.solve(A, (X * W[:, None]).T @ y)


def huber(X, y, lam=1e-3, k=1.5, it=20):
    """robust (Huber IRLS) ridge: protonated sites that fell apart during the
    crude optimisation are outliers, not the typical site"""
    w = ridge(X, y, lam)
    for _ in range(it):
        r = y - X @ w
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        u = np.abs(r) / (k * s)
        w = ridge(X, y, lam, np.where(u <= 1, 1.0, 1.0 / u))
    return w


def main():
    prefixes = sys.argv[1:]
    mols, sites, cuts = {}, defaultdict(list), defaultdict(list)
    for p in prefixes:
        for r in csv.DictReader(open(p + "_mol.csv")):
            if r["status"] == "ok" and r["E0"]:
                mols[r["molecule"]] = (r["smiles"], float(r["E0"]))
        for r in csv.DictReader(open(p + "_sites.csv")):
            sites[r["molecule"]].append(r)
        for r in csv.DictReader(open(p + "_cuts.csv")):
            cuts[r["molecule"]].append(r)
    print(f"molecules ok: {len(mols)}")

    # ---------------- proton affinities
    PX, PY, Ptype = [], [], []
    nmig = ntot = 0
    best_mh = {}
    for name, (smi, e0) in mols.items():
        m = Chem.MolFromSmiles(smi)
        dm = Chem.GetDistanceMatrix(m)
        nh = m.GetNumHeavyAtoms()
        for r in sites[name]:
            if not r["E_MH"]:
                continue
            e = float(r["E_MH"])
            best_mh[name] = min(best_mh.get(name, 1e9), e)
            i, fin = int(r["atom"]), int(r["final_atom"])
            ntot += 1
            if fin != i:
                nmig += 1
                continue
            a = m.GetAtomWithIdx(i)
            t = pa_type(m, a)
            x = [0.0] * len(PA_TYPES)
            x[PA_TYPES.index(t)] = 1.0
            x += [math.log(nh), arom_donors(m, a, dm) if t == "C_arom" else 0.0]
            PX.append(x)
            PY.append((e0 - e) * H)
            Ptype.append(t)
    PX, PY = np.array(PX), np.array(PY)
    print(f"\nPA sites: {ntot}, proton moved during opt {nmig}, used {len(PY)}")
    w = huber(np.hstack([np.ones((len(PY), 1)), PX]), PY, 1e-2)  # col0 intercept, then types
    pred = np.hstack([np.ones((len(PY), 1)), PX]) @ w
    print(f"PA formula R2 {r2(PY, pred):.3f}  RMSE {np.sqrt(np.mean((PY - pred) ** 2)):.2f} kcal/mol")
    # report as type constants at ln(n)=ln(30)
    print(f"  PA = T[type] + {w[-2]:.2f}*ln(nheavy/30) + {w[-1]:.2f}*donors_o/p (aromatic C)")
    pa_ln, pa_don = float(w[-2]), float(w[-1])
    pa_const = [float(w[0] + w[1 + k] + w[-2] * math.log(30)) for k in range(len(PA_TYPES))]
    pmed = np.median(PY)
    global PA_CONST
    PA_CONST = pa_const
    for k, t in enumerate(PA_TYPES):
        sel = np.array([tt == t for tt in Ptype])
        if sel.sum() == 0:
            pa_const[k] = float(pmed)  # unseen type: typical site
            continue
        const = w[0] + w[1 + k] + w[-2] * math.log(30)
        print(f"  T[{t:13s}] = {const:7.1f}   n={sel.sum():5d}  median obs {np.median(PY[sel]):6.1f}  "
              f"resid sd {np.std(PY[sel] - pred[sel]):5.1f}")

    # ---------------- cleavage energies
    rows = []
    for name, (smi, e0) in mols.items():
        if name not in best_mh:
            continue
        m = Chem.MolFromSmiles(smi)
        ref = best_mh[name]
        # molecule's most basic site by the PA table (mobile-proton term)
        pamax = max([pa_const[PA_TYPES.index(pa_type(m, a_))] for a_ in m.GetAtoms() if is_site(a_)] or [float(pmed)])
        loc = {int(r["atom"]): float(r["E_MH"]) for r in sites[name] if r["E_MH"] and r["atom"] == r["final_atom"]}
        for r in cuts[name]:
            try:
                e1c, e1h, e2c, e2h = (float(r[k]) for k in ("E1_cat", "E1_H", "E2_cat", "E2_H"))
            except ValueError:
                continue
            a1, a2, n1, n2 = int(r["a1"]), int(r["a2"]), int(r["n1"]), int(r["n2"])
            b = m.GetBondWithIdx(int(r["bond"]))
            cls = bond_class(m, b)
            for a, bb, ec, eh, nA, nB in ((a1, a2, e1c, e2h, n1, n2), (a2, a1, e2c, e1h, n2, n1)):
                dE = (ec + eh - ref) * H
                dl = (ec + eh - loc[bb]) * H if bb in loc else float("nan")
                rows.append((name, int(r["bond"]), cls, a, bb, dE, dl, dfeat(m, a, bb, nA, nB, pamax)))
    print(f"\ndirectional cuts: {len(rows)} from {len(set(r[0] for r in rows))} molecules")
    y = np.array([r[5] for r in rows])
    X = np.array([r[7] for r in rows])
    cls = np.array([r[2] for r in rows])
    ok = np.isfinite(y) & (y < 400) & (y > -100)
    print(f"finite and sane: {ok.sum()}")
    # class-only baseline (per class mean, direction ignored)
    pc = np.zeros(len(y))
    for c in set(cls[ok]):
        s = ok & (cls == c)
        pc[s] = y[s].mean()
    print(f"class means only: R2 {r2(y[ok], pc[ok]):.3f}")
    w = ridge(X[ok], y[ok], 1e-2)
    p = X @ w
    print(f"directional formula ({len(FEAT)} terms): R2 {r2(y[ok], p[ok]):.3f}  RMSE {np.sqrt(np.mean((y[ok] - p[ok]) ** 2)):.1f}")
    for f, c in zip(FEAT, w):
        print(f"  {f:10s} {c:8.2f}")
    # bond level: easier direction
    by = defaultdict(list)
    for k, r in enumerate(rows):
        if ok[k]:
            by[(r[0], r[1])].append(k)
    yb = np.array([min(y[k] for k in v) for v in by.values()])
    pb = np.array([min(p[k] for k in v) for v in by.values()])
    cb = np.array([cls[v[0]] for v in by.values()])
    print(f"\nbond level (min over directions): {len(yb)} bonds, R2 {r2(yb, pb):.3f}")
    pcb = np.zeros(len(yb))
    for c in set(cb):
        pcb[cb == c] = yb[cb == c].mean()
    print(f"  class means only R2 {r2(yb, pcb):.3f};  formula within-class R2 (after class means) "
          f"{1 - np.sum((yb - pb - (pcb - np.array([pb[cb == c].mean() for c in cb]))) ** 2) / np.sum((yb - pcb) ** 2):.3f}")
    print(f"  {'class':12s} {'n':>5s} {'median':>7s} {'sd':>6s} {'resid sd':>8s}")
    for c in sorted(set(cb)):
        s = cb == c
        print(f"  {CLS[c]:12s} {s.sum():5d} {np.median(yb[s]):7.1f} {np.std(yb[s]):6.1f} {np.std(yb[s] - pb[s]):8.1f}")
    # direction: does the formula pick the right side?
    agree = tot = 0
    for v in by.values():
        if len(v) == 2:
            tot += 1
            agree += (y[v[0]] < y[v[1]]) == (p[v[0]] < p[v[1]])
    print(f"  direction (which side keeps the charge) agreement {agree}/{tot} = {agree / max(1, tot):.3f}")
    # local reference (C-X, proton on X): compare with bde.py
    for c in (1, 2, 3):
        s = ok & (cls == c) & np.array([m_ is not None for m_ in rows]) & np.isfinite([r[6] for r in rows])
        s &= np.array([r[7][1] == 0 and r[7][2] == 0 and r[7][3] == 0 for r in rows])  # carbon keeps the charge
        if s.sum():
            print(f"  local-ref heterolysis {CLS[c]:10s} n={s.sum():4d} median {np.median([rows[k][6] for k in np.where(s)[0]]):6.1f}")
    # ---- bond-level formula: fit on the QM-preferred direction rows; apply in
    # the direction the directional formula prefers (what phys_qm.cpp does)
    keys = [k for k, v in by.items() if len(v) == 2]
    lo = [min(by[k], key=lambda j: y[j]) for k in keys]           # QM-preferred row
    lo_f = [min(by[k], key=lambda j: p[j]) for k in keys]         # formula-preferred row
    mol_of = [k[0] for k in keys]
    fold = np.array([zlib.crc32(mn.encode()) % 2 for mn in mol_of])
    # REF=local: bond-level target with the proton on the leaving atom when that site exists (C-X), else global
    ref = os.environ.get('REF', 'global')
    yt = np.array([r[6] if (ref == 'local' and np.isfinite(r[6])) else y[j] for j, r in enumerate(rows)])
    print(f'bond-level reference: {ref}')
    Xl, yl = X[lo], yt[lo]
    wb = ridge(Xl, yl, 1.0)
    yb2 = yt[lo]
    pb2 = X[lo_f] @ wb
    cb2 = cls[lo]
    print(f"\nbond-level formula (fit on preferred-direction rows, applied in formula direction): "
          f"{len(keys)} bonds, R2 {r2(yb2, pb2):.3f}  RMSE {np.sqrt(np.mean((yb2 - pb2) ** 2)):.1f}")
    # 2-fold CV by molecule (directional + bond formulas both refit)
    pcv = np.zeros(len(keys))
    for fo in (0, 1):
        trm = set(mn for mn, f_ in zip(mol_of, fold) if f_ != fo)
        tr_rows = np.array([ok[j] and rows[j][0] in trm for j in range(len(rows))])
        wd_ = ridge(X[tr_rows], y[tr_rows], 1e-2)
        trk = [i for i in range(len(keys)) if fold[i] != fo]
        wb_ = ridge(X[[lo[i] for i in trk]], yt[[lo[i] for i in trk]], 1.0)
        for i in range(len(keys)):
            if fold[i] == fo:
                j = min(by[keys[i]], key=lambda jj: X[jj] @ wd_)
                pcv[i] = X[j] @ wb_
    print(f"  2-fold CV by molecule: R2 {r2(yb2, pcv):.3f}")
    pcm = np.zeros(len(keys))
    for c in set(cb2):
        pcm[cb2 == c] = yb2[cb2 == c].mean()
    print(f"  class means only R2 {r2(yb2, pcm):.3f};  class means + formula deviation: "
          f"R2 {r2(yb2, pcm + pb2 - np.array([pb2[cb2 == c].mean() for c in cb2])):.3f}")
    print(f"  {'class':12s} {'n':>5s} {'QMmed':>6s} {'QMsd':>5s} {'resid':>6s} {'corr':>5s}")
    med = [0.0] * 10
    for c in sorted(set(cb2)):
        s = cb2 == c
        med[c] = float(np.median(pb2[s]))
        cc = np.corrcoef(yb2[s], pb2[s])[0, 1] if s.sum() > 2 else float("nan")
        print(f"  {CLS[c]:12s} {s.sum():5d} {np.median(yb2[s]):6.1f} {np.std(yb2[s]):5.1f} "
              f"{np.std(yb2[s] - pb2[s]):6.1f} {cc:5.2f}")
    print("  bond-level coefficients:")
    for f, c in zip(FEAT, wb):
        print(f"    {f:10s} {c:8.2f}")
    with open(prefixes[0] + ("_coef.txt" if ref == "global" else "_coefL.txt"), "w") as F:
        F.write("dir " + " ".join(f"{v:.4f}" for v in w) + "\n")
        F.write("bond " + " ".join(f"{v:.4f}" for v in wb) + "\n")
        F.write("med " + " ".join(f"{v:.4f}" for v in med) + "\n")
        F.write("pa " + " ".join(f"{v:.4f}" for v in pa_const) + f" {pa_ln:.4f} {pa_don:.4f} {float(pmed):.4f}\n")
    print(f"wrote {prefixes[0]}_coef.txt")


if __name__ == "__main__":
    main()
