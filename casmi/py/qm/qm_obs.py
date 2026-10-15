"""Does the xTB cleavage energy predict which bonds' fragments are seen in the
real spectra? Model-free check, independent of phys_rank.

For each true fqry molecule with QM data and [M+H]+ spectra: every acyclic
single bond, "observed" = a fragment of that single cut (either side, H shift
-2..+2) matches a peak (0.01 Da) of any of the molecule's [M+H]+ spectra.
Report per class the AUC of -dE (lower energy -> observed), for the raw QM
bond energy (global and local reference) and for the formula.

    python qm_obs.py <qm prefix> <coef.txt>
"""
import csv, struct, sys
from collections import defaultdict
import numpy as np
from rdkit import Chem
sys.path.insert(0, r"C:\prog\casmi\qm")
import qm_fit as Q

B = "C:/prog/casmi/"
H_ = 627.509
MH, EL = 1.00782503207, 0.00054858
pfx, coefp = sys.argv[1], sys.argv[2]

# spectra
names = [l.split("\t")[0] for l in open(B + "fqry_mols.txt", encoding="utf-8")]
hb = open(B + "fqry_head.bin", "rb").read()
pb = open(B + "fqry_peaks.bin", "rb").read()
heads = [struct.unpack_from("<fiiii", hb, 20 * i) for i in range(len(hb) // 20)]
peaks = defaultdict(list)
off = 0
for prec, mol, _, n, tag in heads:
    mz = np.frombuffer(pb, dtype="<f4", count=n, offset=off)
    off += 4 * n * 2
    a = -1 if tag < 0 else (tag // 16 + 1 if tag % 16 == 15 else tag // 16)
    if a == 0:  # [M+H]+
        peaks[names[mol]].append(np.sort(mz.astype(np.float64)))
print("molecules with [M+H]+ spectra", len(peaks))

coef = {}
for l in open(coefp):
    t = l.split()
    coef[t[0]] = [float(x) for x in t[1:]]
wd, wb = np.array(coef["dir"]), np.array(coef["bond"])
Q.PA_CONST = coef["pa"][:len(Q.PA_TYPES)]
pmed = coef["pa"][-1]

mols = {r["molecule"]: r for r in csv.DictReader(open(pfx + "_mol.csv")) if r["status"] == "ok" and r["E0"]}
sites, cuts = defaultdict(list), defaultdict(list)
for r in csv.DictReader(open(pfx + "_sites.csv")):
    sites[r["molecule"]].append(r)
for r in csv.DictReader(open(pfx + "_cuts.csv")):
    cuts[r["molecule"]].append(r)

pt = Chem.GetPeriodicTable()


def frag_mass(m, atoms):
    s = 0.0
    for i in atoms:
        a = m.GetAtomWithIdx(i)
        s += pt.GetMostCommonIsotopeMass(a.GetAtomicNum()) + a.GetTotalNumHs() * MH
    return s


def seen(mass, spl):
    for sp in spl:
        for k in range(-2, 3):
            t = mass + k * MH - EL
            j = np.searchsorted(sp, t - 0.01)
            if j < len(sp) and sp[j] <= t + 0.01:
                return True
    return False


rows = []  # cls, observed, E_glob, E_loc, E_formula
for name, mr in mols.items():
    if name not in peaks:
        continue
    m = Chem.MolFromSmiles(mr["smiles"])
    E_mh = [float(r["E_MH"]) for r in sites[name] if r["E_MH"]]
    if not E_mh:
        continue
    ref = min(E_mh)
    loc = {int(r["atom"]): float(r["E_MH"]) for r in sites[name] if r["E_MH"] and r["atom"] == r["final_atom"]}
    pmx = max([Q.PA_CONST[Q.PA_TYPES.index(Q.pa_type(m, a_))] for a_ in m.GetAtoms() if Q.is_site(a_)] or [pmed])
    for r in cuts[name]:
        try:
            e1c, e1h, e2c, e2h = (float(r[k]) for k in ("E1_cat", "E1_H", "E2_cat", "E2_H"))
        except ValueError:
            continue
        a1, a2, n1, n2 = int(r["a1"]), int(r["a2"]), int(r["n1"]), int(r["n2"])
        b = m.GetBondWithIdx(int(r["bond"]))
        cls = Q.bond_class(m, b)
        d1 = (e1c + e2h - ref) * H_  # a1 side keeps the charge
        d2 = (e2c + e1h - ref) * H_
        l1 = (e1c + e2h - loc[a2]) * H_ if a2 in loc else np.nan
        l2 = (e2c + e1h - loc[a1]) * H_ if a1 in loc else np.nan
        el = np.nanmin([l1, l2]) if not (np.isnan(l1) and np.isnan(l2)) else min(d1, d2)
        f1 = np.array(Q.dfeat(m, a1, a2, n1, n2, pmx))
        f2 = np.array(Q.dfeat(m, a2, a1, n2, n1, pmx))
        ef = (f1 @ wb) if (f1 @ wd) <= (f2 @ wd) else (f2 @ wb)
        em = Chem.RWMol(m)
        em.RemoveBond(a1, a2)
        fr = Chem.GetMolFrags(em, asMols=False)
        obs = any(seen(frag_mass(m, f), peaks[name]) for f in fr)
        # also: which side is seen (charge retention check)
        s1 = seen(frag_mass(m, fr[0] if a1 in fr[0] else fr[1]), peaks[name])
        s2 = seen(frag_mass(m, fr[1] if a1 in fr[0] else fr[0]), peaks[name])
        fa = fr[0] if a1 in fr[0] else fr[1]
        fb = fr[1] if a1 in fr[0] else fr[0]
        def spa(atoms):
            v = [Q.PA_CONST[Q.PA_TYPES.index(Q.pa_type(m, m.GetAtomWithIdx(i)))] for i in atoms
                 if Q.is_site(m.GetAtomWithIdx(i))]
            return (max(v) + coef["pa"][len(Q.PA_TYPES)] * np.log(len(atoms) / 30.0)) if v else -1e9
        side = dict(size=len(fa) - len(fb), pa=spa(fa) - spa(fb), form=(f2 @ wd) - (f1 @ wd), qm=d2 - d1)
        rows.append((cls, obs, min(d1, d2), el, ef, d1 - d2, s1, s2, name, side))


def auc2(score, lab):
    score, lab = np.asarray(score, float), np.asarray(lab, bool)
    pos, neg = score[lab], score[~lab]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return gt + 0.5 * eq


print(f"bonds {len(rows)} from {len(set(r[8] for r in rows))} molecules")
print(f"{'class':12s} {'n':>5s} {'obs%':>5s} {'AUC glob':>8s} {'AUC loc':>8s} {'AUC form':>8s}  (AUC of -dE for 'fragment seen')")
cl = np.array([r[0] for r in rows])
ob = np.array([r[1] for r in rows])
for c in sorted(set(cl)) + ["all"]:
    s = (cl == c) if c != "all" else np.ones(len(rows), bool)
    if s.sum() < 10:
        continue
    g = [-rows[i][2] for i in np.where(s)[0]]
    l = [-rows[i][3] for i in np.where(s)[0]]
    f = [-rows[i][4] for i in np.where(s)[0]]
    nm = Q.CLS[c] if c != "all" else "all"
    print(f"{nm:12s} {s.sum():5d} {100 * ob[s].mean():5.1f} {auc2(g, ob[s]):8.3f} {auc2(l, ob[s]):8.3f} {auc2(f, ob[s]):8.3f}")
# within-molecule: per molecule, AUC over its bonds, averaged (removes molecule-level shifts)
for key, j in (("glob", 2), ("loc", 3), ("form", 4)):
    a = []
    for nm in set(r[8] for r in rows):
        idx = [i for i, r in enumerate(rows) if r[8] == nm]
        v = auc2([-rows[i][j] for i in idx], [rows[i][1] for i in idx])
        if not np.isnan(v):
            a.append(v)
    print(f"within-molecule mean AUC ({key}): {np.mean(a):.3f} over {len(a)} molecules")
# charge side: when exactly one side is seen, does the QM-preferred cation side match?
ok = tot = 0
for r in rows:
    if r[6] != r[7]:
        tot += 1
        ok += (r[5] < 0) == r[6]  # d1 < d2: a1 side keeps the charge
print(f"charge side (one side seen): QM directional energy picks the seen side {ok}/{tot} = {ok / max(1, tot):.3f}")
for k in ("size", "pa", "form", "qm"):
    ok = tot = 0
    for r in rows:
        if r[6] != r[7] and r[9][k] != 0:
            tot += 1
            ok += (r[9][k] > 0) == r[6]
    print(f"  side predictor {k:5s}: {ok}/{tot} = {ok / max(1, tot):.3f}")
# conditional: among bonds where size and qm disagree
ok = tot = 0
for r in rows:
    if r[6] != r[7] and (r[9]["size"] > 0) != (r[9]["qm"] > 0) and r[9]["size"] != 0:
        tot += 1
        ok += (r[9]["qm"] > 0) == r[6]
print(f"  where size and QM disagree, QM right {ok}/{tot} = {ok / max(1, tot):.3f}")
