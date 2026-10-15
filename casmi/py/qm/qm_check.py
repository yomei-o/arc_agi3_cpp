"""Compare phys_qm dump (C++) with qm_fit.py's descriptors for the same coefficients.
    python qm_check.py <coef.txt> <mols.txt> <dump.txt>"""
import math, sys
import numpy as np
from rdkit import Chem
sys.path.insert(0, r"C:\prog\casmi\qm")
import qm_fit as Q

coef = {}
for l in open(sys.argv[1]):
    t = l.split()
    coef[t[0]] = [float(x) for x in t[1:]]
wd, wb, pa = np.array(coef["dir"]), np.array(coef["bond"]), coef["pa"]
pa_const, pmed = pa[:len(Q.PA_TYPES)], pa[-1]
Q.PA_CONST = pa_const
mols = {}
for l in open(sys.argv[2], encoding="utf-8"):
    a, _, s = l.rstrip("\r\n").partition("\t")
    mols[a] = s
nb = ns = bad = 0
for l in open(sys.argv[3]):
    t = l.split()
    m = Chem.MolFromSmiles(mols[t[1]])
    if t[0] == "S":
        a = m.GetAtomWithIdx(int(t[2]))
        tt = Q.PA_TYPES.index(Q.pa_type(m, a))
        ns += 1
        if tt != int(t[3]) or not Q.is_site(a):
            bad += 1
            print("site mismatch", t, Q.pa_type(m, a))
        continue
    nb += 1
    x, y = int(t[3]), int(t[4])
    em = Chem.RWMol(m)
    em.RemoveBond(x, y)
    fr = Chem.GetMolFrags(em, asMols=False)
    nx = len(fr[0]) if x in fr[0] else len(fr[1])
    ny = m.GetNumAtoms() - nx
    pmx = max([pa_const[Q.PA_TYPES.index(Q.pa_type(m, a_))] for a_ in m.GetAtoms() if Q.is_site(a_)] or [pmed])
    fxy = np.array(Q.dfeat(m, x, y, nx, ny, pmx))
    fyx = np.array(Q.dfeat(m, y, x, ny, nx, pmx))
    py = [fxy @ wd, fyx @ wd, fxy @ wb, fyx @ wb]
    cc = [float(v) for v in t[5:9]]
    if max(abs(p - c) for p, c in zip(py, cc)) > 0.05:
        bad += 1
        if bad < 20:
            print("bond mismatch", t[1], t[2], [round(v, 2) for v in py], cc)
print(f"sites {ns} bonds {nb} mismatches {bad}")
