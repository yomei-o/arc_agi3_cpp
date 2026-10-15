"""Aryl-C cleavage through ipso protonation (arenium ion), vs direct heterolysis.

ts_scan.py found direct aryl-C heterolysis barriers of ~108 kcal/mol, above
what CID deposits (1-3 eV = 23-70 kcal/mol), yet these fragments are seen.
Textbook route: the mobile proton moves onto the ring carbon carrying the
substituent (ipso), making a cyclohexadienyl cation whose ipso-substituent
bond is no longer part of an aromatic system, then that bond breaks.

For each aryl-C bond in ts_scan's CSV:
  E_ref : the molecule protonated at the same N/O site ts_scan used (its E0)
  arenium: Kekule form, the ring double bond at the ipso carbon made single,
           ipso gets an H, its ring partner carries the + charge; xtb opt -> E_ar
  scan of the ipso-substituent bond from the arenium -> E_max
  proton transfer cost = E_ar - E_ref ; total barrier = E_max - E_ref
All kcal/mol.

    python ts_ipso.py <ts_scan.csv> <out.csv> [workers]
"""
import csv, os, sys, tempfile
from concurrent.futures import ProcessPoolExecutor

from rdkit import Chem
from rdkit.Chem import AllChem

import ts_scan as T


def arenium(m, ipso, sub):
    rw = Chem.RWMol(m)
    Chem.Kekulize(rw, clearAromaticFlags=True)
    a = rw.GetAtomWithIdx(ipso)
    partner = None
    for b in a.GetBonds():
        o = b.GetOtherAtomIdx(ipso)
        if o != sub and b.GetBondType() == Chem.BondType.DOUBLE:
            b.SetBondType(Chem.BondType.SINGLE)
            partner = o
            break
    if partner is None:
        return None
    # H counts from the untouched molecule: after SetNoImplicit, GetTotalNumHs()
    # no longer counts the implicit H (that dropped one H from the partner carbon)
    a.SetNumExplicitHs(m.GetAtomWithIdx(ipso).GetTotalNumHs() + 1)
    a.SetNoImplicit(True)
    p = rw.GetAtomWithIdx(partner)
    p.SetFormalCharge(1)
    p.SetNumExplicitHs(m.GetAtomWithIdx(partner).GetTotalNumHs())
    p.SetNoImplicit(True)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def one(r):
    m = Chem.MolFromSmiles(r["smiles"])
    i, j = int(r["i"]), int(r["j"])
    ai, aj = m.GetAtomWithIdx(i), m.GetAtomWithIdx(j)
    if ai.GetIsAromatic() and aj.GetIsAromatic():
        return None          # biaryl: skip for now
    ipso, sub = (i, j) if ai.GetIsAromatic() else (j, i)
    if m.GetAtomWithIdx(ipso).GetAtomicNum() != 6:
        return None
    try:
        p = arenium(m, ipso, sub)
    except Exception:
        return None
    if p is None:
        return None
    p = Chem.AddHs(p)
    if AllChem.EmbedMolecule(p, randomSeed=11) != 0:
        return None
    AllChem.MMFFOptimizeMolecule(p)
    with tempfile.TemporaryDirectory() as wd:
        with open(os.path.join(wd, "m.xyz"), "w") as f:
            f.write(T.to_xyz(p))
        e_ar = T.total_energy(T.run_xtb(wd, ["m.xyz", "--opt", "normal", "--gfn", "2", "--chrg", "1"]))
        opt = os.path.join(wd, "xtbopt.xyz")
        if e_ar is None or not os.path.exists(opt):
            return None
        xyz = open(opt).read().splitlines()
        ci = list(map(float, xyz[2 + ipso].split()[1:4]))
        cj = list(map(float, xyz[2 + sub].split()[1:4]))
        r0 = sum((a - c) ** 2 for a, c in zip(ci, cj)) ** 0.5
        with open(os.path.join(wd, "scan.inp"), "w") as f:
            f.write("$constrain\n force constant=1.0\n"
                    f" distance: {ipso + 1}, {sub + 1}, {r0:.4f}\n"
                    f"$scan\n 1: {r0:.4f}, {r0 + T.STRETCH:.4f}, {T.NSTEP}\n$end\n")
        os.replace(opt, os.path.join(wd, "start.xyz"))
        T.run_xtb(wd, ["start.xyz", "--opt", "loose", "--gfn", "2", "--chrg", "1", "--input", "scan.inp"])
        log = os.path.join(wd, "xtbscan.log")
        if not os.path.exists(log):
            return None
        E, _ = T.read_scan(log)
    if len(E) < 3:
        return None
    e_ref = float(r["E0"])
    k = max(range(len(E)), key=lambda q: E[q])
    return dict(name=r["name"], i=i, j=j, direct=float(r["barrier"]),
                pt_cost=(e_ar - e_ref) * T.HARTREE, ipso_scan=(E[k] - e_ar) * T.HARTREE,
                ipso_total=(E[k] - e_ref) * T.HARTREE, ipso_reaction=(E[-1] - e_ref) * T.HARTREE,
                plateau=int(k == len(E) - 1))


def main():
    rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["cls"] == "aryl-C"]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    cols = "name i j direct pt_cost ipso_scan ipso_total ipso_reaction plateau".split()
    print(f"{len(rows)} aryl-C bonds", flush=True)
    res = []
    with open(sys.argv[2], "w", newline="") as f, ProcessPoolExecutor(workers) as ex:
        w = csv.DictWriter(f, cols); w.writeheader()
        for r in ex.map(one, rows):
            if r is None:
                continue
            w.writerow(r); f.flush(); res.append(r)
            print(f"direct {r['direct']:6.1f}  | ipso: proton move {r['pt_cost']:6.1f} + scan {r['ipso_scan']:6.1f}"
                  f" = {r['ipso_total']:6.1f}{'  plateau' if r['plateau'] else ''}", flush=True)
    if res:
        import statistics as S
        print(f"\n{len(res)} bonds: direct median {S.median(x['direct'] for x in res):.1f}, "
              f"ipso total median {S.median(x['ipso_total'] for x in res):.1f}, "
              f"ipso lower in {sum(x['ipso_total'] < x['direct'] for x in res)}/{len(res)}")


if __name__ == "__main__":
    main()
