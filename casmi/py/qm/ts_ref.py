"""Reference energy of [M+H]+: the most stable N/O protonation site of each molecule.

ts_scan/ts_ipso measured barriers from the protonation site next to the bond;
that is not the ion's ground state, which put several kcal/mol of noise (and
negative "proton move" costs) into the barriers. Here every neutral N/O atom
(up to MAXS sites) is protonated and optimised with GFN2-xTB; the lowest
energy is the molecule's reference.

    python ts_ref.py <ts_scan.csv> <out.csv> [workers] [names_from_csv2]
Output: name n_sites best_site E_ref E_site_used (energies in Hartree)
"""
import csv, os, sys, tempfile
from concurrent.futures import ProcessPoolExecutor

from rdkit import Chem
from rdkit.Chem import AllChem

import ts_scan as T

MAXS = 10


def opt_energy(m, site):
    try:
        p = Chem.AddHs(T.protonate(m, site))
    except Exception:
        return None
    if AllChem.EmbedMolecule(p, randomSeed=11) != 0:
        return None
    AllChem.MMFFOptimizeMolecule(p)
    with tempfile.TemporaryDirectory() as wd:
        with open(os.path.join(wd, "m.xyz"), "w") as f:
            f.write(T.to_xyz(p))
        return T.total_energy(T.run_xtb(wd, ["m.xyz", "--opt", "normal", "--gfn", "2", "--chrg", "1"]))


def one(item):
    name, smi = item
    m = Chem.MolFromSmiles(smi)
    sites = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() in (7, 8) and a.GetFormalCharge() == 0][:MAXS]
    best = None
    for s in sites:
        e = opt_energy(m, s)
        if e is not None and (best is None or e < best[1]):
            best = (s, e)
    if best is None:
        return None
    return dict(name=name, n_sites=len(sites), best_site=best[0], E_ref=best[1])


def main():
    names = None
    if len(sys.argv) > 4:
        names = {r["name"] for r in csv.DictReader(open(sys.argv[4]))}
    mols = {}
    for r in csv.DictReader(open(sys.argv[1])):
        if names is None or r["name"] in names:
            mols[r["name"]] = r["smiles"]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    print(f"{len(mols)} molecules", flush=True)
    with open(sys.argv[2], "w", newline="") as f, ProcessPoolExecutor(workers) as ex:
        w = csv.DictWriter(f, ["name", "n_sites", "best_site", "E_ref"]); w.writeheader()
        for k, r in enumerate(ex.map(one, mols.items())):
            if r:
                w.writerow(r); f.flush()
            if k % 10 == 0:
                print(k, flush=True)


if __name__ == "__main__":
    main()
