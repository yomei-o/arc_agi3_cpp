"""GFN2-xTB proton affinities and even-electron cleavage energies, per molecule.

Protocol (same level as bde.py): ETKDG + MMFF geometry, xTB --opt crude.
  neutral M                                   -> E0
  every N/O/S atom and aromatic C-H carbon    -> E(MH+ protonated there), PA = E0 - E(MH+)
      (symmetry-equivalent atoms computed once; the final nearest heavy atom of
       the added proton is recorded, so proton migration during opt is visible)
  every acyclic single bond between heavy atoms, both charge directions:
      MH+ -> A+ + B-H     (A keeps the charge, B leaves as a neutral closed shell
                           with an H on its cut atom)
      record E(A+) and E(B-H); the analysis picks the reference MH+ (local site
      or global best).  For a C-X bond with A = C side this is bde.py's
      heterolysis R-XH+ -> R+ + HX.

    python qm2.py <mols.txt> <outprefix> [workers] [max_heavy] [first] [count]

Resumable: molecules already in <outprefix>_mol.csv are skipped; results are
appended per molecule (mol row written last).
"""
import csv, os, subprocess, sys, tempfile, time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
XTB = r"C:\prog\xtb\xtb-6.7.1\bin\xtb.exe"
HARTREE = 627.509
# QM_ONLY_ARC=1: aromatic C-H protonation sites only, no cuts (the first run missed them:
# GetTotalNumHs() on the H-explicit molecule is 0)
ONLY_ARC = os.environ.get("QM_ONLY_ARC") == "1"


def xtb_opt(symbols, coords, chrg, wd, uhf=0):
    path = os.path.join(wd, "m.xyz")
    with open(path, "w") as f:
        f.write(f"{len(symbols)}\n\n")
        for s, p in zip(symbols, coords):
            f.write(f"{s} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
    opt = os.path.join(wd, "xtbopt.xyz")
    if os.path.exists(opt):
        os.remove(opt)
    try:
        r = subprocess.run([XTB, "m.xyz", "--opt", "crude", "--gfn", "2", "--uhf", str(uhf), "--chrg", str(chrg)],
                           cwd=wd, capture_output=True, timeout=900,
                           env=dict(os.environ, OMP_NUM_THREADS="1", OMP_STACKSIZE="500M"))
    except subprocess.TimeoutExpired:
        return None, None
    e = None
    for line in r.stdout.decode("utf-8", errors="replace").splitlines():
        if "TOTAL ENERGY" in line:
            e = float(line.split()[3])
    if e is None or not os.path.exists(opt):
        return e, None
    lines = open(opt).read().splitlines()[2:2 + len(symbols)]
    xyz = np.array([[float(v) for v in l.split()[1:4]] for l in lines])
    return e, xyz


def one(args):
    name, smi, max_heavy = args
    t0 = time.time()
    res = {"name": name, "smiles": smi, "sites": [], "cuts": [], "E0": None, "nheavy": 0, "status": "ok"}
    m = Chem.MolFromSmiles(smi)
    if m is None:
        res["status"] = "parse"
        return res
    res["nheavy"] = m.GetNumHeavyAtoms()
    if m.GetNumHeavyAtoms() > max_heavy:
        res["status"] = "big"
        return res
    if any(a.GetNumRadicalElectrons() for a in m.GetAtoms()):
        res["status"] = "radical"
        return res
    mh = Chem.AddHs(m)
    if AllChem.EmbedMolecule(mh, randomSeed=7) != 0:
        if AllChem.EmbedMolecule(mh, randomSeed=7, useRandomCoords=True) != 0:
            res["status"] = "embed"
            return res
    try:
        AllChem.MMFFOptimizeMolecule(mh)
    except Exception:
        pass
    sym = [a.GetSymbol() for a in mh.GetAtoms()]
    q0 = Chem.GetFormalCharge(mh)
    X0 = np.array(mh.GetConformer().GetPositions())
    with tempfile.TemporaryDirectory() as wd:
        e0, X = xtb_opt(sym, X0, q0, wd)
        if e0 is None or X is None:
            res["status"] = "xtb0"
            return res
        res["E0"] = e0
        heavy = [a.GetIdx() for a in m.GetAtoms()]
        rank = list(Chem.CanonicalRankAtoms(m, breakTies=False))

        # ---- protonation sites
        done_rank = {}
        for i in heavy:
            a = mh.GetAtomWithIdx(i)
            z = a.GetAtomicNum()
            # H count from the heavy-atom graph: after AddHs, GetTotalNumHs() is 0 (the Hs are neighbours)
            nH = m.GetAtomWithIdx(i).GetTotalNumHs()
            arC = z == 6 and a.GetIsAromatic() and nH > 0
            if ONLY_ARC and not arC:
                continue
            if not (z in (7, 8, 16) or arC):
                continue
            if z == 7 and a.GetFormalCharge() > 0 and nH == 0 and not any(
                    nb.GetFormalCharge() < 0 for nb in a.GetNeighbors()):
                continue  # quaternary N+: no lone pair
            if rank[i] in done_rank:
                res["sites"].append((i, z, rank[i]) + done_rank[rank[i]][3:])
                continue
            p = X[i]
            nbh = [X[n.GetIdx()] for n in a.GetNeighbors()]
            if z == 6:
                ring_nb = [X[n.GetIdx()] for n in a.GetNeighbors() if n.GetAtomicNum() > 1][:2]
                if len(ring_nb) == 2:
                    d = np.cross(ring_nb[0] - p, ring_nb[1] - p)
                else:
                    d = np.array([0.0, 0.0, 1.0])
                bl = 1.10
            else:
                d = p - np.mean(nbh, axis=0) if nbh else np.array([1.0, 0.0, 0.0])
                if np.linalg.norm(d) < 0.2 and len(nbh) >= 1:  # linear: any perpendicular
                    d = np.cross(nbh[0] - p, np.array([0.3, 0.5, 0.8]))
                bl = 1.02
            d = d / (np.linalg.norm(d) + 1e-9)
            Xp = np.vstack([X, p + bl * d])
            e, Xo = xtb_opt(sym + ["H"], Xp, q0 + 1, wd)
            fin = -1
            if Xo is not None:
                h = Xo[-1]
                dd = [np.linalg.norm(Xo[j] - h) for j in heavy]
                fin = heavy[int(np.argmin(dd))]
            row = (i, z, rank[i], e, fin)
            done_rank[rank[i]] = row
            res["sites"].append(row)

        # ---- cuts: acyclic single bonds between heavy atoms, both charge directions
        ri = mh.GetRingInfo()
        frag_cache = {}
        if ONLY_ARC:
            res["sec"] = time.time() - t0
            return res
        for b in m.GetBonds():
            if b.GetBondType() != Chem.BondType.SINGLE or ri.NumBondRings(b.GetIdx()) > 0:
                continue
            i1, i2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            em = Chem.RWMol(mh)
            em.RemoveBond(i1, i2)
            frags = Chem.GetMolFrags(em, asMols=False)
            if len(frags) != 2:
                continue
            side = {}
            for cut_atom, other in ((i1, i2), (i2, i1)):
                atoms = frags[0] if cut_atom in frags[0] else frags[1]
                qs = sum(mh.GetAtomWithIdx(j).GetFormalCharge() for j in atoms)
                key = tuple(sorted(atoms))
                if key not in frag_cache:
                    s_sym = [sym[j] for j in atoms]
                    s_X = X[list(atoms)]
                    eC, _ = xtb_opt(s_sym, s_X, qs + 1, wd)
                    d = X[other] - X[cut_atom]
                    d = d / (np.linalg.norm(d) + 1e-9)
                    bl = 1.09 if mh.GetAtomWithIdx(cut_atom).GetAtomicNum() == 6 else 1.0
                    eH, _ = xtb_opt(s_sym + ["H"], np.vstack([s_X, X[cut_atom] + bl * d]), qs, wd)
                    frag_cache[key] = (eC, eH, sum(1 for j in atoms if sym[j] != "H"))
                side[cut_atom] = frag_cache[key]
            res["cuts"].append((b.GetIdx(), i1, i2, side[i1][0], side[i1][1], side[i1][2],
                                side[i2][0], side[i2][1], side[i2][2]))
    res["sec"] = time.time() - t0
    return res


def main():
    mols_path, prefix = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    max_heavy = int(sys.argv[4]) if len(sys.argv) > 4 else 50
    first = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    count = int(sys.argv[6]) if len(sys.argv) > 6 else 10 ** 9
    items = []
    for line in open(mols_path, encoding="utf-8"):
        a, _, b = line.rstrip("\r\n").partition("\t")
        if a and b:
            items.append((a, b))
    items = items[first:first + count]
    fm, fs, fc = prefix + "_mol.csv", prefix + "_sites.csv", prefix + "_cuts.csv"
    done = set()
    if os.path.exists(fm):
        for r in csv.reader(open(fm)):
            if r and r[0] != "molecule":
                done.add(r[0])
    new = not os.path.exists(fm)
    todo = [(a, b, max_heavy) for a, b in items if a not in done]
    print(f"{len(items)} molecules, {len(done)} done, {len(todo)} to do", flush=True)
    with open(fm, "a", newline="") as Fm, open(fs, "a", newline="") as Fs, open(fc, "a", newline="") as Fc:
        wm, ws, wc = csv.writer(Fm), csv.writer(Fs), csv.writer(Fc)
        if new:
            wm.writerow(["molecule", "smiles", "nheavy", "status", "E0", "sec"])
            ws.writerow(["molecule", "atom", "z", "rank", "E_MH", "final_atom"])
            wc.writerow(["molecule", "bond", "a1", "a2", "E1_cat", "E1_H", "n1", "E2_cat", "E2_H", "n2"])
        t0 = time.time()
        with ProcessPoolExecutor(workers) as ex:
            futs = [ex.submit(one, t) for t in todo]
            for k, fu in enumerate(as_completed(futs)):
                try:
                    r = fu.result()
                except Exception as exc:
                    print("error", exc, flush=True)
                    continue
                for s in r["sites"]:
                    ws.writerow([r["name"]] + list(s))
                for c in r["cuts"]:
                    wc.writerow([r["name"]] + list(c))
                Fs.flush(); Fc.flush()
                wm.writerow([r["name"], r["smiles"], r["nheavy"], r["status"], r["E0"], f"{r.get('sec', 0):.1f}"])
                Fm.flush()
                print(f"{k + 1}/{len(todo)} {r['name']} n={r['nheavy']} {r['status']} sites={len(r['sites'])} "
                      f"cuts={len(r['cuts'])} {r.get('sec', 0):.0f}s total={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
