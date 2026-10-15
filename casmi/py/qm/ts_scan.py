"""Barriers of charge-directed bond cleavage in [M+H]+, by relaxed bond scans (GFN2-xTB).

Why: the physics model's bond-class barriers (phys_rank) were fitted to spectra,
and xTB bond dissociation energies (bde.py) did not reproduce their order -
a BDE is the energy of the products, not the height of the hill on the way.
CID fragments an even-electron cation, so the step that matters is a
heterolysis assisted by the added proton. This measures that hill directly.

For each chosen bond X-Y of a molecule:
  1. protonate the bond's heteroatom (N or O; for C-C bonds the nearest N/O
     within 2 bonds), charge +1
  2. ETKDG + MMFF geometry, GFN2-xTB optimisation       -> E0
  3. relaxed scan: d(X-Y) from r0 to r0 + STRETCH in NSTEP steps, every other
     coordinate optimised at each step (xtb $constrain/$scan)
  4. barrier  = max_k E_k - E0                (top of the scan = approximate TS)
     reaction = E_last - E0                   (separated ion-neutral pair)
     If the maximum is the last point the cleavage has no hill within the scan
     (barrier = endothermicity, flagged `plateau`).
Units kcal/mol. The scan top geometry is saved for later DFT single points.

    python ts_scan.py <mols.txt> <n_molecules> <out.csv> [workers] [max_bonds_per_class_per_mol]

Output columns: name smiles cls i j prot E0 Emax Eend kmax barrier reaction plateau
"""
import csv, os, random, subprocess, sys, tempfile
from concurrent.futures import ProcessPoolExecutor

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
XTB = r"C:\prog\xtb\xtb-6.7.1\bin\xtb.exe"
HARTREE = 627.509
STRETCH, NSTEP = 2.0, 16
OUTDIR = r"C:\prog\casmi\ts_scan_geo"


def has_double_o(c):
    return any(b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(c).GetAtomicNum() == 8
               for b in c.GetBonds())


def anomeric(m, c):
    if c.GetAtomicNum() != 6 or c.GetIsAromatic():
        return False
    n_o = sum(1 for b in c.GetBonds()
              if b.GetOtherAtom(c).GetAtomicNum() == 8 and b.GetBondType() == Chem.BondType.SINGLE)
    return n_o >= 2 and m.GetRingInfo().NumAtomRings(c.GetIdx()) > 0


def bond_class(m, b):
    """Same classes as phys_rank.cpp bondClass() (C-glycosidic folded into C-C here)."""
    x, y = b.GetBeginAtom(), b.GetEndAtom()
    zx, zy = x.GetAtomicNum(), y.GetAtomicNum()
    if zx == 6 and zy == 6:
        return "aryl-C" if (x.GetIsAromatic() or y.GetIsAromatic()) else "C-C"
    c = x if zx == 6 else (y if zy == 6 else None)
    if c is None:
        return "other"
    h = y if c is x else x
    if h.GetAtomicNum() in (7, 8) and has_double_o(c):
        return "acyl"
    if h.GetAtomicNum() == 8 and anomeric(m, c):
        return "glycosidic"
    if h.GetAtomicNum() in (7, 8, 16):
        return "C-het"
    return "other"


def protonation_site(m, b):
    """Heteroatom of the bond; for C-C the nearest N/O (<= 2 bonds) with a free lone pair."""
    for a in (b.GetBeginAtom(), b.GetEndAtom()):
        if a.GetAtomicNum() in (7, 8) and a.GetFormalCharge() == 0:
            return a.GetIdx()
    best = None
    for a in (b.GetBeginAtom(), b.GetEndAtom()):
        for n1 in a.GetNeighbors():
            for c in [n1] + list(n1.GetNeighbors()):
                if c.GetAtomicNum() in (7, 8) and c.GetFormalCharge() == 0 and not c.GetIsAromatic():
                    d = Chem.GetShortestPath(m, a.GetIdx(), c.GetIdx())
                    if best is None or len(d) < best[0]:
                        best = (len(d), c.GetIdx())
    return None if best is None else best[1]


def protonate(m, site):
    rw = Chem.RWMol(m)
    a = rw.GetAtomWithIdx(site)
    a.SetFormalCharge(1)
    a.SetNumExplicitHs(a.GetTotalNumHs() + 1)
    a.SetNoImplicit(True)
    p = rw.GetMol()
    Chem.SanitizeMol(p)
    return p


def to_xyz(mol):
    conf = mol.GetConformer()
    lines = [str(mol.GetNumAtoms()), ""]
    for a in mol.GetAtoms():
        q = conf.GetAtomPosition(a.GetIdx())
        lines.append(f"{a.GetSymbol()} {q.x:.5f} {q.y:.5f} {q.z:.5f}")
    return "\n".join(lines) + "\n"


def run_xtb(wd, args):
    r = subprocess.run([XTB] + args, cwd=wd, capture_output=True, timeout=3600,
                       env=dict(os.environ, OMP_NUM_THREADS="1", OMP_STACKSIZE="500M"))
    return r.stdout.decode("utf-8", errors="replace")


def total_energy(out):
    e = None
    for line in out.splitlines():
        if "TOTAL ENERGY" in line:
            e = float(line.split()[3])
    return e


def read_scan(path):
    """xtbscan.log: concatenated xyz frames, comment line ' energy: E gnorm: ...'"""
    E, frames = [], []
    with open(path) as f:
        lines = f.read().splitlines()
    k = 0
    while k < len(lines):
        if not lines[k].strip():
            k += 1; continue
        n = int(lines[k].split()[0])
        comment = lines[k + 1]
        tok = comment.split()
        E.append(float(tok[tok.index("energy:") + 1]))
        frames.append("\n".join(lines[k:k + 2 + n]) + "\n")
        k += 2 + n
    return E, frames


def scan_one(job):
    name, smi, bidx, cls, site = job
    m = Chem.MolFromSmiles(smi)
    b = m.GetBondWithIdx(bidx)
    i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
    try:
        p = Chem.AddHs(protonate(m, site))
    except Exception:
        return None
    if AllChem.EmbedMolecule(p, randomSeed=11) != 0:
        return None
    AllChem.MMFFOptimizeMolecule(p)
    with tempfile.TemporaryDirectory() as wd:
        with open(os.path.join(wd, "m.xyz"), "w") as f:
            f.write(to_xyz(p))
        out = run_xtb(wd, ["m.xyz", "--opt", "normal", "--gfn", "2", "--chrg", "1"])
        e0 = total_energy(out)
        opt = os.path.join(wd, "xtbopt.xyz")
        if e0 is None or not os.path.exists(opt):
            return None
        xyz = open(opt).read().splitlines()
        ci = list(map(float, xyz[2 + i].split()[1:4]))
        cj = list(map(float, xyz[2 + j].split()[1:4]))
        r0 = sum((a - c) ** 2 for a, c in zip(ci, cj)) ** 0.5
        # xtb atom numbers are 1-based
        with open(os.path.join(wd, "scan.inp"), "w") as f:
            f.write("$constrain\n force constant=1.0\n"
                    f" distance: {i + 1}, {j + 1}, {r0:.4f}\n"
                    f"$scan\n 1: {r0:.4f}, {r0 + STRETCH:.4f}, {NSTEP}\n$end\n")
        os.replace(opt, os.path.join(wd, "start.xyz"))
        run_xtb(wd, ["start.xyz", "--opt", "loose", "--gfn", "2", "--chrg", "1", "--input", "scan.inp"])
        log = os.path.join(wd, "xtbscan.log")
        if not os.path.exists(log):
            return None
        E, frames = read_scan(log)
        if len(E) < 3:
            return None
        kmax = max(range(len(E)), key=lambda k: E[k])
        os.makedirs(OUTDIR, exist_ok=True)
        with open(os.path.join(OUTDIR, f"{name}_{bidx}_top.xyz"), "w") as f:
            f.write(frames[kmax])
    return dict(name=name, smiles=smi, cls=cls, i=i, j=j, prot=site, E0=e0, Emax=E[kmax], Eend=E[-1], kmax=kmax,
                barrier=(E[kmax] - e0) * HARTREE, reaction=(E[-1] - e0) * HARTREE, plateau=int(kmax == len(E) - 1))


def jobs_for(name, smi, per_class, rnd):
    m = Chem.MolFromSmiles(smi)
    if m is None or m.GetNumHeavyAtoms() > 35:
        return []
    ri = m.GetRingInfo()
    by = {}
    for b in m.GetBonds():
        if ri.NumBondRings(b.GetIdx()) or b.GetBondType() != Chem.BondType.SINGLE:
            continue
        cls = bond_class(m, b)
        if cls == "other":
            continue
        site = protonation_site(m, b)
        if site is None:
            continue
        by.setdefault(cls, []).append((name, smi, b.GetIdx(), cls, site))
    out = []
    for v in by.values():
        rnd.shuffle(v)
        out += v[:per_class]
    return out


def main():
    mols, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    per_class = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    rnd = random.Random(0)
    rows = [l.rstrip("\r\n").split("\t") for l in open(mols, encoding="utf-8")]
    rnd.shuffle(rows)
    jobs = []
    for name, smi in rows[:n]:
        jobs += jobs_for(name, smi, per_class, rnd)
    print(f"{len(jobs)} scans from {n} molecules", flush=True)
    cols = "name smiles cls i j prot E0 Emax Eend kmax barrier reaction plateau".split()
    with open(out, "w", newline="") as f, ProcessPoolExecutor(workers) as ex:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for k, r in enumerate(ex.map(scan_one, jobs)):
            if r is None:
                continue
            w.writerow(r); f.flush()
            print(f"{k + 1}/{len(jobs)} {r['cls']:10s} barrier {r['barrier']:6.1f} reaction {r['reaction']:6.1f}"
                  f"{'  plateau' if r['plateau'] else ''}", flush=True)


if __name__ == "__main__":
    main()
