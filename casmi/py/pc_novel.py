"""Why did the PubChem channel not move the LB? Test on "novel natural products".

Holdout truths are spectral-library molecules (in train), which PubChem knows
well (MassBank/GNPS deposits inflate #SID). A hidden-test molecule that is NOT
in any library is more like a COCONUT-only natural product. For a random
sample of each kind, find the molecule in PubChem's +-3 ppm neutral-mass window
(stereo-blind canonical SMILES) and record:
  found        is it in PubChem at all?          (hypothesis b: not in PubChem)
  pop          its popularity log1p(#SID)+log1p(#PMID)
  pop rank     its rank by popularity within the window (hypothesis a: leak)
Sets: train (library) molecules vs COCONUT-only pool molecules (pool730 minus train).

    python pc_novel.py <n_per_set> <workers>
"""
import bisect, random, sys
from multiprocessing import Pool
import numpy as np

D = r"C:\prog\casmi"
PPM = 3.0


def init():
    global M, OFF, POP, FS
    M = np.fromfile(fr"{D}\pc_mass.bin", dtype=np.float32)
    OFF = np.memmap(fr"{D}\pc_off.bin", dtype=np.uint64, mode="r")
    POP = np.memmap(fr"{D}\pc_pop.bin", dtype=np.float32, mode="r")
    FS = open(fr"{D}\pc_smi.bin", "rb")


def key(m):
    from rdkit import Chem
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m)


def work(item):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Descriptors import ExactMolWt
    RDLogger.DisableLog("rdApp.*")
    name, smi = item
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    if Chem.GetFormalCharge(m) != 0:
        return None
    mass = ExactMolWt(m)
    tk = key(m)
    lo = int(np.searchsorted(M, np.float32(mass * (1 - PPM * 1e-6)), "left"))
    hi = int(np.searchsorted(M, np.float32(mass * (1 + PPM * 1e-6)), "right"))
    if hi <= lo:
        return (name, 0, np.nan, np.nan, 0)
    o0, o1 = int(OFF[lo]), int(OFF[hi])
    FS.seek(o0)
    buf = FS.read(o1 - o0).decode("ascii", "replace")
    form = Chem.rdMolDescriptors.CalcMolFormula(m)
    found = -1
    for k in range(lo, hi):
        s = buf[int(OFF[k]) - o0:int(OFF[k + 1]) - o0]
        c = Chem.MolFromSmiles(s)
        if c is None or Chem.rdMolDescriptors.CalcMolFormula(c) != form:
            continue
        if key(c) == tk:
            found = k
            break
    pops = np.asarray(POP[lo:hi], dtype=float)
    if found < 0:
        return (name, 0, np.nan, np.nan, hi - lo)
    p = float(POP[found])
    rank = int((pops > p).sum())
    return (name, 1, p, rank, hi - lo)


if __name__ == "__main__":
    n, workers = int(sys.argv[1]), int(sys.argv[2])
    train = set(l.split("\t")[0] for l in open(fr"{D}\train_mols.txt", encoding="utf-8"))
    pool = [l.rstrip("\r\n").split("\t") for l in open(fr"{D}\pool730_mols.txt", encoding="utf-8")]
    rnd = random.Random(0)
    sets = {"train (library)": [p for p in pool if p[0] in train], "COCONUT-only": [p for p in pool if p[0] not in train]}
    for st in ("tqry", "cqry"):
        sets[st + " truths"] = [l.rstrip("\r\n").split("\t") for l in open(fr"{D}\{st}_mols.txt", encoding="utf-8")]
    with Pool(workers, initializer=init) as P:
        for lab, items in sets.items():
            sample = rnd.sample(items, min(n, len(items)))
            R = [r for r in P.map(work, sample, chunksize=4) if r is not None]
            f = np.array([r[1] for r in R]); pop = np.array([r[2] for r in R if r[1]]); rk = np.array([r[3] for r in R if r[1]])
            win = np.array([r[4] for r in R])
            print(f"{lab:16s} n {len(R)}  in PubChem {100*f.mean():.1f}%  window median {np.median(win):.0f}"
                  f"  | found: pop mean {pop.mean():.2f} median {np.median(pop):.2f}  pop-rank<1 {100*np.mean(rk<1):.1f}%"
                  f" <5 {100*np.mean(rk<5):.1f}% <25 {100*np.mean(rk<25):.1f}%  MRR(pop only) {np.mean([1/(r+1) if r<25 else 0 for r in rk]):.3f}"
                  f"  MRR incl. not-found {np.sum([1/(r+1) if r<25 else 0 for r in rk])/len(R):.3f}", flush=True)
