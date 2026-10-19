"""PubChem popularity log1p(#SID)+log1p(#PMID) of molecule groups (lookup code from pc_novel.py).

    python pop_sample.py stats <n_per_group> <workers>
        popularity of random samples of train molecule groups (from h2_molstats.tsv)
    python pop_sample.py keys <keys.txt> <out.tsv> <workers>
        popularity of every molecule listed (first field) -> out.tsv: key, found, pop
"""
import random, sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, r"C:\prog\casmi")
import pc_novel  # noqa: E402

D = r"C:\prog\casmi"


def smiles_of():
    return dict(l.rstrip("\r\n").split("\t", 1) for l in open(fr"{D}\train_mols.txt", encoding="utf-8") if "\t" in l)


def groups():
    G = {"all train": [], "coconut n=1": [], "coconut n>=2": [], "enveda-180": [], "enveda-np-examples": [],
         "timsTOF-range": [], "non-coconut non-enveda": []}
    for l in open(fr"{D}\h2_molstats.tsv", encoding="utf-8"):
        if l.startswith("key\t"):
            continue
        k, na, nt, nta, coc, old, libs = l.rstrip("\n").split("\t")
        if old:
            continue
        G["all train"].append(k)
        if coc == "1":
            G["coconut n=1" if na == "1" else "coconut n>=2"].append(k)
        if "enveda-180" in libs:
            G["enveda-180"].append(k)
        if "enveda-np-examples" in libs:
            G["enveda-np-examples"].append(k)
        if int(nt) >= 1:
            G["timsTOF-range"].append(k)
        if coc == "0" and "enveda" not in libs:
            G["non-coconut non-enveda"].append(k)
    return G


def summarize(lab, R):
    f = np.array([r[1] for r in R])
    pop = np.array([r[2] for r in R if r[1]])
    rk = np.array([r[3] for r in R if r[1]])
    print(f"{lab:24s} n {len(R)} in PubChem {100*f.mean():.1f}% | pop mean {pop.mean():.2f} median {np.median(pop):.2f}"
          f" q25 {np.quantile(pop,.25):.2f} q75 {np.quantile(pop,.75):.2f}"
          f" | pop-only MRR {np.sum([1/(r+1) if r < 25 else 0 for r in rk])/len(R):.3f}", flush=True)


if __name__ == "__main__":
    smi = smiles_of()
    if sys.argv[1] == "stats":
        n, workers = int(sys.argv[2]), int(sys.argv[3])
        rnd = random.Random(0)
        with Pool(workers, initializer=pc_novel.init) as P:
            for lab, ks in groups().items():
                s = rnd.sample(ks, min(n, len(ks)))
                R = [r for r in P.map(pc_novel.work, [(k, smi[k]) for k in s if smi.get(k)], chunksize=4) if r is not None]
                summarize(lab, R)
    else:
        keys = [l.split("\t")[0].strip() for l in open(sys.argv[2], encoding="utf-8") if l.strip()]
        workers = int(sys.argv[4])
        with Pool(workers, initializer=pc_novel.init) as P:
            R = P.map(pc_novel.work, [(k, smi[k]) for k in keys if smi.get(k)], chunksize=4)
        with open(sys.argv[3], "w") as o:
            for r in R:
                if r is not None:
                    o.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\n")
        summarize(sys.argv[2], [r for r in R if r is not None])
