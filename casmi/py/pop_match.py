"""Draw COCONUT train molecules whose PubChem popularity matches novel natural products.

Target distribution: COCONUT-only pool730 molecules (in COCONUT, not in train) =
what an unpublished natural product looks like to PubChem. Source: COCONUT train
molecules with >= 1 spectrum, not in older holdouts or in the --exclude files.
Popularity log1p(#SID)+log1p(#PMID), lookup from pc_novel.py; "not in PubChem"
is its own bin. Importance weights per bin, 400 drawn without replacement.

    python pop_match.py <out_keys.txt> <n_source> <n_target> <n_out> <exclude1,exclude2,..> <workers> [seed]
"""
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, r"C:\prog\casmi")
import pc_novel  # noqa: E402

D = r"C:\prog\casmi"
EDGES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]


def binof(r):
    if not r[1]:
        return 0
    return 1 + int(np.searchsorted(EDGES, r[2], "right"))


if __name__ == "__main__":
    out, n_src, n_tgt, n_out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    excl_files, workers = sys.argv[5], int(sys.argv[6])
    rng = np.random.default_rng(int(sys.argv[7]) if len(sys.argv) > 7 else 2603)
    train = dict(l.rstrip("\r\n").split("\t", 1) for l in open(fr"{D}\train_mols.txt", encoding="utf-8") if "\t" in l)
    ban = set()
    for fn in excl_files.split(","):
        ban |= {l.split("\t")[0].strip() for l in open(fr"{D}\{fn}", encoding="utf-8") if l.strip()}
    src = []
    for l in open(fr"{D}\h2_molstats.tsv", encoding="utf-8"):
        if l.startswith("key\t"):
            continue
        k, na, nt, nta, coc, old, libs = l.rstrip("\n").split("\t")
        if coc == "1" and not old and k not in ban and train.get(k):
            src.append(k)
    coco = set(open(fr"{D}\coconut_keys.txt").read().split())
    tgt = [l.rstrip("\r\n").split("\t") for l in open(fr"{D}\pool730_mols.txt", encoding="utf-8")]
    tgt = [t for t in tgt if len(t) == 2 and t[0] not in train and t[0] in coco]
    print(f"source pool {len(src)}  target pool {len(tgt)}", flush=True)
    s_idx = rng.choice(len(src), size=min(n_src, len(src)), replace=False)
    t_idx = rng.choice(len(tgt), size=min(n_tgt, len(tgt)), replace=False)
    with Pool(workers, initializer=pc_novel.init) as P:
        RS = [r for r in P.map(pc_novel.work, [(src[i], train[src[i]]) for i in s_idx], chunksize=4) if r is not None]
        RT = [r for r in P.map(pc_novel.work, [tuple(tgt[i]) for i in t_idx], chunksize=4) if r is not None]
    nb = len(EDGES) + 2
    hs = np.bincount([binof(r) for r in RS], minlength=nb) / len(RS)
    ht = np.bincount([binof(r) for r in RT], minlength=nb) / len(RT)
    w = np.where(hs > 0, ht / np.maximum(hs, 1e-12), 0.0)
    p = np.array([w[binof(r)] for r in RS])
    p /= p.sum()
    k = min(n_out, int((p > 0).sum()))
    pick = rng.choice(len(RS), size=k, replace=False, p=p)
    sel = [RS[i] for i in pick]
    with open(fr"{D}\{out}", "w") as o:
        o.write("\n".join(r[0] for r in sel) + "\n")

    def desc(lab, R):
        f = np.array([r[1] for r in R]); pop = np.array([r[2] for r in R if r[1]])
        rk = np.array([r[3] for r in R if r[1]])
        print(f"{lab:22s} n {len(R)} in PubChem {100*f.mean():.1f}% pop mean {pop.mean():.2f} median {np.median(pop):.2f}"
              f" q75 {np.quantile(pop,.75):.2f}  pop-only MRR {np.sum([1/(r+1) if r < 25 else 0 for r in rk])/len(R):.3f}")
    desc("target (COCONUT-only)", RT)
    desc("source (COCONUT train)", RS)
    desc("selected", sel)
    print("bin hist target", np.round(ht, 3).tolist())
    print("bin hist source", np.round(hs, 3).tolist())
    print("selected unique", len(set(r[0] for r in sel)), "max weight", w.max().round(2))
