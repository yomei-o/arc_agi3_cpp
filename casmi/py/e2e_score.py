"""e2e scoring / inspection of submission.csv files with the official identity
(stereo stripped -> RDKit canonical tautomer -> InChIKey[:14]; same as pp_proto.compute_key).

  e2e_score.py score <truth.tsv> <sub.csv> [more.csv ...]      truth.tsv: molecule_id set ik14 smiles
  e2e_score.py names <qry_mols.txt> <eval.csv> [...]            eval-mode CSV (names = IK14 / P<cid> / G<k>):
                                                                 plain name MRR (as mrr.py)
  e2e_score.py inspect <test.parquet|-> <sub.csv> [...]         sanity counts of a Kaggle submission
Key cache: e2e_keycache.pkl next to this script (4 worker processes).
"""
import csv, os, pickle, sys, random
from collections import Counter
from multiprocessing import Pool

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
_te = None
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2e_keycache.pkl")


def compute_key(smi):
    global _te
    if _te is None:
        _te = rdMolStandardize.TautomerEnumerator()
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return (None, None, 0, 0)
    try:
        ik = Chem.MolToInchiKey(m)[:14] or None
    except Exception:
        ik = None
    try:
        m2 = Chem.Mol(m)
        Chem.RemoveStereochemistry(m2)
        k = Chem.MolToInchiKey(_te.Canonicalize(m2))[:14] or None
    except Exception:
        k = None
    frag = len(Chem.GetMolFrags(m))
    chg = sum(abs(a.GetFormalCharge()) for a in m.GetAtoms())
    return (k, ik, frag, chg)


def load_cache():
    try:
        return pickle.load(open(CACHE, "rb"))
    except Exception:
        return {}


def keys_for(smis):
    cache = load_cache()
    todo = sorted(set(s for s in smis if s not in cache))
    if todo:
        with Pool(4) as p:
            for s, v in zip(todo, p.map(compute_key, todo, chunksize=32)):
                cache[s] = v
        pickle.dump(cache, open(CACHE + ".tmp", "wb")); os.replace(CACHE + ".tmp", CACHE)
        print(f"  computed {len(todo)} keys", flush=True)
    return cache


def read_sub(p):
    c = {}
    with open(p, encoding="utf-8") as f:
        r = csv.reader(f); next(r)
        for row in r:
            if row:
                c[row[0]] = [s for s in (row[1] if len(row) > 1 else "").split(";") if s]
    return c


def boot(rr):
    n = len(rr); rng = random.Random(1)
    bs = sorted(sum(rng.choice(rr) for _ in range(n)) / n for _ in range(1000))
    return bs[25], bs[974]


def score(truth_path, subs):
    tr = [l.rstrip("\r\n").split("\t") for l in open(truth_path, encoding="utf-8") if l.strip()]
    tk = keys_for([t[3] for t in tr])
    for p in subs:
        c = read_sub(p)
        allsmi = [s for t in tr for s in c.get(t[0], [])[:25]]
        K = keys_for(allsmi)
        by = {}
        for mid, st, ik14, smi in tr:
            truthkey = tk[smi][0] or ik14
            l = c.get(mid, [])[:25]
            v_off = v_ik = 0.0
            for i, s in enumerate(l):
                k, ik, _, _ = K[s]
                if v_off == 0 and k is not None and k == truthkey:
                    v_off = 1.0 / (i + 1)
                if v_ik == 0 and ik == ik14:
                    v_ik = 1.0 / (i + 1)
            by.setdefault(st, []).append((v_off, v_ik, len(l), mid in c))
        for st, rows in by.items():
            off = [r[0] for r in rows]; ikr = [r[1] for r in rows]
            lo, hi = boot(off)
            print(f"{p} [{st}] n={len(rows)} official MRR@25 {sum(off)/len(off):.4f} [{lo:.4f},{hi:.4f}]  "
                  f"plain-IK14 {sum(ikr)/len(ikr):.4f}  hits {sum(1 for x in off if x > 0)}  "
                  f"top1 {sum(1 for x in off if x == 1)}  mean len {sum(r[2] for r in rows)/len(rows):.2f}  "
                  f"rows present {sum(r[3] for r in rows)}", flush=True)


def names(qmols, subs):
    tr = [l.rstrip("\r\n").split("\t")[0] for l in open(qmols, encoding="utf-8") if l.strip()]
    for p in subs:
        c = read_sub(p)
        rr = []
        for k in tr:
            l = c.get(k, [])[:25]
            rr.append(next((1.0 / (i + 1) for i, s in enumerate(l) if s == k), 0.0))
        print(f"{p}: name MRR@25 {sum(rr)/len(rr):.4f} n={len(rr)}")


def inspect(test_pq, subs):
    tmols = None
    if test_pq != "-":
        import pyarrow.parquet as pq
        tmols = set(pq.read_table(test_pq, columns=["molecule_id"]).column(0).to_pylist())
    for p in subs:
        c = read_sub(p)
        allsmi = [s for l in c.values() for s in l]
        K = keys_for(allsmi)
        n_rows = len(c)
        lens = Counter(min(len(l), 26) for l in c.values())
        bad = sum(1 for s in allsmi if K[s][1] is None)
        nokey = sum(1 for s in allsmi if K[s][0] is None)
        multi = sum(1 for s in allsmi if K[s][2] > 1)
        chg = sum(1 for s in allsmi if K[s][3] > 0)
        cco = sum(1 for s in allsmi if s == "CCO")
        dup_rows = dup_slots = dup_exact = 0
        for l in c.values():
            ks = [K[s][0] for s in l[:25] if K[s][0]]
            d = len(ks) - len(set(ks))
            dup_slots += d; dup_rows += d > 0
            dup_exact += len(l[:25]) - len(set(l[:25]))
        same = Counter(";".join(l) for l in c.values())
        top1 = Counter(l[0] if l else "" for l in c.values())
        print(f"== {p}: rows {n_rows}, candidates {len(allsmi)}, unique smiles {len(set(allsmi))}")
        print(f"   list length histogram {dict(sorted(lens.items()))}  (<25: {sum(v for k, v in lens.items() if k < 25)}, >25: {lens.get(26, 0)})")
        print(f"   unparsable {bad}, no official key {nokey}, multi-component {multi}, charged {chg}, 'CCO' {cco}")
        print(f"   official-identity duplicates: {dup_slots} slots in {dup_rows} rows; exact-string duplicates {dup_exact}")
        print(f"   most repeated full list x{same.most_common(1)[0][1]}; most repeated top-1 x{top1.most_common(1)[0][1]} ({top1.most_common(1)[0][0][:60]})")
        if tmols is not None:
            print(f"   coverage: test molecules {len(tmols)}, in sub {len(tmols & set(c))}, extra ids {len(set(c) - tmols)}")
        ex = [s for s in allsmi if K[s][2] > 1 or K[s][3] > 0][:8]
        if ex:
            print("   examples multi/charged:", ex)


if __name__ == "__main__":
    if os.name == "nt":  # share the 4 cores of the e2e run (children inherit the affinity)
        import ctypes
        ctypes.windll.kernel32.SetProcessAffinityMask(ctypes.windll.kernel32.GetCurrentProcess(), 0xF0)
    mode = sys.argv[1]
    if mode == "score":
        score(sys.argv[2], sys.argv[3:])
    elif mode == "names":
        names(sys.argv[2], sys.argv[3:])
    else:
        inspect(sys.argv[2], sys.argv[3:])
