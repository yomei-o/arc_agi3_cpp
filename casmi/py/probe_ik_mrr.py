"""probe_ik_mrr.py <qry>_mols.txt <list_smi.csv> [more.csv ...]
MRR@25 of SMILES lists scored like Kaggle: InChIKey first block (14 chars) of each SMILES vs the
query name (= truth IK14). Prints MRR, hits, mean list length and a 95% bootstrap CI (2000 resamples).
"""
import csv, sys, random
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

truth = [l.split("\t")[0].strip() for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
for p in sys.argv[2:]:
    c = {}
    with open(p, encoding="utf-8") as f:
        r = csv.reader(f); next(r)
        for row in r:
            c[row[0]] = [s for s in row[1].split(";") if s] if len(row) > 1 else []
    rr, lens, nbad = [], [], 0
    for k in truth:
        l = c.get(k, [])[:25]
        lens.append(len(l))
        v = 0.0
        for i, s in enumerate(l):
            m = Chem.MolFromSmiles(s)
            if m is None:
                nbad += 1; continue
            ik = Chem.MolToInchiKey(m)
            if ik[:14] == k[:14]:
                v = 1.0 / (i + 1); break
        rr.append(v)
    n = len(rr)
    mean = sum(rr) / max(1, n)
    rng = random.Random(1)
    bs = sorted(sum(rng.choice(rr) for _ in range(n)) / n for _ in range(2000))
    print(f"{p}: IK14 MRR@25 {mean:.4f}  95% CI [{bs[50]:.4f}, {bs[1949]:.4f}]  hits {sum(1 for x in rr if x > 0)}/{n}"
          f"  top1 {sum(1 for x in rr if x == 1.0)}  mean len {sum(lens)/max(1,n):.2f}  unparsable {nbad}")
