"""probe_leak.py <qry>_mols.txt <list_smi.csv>: print the list entries whose InChIKey14 equals the query's
truth (for the answer-in-pool sanity run these are leaks: same IK14, different stereo-blind canonical SMILES)."""
import csv, sys
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
ts = {}
for l in open(sys.argv[1], encoding="utf-8"):
    a = l.rstrip("\n").split("\t")
    if a and a[0]:
        ts[a[0][:14]] = a[-1]
with open(sys.argv[2], encoding="utf-8") as f:
    rd = csv.reader(f); next(rd)
    for row in rd:
        k = row[0][:14]
        for i, x in enumerate(row[1].split(";") if len(row) > 1 else []):
            m = Chem.MolFromSmiles(x)
            if m is not None and Chem.MolToInchiKey(m)[:14] == k:
                print(k, "rank", i + 1, "| list:", x, "| truth:", ts.get(k))
