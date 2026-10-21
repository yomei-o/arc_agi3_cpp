"""e2e: adduct / precursor / truth mass of the molecules whose pool list came back empty."""
import sys, csv
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
t = pq.read_table(sys.argv[1]).to_pandas()
tr = {l.split("\t")[0]: l.rstrip("\n").split("\t") for l in open(sys.argv[2], encoding="utf-8")}
pool = {r[0]: [s for s in r[1].split(";") if s] for r in list(csv.reader(open(sys.argv[3], encoding="utf-8")))[1:]}
for mid, l in pool.items():
    if len(l) >= 12:
        continue
    m = Chem.MolFromSmiles(tr[mid][3])
    d = t[t.molecule_id == mid]
    print(mid, len(l), tr[mid][1], "M=%.4f" % ExactMolWt(m), list(zip(d.adduct, d.precursor_mz.round(4), d.instrument_type)))
