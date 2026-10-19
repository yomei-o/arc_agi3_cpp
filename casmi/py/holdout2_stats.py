"""Per-molecule statistics of train.parquet for designing unbiased holdouts.

For each InChIKey14: total spectra, timsTOF spectra with precursor 245-461,
instruments, ingest libraries, COCONUT membership, and whether it is in one of
the existing holdouts (cqry/tqry/sqry/oqry/fqry/dqry). Writes
<root>/h2_molstats.tsv and prints a summary.

    python holdout2_stats.py [root]
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\prog\casmi")
f = pq.ParquetFile(ROOT / "train.parquet")
n_all = Counter()
n_tims = Counter()          # timsTOF and 245 <= precursor <= 461
n_instr = defaultdict(Counter)
libs = defaultdict(Counter)
for b in f.iter_batches(batch_size=400000,
                        columns=["inchikey14", "precursor_mz", "instrument_type", "ingest_lib"]):
    d = b.to_pandas()
    d = d[d.inchikey14.notna()]
    for k, p, ins, lib in zip(d.inchikey14, d.precursor_mz, d.instrument_type, d.ingest_lib):
        n_all[k] += 1
        n_instr[k][ins] += 1
        libs[k][lib] += 1
        if ins == "timsTOF" and 245.0 <= p <= 461.0:
            n_tims[k] += 1

coconut = set((ROOT / "coconut_keys.txt").read_text().split())
old = {}
for s in ("cqry", "tqry", "sqry", "oqry", "fqry", "dqry", "qry"):
    p = ROOT / f"{s}_mols.txt"
    if p.exists():
        for l in open(p, encoding="utf-8"):
            if l.strip():
                old.setdefault(l.split("\t")[0], s)

with open(ROOT / "h2_molstats.tsv", "w", encoding="utf-8") as o:
    o.write("key\tn_all\tn_tims\tn_timsall\tcoconut\told\tlibs\n")
    for k in n_all:
        o.write(f"{k}\t{n_all[k]}\t{n_tims[k]}\t{n_instr[k]['timsTOF']}\t{int(k in coconut)}\t{old.get(k, '')}\t"
                + ",".join(f"{a}:{c}" for a, c in libs[k].most_common()) + "\n")

keys = list(n_all)
print("molecules", len(keys), "spectra", sum(n_all.values()))
print("ingest libs (spectra):", Counter({l: c for k in keys for l, c in libs[k].items()}).most_common())
tims = [k for k in keys if n_tims[k] >= 1]
print(f"timsTOF 245-461 >=1: {len(tims)}  >=2: {sum(n_tims[k] >= 2 for k in tims)}  >=3: {sum(n_tims[k] >= 3 for k in tims)}")
print("  of which COCONUT:", sum(k in coconut for k in tims))
print("  timsTOF mols with non-timsTOF spectra too:", sum(n_all[k] > n_instr[k]['timsTOF'] for k in tims))
print("  ingest libs of timsTOF mols:", Counter(l for k in tims for l in libs[k]).most_common(10))
coc = [k for k in keys if k in coconut]
print(f"COCONUT >=1: {len(coc)}  >=2: {sum(n_all[k] >= 2 for k in coc)}  ==1: {sum(n_all[k] == 1 for k in coc)}")
print("spectra/mol distribution all:", sorted(Counter(min(n_all[k], 10) for k in keys).items()))
print("spectra/mol distribution coconut:", sorted(Counter(min(n_all[k], 10) for k in coc).items()))
print("timsTOF-range spectra/mol distribution (tims mols):", sorted(Counter(min(n_tims[k], 10) for k in tims).items()))
