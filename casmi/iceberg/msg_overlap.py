r"""msg_overlap.py <root> <MassSpecGym.tsv> <qry> ...: which holdout answers are MassSpecGym molecules (ICEBERG /
GLACIER training data), by InChIKey first block and MassSpecGym fold. Writes <root>\iceberg\msg_ik14_<fold>.txt
and prints per holdout: n, in MSG (any fold), in MSG train/val, in MSG test; and the same for train.parquet
molecules by ingest_lib."""
import sys
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

root, tsv = Path(sys.argv[1]), sys.argv[2]
m = pd.read_csv(tsv, sep="\t", usecols=["inchikey", "fold"])
m["ik14"] = m.inchikey.str[:14]
fold = {f: set(g.ik14) for f, g in m.groupby("fold")}
allk = set(m.ik14)
trainval = fold.get("train", set()) | fold.get("val", set())
for f, s in fold.items():
    (root / "iceberg" / f"msg_ik14_{f}.txt").write_text("\n".join(sorted(s)))
print("MassSpecGym:", len(m), "spectra,", len(allk), "molecules;", {f: len(s) for f, s in fold.items()})
for q in sys.argv[3:]:
    names = [l.split("\t")[0] for l in (root / f"{q}_mols.txt").read_text(encoding="utf-8").splitlines()]
    n = len(names)
    a = sum(k in allk for k in names)
    tv = sum(k in trainval for k in names)
    te = sum(k in fold.get("test", set()) for k in names)
    print(f"{q}: {n} answers, in MSG {a} ({a / n:.1%}), train/val {tv} ({tv / n:.1%}), test {te}")
t = pq.read_table(root / "train.parquet", columns=["inchikey14", "ingest_lib"]).to_pandas().drop_duplicates()
for lib, g in t.groupby("ingest_lib"):
    ks = set(g.inchikey14)
    if len(ks) < 200:
        continue
    print(f"train.parquet {lib}: {len(ks)} molecules, in MSG {sum(k in allk for k in ks) / len(ks):.1%}, "
          f"train/val {sum(k in trainval for k in ks) / len(ks):.1%}")
