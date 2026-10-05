import sys
from pathlib import Path
root = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\prog\casmi")
lib = {l.split("\t")[0] for l in (root / "lib_mols.txt").read_text(encoding="utf-8").splitlines()}
qry_names = [l.split("\t")[0] for l in (root / "qry_mols.txt").read_text(encoding="utf-8").splitlines()]
missing = [n for n in qry_names if n not in lib]
print("qry molecules", len(qry_names), "missing from library", len(missing))
