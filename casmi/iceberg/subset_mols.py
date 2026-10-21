"""subset_mols.py <mols.txt> <scores.json> <out mols.txt>: keep the holdout answers whose query is in the score file."""
import json, sys
keep = set(json.load(open(sys.argv[2])))
lines = [l for l in open(sys.argv[1], encoding="utf-8") if l.split("\t")[0] in keep]
open(sys.argv[3], "w", encoding="utf-8").writelines(lines)
print(len(lines), "answers kept")
