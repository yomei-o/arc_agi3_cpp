"""check that e2e_rerank's merge reproduces the notebook's submission.csv from its own pool_candidates.csv.
usage: python e2e_remerge_check.py <work dir>"""
import csv, sys
from pathlib import Path
W = Path(sys.argv[1])


def rd(p):
    with open(p, encoding="utf-8") as f:
        return {r[0]: r[1] for r in csv.reader(f) if r and r[0] != "molecule_id"}


d, p, s = rd(W / "direct_submission.csv"), rd(W / "pool_candidates.csv"), rd(W / "submission.csv")
bad = 0
for m, v in s.items():
    merged, seen = [], set()
    for x in [x for x in p.get(m, "").split(";") if x] + [x for x in d.get(m, "").split(";") if x]:
        if x not in seen:
            seen.add(x); merged.append(x)
        if len(merged) == 25:
            break
    bad += (";".join(merged) if merged else "CCO") != v
print("rows", len(s), "differing", bad)
