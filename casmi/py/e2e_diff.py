"""e2e: truth rank (official identity) in e2e vs eval list for the molecules whose lists differ, plus where
   the extra copies of the truth come from (count of truth-identical entries in each list)."""
import csv, sys, pickle, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2e_score import keys_for
tr = [l.rstrip("\n").split("\t") for l in open(sys.argv[1], encoding="utf-8")]
rd = lambda p: {r[0]: [s for s in r[1].split(";") if s] for r in list(csv.reader(open(p, encoding="utf-8")))[1:]}
A, B = rd(sys.argv[2]), rd(sys.argv[3])
K = keys_for([t[3] for t in tr] + [s for d in (A, B) for l in d.values() for s in l])
for mid, st, ik, smi in tr:
    a, b = A.get(mid, []), B.get(mid, [])
    if a == b: continue
    tk = K[smi][0]
    ra = [i + 1 for i, s in enumerate(a) if K[s][0] == tk]
    rb = [i + 1 for i, s in enumerate(b) if K[s][0] == tk]
    print(mid, st, "e2e truth at", ra, " eval truth at", rb, " e2e-only", len(set(a) - set(b)), "eval-only", len(set(b) - set(a)))
