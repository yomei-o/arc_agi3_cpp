"""e2e: the notebook merge (pool list, then the library list, exact-string dedup) vs the same merge with
   official-identity dedup (stereo-free canonical tautomer IK14) -> writes <out> for e2e_score.py.
   e2e_dedup.py pool_candidates.csv direct_submission.csv out.csv"""
import csv, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2e_score import keys_for
rd = lambda p: {r[0]: [s for s in r[1].split(";") if s] for r in list(csv.reader(open(p, encoding="utf-8")))[1:]}
P, D = rd(sys.argv[1]), rd(sys.argv[2])
K = keys_for([s for d in (P, D) for l in d.values() for s in l])
n_rows = n_moved = 0
with open(sys.argv[3], "w", encoding="utf-8") as f:
    f.write("molecule_id,smiles\n")
    for mid, d in D.items():
        out, seen = [], set()
        for s in P.get(mid, []) + d:
            k = K[s][0] or s
            if k in seen: continue
            seen.add(k); out.append(s)
            if len(out) == 25: break
        plain, ss = [], set()
        for s in P.get(mid, []) + d:
            if s not in ss: ss.add(s); plain.append(s)
            if len(plain) == 25: break
        n_rows += out != plain
        f.write(mid + "," + ";".join(out) + "\n")
print("rows changed by official dedup:", n_rows)
