"""e2e: per-molecule comparison of the e2e submission (molecule ids) with eval-mode CSVs (IK14 names).
   e2e_cmp.py truth.tsv e2e_sub.csv eval_hcqry.csv eval_hxqry.csv"""
import csv, sys
tr = [l.rstrip("\n").split("\t") for l in open(sys.argv[1], encoding="utf-8")]
def rd(p):
    return {r[0]: [s for s in r[1].split(";") if s] for r in list(csv.reader(open(p, encoding="utf-8")))[1:]}
e2e = rd(sys.argv[2]); ev = {}
for p in sys.argv[3:]:
    ev.update(rd(p))
same = diff = 0; ex = []
for mid, st, ik, smi in tr:
    a, b = e2e.get(mid, []), ev.get(ik, [])
    if a == b: same += 1
    else:
        diff += 1
        if len(ex) < 6:
            k = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
            ex.append((mid, st, ik, len(a), len(b), k, a[k] if k < len(a) else None, b[k] if k < len(b) else None))
print("identical lists", same, "different", diff)
for e in ex: print(e)
# also write the eval lists under the e2e molecule ids (for e2e_score.py score)
with open(sys.argv[2] + ".evalmap.csv", "w", encoding="utf-8") as f:
    f.write("molecule_id,smiles\n")
    for mid, st, ik, smi in tr:
        f.write(mid + "," + ";".join(ev.get(ik, [])) + "\n")
