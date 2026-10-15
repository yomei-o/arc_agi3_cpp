# Write the score-mode merged CSV for one setting from phys_pc's plain CSV +
# pcdet dump (eval names: the truth's own name for a PubChem pick equal to the
# truth, P<cid> otherwise), so mrr.py re-scores exactly what the setting outputs.
#   python pcslot_write.py <dir> <tag> <k> <sw> <bias> <pmin> <gate> <out.csv> [pcpop=2]
import csv, sys
sys.path.insert(0, r"C:\prog\casmi\pcslot")
import pcslot_grid as G

d, tag = sys.argv[1], sys.argv[2]
k, sw, bias, pmin, gate = int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]), float(sys.argv[7])
out = sys.argv[8]
pcpop = float(sys.argv[9]) if len(sys.argv) > 9 else 2.0
plain, Q = G.load(d, tag)
rk = G.ranker(w_pop=pcpop)
with open(out, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["molecule_id", "smiles"])
    for name, L in plain.items():
        q = Q.get(name)
        pick, ps = [], []
        if q is not None and q["lib"] < gate:
            for c in rk(q)[:k]:
                pick.append(name if c["truth"] else "P" + c["cid"])
                ps.append(c["zp"] + 0.25 * c["zm"] + c["zr"] + 24 * c["ana"] + sw * c["pop"] + bias)
        S = q["topS"] if q is not None else []
        res, a, b = [], 0, 0
        for pos in range(1, 26):
            if b < len(pick) and pos >= pmin and (a >= len(S) or ps[b] > S[a]): res.append(pick[b]); b += 1
            elif a < len(L): res.append(L[a]); a += 1
            else: break
        w.writerow([name, ";".join(res)])
