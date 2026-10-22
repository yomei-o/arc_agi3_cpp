"""tune38_check: the offline re-scoring (tune38_core) at the v37 weights vs the C++ run itself.
usage: python tune38_check.py <dir with <tag>_td.txt, _det.txt, .csv, _ins.csv, <qry>_mols.txt> tag:qry[,...]
Prints max |score - C++ final score|, pool-list and final-list differences vs the C++ CSVs, MRR both ways."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tune38_core as tc  # noqa: E402


def main():
    D = sys.argv[1]
    for x in sys.argv[2].split(","):
        tag, qry = x.split(":")
        Q = tc.load(D, tag, f"{D}/{qry}_mols.txt")
        cp, ci = tc.csv_lists(f"{D}/{tag}.csv"), tc.csv_lists(f"{D}/{tag}_ins.csv")
        md, dp, di, rc, rs = 0.0, 0, 0, [], []
        for q in Q:
            if q["n"] == 0:
                continue
            sc = tc.scores(q, tc.V37)
            md = max(md, float(abs(sc - q["final_cpp"]).max()))
            P, F = tc.final_list(q, tc.V37, sc)
            dp += cp.get(q["name"], [])[:25] != P
            di += ci.get(q["name"], [])[:25] != F
            rs.append(tc.rr(q, F))
            rc.append(tc.rr(q, ci.get(q["name"], [])[:25]))
        print(f"{tag} ({qry}): queries {len(Q)} with dump {len(rs)}, max |score diff| {md:.2e}, pool-list diffs {dp}, "
              f"final-list diffs {di}, final MRR sim {sum(rs)/len(Q):.4f} C++ {sum(rc)/len(Q):.4f}", flush=True)


if __name__ == "__main__":
    main()
