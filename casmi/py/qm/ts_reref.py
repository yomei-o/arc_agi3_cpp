"""Re-reference ipso barriers to the molecule's most stable protonation (ts_ref.py).

new = old + (E0_used - E_ref) * 627.509, for pt_cost and ipso_total (and the
direct barrier). Writes a CSV with the same columns as ts_ipso.py so that
ts_ipso_vs_obs.py / ts_ipso_subst.py run on it unchanged.

    python ts_reref.py <ts_ipso.csv> <ts_scan.csv> <ts_ref.csv> <out.csv>
"""
import csv, sys

H = 627.509
E0 = {}
for r in csv.DictReader(open(sys.argv[2])):
    E0[(r["name"], r["i"], r["j"])] = float(r["E0"])
ref = {r["name"]: float(r["E_ref"]) for r in csv.DictReader(open(sys.argv[3]))}
rows = list(csv.DictReader(open(sys.argv[1])))
out, shifts = [], []
for r in rows:
    k = (r["name"], r["i"], r["j"])
    if k not in E0 or r["name"] not in ref:
        continue
    d = (E0[k] - ref[r["name"]]) * H        # >= 0 when the reference is the true minimum
    shifts.append(d)
    r = dict(r)
    for c in ("direct", "pt_cost", "ipso_total", "ipso_reaction"):
        r[c] = f"{float(r[c]) + d:.3f}"
    out.append(r)
with open(sys.argv[4], "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0].keys())); w.writeheader(); w.writerows(out)
shifts.sort()
print(f"{len(out)}/{len(rows)} rows re-referenced; shift (site used above the most stable site) "
      f"median {shifts[len(shifts)//2]:.1f}, max {shifts[-1]:.1f}, min {shifts[0]:.1f} kcal/mol; "
      f"negative proton-move costs now {sum(float(r['pt_cost']) < 0 for r in out)}")
