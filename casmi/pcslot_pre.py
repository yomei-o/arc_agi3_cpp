# Prefilter study on a pcall dump (name cid rev pop truth per window candidate):
# truth rank within the PubChem window under rev + w*pop orderings.
import sys
from collections import defaultdict
Q = defaultdict(list)
for line in open(sys.argv[1], encoding="utf-8"):
    n, cid, r, p, t = line.split("\t")
    Q[n].append((float(r), float(p), int(t)))
nq = int(sys.argv[2]) if len(sys.argv) > 2 else len(Q)
print("queries with dump", len(Q), "truth in window", sum(any(x[2] for x in v) for v in Q.values()))
for w in (0, 0.5, 1, 2, 3, 5, 8, 12, 20, 50, 1e6):
    ranks = []
    for v in Q.values():
        tr = [x for x in v if x[2]]
        if not tr: continue
        s = max(x[0] + w * x[1] for x in tr) if w < 1e5 else max(x[1] for x in tr)
        # pessimistic ties (pop alone ties a lot): every other candidate scoring >= the truth counts above it
        r = sum(1 for x in v if not x[2] and (x[0] + w * x[1] if w < 1e5 else x[1]) >= s)
        ranks.append(r)
    m = sum(1 / (r + 1) for r in ranks if r < 25) / nq
    print("w_pop %-6g: <1 %3d <5 %3d <25 %3d <100 %3d <300 %3d <1000 %3d  MRR(window only) %.3f" % (
        w, sum(r < 1 for r in ranks), sum(r < 5 for r in ranks), sum(r < 25 for r in ranks), sum(r < 100 for r in ranks),
        sum(r < 300 for r in ranks), sum(r < 1000 for r in ranks), m))
# pop distribution: truth vs others
import statistics
tp = [x[1] for v in Q.values() for x in v if x[2]]
op = [x[1] for v in Q.values() for x in v if not x[2]]
print("pop truth median %.2f mean %.2f | others median %.2f mean %.2f; frac others pop==0 %.3f" % (
    statistics.median(tp), statistics.mean(tp), statistics.median(op), statistics.mean(op), sum(p == 0 for p in op) / len(op)))
