"""e2e_pair: paired comparison of two e2e submission.csv files with the official identity (same keys and truth key
as e2e_score.score). Per set of truth.tsv: MRR@25 of A and B, B - A with a 2000-resample paired bootstrap 95% CI,
and how many molecules got better / worse.
usage: python e2e_pair.py <truth.tsv> <A.csv> <B.csv>
"""
import os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_score as es

if __name__ == "__main__":
    tr = [l.rstrip("\r\n").split("\t") for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
    tk = es.keys_for([t[3] for t in tr])
    A, B = es.read_sub(sys.argv[2]), es.read_sub(sys.argv[3])
    K = es.keys_for([s for c in (A, B) for t in tr for s in c.get(t[0], [])[:25]])

    def rr(lst, truthkey):
        for i, s in enumerate(lst[:25]):
            if K[s][0] is not None and K[s][0] == truthkey:
                return 1.0 / (i + 1)
        return 0.0

    by = {}
    for mid, st, ik14, smi in tr:
        truthkey = tk[smi][0] or ik14
        by.setdefault(st, []).append((rr(A.get(mid, []), truthkey), rr(B.get(mid, []), truthkey)))
    by["all"] = [x for v in list(by.values()) for x in v]
    rng = random.Random(1)
    for st, v in by.items():
        n = len(v)
        a = sum(x for x, _ in v) / n; b = sum(y for _, y in v) / n
        d = [y - x for x, y in v]
        bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(2000))
        print(f"[{st}] n={n}  A {a:.4f}  B {b:.4f}  B-A {b - a:+.4f} [{bs[50]:+.4f},{bs[1949]:+.4f}]  "
              f"better {sum(1 for x in d if x > 1e-12)} worse {sum(1 for x in d if x < -1e-12)}", flush=True)
