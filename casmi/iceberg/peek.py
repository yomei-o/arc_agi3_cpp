"""peek.py <scores.json>: per molecule, mean (entropy, cosine) of each candidate in input order (smoke check)."""
import json, sys
r = json.load(open(sys.argv[1]))
for m, v in r.items():
    print(m, [(round(sum(x[0] for x in v[s]) / max(1, len(v[s])), 3), round(sum(x[1] for x in v[s]) / max(1, len(v[s])), 3), len(v[s])) for s in v])
