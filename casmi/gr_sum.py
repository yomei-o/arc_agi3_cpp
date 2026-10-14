import sys, json, collections, statistics
R = [json.loads(l) for l in open(sys.argv[1])]
n = len(R)
print('queries', n, ' with >=1 analog', sum(r['nan'] > 0 for r in R),
      ' with a Δ-matched analog', sum(r['nmatch_an'] > 0 for r in R), ' iso(analog==truth)', sum(r['iso_hit'] is not None for r in R))
print('K   rec1    rec1+2   cand1(med/mean)   cand1+2(med/mean)')
for K in ('10', '20', '40', '80', '200'):
    if K not in R[0]['cnt']:
        continue
    h1 = sum(any(h['r1'] < int(K) for h in r['hits']) for r in R)
    h2 = sum(any(min(h['r1'], h['r2']) < int(K) for h in r['hits']) for r in R)
    c1 = [r['cnt'][K][0] for r in R]
    c2 = [r['cnt'][K][1] for r in R]
    print(f"{K:>3} {h1/n:.3f}  {h2/n:.3f}   {statistics.median(c1):6.0f} / {sum(c1)/n:7.0f}   {statistics.median(c2):6.0f} / {sum(c2)/n:7.0f}")
# by op (credit each op string of hit products, first hit per query)
ops1 = collections.Counter(); ops2 = collections.Counter()
for r in R:
    s1 = set(); s2 = set()
    for h in r['hits']:
        for o in h['ops']:
            (s1 if '/' not in o else s2).add(o)
    for o in s1: ops1[o] += 1
    for o in s2: ops2[o] += 1
print('one-step hits by type:', ops1.most_common(30))
print('two-step hits by type:', ops2.most_common(30))
# sim of the analog producing the truth
sims = [max(h['sim'] for h in r['hits']) for r in R if r['hits']]
bins = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
print('hit analog sim:', [(bins[i], sum(bins[i] <= s < bins[i+1] for s in sims)) for i in range(len(bins)-1)])
# recall vs the query's top analog sim
for lo, hi in ((0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)):
    sub = [r for r in R if lo <= r['top_sim'] < hi]
    if sub:
        print(f'top analog sim [{lo},{hi}): n={len(sub)} rec1+2@80={sum(bool(r["hits"]) for r in sub)/len(sub):.3f}')

for cap in (25, 50, 100, 200, 500, 1000, 3000):
    print(f'recall within first {cap} candidates (sim-ordered): {sum(any(h.get("pos", 1e9) < cap for h in r["hits"]) for r in R)/n:.3f}')
