# gentfm_eval.py: does the own Transformer (TFM) reverse score rank generated
# candidates better than the v4k MLP (REV2)? Works on phys_gentfm feature dumps
# (columns: phys mlp rev lib ana inLib name g0..g5 tfm-columns... smiles).
#   python gentfm_eval.py dump1.txt[:label] dump2.txt[:label] ...
# MRR@25 with ties averaged (expected reciprocal rank over a random order of the
# tied candidates). Base = v20 hand fusion + gen2 hand weights:
#   z(phys) + 0.25 z(mlp) + wr z(rev) + 8 lib^32 + 24 ana + gw . gen
# Variants: TFM column replaces rev / is added (+tw z(tfm)), controls: the TFM
# column shuffled within the query (mean of 5 seeds), reversed sign.
import sys
import numpy as np

GW = np.array([-2, 0, -1, 4, 0, 1.5])
TNAMES = ['A_logit', 'A_tan', 'B_logit', 'B_tan', 'C_logit', 'C_tan', 'ABC_logit', 'ABC_tan']


def load(p):
    Q = []
    f = open(p)
    for line in f:
        h = line.split()
        t, n = int(h[2]), int(h[3])
        rows = [f.readline().split() for _ in range(n)]
        if n == 0:
            Q.append(None); continue
        nt = len(rows[0]) - 14
        X = np.array([[float(v) for v in r[:6]] for r in rows])
        G = np.array([[float(v) for v in r[7:13]] for r in rows])
        T = np.array([[float(v) for v in r[13:13 + nt]] for r in rows]).reshape(n, nt)
        Q.append(dict(X=X, G=G, T=T, t=t, gen=G[:, 0] > 0))
    return Q


def z(v):
    return (v - v.mean()) / (v.std() + 1e-6)


def rr_avg(s, t, mask=None):
    if t < 0:
        return 0.0
    if mask is not None:
        if not mask[t]:
            return 0.0
        s = s[mask]
        t = int(mask[:t].sum())
    st = s[t]
    better = int((s > st).sum())
    ties = int((s == st).sum()) - 1
    return float(np.mean([1.0 / (better + 1 + j) if better + j < 25 else 0.0 for j in range(ties + 1)]))


def mrr(Q, score, mask_gen=False, only_gen_truth=False):
    tot, n = 0.0, 0
    for q in Q:
        if q is None:
            n += 0 if only_gen_truth else 1
            continue
        if only_gen_truth and not (q['t'] >= 0 and q['gen'][q['t']]):
            continue
        n += 1
        tot += rr_avg(score(q), q['t'], q['gen'] if mask_gen else None)
    return tot / max(1, n), n


def base(q, wr=1.0):
    X = q['X']
    return z(X[:, 0]) + 0.25 * z(X[:, 1]) + wr * z(X[:, 2]) + 8 * np.maximum(X[:, 3], 0) ** 32 + 24 * X[:, 4] + q['G'] @ GW


def shuf(v, seed):
    r = np.random.default_rng(seed)
    return r.permutation(v)


def report(path, label):
    Q = load(path)
    nq = len(Q)
    nt = next(q for q in Q if q is not None)['T'].shape[1]
    ngt = sum(1 for q in Q if q is not None and q['t'] >= 0 and q['gen'][q['t']])
    print(f'=== {label}: {nq} queries, truth = generated candidate in {ngt}, TFM columns {nt}')
    cols = list(range(nt))
    # 1) single scores alone, all candidates and generated-only ranking
    print('alone (all cands | among generated only, queries with generated truth):')
    for name, fn in [('MLP rev2', lambda q: q['X'][:, 2]), ('phys', lambda q: q['X'][:, 0]),
                     ('fwdMLP', lambda q: q['X'][:, 1])] + [(TNAMES[c], (lambda c: lambda q: q['T'][:, c])(c)) for c in cols]:
        a, _ = mrr(Q, fn)
        g, n = mrr(Q, fn, mask_gen=True, only_gen_truth=True)
        print(f'  {name:10s} all {a:.4f} | gen-only {g:.4f} (n={n})')
    b, _ = mrr(Q, base)
    bg, n = mrr(Q, base, mask_gen=True, only_gen_truth=True)
    b0, _ = mrr(Q, lambda q: base(q, 0.0))
    print(f'base fusion (wr=1 MLP): {b:.4f} | gen-only {bg:.4f}; without rev (wr=0): {b0:.4f}')
    for c in cols:
        line = f'  {TNAMES[c]:10s}'
        # replace rev by tfm
        for wr in (1, 1.5, 2, 3):
            m, _ = mrr(Q, lambda q: base(q, 0.0) + wr * z(q['T'][:, c]))
            line += f' rep{wr}={m:.4f}'
        for tw in (0.5, 1, 2, 3):
            m, _ = mrr(Q, lambda q: base(q) + tw * z(q['T'][:, c]))
            line += f' add{tw}={m:.4f}'
        print(line)
    # controls for the two main columns (ensemble logit/tanimoto or A)
    for c in ([nt - 2, nt - 1] if nt > 2 else cols):
        for tw in (1, 2):
            mr = mrr(Q, lambda q: base(q) - tw * z(q['T'][:, c]))[0]
            ms = np.mean([mrr(Q, lambda q: base(q) + tw * z(shuf(q['T'][:, c], s)))[0] for s in range(5)])
            mrep = np.mean([mrr(Q, lambda q: base(q, 0.0) + tw * z(shuf(q['T'][:, c], s)))[0] for s in range(5)])
            mreal = mrr(Q, lambda q: base(q) + tw * z(q['T'][:, c]))[0]
            print(f'  control {TNAMES[c]} tw={tw}: real add {mreal:.4f} | reversed sign {mr:.4f} | shuffled add {ms:.4f} | shuffled replace {mrep:.4f}')
    # paired bootstrap of the per-query difference vs base (add tw=1, replace wr=2) for the ensemble columns
    rng = np.random.default_rng(7)
    def per(score):
        return np.array([0.0 if q is None else rr_avg(score(q), q['t']) for q in Q])
    rb = per(base)
    for c in ([nt - 2, nt - 1] if nt > 2 else cols):
        for lab2, fn in [('add1', lambda q: base(q) + z(q['T'][:, c])), ('rep2', lambda q: base(q, 0.0) + 2 * z(q['T'][:, c]))]:
            d = per(fn) - rb
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            print(f'  bootstrap {TNAMES[c]} {lab2}: delta {d.mean():+.4f}  95% CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]  (queries up {int((d > 1e-9).sum())}, down {int((d < -1e-9).sum())})')
    return Q


if __name__ == '__main__':
    for a in sys.argv[1:]:
        p, lab = a.rsplit(':', 1) if ':' in a[2:] else (a, a)
        report(p, lab)
