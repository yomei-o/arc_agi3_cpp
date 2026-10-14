# gen_fuse.py: conditional-logit fusion with generation features.
#   python gen_fuse.py <dir> <pool730_mols.txt>
# expects gen_f_{c,t}{A,B}.txt, f21_{c,t}qry.txt in dir and fwdmodel/gen/gc_{c,t}qry.txt
import sys, math
import numpy as np
d = sys.argv[1]
poolnames = set(l.split('\t')[0] for l in open(sys.argv[2], encoding='utf8'))

def read_feat(p):
    Q = []
    f = open(p)
    for line in f:
        h = line.split()
        name, t, n = h[1], int(h[2]), int(h[3])
        rows = [f.readline().split() for _ in range(n)]
        X = np.array([[float(v) for v in r[:6]] for r in rows], dtype=np.float64).reshape(-1, 6)
        Q.append((name, t, X, [r[6] for r in rows]))
    return Q

def read_gc(p):
    G = {}
    cur = None
    for l in open(p, encoding='utf8'):
        if l.startswith('G '):
            cur = G.setdefault(l.split()[1], {})
            continue
        x = l.rstrip('\n').split('\t')
        cur[x[0]] = (float(x[3]), int(x[4]), int(x[5]), int(x[6]), x[7])
    return G

def z(v):
    s = v.std()
    return (v - v.mean()) / (s + 1e-6)

NB = 8
def build(Q, G, gen, poolnames=poolnames):
    out = []
    for name, t, X, names in Q:
        if len(names) == 0:
            out.append((None, t)); continue
        lib = np.maximum(X[:, 3], 0)
        inlib = X[:, 5]
        cols = [z(X[:, 0]), z(X[:, 1]), z(X[:, 2]), lib ** 32, lib ** 8, lib, X[:, 4], inlib]
        if gen:
            g = G.get(name, {})
            isg = np.array([1.0 if n in g else 0.0 for n in names])
            sim = np.array([g[n][0] if n in g else 0.0 for n in names])
            two = np.array([1.0 if n in g and g[n][2] == 2 else 0.0 for n in names])
            sup = np.array([math.log1p(g[n][3]) if n in g else 0.0 for n in names])
            new = np.array([0.0 if n in poolnames else 1.0 for n in names])
            cols += [isg, isg * sim ** 3, two, sup, new, new * X[:, 4]]
        out.append((np.stack(cols, 1), t))
    return out

def fit(data, iters=600, l2=1e-3):
    data = [(X, t) for X, t in data if X is not None and t >= 0]
    w = np.zeros(data[0][0].shape[1])
    m = np.zeros_like(w); v = np.zeros_like(w)
    for it in range(1, iters + 1):
        g = -l2 * w
        for X, t in data:
            s = X @ w
            s -= s.max()
            p = np.exp(s); p /= p.sum()
            g += X[t] - p @ X
        g /= len(data)
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        w += 0.05 * (m / (1 - 0.9 ** it)) / (np.sqrt(v / (1 - 0.999 ** it)) + 1e-8)
    return w

def mrr(data, w):
    tot = 0.0
    for X, t in data:
        if X is None or t < 0:
            continue
        s = X[:, :len(w)] @ w
        better = int((s >= s[t]).sum()) - 1
        if better < 25:
            tot += 1.0 / (better + 1)
    return tot / len(data)

S = {}
for k in ('c', 't'):
    q = 'cqry' if k == 'c' else 'tqry'
    G = read_gc(f'{d}/fwdmodel/gen/gc_{q}.txt')
    # 'new' = not a pool member once this holdout's answers are taken out
    # (otherwise a regenerated answer is flagged as a pool member: a leak)
    pn = poolnames - set(l.split('	')[0] for l in open(f'{d}/{q}_mols.txt', encoding='utf8'))
    S[k] = {
        'base': build(read_feat(f'{d}/f21_{q}.txt'), G, False),
        'A': build(read_feat(f'{d}/gen_f_{k}A.txt'), G, True, poolnames),
        'B': build(read_feat(f'{d}/gen_f_{k}B.txt'), G, True, pn),
    }
    nA = np.mean([X.shape[0] for X, t in S[k]['A'] if X is not None])
    nb = np.mean([X.shape[0] for X, t in S[k]['base'] if X is not None])
    print(f'{q}: cands/query base {nb:.0f}, +gen {nA:.0f}; truth in B-pool cands: '
          f'{sum(t >= 0 for X, t in S[k]["B"])}/{len(S[k]["B"])}', flush=True)
for tr, te in (('c', 't'), ('t', 'c')):
    wb = fit(S[tr]['base'])
    wg = fit(S[tr]['A'] + S[tr]['B'])
    # base weights on the augmented pool (no gen features) = naive merge
    A0 = mrr(S[te]['base'], wb)
    An = mrr([(X[:, :NB], t) if X is not None else (X, t) for X, t in S[te]['A']], wb)
    Bn = mrr([(X[:, :NB], t) if X is not None else (X, t) for X, t in S[te]['B']], wb)
    Ag = mrr(S[te]['A'], wg)
    Bg = mrr(S[te]['B'], wg)
    print(f'train {tr} -> test {te}:')
    print(f'  answer in pool   : no gen {A0:.4f} | +gen naive {An:.4f} | +gen with gen features {Ag:.4f}')
    print(f'  answer NOT in pool: no gen 0.0000 | +gen naive {Bn:.4f} | +gen with gen features {Bg:.4f}')
    for p in (0.5, 0.7):
        print(f'  mix p_inpool={p}: no gen {p*A0:.4f} | +gen {p*Ag+(1-p)*Bg:.4f}')
    print('  gen weights', np.round(wg, 3).tolist())
