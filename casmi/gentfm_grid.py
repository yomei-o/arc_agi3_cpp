# gentfm_grid.py: grid over TFM column / wr / tw on the six phys_gentfm dumps in the cwd (f_c_g, f_c_rm, f_s_g, f_t_g, f_t_rm, f_o_g); selection = (cqry + cqry_rm + 2 sqry)/4
import sys, numpy as np
sys.path.insert(0, r'C:\prog\claude4\arc\arc3\casmi')
import gentfm_eval as g
S = {k: g.load(f) for k, f in [('c','f_c_g.txt'),('crm','f_c_rm.txt'),('s','f_s_g.txt'),('t','f_t_g.txt'),('trm','f_t_rm.txt'),('o','f_o_g.txt')]}
res = []
for c in [0,1,6,7]:
    for wr in [0, 0.5, 1]:
        for tw in [0.5,1,1.5,2,3]:
            f = lambda q: g.base(q, wr) + tw * g.z(q['T'][:, c])
            m = {k: g.mrr(Q, f)[0] for k, Q in S.items()}
            sel = (m['c'] + m['crm'] + 2*m['s'])/4
            res.append((sel, g.TNAMES[c], wr, tw, m))
b = {k: g.mrr(Q, g.base)[0] for k, Q in S.items()}
print('base', ' '.join(f'{k} {v:.4f}' for k, v in b.items()))
res.sort(key=lambda r: -r[0])
for r in res[:12]:
    print(f'sel {r[0]:.4f} {r[1]:9s} wr {r[2]} tw {r[3]} |', ' '.join(f'{k} {v:.4f}' for k, v in r[4].items()))
print('best per column:')
for name in ['A_logit','A_tan','ABC_logit','ABC_tan']:
    r = next(r for r in res if r[1]==name)
    print(f'sel {r[0]:.4f} {r[1]:9s} wr {r[2]} tw {r[3]} |', ' '.join(f'{k} {v:.4f}' for k, v in r[4].items()))
