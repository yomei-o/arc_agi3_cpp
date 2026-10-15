# gentfm_diag.py: in the answer-removed setting, who outranks a generated truth (pool vs generated candidates, per-term gap). Usage: python gentfm_diag.py f_c_rm.txt ...
import sys, numpy as np
sys.path.insert(0, r'C:\prog\claude4\arc\arc3\casmi')
from gentfm_eval import load, base, z, GW
for p in sys.argv[1:]:
    Q = load(p)
    rows = []
    comp = []
    for q in Q:
        if q is None or q['t'] < 0 or not q['gen'][q['t']]: continue
        s = base(q); t = q['t']
        above = s > s[t]
        npool = int((above & ~q['gen']).sum()); ngen = int((above & q['gen']).sum())
        rows.append((npool, ngen, len(s), int(q['gen'].sum())))
        X = q['X']
        parts = np.stack([z(X[:,0]), 0.25*z(X[:,1]), z(X[:,2]), 8*np.maximum(X[:,3],0)**32, 24*X[:,4], q['G']@GW], 1)
        pool_above = np.where(above & ~q['gen'])[0]
        if len(pool_above):
            top = pool_above[np.argmax(s[pool_above])]
            comp.append(parts[top] - parts[t])
    r = np.array(rows)
    print(p, 'gen-truth queries', len(r))
    print('  mean #pool above truth %.1f (median %d), #gen above %.1f (median %d); cands %.0f, gen %.0f' % (r[:,0].mean(), np.median(r[:,0]), r[:,1].mean(), np.median(r[:,1]), r[:,2].mean(), r[:,3].mean()))
    print('  queries where truth beaten by >=1 pool: %d, by >=1 gen: %d, truth top1: %d' % ((r[:,0]>0).sum(), (r[:,1]>0).sum(), ((r[:,0]+r[:,1])==0).sum()))
    c = np.array(comp)
    print('  top pool competitor minus truth, mean per term [phys, mlp, rev, lib, ana, gen]:', np.round(c.mean(0), 2))
