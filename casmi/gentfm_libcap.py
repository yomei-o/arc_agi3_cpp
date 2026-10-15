# gentfm_libcap.py: effect of capping libSim at 1 (the library cosine can exceed 1: best-peak matching is not one-to-one)
import sys, numpy as np
sys.path.insert(0, r'C:\prog\claude4\arc\arc3\casmi')
import gentfm_eval as g
for p in sys.argv[1:]:
    Q = g.load(p)
    m0 = g.mrr(Q, g.base)[0]
    def capped(q):
        X = q['X'].copy(); X[:, 3] = np.minimum(X[:, 3], 1.0)
        return g.z(X[:,0]) + 0.25*g.z(X[:,1]) + g.z(X[:,2]) + 8*np.maximum(X[:,3],0)**32 + 24*X[:,4] + q['G']@g.GW
    nq = sum(1 for q in Q if q is not None and (q['X'][:,3] > 1).any())
    print(p, 'base %.4f  lib capped at 1: %.4f  (queries with lib>1: %d)' % (m0, g.mrr(Q, capped)[0], nq))
