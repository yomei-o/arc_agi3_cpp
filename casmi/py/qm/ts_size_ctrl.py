"""Does the xTB scan barrier add information beyond bond class + fragment size?

Baseline: logistic regression on bond class + size bin of the smaller piece
(heavy atoms: 1, 2, 3-4, 5-8, 9-15, 16+). Then + barrier, + reaction energy
(both z-scored within class), and the barrier shuffled within class (control).
5-fold CV by molecule, 20 different fold splits; held-out log-likelihood gain
per cut (mnat) and pooled AUC.

    python ts_size_ctrl.py <ts_scan.csv> <huckel_fqry2.json>
"""
import csv, json, sys
import numpy as np

cut = {}
for m in json.load(open(sys.argv[2])):
    for c in m["cuts"]:
        b = m["bonds"][c["b"]]
        cut[(m["name"], min(b["i"], b["j"]), max(b["i"], b["j"]))] = (int(max(c["oA"], c["oB"]) > 0), min(len(c["A"]), len(c["B"])))


def sbin(k):
    return 0 if k <= 1 else 1 if k == 2 else 2 if k <= 4 else 3 if k <= 8 else 4 if k <= 15 else 5


R = []
for r in csv.DictReader(open(sys.argv[1])):
    i, j = int(r["i"]), int(r["j"])
    v = cut.get((r["name"], min(i, j), max(i, j)))
    if v:
        R.append((r["name"], r["cls"], sbin(v[1]), float(r["barrier"]), float(r["reaction"]), v[0]))
classes = sorted({x[1] for x in R})
mol = np.array([x[0] for x in R]); y = np.array([x[5] for x in R], float)
X0 = np.zeros((len(R), 1 + len(classes) + 6)); X0[:, 0] = 1
for k, x in enumerate(R):
    X0[k, 1 + classes.index(x[1])] = 1; X0[k, 1 + len(classes) + x[2]] = 1


def zc(col):
    v = np.array([x[col] for x in R], float); out = np.zeros(len(R))
    for c in classes:
        idx = np.array([x[1] == c for x in R])
        out[idx] = (v[idx] - v[idx].mean()) / (v[idx].std() + 1e-9)
    return out


bar, rea = zc(3), zc(4)
rng = np.random.default_rng(1)
shuf = bar.copy()
for c in classes:
    idx = np.where(np.array([x[1] == c for x in R]))[0]
    shuf[idx] = bar[rng.permutation(idx)]


def fit(X, t, l2=1.0):
    w = np.zeros(X.shape[1])
    for _ in range(30):
        p = 1 / (1 + np.exp(-X @ w))
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(w))
        w -= np.linalg.solve(H, X.T @ (p - t) + l2 * w)
    return w


def cv(X, seeds=20):
    um = np.unique(mol); lls, aucs = [], []
    for s in range(seeds):
        r = np.random.default_rng(s); perm = r.permutation(um); fold = {m: k % 5 for k, m in enumerate(perm)}
        fo = np.array([fold[m] for m in mol]); pred = np.zeros(len(y))
        for k in range(5):
            w = fit(X[fo != k], y[fo != k]); pred[fo == k] = 1 / (1 + np.exp(-X[fo == k] @ w))
        pred = np.clip(pred, 1e-6, 1 - 1e-6)
        lls.append(np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred)))
        P, N = pred[y == 1], pred[y == 0]
        aucs.append(np.mean((P[:, None] > N[None, :]) + 0.5 * (P[:, None] == N[None, :])))
    return np.array(lls), np.mean(aucs)


print(f"labelled cuts {len(R)}, observed {int(y.sum())}")
base, ab = cv(X0)
print(f"baseline class + size bin: ll/cut {base.mean():.4f}  AUC {ab:.3f}")
for name, cols in (("+ barrier", [bar]), ("+ reaction energy", [rea]), ("+ barrier + reaction", [bar, rea]),
                   ("+ barrier SHUFFLED in class", [shuf])):
    ll, a = cv(np.column_stack([X0] + cols))
    d = 1000 * (ll - base)
    print(f"  {name:28s} gain {d.mean():+6.2f} mnat/cut (split-to-split sd {d.std():.2f})  AUC {a:.3f}")
