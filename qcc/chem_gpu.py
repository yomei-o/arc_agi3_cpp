"""Benchmark III on the GPU (PyTorch, RTX 3090): compact seed + L-BFGS fit of the verifier's error_2.

Target M = tei + (S (x) I + I (x) S)/2 as an N^2 x N^2 matrix (dfthc_math error_2). A table (R, B, C): unit vectors
u_rb, weights w_rbc, leaves X_rc = sum_b w_rbc u_rb u_rb^T, error_2 = ||sum_rc vec X vec X^T - M||_F.

seed:  pivoted Cholesky of M -> R*C local leaves (columns of the residual), balanced k-means of the leaves'
       orbital profiles into R groups of C, per group B shared vectors by simultaneous OMP over the leaves'
       eigenvectors, least-squares weights.
fit:   L-BFGS on error_2^2 + kappa * D(w), D(w) = 1/2 sum_rc [(sum_b |w|)^2 - (sum_b w)^2] (sign mixing part of Lambda),
       rows of U normalized inside the loss.
Writes the trainer/verifier npz (U, W = w, H2_I = W_native + sum_b w with W_native kept, H1_I, Hsym_lower fixed).

    python chem_gpu.py seed R B C out.npz [kmeans_starts]
    python chem_gpu.py fit in.npz out.npz rounds iters [kappa]
"""
import sys, time, math
import numpy as np
import torch

sys.path.insert(0, r"C:\prog\qcc\verifiers")
from dfthcsos.scoreinfra.reference import load_reference, load_fixed_shifts, PACKAGED_FCIDUMP
from dfthcsos.scoreinfra.dfthc_math import symmetric_from_lower

torch.backends.cuda.matmul.allow_tf32 = False
dev = torch.device("cuda")
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def target():
    ref = load_reference(PACKAGED_FCIDUMP)
    tei = np.asarray(ref["tei"], dtype=np.float64)
    n = tei.shape[0]
    fx = load_fixed_shifts()
    S = symmetric_from_lower(fx["Hsym_lower"], n).astype(np.float64)
    tgt = tei.copy()
    d = np.arange(n)
    tgt[:, :, d, d] += 0.5 * S[:, :, None]
    tgt[d, d, :, :] += 0.5 * S[None, :, :]
    M = tgt.reshape(n * n, n * n)
    return 0.5 * (M + M.T), n, fx


def err2(Ut, wt, Mt):
    """error_2 (exact, float64) for torch tensors U (R,B,N) normalized inside, w (R,B,C)."""
    Un = Ut / Ut.norm(dim=-1, keepdim=True)
    R, B, N = Un.shape
    A = torch.einsum("rbc,rbn,rbm->rcnm", wt, Un, Un).reshape(-1, N * N)
    return torch.linalg.norm(A.T @ A - Mt)


def seed(R, B, C, out, starts=50):
    M, N, fx = target()
    # pivoted Cholesky leaves
    Mt = torch.tensor(M, device=dev, dtype=torch.float64)
    diag = torch.diagonal(Mt).clone()
    Ls = []
    res = Mt.clone()
    for k in range(R * C):
        p = int(torch.argmax(diag))
        col = res[:, p] / torch.sqrt(res[p, p])
        Ls.append(col.clone())
        res -= torch.outer(col, col)
        diag = torch.diagonal(res).clone()
    L = torch.stack(Ls)  # (RC, N^2)
    log(f"Cholesky {R * C} leaves, residual {1000 * torch.linalg.norm(res).item():.1f} mHa")
    Lm = L.reshape(-1, N, N)
    Lm = 0.5 * (Lm + Lm.transpose(1, 2))
    prof = (Lm ** 2).sum(-1)
    prof = prof / prof.sum(-1, keepdim=True)  # orbital profile of each leaf
    P = prof.cpu().numpy()
    best = None
    rng = np.random.default_rng(0)
    for s in range(starts):
        cent = P[rng.choice(len(P), R, replace=False)]
        for it in range(30):
            d2 = ((P[:, None, :] - cent[None]) ** 2).sum(-1)
            # balanced assignment: greedy by margin, capacity C
            order = np.argsort(d2.min(1) - np.sort(d2, 1)[:, 1])
            cap = np.full(R, C)
            lab = np.empty(len(P), int)
            for i in order:
                for g in np.argsort(d2[i]):
                    if cap[g] > 0:
                        lab[i] = g
                        cap[g] -= 1
                        break
            cent = np.stack([P[lab == g].mean(0) for g in range(R)])
        gp = np.stack([P[lab == g].sum(0) for g in range(R)])
        gp = gp / gp.sum(1, keepdims=True)
        comp = np.mean(np.exp(-(gp * np.log(gp + 1e-30)).sum(1)))  # effective orbitals per group
        if best is None or comp < best[0]:
            best = (comp, lab.copy())
    comp, lab = best
    log(f"k-means: best of {starts} starts, {comp:.1f} orbitals per group")
    U = torch.zeros(R, B, N, device=dev, dtype=torch.float64)
    W = torch.zeros(R, B, C, device=dev, dtype=torch.float64)
    for g in range(R):
        G = Lm[torch.tensor(np.flatnonzero(lab == g), device=dev)]  # (C, N, N)
        ev, vec = torch.linalg.eigh(G)
        cand = vec.transpose(1, 2).reshape(-1, N)  # candidate vectors (C*N, N)
        resid = G.clone()
        chosen = []
        for t in range(B):
            proj = torch.einsum("kn,cnm,km->kc", cand, resid, cand)
            score = (proj ** 2).sum(1)
            if chosen:
                score[torch.tensor(chosen, device=dev)] = -1
            k = int(torch.argmax(score))
            chosen.append(k)
            V = cand[torch.tensor(chosen, device=dev)]
            Gr = (V @ V.T) ** 2
            rhs = torch.einsum("kn,cnm,km->kc", V, G, V)
            wg = torch.linalg.solve(Gr + 1e-10 * torch.eye(len(chosen), device=dev, dtype=torch.float64), rhs)
            resid = G - torch.einsum("kc,kn,km->cnm", wg, V, V)
        U[g] = cand[torch.tensor(chosen, device=dev)]
        W[g] = wg
    e = err2(U, W, Mt).item()
    log(f"seed error_2 {1000 * e:.1f} mHa")
    np.savez(out, U=U.cpu().numpy(), W=W.cpu().numpy(), H2_I=W.sum(1).cpu().numpy(),
             H1_I=np.asarray(fx["H1_I"]).reshape(()), Hsym_lower=np.asarray(fx["Hsym_lower"]))
    log("wrote", out)


def fit(src, out, rounds, iters, kappa=5e-5):
    M, N, fx = target()
    Mt32 = torch.tensor(M, device=dev, dtype=torch.float32)
    Mt64 = torch.tensor(M, device=dev, dtype=torch.float64)
    z = dict(np.load(src))
    w0 = np.asarray(z["W"], dtype=np.float64)
    Wnat = np.asarray(z["H2_I"], dtype=np.float64) - w0.sum(1)  # native identity offsets, kept fixed
    U = torch.tensor(z["U"], device=dev, dtype=torch.float32, requires_grad=True)
    w = torch.tensor(w0, device=dev, dtype=torch.float32, requires_grad=True)
    R, B, C = w.shape

    def objective():
        Un = U / U.norm(dim=-1, keepdim=True)
        A = torch.einsum("rbc,rbn,rbm->rcnm", w, Un, Un).reshape(-1, N * N)
        e = ((A.T @ A - Mt32) ** 2).sum()
        D = 0.5 * ((w.abs().sum(1)) ** 2 - (w.sum(1)) ** 2).sum()
        return e + kappa * D, e, D

    for r in range(rounds):
        opt = torch.optim.LBFGS([U, w], lr=1, max_iter=iters, history_size=100, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-12, tolerance_change=1e-15)

        def closure():
            opt.zero_grad()
            l, _, _ = objective()
            l.backward()
            return l
        opt.step(closure)
        with torch.no_grad():
            e64 = err2(U.double(), w.double(), Mt64).item()
            _, _, D = objective()
        log(f"round {r}: error_2 {1000 * e64:.3f} mHa, D(w) {D.item():.2f}")
        wn = w.detach().cpu().numpy().astype(np.float64)
        np.savez(out, U=(U / U.norm(dim=-1, keepdim=True)).detach().cpu().numpy().astype(np.float64), W=wn,
                 H2_I=Wnat + wn.sum(1), H1_I=np.asarray(fx["H1_I"]).reshape(()), Hsym_lower=np.asarray(fx["Hsym_lower"]))
    log("wrote", out)


if __name__ == "__main__":
    if sys.argv[1] == "seed":
        seed(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5],
             int(sys.argv[6]) if len(sys.argv) > 6 else 50)
    else:
        fit(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]),
            float(sys.argv[6]) if len(sys.argv) > 6 else 5e-5)
