"""Progressive rounding of a Benchmark III table to brot-bit angle words (GPU, PyTorch). Own implementation.

The vectors are held as Givens-tree angles theta/pi on the log-depth tree (the verifier's decode). Stages freeze a
growing share of the angles at their rounded grid values, cheapest first, and re-fit the free angles and all weights
w with L-BFGS on error_2^2 + kappa D(w) between stages. Cost of rounding an angle = |u(theta + delta e_t) - u(theta)|^2
times the row's weight energy sum_c w^2 (delta = offset to the nearest grid point). The schedule is finer early,
where most of the rounding noise enters.
    python chem_round.py in.npz out.npz [brot] [stages] [iters_per_stage] [kappa]
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, r"C:\prog\qcc\verifiers")
sys.path.insert(0, r"C:\prog\qcc\starter\starter_kit")
from chem_gpu import target, err2, log, dev
from starter.task2_angles.trees import tree_angles, log_depth_tree_edges

src, out = sys.argv[1], sys.argv[2]
brot = int(sys.argv[3]) if len(sys.argv) > 3 else 8
stages = int(sys.argv[4]) if len(sys.argv) > 4 else 40
iters = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
kappa = float(sys.argv[6]) if len(sys.argv) > 6 else 5e-5
M, N, fx = target()
Mt32 = torch.tensor(M, device=dev, dtype=torch.float32)
Mt64 = torch.tensor(M, device=dev, dtype=torch.float64)
z = dict(np.load(src))
U0 = np.asarray(z["U"], dtype=np.float64)
w0 = np.asarray(z["W"], dtype=np.float64)
Wnat = np.asarray(z["H2_I"], dtype=np.float64) - w0.sum(1)
R, B, _ = U0.shape
C = w0.shape[2]
edges = np.asarray(log_depth_tree_edges(N))
STEP = 1.0 / (1 << (brot - 1))
th0 = tree_angles((U0 / np.linalg.norm(U0, axis=-1, keepdims=True)).reshape(-1, N), edges)  # (RB, N-1) theta/pi


def decode(th):
    v = torch.zeros(th.shape[0], N, device=dev, dtype=th.dtype)
    v[:, 0] = 1.0
    for t in range(N - 1):
        p, c = int(edges[t, 0]), int(edges[t, 1])
        cs, sn = torch.cos(np.pi * th[:, t]), torch.sin(np.pi * th[:, t])
        vp, vc = v[:, p].clone(), v[:, c].clone()
        v[:, p] = cs * vp - sn * vc
        v[:, c] = sn * vp + cs * vc
    return v


free = torch.tensor(np.asarray(th0), device=dev, dtype=torch.float32, requires_grad=True)
fixed = torch.zeros_like(free.detach())
mask = torch.zeros_like(fixed, dtype=torch.bool)  # True = frozen
w = torch.tensor(w0, device=dev, dtype=torch.float32, requires_grad=True)


def angles():
    return torch.where(mask, fixed, free)


def objective():
    Un = decode(angles()).reshape(R, B, N)
    A = torch.einsum("rbc,rbn,rbm->rcnm", w, Un, Un).reshape(-1, N * N)
    e = ((A.T @ A - Mt32) ** 2).sum()
    D = 0.5 * ((w.abs().sum(1)) ** 2 - (w.sum(1)) ** 2).sum()
    return e + kappa * D


def refit(n):
    opt = torch.optim.LBFGS([free, w], lr=1, max_iter=n, history_size=100, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-12, tolerance_change=1e-15)

    def closure():
        opt.zero_grad()
        l = objective()
        l.backward()
        free.grad[mask] = 0
        return l
    opt.step(closure)


def report(tag):
    with torch.no_grad():
        Un = decode(angles().double()).reshape(R, B, N)
        e = err2(Un, w.double(), Mt64).item()
    log(f"{tag}: error_2 {1000 * e:.3f} mHa, frozen {100 * mask.float().mean().item():.1f}%")
    return e


report("start (continuous)")
# freeze schedule: cumulative fraction after each stage, quadratic so early stages are fine
fr = [(s / stages) ** 2 for s in range(1, stages + 1)]
total = mask.numel()
for s, f in enumerate(fr):
    with torch.no_grad():
        th = angles()
        q = torch.round(th / STEP) * STEP
        delta = q - th
        base = decode(th)
        wen = (w ** 2).sum(-1).reshape(-1)  # row weight energy
        cost = torch.full_like(th, float("inf"))
        for t in range(N - 1):
            col = ~mask[:, t]
            if not col.any():
                continue
            th2 = th.clone()
            th2[:, t] = q[:, t]
            dv = ((decode(th2) - base) ** 2).sum(-1)
            cost[:, t] = torch.where(col, dv * wen, cost[:, t])
        k = int(round(f * total)) - int(mask.sum())
        if k > 0:
            idx = torch.argsort(cost.reshape(-1))[:k]
            mflat = mask.reshape(-1)
            fflat = fixed.reshape(-1)
            mflat[idx] = True
            fflat[idx] = q.reshape(-1)[idx]
    refit(iters)
    if s % 5 == 0 or s == stages - 1:
        report(f"stage {s + 1}/{stages}")
refit(iters)
e = report("final")
with torch.no_grad():
    words = (torch.round(angles() / STEP).long() % (1 << brot)).cpu().numpy()
    Uq = decode(torch.tensor(words * STEP, device=dev, dtype=torch.float64)).reshape(R, B, N).cpu().numpy()
wn = w.detach().cpu().numpy().astype(np.float64)
np.savez(out, U=Uq, W=wn, H2_I=Wnat + wn.sum(1), H1_I=np.asarray(fx["H1_I"]).reshape(()),
         Hsym_lower=np.asarray(fx["Hsym_lower"]), angle_words_two_body=words)
log("wrote", out)
