"""Adam continuation of a Benchmark III table on the GPU (escape the L-BFGS stall): error_2^2 + kappa D(w),
cosine-decayed learning rate.  python chem_adam.py in.npz out.npz steps lr0 [kappa]"""
import sys, math
import numpy as np, torch
sys.path.insert(0, r"C:\prog\qcc")
from chem_gpu import target, err2, log, dev

src, out, steps, lr0 = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
kappa = float(sys.argv[5]) if len(sys.argv) > 5 else 5e-5
M, N, fx = target()
Mt = torch.tensor(M, device=dev, dtype=torch.float32)
M64 = torch.tensor(M, device=dev, dtype=torch.float64)
z = dict(np.load(src))
Wnat = np.asarray(z["H2_I"]) - np.asarray(z["W"]).sum(1)
U = torch.tensor(z["U"], device=dev, dtype=torch.float32, requires_grad=True)
w = torch.tensor(z["W"], device=dev, dtype=torch.float32, requires_grad=True)
R, B, C = w.shape
opt = torch.optim.Adam([U, w], lr=lr0)
best = (float(err2(U.double(), w.double(), M64)), None)
log(f"start error_2 {1000 * best[0]:.3f} mHa")
for i in range(steps + 1):
    for g in opt.param_groups:
        g["lr"] = lr0 * 0.5 * (1 + math.cos(math.pi * i / steps))
    opt.zero_grad()
    Un = U / U.norm(dim=-1, keepdim=True)
    P = (Un[:, :, :, None] * Un[:, :, None, :]).reshape(R, B, N * N)
    A = torch.bmm(w.transpose(1, 2), P).reshape(R * C, N * N)
    e = ((A.T @ A - Mt) ** 2).sum()
    D = 0.5 * ((w.abs().sum(1)) ** 2 - (w.sum(1)) ** 2).sum()
    (e + kappa * D).backward()
    opt.step()
    if i % max(1, steps // 20) == 0:
        e64 = float(err2(U.detach().double(), w.detach().double(), M64))
        if e64 < best[0]:
            best = (e64, (U.detach().clone(), w.detach().clone()))
        log(f"  step {i}: error_2 {1000 * e64:.3f} mHa, lr {opt.param_groups[0]['lr']:.2e}")
if best[1] is not None:
    Ub, wb = best[1]
    wn = wb.cpu().numpy().astype(np.float64)
    np.savez(out, U=(Ub / Ub.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float64), W=wn,
             H2_I=Wnat + wn.sum(1), H1_I=z["H1_I"], Hsym_lower=z["Hsym_lower"])
log("best", 1000 * best[0], "wrote", out)
