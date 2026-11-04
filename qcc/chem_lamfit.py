"""lambda_eff refit at fixed vectors (Benchmark III, GPU float64): optimize the native identity offsets W (R, C) and
the weights w (R, B, C) for the verifier's lambda_eff = sqrt(E_gap (2 Lambda - E_gap)) with
  h1' = H1_THC - (eta/2) S + beta_1 I - sum_rc W_rc X_rc,   lambda_h1 = sum |eig h1'|,
  Lambda = lambda_h1 + 1/4 sum_rc (sum_b |w| + |W + sum_b w|)^2,
  E_SOS = ecore - eta beta_1 - (lambda_h1 - tr h1') - 1/2 sum W^2,   E_gap = E_upper - E_SOS
(dfthc_math.evaluate_dfthc_vectors), with quadratic penalties keeping error_2 <= target and E_gap >= floor + margin.
The vectors (angle words) are not touched.
    python chem_lamfit.py in.npz out.npz [err_target_mHa] [steps] [lr] [lsinit] [wonly]
"""
import sys
import numpy as np
import torch

sys.path.insert(0, r"C:\prog\qcc\verifiers")
from chem_gpu import target, err2, log, dev
from dfthcsos.scoreinfra.reference import load_reference, PACKAGED_FCIDUMP
from dfthcsos.scoreinfra.dfthc_math import symmetric_from_lower

src, out = sys.argv[1], sys.argv[2]
etgt = float(sys.argv[3]) / 1000 if len(sys.argv) > 3 else 0.0195
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
lr = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-3
E_UPPER, GAP_FLOOR = -22140.27335, 3.914898
M, N, fx = target()
ref = load_reference(PACKAGED_FCIDUMP)
tei = np.asarray(ref["tei"], dtype=np.float64)
h1_thc = torch.tensor(np.asarray(ref["oei"]) - 0.5 * np.einsum("prrq->pq", tei), device=dev, dtype=torch.float64)
S = torch.tensor(symmetric_from_lower(fx["Hsym_lower"], N).astype(np.float64), device=dev)
beta1 = float(fx["H1_I"]) - 0.5 * float(torch.trace(S))
eta, ecore = int(ref["electrons"]), float(ref["ecore"])
Mt = torch.tensor(M, device=dev, dtype=torch.float64)
z = dict(np.load(src))
U = torch.tensor(z["U"], device=dev, dtype=torch.float64)
U = U / U.norm(dim=-1, keepdim=True)
w = torch.tensor(z["W"], device=dev, dtype=torch.float64, requires_grad=True)
Wn = torch.tensor(np.asarray(z["H2_I"]) - np.asarray(z["W"]).sum(1), device=dev, dtype=torch.float64, requires_grad=True)
R, B, C = w.shape

base = h1_thc - 0.5 * eta * S + beta1 * torch.eye(N, device=dev, dtype=torch.float64)
if "lsinit" in sys.argv[6:]:
    # least-squares offsets: sum_rc W_rc X_rc ~ base, i.e. h1' ~ 0 (small lambda_h1 and no negative eigenvalues)
    with torch.no_grad():
        Xf = torch.einsum("rbc,rbn,rbm->rcnm", w, U, U).reshape(R * C, N * N)
        sol = torch.linalg.lstsq(Xf.T, base.reshape(-1, 1)).solution.reshape(R, C)
        Wn.copy_(sol)
    log("offsets initialized by least squares")


def metrics():
    X = torch.einsum("rbc,rbn,rbm->rcnm", w, U, U)
    h1p = base - torch.einsum("rc,rcnm->nm", Wn, X)
    ev = torch.linalg.eigvalsh(0.5 * (h1p + h1p.T))
    lam1 = ev.abs().sum()
    lam2 = 0.25 * ((w.abs().sum(1) + (Wn + w.sum(1)).abs()) ** 2).sum()
    Lam = lam1 + lam2
    esos = ecore - eta * beta1 - (lam1 - ev.sum()) - 0.5 * (Wn ** 2).sum()
    gap = E_UPPER - esos
    A = X.reshape(-1, N * N)
    e2 = torch.linalg.norm(A.T @ A - Mt)
    leff = torch.sqrt(torch.clamp(gap * (2 * Lam - gap), min=1e-12))
    return leff, e2, gap, Lam


with torch.no_grad():
    l0 = metrics()
log(f"start: lambda_eff {l0[0].item():.3f}, error_2 {1000 * l0[1].item():.3f} mHa, E_gap {l0[2].item():.4f}, Lambda {l0[3].item():.2f}")
MODES = sys.argv[6:] if len(sys.argv) > 6 else []
opt = torch.optim.Adam([Wn] if 'wonly' in MODES else [w, Wn], lr=lr)  # wonly: offsets only, error_2 untouched
best = None
for i in range(steps + 1):
    opt.zero_grad()
    leff, e2, gap, Lam = metrics()
    pen = 1e7 * torch.relu(e2 - etgt) ** 2 + 1e4 * torch.relu(GAP_FLOOR + 0.002 - gap) ** 2
    (leff + pen).backward()
    opt.step()
    ok = e2.item() <= etgt and gap.item() > GAP_FLOOR + 0.001
    if ok and (best is None or leff.item() < best[0]):
        best = (leff.item(), w.detach().clone(), Wn.detach().clone())
    if i % max(1, steps // 10) == 0:
        log(f"  step {i}: lambda_eff {leff.item():.3f}, error_2 {1000 * e2.item():.3f} mHa, E_gap {gap.item():.4f}, "
            f"Lambda {Lam.item():.2f}, feasible best {best[0] if best else None}")
if best is None:
    log("no feasible point; writing the last iterate")
    wb, Wb = w.detach(), Wn.detach()
else:
    wb, Wb = best[1], best[2]
wn = wb.cpu().numpy()
z["W"], z["H2_I"] = wn, Wb.cpu().numpy() + wn.sum(1)
np.savez(out, **z)
log("wrote", out, "best lambda_eff", best[0] if best else None)
