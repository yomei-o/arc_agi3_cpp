"""error_2 of a table in float32 (direct form, as the fit uses) vs float64, and gradient agreement."""
import sys
import numpy as np, torch
sys.path.insert(0, r"C:\prog\qcc")
from chem_gpu import target, dev
M, N, fx = target()
z = np.load(sys.argv[1])
for dt in (torch.float32, torch.float64):
    U = torch.tensor(z["U"], device=dev, dtype=dt, requires_grad=True)
    w = torch.tensor(z["W"], device=dev, dtype=dt, requires_grad=True)
    Mt = torch.tensor(M, device=dev, dtype=dt)
    Un = U / U.norm(dim=-1, keepdim=True)
    A = torch.einsum("rbc,rbn,rbm->rcnm", w, Un, Un).reshape(-1, N * N)
    e = ((A.T @ A - Mt) ** 2).sum()
    e.backward()
    print(dt, "error_2 %.4f mHa" % (1000 * e.item() ** 0.5), "grad norm U %.4e w %.4e" % (U.grad.norm().item(), w.grad.norm().item()))
