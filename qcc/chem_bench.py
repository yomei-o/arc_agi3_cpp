"""Time the error_2 objective + backward: einsum form vs pair-matrix bmm form (GPU)."""
import time, torch
dev = torch.device("cuda")
R, B, C, N = 9, 143, 40, 76
U = torch.randn(R, B, N, device=dev, requires_grad=True)
w = torch.randn(R, B, C, device=dev, requires_grad=True)
M = torch.randn(N * N, N * N, device=dev)


def f_einsum():
    Un = U / U.norm(dim=-1, keepdim=True)
    A = torch.einsum("rbc,rbn,rbm->rcnm", w, Un, Un).reshape(-1, N * N)
    return ((A.T @ A - M) ** 2).sum()


def f_bmm():
    Un = U / U.norm(dim=-1, keepdim=True)
    P = (Un[:, :, :, None] * Un[:, :, None, :]).reshape(R, B, N * N)
    A = torch.bmm(w.transpose(1, 2), P).reshape(R * C, N * N)
    return ((A.T @ A - M) ** 2).sum()


def f_gram():
    # ||A^T A - M||^2 = ||A A^T||^2 - 2 tr(A M A^T) + ||M||^2 (M symmetric)
    Un = U / U.norm(dim=-1, keepdim=True)
    P = (Un[:, :, :, None] * Un[:, :, None, :]).reshape(R, B, N * N)
    A = torch.bmm(w.transpose(1, 2), P).reshape(R * C, N * N)
    G = A @ A.T
    return (G ** 2).sum() - 2 * ((A @ M) * A).sum()


for name, f in (("einsum", f_einsum), ("bmm", f_bmm), ("gram", f_gram)):
    for _ in range(3):
        f().backward()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(20):
        f().backward()
    torch.cuda.synchronize()
    print(name, "%.1f ms per objective+backward" % (1000 * (time.time() - t) / 20))
