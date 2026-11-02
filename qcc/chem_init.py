"""Benchmark III physics initializer: DF-THC factorization from the eigen-decomposition of the shifted target.

1. Target supermatrix M = tei + (S (x) I + I (x) S)/2 (as the verifier's error_2), eigen-matrices
   L_k = sqrt(lam_k) mat(v_k) for the top K = R*C eigenvalues (Eckart-Young optimum for K terms).
2. Group them into R groups of C (GROUPING: "block" = consecutive by eigenvalue, "rr" = round robin).
3. Per group, a common set of B unit vectors u_b with weights w_bc: L_c ~ sum_b w_bc u_b u_b^T (INDSCAL), fitted by
   Adam on sum_c ||L_c - X_c||_F^2 (JAX), u initialized from the groups' leading eigenvectors.
4. Reports the verifier's error_2 = ||sum X (x) X - M||_F exactly and writes the trainer's npz format
   (U (R,B,N), W (R,B,C), H2_I = 0, fixed H1_I / Hsym_lower).
Run in C:\\prog\\qcc\\verifiers:  python chem_init.py R B C [steps] [block|rr] [out.npz]
"""
import sys, time
import numpy as np
import jax, jax.numpy as jnp, optax
from dfthcsos.scoreinfra.reference import load_reference, load_fixed_shifts, PACKAGED_FCIDUMP
from dfthcsos.scoreinfra.dfthc_math import symmetric_from_lower

jax.config.update("jax_enable_x64", True)
R, B, C = (int(v) for v in sys.argv[1:4])
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
GROUPING = sys.argv[5] if len(sys.argv) > 5 else "block"
OUT = sys.argv[6] if len(sys.argv) > 6 else f"init_{R}_{B}_{C}_{GROUPING}.npz"

t0 = time.time()
ref = load_reference(PACKAGED_FCIDUMP)
tei = np.asarray(ref["tei"], dtype=np.float64)
N = tei.shape[0]
fx = load_fixed_shifts()
S = symmetric_from_lower(fx["Hsym_lower"], N).astype(np.float64)
tgt = tei.copy()
d = np.arange(N)
tgt[:, :, d, d] += 0.5 * S[:, :, None]
tgt[d, d, :, :] += 0.5 * S[None, :, :]
M = tgt.reshape(N * N, N * N)
M = 0.5 * (M + M.T)
lam, vec = np.linalg.eigh(M)
order = np.argsort(lam)[::-1][: R * C]
Ls = np.stack([np.sqrt(max(lam[k], 0.0)) * vec[:, k].reshape(N, N) for k in order])
Ls = 0.5 * (Ls + np.swapaxes(Ls, 1, 2))
print(f"eigen-decomposition {time.time() - t0:.0f}s; Eckart-Young error with {R * C} terms: "
      f"{1000 * np.sqrt(max(0.0, np.sum(lam ** 2) - np.sum(lam[order] ** 2))):.2f} mHa", flush=True)

if GROUPING == "block":
    gidx = np.arange(R * C).reshape(R, C)
else:
    gidx = np.arange(R * C).reshape(C, R).T
Lg = Ls[gidx]  # (R, C, N, N)

# init u: per group, the B leading eigenvectors (by |eigenvalue|) pooled over the group's matrices
U0 = np.zeros((R, B, N))
for r in range(R):
    cand = []
    for c in range(C):
        e, v = np.linalg.eigh(Lg[r, c])
        for j in range(N):
            cand.append((abs(e[j]), v[:, j]))
    cand.sort(key=lambda t: -t[0])
    picked = []
    for a, v in cand:
        if all(abs(v @ p) < 0.95 for p in picked):
            picked.append(v)
        if len(picked) == B:
            break
    while len(picked) < B:
        picked.append(np.random.default_rng(len(picked)).standard_normal(N))
    U0[r] = np.array(picked)


def weights_ls(U, L):
    """Least-squares w for fixed unit u: (u_a.u_b)^2 Gram system per group."""
    Un = U / jnp.linalg.norm(U, axis=-1, keepdims=True)
    G = jnp.einsum("rbn,rdn->rbd", Un, Un) ** 2
    rhs = jnp.einsum("rbn,rcnm,rbm->rbc", Un, L, Un)
    return jnp.linalg.solve(G + 1e-10 * jnp.eye(G.shape[-1]), rhs)


MU = float(sys.argv[7]) if len(sys.argv) > 7 else 1e-4


def loss(P, L):
    U, W = P
    Un = U / jnp.linalg.norm(U, axis=-1, keepdims=True)
    X = jnp.einsum("rbc,rbn,rbm->rcnm", W, Un, Un)
    lam2 = 0.25 * jnp.sum(jnp.sum(jnp.abs(W), axis=1) ** 2)  # lambda_H2 with W_rho_c = -sum_b w (|w_B| = 0)
    return jnp.sum((X - L) ** 2) + MU * lam2, (jnp.sum((X - L) ** 2), lam2)


nuc = np.sum(np.abs(np.linalg.eigvalsh(Lg)), axis=-1)  # nuclear norms of the eigen-matrices
print(f"ideal lambda_H2 of exact eigen-matrices (nuclear norms): {0.25 * np.sum(nuc ** 2):.1f}", flush=True)
Lj = jnp.asarray(Lg)
U = jnp.asarray(U0)
P = (U, weights_ls(U, Lj))
opt = optax.adam(1e-3)
st = opt.init(P)
vg = jax.jit(jax.value_and_grad(loss, has_aux=True))
for i in range(STEPS + 1):
    (l, (fit, lam2)), g = vg(P, Lj)
    if i % max(1, STEPS // 10) == 0:
        print(f"  step {i}: group fit {1000 * np.sqrt(float(fit)):.1f} mHa, lambda_H2 {float(lam2):.1f} ({time.time() - t0:.0f}s)", flush=True)
    upd, st = opt.update(g, st)
    P = optax.apply_updates(P, upd)
U, Wfit = P

Un = np.asarray(U / jnp.linalg.norm(U, axis=-1, keepdims=True))
W = np.asarray(Wfit)
X = np.einsum("rbc,rbn,rbm->rcnm", W, Un, Un).reshape(R * C, N * N)
err2 = np.linalg.norm(X.T @ X - M)
print(f"error_2 of the init (unquantized): {1000 * err2:.2f} mHa", flush=True)
np.savez(OUT, U=Un, W=W, H2_I=np.zeros((R, C)), H1_I=np.asarray(fx["H1_I"]).reshape(()),
         Hsym_lower=np.asarray(fx["Hsym_lower"]))
print("wrote", OUT, f"({time.time() - t0:.0f}s)")
