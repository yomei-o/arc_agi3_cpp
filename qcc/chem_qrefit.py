"""Quantization-aware refit for Benchmark III (angles with straight-through rounding + weights): round the vectors u to brot-bit angle words on the log-depth tree, keep
them fixed, and re-fit the continuous weights w (Adam, JAX) on the verifier's error_2 = ||sum X (x) X - M||_F
(+ mu * lambda_H2). M is used through its top eigenpairs. Writes an npz whose U are the quantized vectors, then the
caller checks it with chem_brot_sweep.py. Run in C:\\prog\\qcc\\starter\\starter_kit.
    python chem_qrefit.py factorization.npz out.npz [brot] [steps] [mu] [angle_scale]
"""
import os, sys, time
sys.path.insert(0, os.getcwd())
sys.path.insert(0, r"C:\prog\qcc\verifiers")
import numpy as np
import jax, jax.numpy as jnp, optax
from starter.task1_factorization.solution_io import read_factorization
from starter.task2_angles.encode import angles_to_words, words_to_vectors, TREES
from starter.task2_angles.trees import tree_angles
from dfthcsos.scoreinfra.reference import load_reference, load_fixed_shifts, PACKAGED_FCIDUMP
from dfthcsos.scoreinfra.dfthc_math import symmetric_from_lower

jax.config.update("jax_enable_x64", True)
src, out = sys.argv[1], sys.argv[2]
brot = int(sys.argv[3]) if len(sys.argv) > 3 else 8
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 400
mu = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-5
t0 = time.time()
params, _ = read_factorization(src)
U = np.asarray(params["U"], dtype=np.float64)
W0 = np.asarray(params["W"], dtype=np.float64)
R, B, N = U.shape
C = W0.shape[2]
edges = TREES["log_depth"](N)
words = angles_to_words(tree_angles(U.reshape(-1, N), edges), brot)
Uq = words_to_vectors(words, edges, brot).reshape(R, B, N)

ref = load_reference(PACKAGED_FCIDUMP)
tei = np.asarray(ref["tei"], dtype=np.float64)
S = symmetric_from_lower(load_fixed_shifts()["Hsym_lower"], N).astype(np.float64)
tgt = tei.copy()
d = np.arange(N)
tgt[:, :, d, d] += 0.5 * S[:, :, None]
tgt[d, d, :, :] += 0.5 * S[None, :, :]
M = tgt.reshape(N * N, N * N)
M = 0.5 * (M + M.T)
lam, V = np.linalg.eigh(M)
keep = np.argsort(np.abs(lam))[::-1][:1500]
Vk, lk, M2 = jnp.asarray(V[:, keep]), jnp.asarray(lam[keep]), float(np.sum(lam ** 2))
Uqj = jnp.asarray(Uq)
TH0 = tree_angles(U.reshape(-1, N), edges)  # theta / pi, continuous
E = np.asarray(edges)
STEP = 1.0 / (1 << (brot - 1))
TH0j = jnp.asarray(TH0)
ASCALE = float(sys.argv[6]) if len(sys.argv) > 6 else 0.1


def vectors(th):
    """Givens tree decode (verifier map) of theta/pi with straight-through rounding to brot bits."""
    thq = th + jax.lax.stop_gradient(jnp.round(th / STEP) * STEP - th)
    v = jnp.zeros((th.shape[0], N)).at[:, 0].set(1.0)
    for t in range(N - 1):
        p, c = int(E[t, 0]), int(E[t, 1])
        cs, sn = jnp.cos(jnp.pi * thq[:, t]), jnp.sin(jnp.pi * thq[:, t])
        vp, vc = v[:, p], v[:, c]
        v = v.at[:, p].set(cs * vp - sn * vc).at[:, c].set(sn * vp + cs * vc)
    return v.reshape(R, B, N)


def parts(P):
    Wv, dth = P
    Uv = vectors(TH0j + ASCALE * dth)  # angle step scaled down relative to the weights
    A = jnp.einsum("rbc,rbn,rbm->rcnm", Wv, Uv, Uv).reshape(R * C, N * N)
    G = A @ A.T
    AV = A @ Vk
    e2 = jnp.sum(G ** 2) - 2 * jnp.sum(AV ** 2 * lk) + M2
    lam2 = 0.25 * jnp.sum(jnp.sum(jnp.abs(Wv), axis=1) ** 2)
    return e2, lam2


def loss(P):
    e2, lam2 = parts(P)
    return e2 + mu * lam2


def exact_err2(Wv, Uv=Uq):
    A = np.einsum("rbc,rbn,rbm->rcnm", np.asarray(Wv), np.asarray(Uv), np.asarray(Uv)).reshape(R * C, N * N)
    return np.linalg.norm(A.T @ A - M)


print(f"quantized u at brot {brot}: error_2 {1000 * exact_err2(W0):.2f} mHa (before refit; {time.time() - t0:.0f}s)", flush=True)
P = (jnp.asarray(W0), jnp.zeros_like(TH0j))
opt = optax.adam(3e-4)
st = opt.init(P)
vg = jax.jit(jax.value_and_grad(loss))
for i in range(steps + 1):
    l, g = vg(P)
    if i % max(1, steps // 8) == 0:
        e2, lam2 = parts(P)
        print(f"  step {i}: error_2 ~{1000 * np.sqrt(max(float(e2), 0)):.2f} mHa, lambda_H2 {float(lam2):.1f} ({time.time() - t0:.0f}s)", flush=True)
    upd, st = opt.update(g, st)
    P = optax.apply_updates(P, upd)
Wv = P[0]
Uq = np.asarray(vectors(TH0j + ASCALE * P[1]))
print(f"after joint QAT: error_2 {1000 * exact_err2(Wv, Uq):.2f} mHa", flush=True)
np.savez(out, U=Uq, W=np.asarray(Wv), H2_I=np.asarray(params["H2_I"]), H1_I=np.asarray(params["H1_I"]),
         Hsym_lower=np.asarray(params["Hsym_lower"]))
print("wrote", out)
