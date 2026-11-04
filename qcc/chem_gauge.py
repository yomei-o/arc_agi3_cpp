"""Gauge optimization for Benchmark III: lower lambda_H2 without touching the error.

Within a group rho, mixing the C leaves by an orthogonal C x C matrix O (w'[b, :] = w[b, :] O, W'[:] = W[:] O)
leaves sum_c X_c (x) X_c, sum_c W_c X_c and sum_c W_c^2 unchanged, hence error_1, error_2, h1_prime and E_SOS, and
the vectors u (the angle words) are untouched. Only lambda_H2 = 1/4 sum_c (|w'_c|_1 + |w_B'_c|)^2 moves
(H2_I stores w_B = W + sum_b w, which rotates with the same O).
O = expm(A - A^T) per group, Adam on A with a smoothed |.|; the exact lambda is reported, and the factorization
with the best exact lambda is written.
    python chem_gauge.py in.npz out.npz [steps] [lr]
"""
import sys, time
import numpy as np
import jax, jax.numpy as jnp, optax
from jax.scipy.linalg import expm

jax.config.update("jax_enable_x64", True)
src, out = sys.argv[1], sys.argv[2]
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
lr = float(sys.argv[4]) if len(sys.argv) > 4 else 3e-3
z = dict(np.load(src))
w = jnp.asarray(z["W"], dtype=jnp.float64)       # (R, B, C) lowercase weights
H2 = jnp.asarray(z["H2_I"], dtype=jnp.float64)   # (R, C) w_B = W_rho_c + sum_b w (verifier native_inputs)
R, B, C = w.shape


def lam_h2(wv, Wv, eps=0.0):
    a = (lambda x: jnp.sqrt(x * x + eps)) if eps else jnp.abs
    return 0.25 * jnp.sum((jnp.sum(a(wv), axis=1) + a(Wv)) ** 2)  # Wv = H2_I = w_B (verifier convention)


def rotate(A):
    O = jax.vmap(lambda a: expm(a - a.T))(A)                   # (R, C, C)
    return jnp.einsum("rbc,rcd->rbd", w, O), jnp.einsum("rc,rcd->rd", H2, O)


def loss(A, eps):
    wv, Wv = rotate(A)
    return lam_h2(wv, Wv, eps)


t0 = time.time()
lam0 = float(lam_h2(w, H2))
print(f"lambda_H2 before {lam0:.2f}", flush=True)
A = jnp.zeros((R, C, C))
opt = optax.adam(lr)
st = opt.init(A)
g = jax.jit(jax.value_and_grad(loss), static_argnums=1)
best, bestA = lam0, A
for i in range(steps + 1):
    eps = 1e-6
    l, gr = g(A, eps)
    upd, st = opt.update(gr, st)
    A = optax.apply_updates(A, upd)
    if i % max(1, steps // 10) == 0:
        wv, Wv = rotate(A)
        ex = float(lam_h2(wv, Wv))
        if ex < best:
            best, bestA = ex, A
        print(f"  step {i}: lambda_H2 {ex:.2f} ({time.time() - t0:.0f}s)", flush=True)
wv, Wv = rotate(bestA)
z["W"], z["H2_I"] = np.asarray(wv), np.asarray(Wv)
np.savez(out, **z)
print(f"lambda_H2 {lam0:.2f} -> {best:.2f}; wrote {out}", flush=True)
