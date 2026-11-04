"""Discrete gradient-guided coordinate descent on the brot-bit angle words (Benchmark III).

The vectors u are held as integer words on the log-depth tree (exactly what the circuits load). Each round:
  1. refit the continuous weights w for a few Adam steps (vectors fixed),
  2. gradient of the error at the quantized angles, d err / d theta,
  3. move the k angles with the largest |gradient| one word against it,
  4. keep the move only if the exact objective dropped (else revert and halve k; on success grow k).
Monotone, unlike straight-through training. Objective: error_2^2 + mu * lambda_H2.
Run in C:\\prog\\qcc\\starter\\starter_kit:
    python chem_qcd.py factorization.npz out.npz [brot] [rounds] [mu]
"""
import os, sys, time
sys.path.insert(0, os.getcwd())
sys.path.insert(0, r"C:\prog\qcc\verifiers")
import numpy as np
import jax, jax.numpy as jnp, optax
from starter.task1_factorization.solution_io import read_factorization
from starter.task2_angles.encode import angles_to_words, TREES
from starter.task2_angles.trees import tree_angles
from dfthcsos.scoreinfra.reference import load_reference, load_fixed_shifts, PACKAGED_FCIDUMP
from dfthcsos.scoreinfra.dfthc_math import symmetric_from_lower

jax.config.update("jax_enable_x64", True)
src, out = sys.argv[1], sys.argv[2]
brot = int(sys.argv[3]) if len(sys.argv) > 3 else 8
rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 300
mu = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-5
t0 = time.time()
params, _ = read_factorization(src)
U = np.asarray(params["U"], dtype=np.float64)
W0 = np.asarray(params["W"], dtype=np.float64)
R, B, N = U.shape
C = W0.shape[2]
edges = np.asarray(TREES["log_depth"](N))
STEP = 1.0 / (1 << (brot - 1))
words = angles_to_words(tree_angles(U.reshape(-1, N), edges), brot).astype(np.int64)

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


def vectors(th):
    """Givens tree decode of theta/pi (the verifier's map)."""
    v = jnp.zeros((th.shape[0], N)).at[:, 0].set(1.0)
    for t in range(N - 1):
        p, c = int(edges[t, 0]), int(edges[t, 1])
        cs, sn = jnp.cos(jnp.pi * th[:, t]), jnp.sin(jnp.pi * th[:, t])
        vp, vc = v[:, p], v[:, c]
        v = v.at[:, p].set(cs * vp - sn * vc).at[:, c].set(sn * vp + cs * vc)
    return v.reshape(R, B, N)


def objective(Wv, th):
    Uv = vectors(th)
    A = jnp.einsum("rbc,rbn,rbm->rcnm", Wv, Uv, Uv).reshape(R * C, N * N)
    G = A @ A.T
    AV = A @ Vk
    e2 = jnp.sum(G ** 2) - 2 * jnp.sum(AV ** 2 * lk) + M2
    lam2 = 0.25 * jnp.sum(jnp.sum(jnp.abs(Wv), axis=1) ** 2)
    return e2 + mu * lam2, (e2, lam2)


f = jax.jit(objective)
gw = jax.jit(jax.grad(lambda Wv, th: objective(Wv, th)[0], argnums=0))
gt = jax.jit(jax.grad(lambda Wv, th: objective(Wv, th)[0], argnums=1))
opt = optax.adam(3e-4)
Wv = jnp.asarray(W0)
st = opt.init(Wv)
th = jnp.asarray(words * STEP, dtype=jnp.float64)
cur, (e2, lam2) = f(Wv, th)
print(f"start: error_2 {1000 * np.sqrt(max(float(e2), 0)):.2f} mHa, lambda_H2 {float(lam2):.1f} ({time.time() - t0:.0f}s)", flush=True)
k, acc = 64, 0
for rnd in range(rounds):
    for _ in range(20):
        upd, st = opt.update(gw(Wv, th), st)
        Wv = optax.apply_updates(Wv, upd)
    cur, _ = f(Wv, th)
    g = np.asarray(gt(Wv, th))
    flat = np.argsort(-np.abs(g).ravel())
    for _try in range(6):
        idx = flat[:k]
        w2 = words.copy().ravel()
        w2[idx] -= np.sign(g.ravel()[idx]).astype(np.int64)
        w2 = (w2 % (1 << brot)).reshape(words.shape)
        th2 = jnp.asarray(w2 * STEP, dtype=jnp.float64)
        new, _ = f(Wv, th2)
        if float(new) < float(cur):
            words, th, cur = w2, th2, new
            acc += k
            k = min(int(k * 1.5) + 1, 4096)
            break
        k = max(1, k // 2)
    if rnd % 10 == 0 or rnd == rounds - 1:
        _, (e2, lam2) = f(Wv, th)
        print(f"  round {rnd}: error_2 {1000 * np.sqrt(max(float(e2), 0)):.2f} mHa, lambda_H2 {float(lam2):.1f}, "
              f"k {k}, words moved {acc} ({time.time() - t0:.0f}s)", flush=True)
Uq = np.asarray(vectors(th))
np.savez(out, U=Uq, W=np.asarray(Wv), H2_I=np.asarray(params["H2_I"]), H1_I=np.asarray(params["H1_I"]),
         Hsym_lower=np.asarray(params["Hsym_lower"]))
print("wrote", out, flush=True)
