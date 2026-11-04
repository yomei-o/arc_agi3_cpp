"""Build and verify a Benchmark III submission from a factorization npz.

Two-body words: the npz's angle_words_two_body if present (from chem_round.py), else the vectors encoded on the
log-depth tree. One-body words: eigenvectors of h1_prime(X of the decoded two-body vectors), encoded the same way.
Circuits: the hosts' plain QROM family (example_submission/angle_k_reference_circuit.py) at brot bits, J words per
circuit. Then the official verifier.
Run in C:\\prog\\qcc with the venv python (kickmix):  python chem_build.py factorization.npz outdir [brot] [J]
"""
import importlib.util, json, os, subprocess, sys
import numpy as np

sys.path.insert(0, r"C:\prog\qcc\verifiers")
sys.path.insert(0, r"C:\prog\qcc\starter\starter_kit")
from dfthcsos.verify import load_inputs, evaluate_words
from dfthcsos.scoreinfra.dfthc_math import reconstruct_x, build_h1_prime
from starter.task2_angles.trees import tree_angles, log_depth_tree_edges
from starter.task2_angles.encode import angles_to_words, words_to_vectors

spec = importlib.util.spec_from_file_location(
    "akrc", r"C:\prog\qcc\verifiers\dfthcsos\example_submission\angle_k_reference_circuit.py")
akrc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(akrc)

f, outdir = sys.argv[1], sys.argv[2]
brot = int(sys.argv[3]) if len(sys.argv) > 3 else 8
J = int(sys.argv[4]) if len(sys.argv) > 4 else 5
inputs = load_inputs(f)
z = np.load(f)
U = np.asarray(z["U"], dtype=np.float64)
R, B, N = U.shape
edges = np.asarray(log_depth_tree_edges(N))
if "angle_words_two_body" in z.files:
    two = np.asarray(z["angle_words_two_body"], dtype=np.int64)
else:
    two = angles_to_words(tree_angles(U.reshape(-1, N), edges), brot)
vq = words_to_vectors(two, edges, brot).reshape(R, B, N)
X = reconstruct_x(vq, inputs.w)
X = np.tril(X) + np.swapaxes(np.tril(X, k=-1), -1, -2)
h1p = build_h1_prime(inputs.ref["h1_thc"], inputs.S, inputs.beta_1, inputs.ref["electrons"], inputs.W, X)
_, ev = np.linalg.eigh(h1p)
one = angles_to_words(tree_angles(ev.T, edges), brot)


def polish_onebody(one, h1p, sweeps=6):
    """Coordinate descent on the one-body words: minimize ||sum_j lam_j v_j v_j^T - h1'||_F (= error_1, with the
    verifier's eigenvalue matching, which small moves do not change). Moves of +-1, +-2 words per angle."""
    lam, vecs = np.linalg.eigh(h1p)
    one = one.copy()
    V = words_to_vectors(one, edges, brot)                      # (N, N) rows
    match = np.abs(V @ vecs).argmax(1)
    lj = lam[match]
    E = np.einsum("j,jp,jq->pq", lj, V, V) - h1p
    e0 = np.linalg.norm(E)
    for sw in range(sweeps):
        moved = 0
        for j in range(N):
            for t in range(N - 1):
                best = None
                for d in (-2, -1, 1, 2):
                    cand = one[j].copy()
                    cand[t] = (cand[t] + d) % (1 << brot)
                    v2 = words_to_vectors(cand[None], edges, brot)[0]
                    D = lj[j] * (np.outer(v2, v2) - np.outer(V[j], V[j]))
                    gain = 2 * np.sum(E * D) + np.sum(D * D)
                    if gain < -1e-15 and (best is None or gain < best[0]):
                        best = (gain, cand, v2, D)
                if best:
                    one[j], V[j] = best[1], best[2]
                    E += best[3]
                    moved += 1
        print(f"  one-body polish sweep {sw}: error_1 ~{1000 * np.linalg.norm(E):.2f} mHa (start {1000 * e0:.2f}), moved {moved}", flush=True)
        if not moved:
            break
    return one


one = polish_onebody(one, h1p)
words = np.vstack([one, two]).astype(np.int64)
m = evaluate_words(inputs, words, edges, brot)
print(json.dumps({k: (float(v) if not isinstance(v, bool) else v) for k, v in m.items()}), flush=True)
os.makedirs(outdir, exist_ok=True)
for old in os.listdir(outdir):  # stale circuits of an earlier J would join the family
    if old.startswith("angle_k_"):
        os.remove(os.path.join(outdir, old))
np.savez(os.path.join(outdir, "submission.npz"), U=U, W=np.asarray(z["W"]), H2_I=np.asarray(z["H2_I"]),
         H1_I=np.asarray(z["H1_I"]), Hsym_lower=np.asarray(z["Hsym_lower"]), tree_edges=edges, angle_words=words)
for i, c in enumerate(akrc.build_angle_k_family(words, brot, J)):
    c.to_file(os.path.join(outdir, f"angle_k_{i}.kmb"), format="kmb")
r = subprocess.run([sys.executable, r"C:\prog\qcc\verifiers\dfthcsos\verify.py", os.path.abspath(outdir)], capture_output=True,
                   text=True, cwd=r"C:\prog\qcc\verifiers")
print(r.stdout[-3000:], r.stderr[-2000:], flush=True)
