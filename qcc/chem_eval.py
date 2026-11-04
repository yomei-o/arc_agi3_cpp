"""Official-verifier metrics of a factorization npz after encoding its vectors at brot bits on the log-depth tree
(dfthcsos.verify.load_inputs / evaluate_words; the starter kit's evaluator reports a different Lambda).
Run in C:\\prog\\qcc\\starter\\starter_kit:  python chem_eval.py factorization.npz [brot ...]
"""
import os, sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, r"C:\prog\qcc\verifiers")
import numpy as np
from dfthcsos.verify import load_inputs, evaluate_words
from starter.task2_angles.encode import angles_to_words, words_to_vectors, TREES
from starter.task2_angles.trees import tree_angles

f = sys.argv[1]
brots = [int(b) for b in sys.argv[2:]] or [8, 12]
inputs = load_inputs(f)
z = np.load(f)
U = z["U"].astype(np.float64)
R, B, N = U.shape
edges = np.asarray(TREES["log_depth"](N))
for brot in brots:
    two = angles_to_words(tree_angles(U.reshape(-1, N), edges), brot)
    # one-body rows: eigenvectors of h1_prime(X) of the decoded two-body vectors (as the starter encoder does)
    from dfthcsos.scoreinfra.dfthc_math import reconstruct_x, build_h1_prime
    vq = words_to_vectors(two, edges, brot).reshape(R, B, N)
    X = reconstruct_x(vq, inputs.w)
    X = np.tril(X) + np.swapaxes(np.tril(X, k=-1), -1, -2)
    h1p = build_h1_prime(inputs.ref["h1_thc"], inputs.S, inputs.beta_1, inputs.ref["electrons"], inputs.W, X)
    _, ev = np.linalg.eigh(h1p)
    one = angles_to_words(tree_angles(ev.T, edges), brot)
    m = evaluate_words(inputs, np.vstack([one, two]), edges, brot)
    print(f"brot {brot}: error_1 {1000 * m['error_1']:.2f} mHa, error_2 {1000 * m['error_2']:.2f} mHa, "
          f"Lambda {m['Lambda']:.2f}, E_gap {m['E_gap']:.4f}, lambda_eff {m['lambda_eff']:.3f}", flush=True)
