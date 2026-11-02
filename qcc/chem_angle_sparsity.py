"""How sparse are the ANGLE_k deltas, and what would a pruned unary iteration cost?

delta[k, a] = batch[k, a] XOR batch[k-1, a] (SCORING_README). A unary iteration over the address tree only has to
visit subtrees that contain a nonzero delta; its Toffoli cost is about the number of visited internal nodes
(one AND per visited node), versus L - 1 for the full iteration. Prints, per batch: nonzero rows, full cost,
pruned cost.  python chem_angle_sparsity.py submission.npz [J]
"""
import math, sys
import numpy as np

z = np.load(sys.argv[1])
J = int(sys.argv[2]) if len(sys.argv) > 2 else 8
words = z["angle_words"].astype(np.int64)          # (L, 75)
L, NA = words.shape
K = math.ceil(NA / J)
pad = np.zeros((L, K * J), dtype=np.int64)
pad[:, :NA] = words
bits = 1 << 64  # pack a batch's J words into one Python int per row
batch = [[tuple(pad[a, k * J:(k + 1) * J]) for a in range(L)] for k in range(K)]
nb = math.ceil(math.log2(L))


def pruned_cost(nonzero):
    """Visited internal nodes of the complete binary address tree restricted to subtrees with a nonzero leaf."""
    cost = 0
    level = set(np.flatnonzero(nonzero).tolist())
    for _ in range(nb):
        parents = {a >> 1 for a in level}
        cost += len(parents)
        level = parents
    return cost


tot_full = tot_pruned = 0
zero_words = np.mean(words == 0)
for k in range(K):
    prev = batch[k - 1] if k else [tuple([0] * J)] * L
    nz = np.array([batch[k][a] != prev[a] for a in range(L)])
    full = L - 1
    pr = pruned_cost(nz)
    tot_full += full
    tot_pruned += min(full, pr)
    print(f"batch {k}: nonzero delta rows {nz.sum()}/{L}, full {full}, pruned ~{pr}")
print(f"L {L}, J {J}, K {K}, zero angle words {100 * zero_words:.1f}%: per pass full {tot_full}, pruned ~{tot_pruned}")
