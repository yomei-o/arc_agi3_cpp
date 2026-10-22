"""mol_cmp: sanity comparison of two _mol.bin files (per-molecule mean logits) of the same query set: finite values,
mean logit, mean sum of p, and per-molecule Pearson correlation of p. usage: python mol_cmp.py <a_mol.bin> <b_mol.bin>"""
import math
import struct
import sys


def load(path):
    raw = open(path, "rb").read()
    n, nb = struct.unpack_from("<ii", raw, 0)
    off, out = 8, {}
    for _ in range(n):
        (i,) = struct.unpack_from("<i", raw, off)
        off += 4
        out[i] = struct.unpack_from(f"<{nb}f", raw, off)
        off += 4 * nb
    return out


sig = lambda x: 1 / (1 + math.exp(-max(-30, min(30, x))))  # noqa: E731
A, B = load(sys.argv[1]), load(sys.argv[2])
for name, M in (("A", A), ("B", B)):
    bad = sum(1 for v in M.values() for x in v if not math.isfinite(x))
    ml = sum(sum(v) / len(v) for v in M.values()) / len(M)
    sp = sum(sum(sig(x) for x in v) for v in M.values()) / len(M)
    print(f"{name}: {len(M)} molecules, non-finite {bad}, mean logit {ml:.3f}, mean sum p {sp:.1f}")
cs = []
for k in A.keys() & B.keys():
    a = [sig(x) for x in A[k]]; b = [sig(x) for x in B[k]]
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    c = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    d = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) + 1e-12
    cs.append(c / d)
cs.sort()
print(f"per-molecule corr(p_A, p_B): n {len(cs)}, median {cs[len(cs) // 2]:.3f}, min {cs[0]:.3f}")
