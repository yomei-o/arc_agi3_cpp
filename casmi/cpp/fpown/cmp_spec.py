# compare per-spectrum logits: fpnet_ggml _spec.bin (exported file) vs tfm_train dumpz (our trainer's forward pass)
# usage: cmp_spec.py a_spec.bin b_dumpz.bin
import sys, numpy as np
def rd(p):
    b = np.fromfile(p, np.int32, 2); n, nb = int(b[0]), int(b[1])
    return np.fromfile(p, np.float32, offset=8).reshape(n, nb)
a, b = rd(sys.argv[1]), rd(sys.argv[2])
assert a.shape == b.shape, (a.shape, b.shape)
rows = np.flatnonzero((np.abs(a).sum(1) > 0) & (np.abs(b).sum(1) > 0))
d = np.abs(a[rows] - b[rows])
print(f"{len(rows)} spectra compared, logit range [{b[rows].min():.2f}, {b[rows].max():.2f}]")
print("per-spectrum max|diff|:", " ".join(f"{x:.2e}" for x in d.max(1)))
print(f"overall max|diff| {d.max():.3e}  mean|diff| {d.mean():.3e}")
