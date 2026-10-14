import sys, numpy as np
r = np.load(sys.argv[1]); pre = sys.argv[2]
a = np.fromfile(pre + "_spec.bin", dtype="<i4", count=2); S = np.fromfile(pre + "_spec.bin", dtype="<f4", offset=8).reshape(a[0], a[1])
d = np.abs(S[r["spec_idx"]] - r["spec_z"]); print("spec maxabs", d.max(), "mean", d.mean(), "ref range", np.abs(r["spec_z"]).max())
h = np.fromfile(pre + "_mol.bin", dtype="<i4", count=2); M = np.fromfile(pre + "_mol.bin", dtype="<f4", offset=8).reshape(h[0], h[1] + 1)
mids = M[:, 0].view("<i4"); Z = M[:, 1:]
for i, m in enumerate(r["mols"]):
    j = np.where(mids == m)[0][0]; print("mol", m, "maxabs", np.abs(Z[j] - r["mol_z"][i]).max())
