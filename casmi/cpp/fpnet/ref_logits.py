# reference: PyTorch FPNet on our <stem>_head/peaks.bin, per-spectrum logits + per-molecule (notebook molecule_logits)
# usage: ref_logits.py <code_dir> <ckpt> <data_dir> <stem> <n_mols> <out.npz>
import sys, numpy as np, torch
sys.path.insert(0, sys.argv[1])
from casmi import fpnet, chem
from casmi.spectra import merge_spectra
torch.set_num_threads(4)
code, ck, dd, stem, nm, out = sys.argv[1:7]; nm = int(nm)
OUR_AD = ["[M+H]+", "[M-H]-", "[M+Na]+", "[M+NH4]+", "[M+K]+", "[M+Cl]-", "[M+CH2O2-H]-", "[M+H-H2O]+", "[2M+H]+", "[2M+Na]+"]
MAP = {"[M+H-H2O]+": "[M-H2O+H]+"}
H = np.fromfile(f"{dd}/{stem}_head.bin", dtype=np.dtype([("p", "<f4"), ("mol", "<i4"), ("off", "<i4"), ("n", "<i4"), ("tag", "<i4")]))
P = np.fromfile(f"{dd}/{stem}_peaks.bin", dtype="<f4")
specs = []; cur = 0
for h in H:
    n = int(h["n"]); mz = P[cur:cur+n]; it = P[cur+n:cur+2*n]; cur += 2*n
    t = int(h["tag"]); a = -1 if t < 0 else (t // 16 + 1 if t % 16 == 15 else t // 16)
    ad = OUR_AD[a] if a >= 0 else "<unk>"; ad = MAP.get(ad, ad)
    mode = -1 if ad.endswith("-") else 1
    specs.append(dict(mol=int(h["mol"]), mz=mz.astype(np.float64), it=it.astype(np.float64), prec=float(h["p"]), adduct=ad, mode=mode))
m = fpnet.load_model(ck, "cpu")
def item(mz, it, prec, ad, nmerged, mode):
    pm, pi = fpnet.prep_peaks(mz, it, prec)
    return dict(mz=pm, it=pi, prec=prec, adduct_ix=chem.adduct_index(ad), ce=0.0, ce_known=0.0, n_merged=nmerged, mode=float(mode))
@torch.no_grad()
def run(items):
    out = []
    for it in items:   # batch of 1: exactly what C++ does
        out.append(m(*fpnet.collate([it], "cpu"))[0][0].numpy())
    return np.array(out)
mols = sorted(set(s["mol"] for s in specs))[:nm]
spec_z, mol_z, spec_idx = [], [], []
for mid in mols:
    S = [(i, s) for i, s in enumerate(specs) if s["mol"] == mid]
    zs = run([item(s["mz"], s["it"], s["prec"], s["adduct"], 1, s["mode"]) for _, s in S])
    spec_z.append(zs); spec_idx += [i for i, _ in S]
    im = []
    for md in (1, -1):
        g = [s for _, s in S if s["mode"] == md]
        if not g: continue
        mm, ii = merge_spectra([(s["mz"], s["it"]) for s in g])
        prec = float(np.median([s["prec"] for s in g])); ads = [s["adduct"] for s in g]
        im.append(item(mm, ii, prec, max(ads, key=ads.count), min(len(g), 8), md))
    zm = run(im)
    mol_z.append(0.5 * (zs.mean(0) + zm.mean(0)))
np.savez(out, spec_z=np.concatenate(spec_z), spec_idx=np.array(spec_idx), mol_z=np.array(mol_z), mols=np.array(mols))
print("ok", len(spec_idx), len(mols))
