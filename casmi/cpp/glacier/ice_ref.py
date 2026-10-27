"""ice_ref.py: reference dumps of the public ICEBERG 2.1 (ms-pred FragGNN + IntenGNN, MassSpecGym checkpoints, MIT)
for the own C++ port (cpp/glacier/ice_model.hpp).

    python ice_ref.py <iceberg pkg dir> <site dir> <out dir> [cases.tsv]
writes <out>/ice_gen_w.bin, <out>/ice_inten_w.bin (GLW1, every state_dict tensor), <out>/ice_hp.json and per case
<out>/iref_<k>.bin + .txt (root canonical smiles = the runner's rm_stereo -> InChI round trip, adduct, instrument, CE):
  leave0      [n_atoms]       gen model's leaving probabilities of the root's atoms at depth 0
  frags       [F, n_atoms]    final fragments (atom masks, order of the gen output), probs [F], brokens [F],
  masses      [F]             masses_no_adduct, forms [F, 18] fragment formula vectors
  spec        [rows, 2]       joint predict_mol output (mass, intensity), post_mz / post_it = the runner's post()
CPU, float32, batch of one, the runner's own model loading (incl. its GatedGraphConv speed patch, same math).
"""
import json, os, struct, sys

import numpy as np

PKG, SITE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
CASES = sys.argv[4] if len(sys.argv) > 4 else None
sys.path.insert(0, PKG)
import ice_runner as R  # noqa: E402

a = R.parse([PKG, os.path.join(OUT, "_in.json"), os.path.join(OUT, "_out.json"), "--site", SITE, "--any-rdkit",
             "--device", "cpu"])
log = R.Log(False)
os.makedirs(OUT, exist_ok=True)
R.setup_env(a, log)
import torch  # noqa: E402
import ms_pred.common as common  # noqa: E402

torch.set_num_threads(4)
jm = R.load_model(os.path.abspath(PKG), "cpu", log)
gen, inten = jm.gen_model_obj, jm.inten_model_obj


def write_arrays(path, arrs):
    with open(path, "wb") as f:
        f.write(b"GLW1")
        f.write(struct.pack("<I", len(arrs)))
        for name, v in arrs:
            v = np.ascontiguousarray(np.asarray(v, dtype=np.float32))
            nb = name.encode()
            f.write(struct.pack("<I", len(nb))); f.write(nb)
            f.write(struct.pack("<I", v.ndim))
            for d in v.shape:
                f.write(struct.pack("<I", d))
            f.write(v.tobytes())


for nm, m in (("ice_gen_w.bin", gen), ("ice_inten_w.bin", inten)):
    write_arrays(os.path.join(OUT, nm), [(k, v.double().float().cpu().numpy()) for k, v in m.state_dict().items()])
json.dump({"gen": {k: str(v) for k, v in dict(gen.hparams).items()},
           "inten": {k: str(v) for k, v in dict(inten.hparams).items()},
           "CHEM_MASSES": np.asarray(common.CHEM_MASSES).reshape(-1).tolist(), "MAX_H": int(common.MAX_H),
           "element_to_position_C": np.asarray(common.element_to_position["C"]).tolist()},
          open(os.path.join(OUT, "ice_hp.json"), "w"), indent=1)

cases = []
if CASES:
    for l in open(CASES, encoding="utf-8"):
        p = l.rstrip("\n").split("\t")
        if len(p) >= 4 and not l.startswith("#"):
            cases.append((p[0], p[1], p[2], float(p[3])))
else:
    cases = [("CC(=O)Oc1ccccc1C(=O)O", "[M+H]+", "QTOF", 20.0),
             ("Cn1cnc2c1c(=O)n(C)c(=O)n2C", "[M+Na]+", "Orbitrap", 40.0),
             ("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "[M+H]+", "QTOF", 35.0),
             ("COc1cc(C=CC(=O)OC2C(O)C(O)C(CO)OC2Oc2ccc(CC=C)cc2)ccc1O", "[M+H]+", "QTOF", 30.0)]

rec = {}
orig_fwd = gen.forward


def hook(*args, **kw):
    out = orig_fwd(*args, **kw)
    if "leave0" not in rec:
        rec["leave0"] = out[0].detach().cpu().numpy()
    return out


gen.forward = hook
for k, (smi, ad, ins, ce) in enumerate(cases):
    root = common.smiles_from_inchi(common.inchi_from_smiles(common.rm_stereo(smi)))
    rec.clear()
    with torch.no_grad():
        fp, _ = gen.predict_mol(root_smi=root, collision_eng=ce, precursor_mz=0.0, adduct=ad, instrument=ins,
                                threshold=0.0, device="cpu", max_nodes=100, canonical_root_smi=True)
        out = jm.predict_mol([root], collision_eng=[float(ce)], precursor_mz=[0.0], adduct=[ad], threshold=0.0,
                             device="cpu", max_nodes=100, instrument=[ins], binned_out=False, canonical_root_smi=True)
    nf = int(fp["nfrags"])
    spec = out["spec"][0].float().cpu().numpy()
    pm, pi = R.post(spec)
    arrs = [("leave0", rec["leave0"]), ("frags", fp["frags"][:nf].float().numpy()), ("probs", fp["probs"][:nf].numpy()),
            ("brokens", fp["brokens"][:nf].numpy()), ("masses", fp["masses_no_adduct"][:nf].numpy()),
            ("forms", fp["frag_form_vecs"][:nf].numpy()), ("root_form", fp["root_form_vec"].numpy()),
            ("spec", spec), ("post_mz", pm), ("post_it", pi)]
    write_arrays(os.path.join(OUT, f"iref_{k}.bin"), arrs)
    with open(os.path.join(OUT, f"iref_{k}.txt"), "w", encoding="utf-8") as f:
        f.write(f"{root}\t{ad}\t{ins}\t{ce}\n")
    print(k, root, "atoms", int(fp["natoms"]), "frags", nf, "rows", spec.shape[0], "post", len(pm))
