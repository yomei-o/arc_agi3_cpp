"""gl_ref.py: reference dumps of the public GLACIER (ms-pred glacier.JointModel, MassSpecGym checkpoint, MIT) for the
own C++ inference port (cpp/glacier/gl_infer.cpp).

    python gl_ref.py <glacier pkg dir> <site dir> <out dir> [cases.tsv]
writes
  <out>/glacier_w.bin   every state_dict tensor: "GLW1", u32 n; per tensor: u32 name len, name, u32 ndim, u32 dims[ndim],
                        float32 data (row-major)
  <out>/glacier_hp.json hyper-parameters + the ms_pred.common tables the featurisation uses
  <out>/ref_<k>.bin     per case (canonical smiles, adduct, instrument token, CE): named float32 arrays in the same
                        format: inputs (x, degree, spatial_pos, edge_input, attn_edge_type, masses, atom_hs, total_hs,
                        root_form_vec, atom_form_vecs, adj, adduct, instrument, ce), intermediates (node_emb, root_tok,
                        frag_logits, card_logits, breakpoints, fragments) and outputs (inten_out = [rows, 2] raw,
                        post_mz, post_it = the runner's post()).
Runs on CPU in float32 (the runner's own model loading and featurisation; batch of one).
"""
import json, os, struct, sys

import numpy as np

PKG, SITE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
CASES = sys.argv[4] if len(sys.argv) > 4 else None
sys.path.insert(0, PKG)
import gl_runner as R  # noqa: E402

a = R.parse([PKG, os.path.join(OUT, "_in.json"), os.path.join(OUT, "_out.json"), "--site", SITE, "--any-rdkit",
             "--device", "cpu"])
log = R.Log(False)
os.makedirs(OUT, exist_ok=True)
R.setup_env(a, log)
import torch  # noqa: E402
import ms_pred.common as common  # noqa: E402
import ms_pred.nn_utils as nn_utils  # noqa: E402

torch.set_num_threads(4)
model = R.load_model(os.path.abspath(PKG), "cpu", log)


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


sd = model.state_dict()
write_arrays(os.path.join(OUT, "glacier_w.bin"), [(k, v.float().cpu().numpy()) for k, v in sd.items()])


def js(v):
    if isinstance(v, dict):
        return {str(k): js(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [js(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)


tabs = {}
for name in ("VALID_ATOM_NUM", "MAX_COMMON_DEGREE", "MAX_ABS_FORMAL_CHARGE", "COMMON_HYBRIDIZATION",
             "COMMON_MAX_HYDROGEN_COUNTS", "COMMON_RING_SIZES", "COMMON_CHIRALITY", "COMMON_BOND_TYPES",
             "VALID_BOND_STEREO", "ELEMENT_TO_GROUP", "ELEMENT_GROUP_DIM", "element_to_group", "COLLISION_PE_DIM",
             "COLLISION_PE_SCALAR", "ion2onehot_pos", "ion_pos2extra_multihot", "instrument2onehot_pos", "ion2mass",
             "ELECTRON_MASS", "NORM_VEC", "ELEMENT_TO_MASS", "VALID_ELEMENTS", "element_to_ind"):
    if hasattr(common, name):
        tabs[name] = js(getattr(common, name))
hp = {k: js(v) for k, v in dict(model.hparams).items()}
json.dump({"hp": hp, "common": tabs, "embedder_num_dim": int(model.embedder.num_dim),
           "positive": {ad: bool(common.is_positive_adduct(ad)) for ad in common.ion2mass}},
          open(os.path.join(OUT, "glacier_hp.json"), "w"), indent=1)

cases = []
if CASES:
    for l in open(CASES, encoding="utf-8"):
        p = l.rstrip("\n").split("\t")
        if len(p) >= 4 and not l.startswith("#"):
            cases.append((p[0], p[1], p[2], float(p[3])))
else:
    cases = [("CC(=O)Oc1ccccc1C(=O)O", "[M+H]+", "QTOF", 20.0),
             ("Cn1cnc2c1c(=O)n(C)c(=O)n2C", "[M+Na]+", "Orbitrap", 40.0),
             ("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "[M+H]+", "QTOF", 35.0)]

fz = R.Featurizer(model)
from ms_pred.glacier import dataset as gl_ds  # noqa: E402

summary = []
for k, (smi, ad, ins, ce) in enumerate(cases):
    ok, why = R.smiles_ok(smi)
    if not ok:
        summary.append((k, smi, "REJECT " + why)); continue
    c = R._CANON[smi][0]
    item = fz.item(c, ad, float(ce), ins)
    batch = gl_ds.IntenDataset.collate_fn([item])
    g = batch["graphormer_input"]
    arrs = [("smiles_len", np.array([len(c)]))]
    for nm in ("x", "degree", "spatial_pos", "edge_input", "attn_edge_type", "attn_bias"):
        arrs.append(("in_" + nm, g[nm][0].numpy()))
    for nm in ("masses", "atom_hs", "total_hs", "root_form_vecs", "atom_form_vecs", "adj_matrices", "adducts",
               "instruments", "collision_engs", "adduct_mass_shifts", "num_atoms"):
        v = torch.as_tensor(batch[nm])
        arrs.append(("in_" + nm, (v[0] if nm in ("masses", "atom_hs", "root_form_vecs", "adj_matrices",
                                                   "adduct_mass_shifts") else v).float().numpy()))
    with torch.inference_mode():
        emb = model.molecular_embedding(batch["adducts"], batch["collision_engs"], graphormer_input=g,
                                        instruments=batch["instruments"])
        rt, ne = emb["root_tokens"], emb["node_embeddings"]
        bp = model.breakpoint_forward(ne, rt, batch["num_atoms"], batch["root_form_vecs"])
        fl, cl = bp["frag_logits"][-1], bp["frag_card_logits"][-1]
        brk = model.breakpoint_inference(fl, cl, batch["num_atoms"])
        frags, fcount = model.breakpoints_to_patterns(brk, batch["adj_matrices"], batch["num_atoms"])
        frags_p = nn_utils.pack_padded_tensor(frags, fcount).bool()
        ip = model.inten_calculation(rt, ne, frags_p, fcount, batch["root_form_vecs"], batch["atom_form_vecs"],
                                     batch["num_atoms"], batch["atom_hs"], batch["total_hs"], batch["adj_matrices"],
                                     batch["adduct_mass_shifts"], batch["masses"])
        full = model.predict_inten_frag_batch(batch)["spec"][0]
    out = ip["output"][0]
    nrows = int(fcount[0]) * batch["adduct_mass_shifts"].shape[1] * model.output_size
    pm, pi = R.post(full.float().cpu().numpy())
    arrs += [("node_emb", ne[0].numpy()), ("root_tok", rt[0].numpy()), ("frag_logits", fl[0].numpy()),
             ("card_logits", cl[0].numpy()), ("breakpoints", brk[0].float().numpy()),
             ("fragments", frags[0, :int(fcount[0])].float().numpy()), ("inten_out", out[:nrows].numpy()),
             ("full_out", full.float().numpy()), ("post_mz", pm), ("post_it", pi)]
    write_arrays(os.path.join(OUT, f"ref_{k}.bin"), arrs)
    with open(os.path.join(OUT, f"ref_{k}.txt"), "w", encoding="utf-8") as f:
        f.write(f"{c}\t{ad}\t{ins}\t{ce}\n")
    summary.append((k, c, f"atoms {int(batch['num_atoms'][0])} frags {int(fcount[0])} rows {nrows} post {len(pm)} "
                          f"max|inten-full| {float((out[:nrows] - full).abs().max()) if full.shape == out[:nrows].shape else -1:.2e}"))
for s in summary:
    print(*s)
