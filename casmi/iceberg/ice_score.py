r"""ice_score.py --model ice|gl --mspred <ms-pred src dir> --cands <json> --queries <q_<qry>.json> --out <json>
             [--device cpu|cuda] [--threads 4] [--bs 32] [--max-mols N] [--ce-default 30] [--adducts "[M+H]+,[M+Na]+"]

Glue around ms-pred (github coleygroup/ms-pred, commit 708148c2a8eb0c32120644436fefd2fe90cd2214, MIT), NOT a copy of it:
predicts each candidate's MS/MS with the MassSpecGym-trained ICEBERG 2.1 (msg_all: gen + inten_contr best.ckpt)
or GLACIER (glacier best.ckpt) at every query spectrum's condition (adduct, instrument, collision energy rounded to
a 5 eV bucket) and scores it against the query spectrum (entropy similarity and cosine, 0.01 Da).

cands  : {mol: [smiles, ...]}        (the candidates to score, any order)
queries: {mol: [{prec, adduct, instr, mode, ce: [..]|null, mz, it}, ...]}   (q_extract.py)
out    : {mol: {smiles: [[ent, cos, ce_known(0/1), spectrum index], ...]}} + out.meta.json (timings / counts).
Only positive-mode spectra whose adduct is in --adducts are scored (the public pipelines use [M+H]+ / [M+Na]+);
a spectrum with no recorded CE is predicted at --ce-default and flagged ce_known=0.
Resumable: an existing --out is loaded and its molecules are skipped; the file is rewritten every chunk.
"""
import argparse
import json
import math
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["ice", "gl"])
ap.add_argument("--mspred", required=True)
ap.add_argument("--ckpt", default=r"C:\prog\casmi\iceberg\dl\dropbox")
ap.add_argument("--cands", required=True)
ap.add_argument("--queries", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cpu")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--max-mols", type=int, default=0)
ap.add_argument("--ce-default", type=float, default=30.0)
ap.add_argument("--adducts", default="[M+H]+,[M+Na]+")
ap.add_argument("--dgl-stub", action="store_true",
                help="GLACIER only: register an empty `dgl` module (its predict path never builds a DGL graph; "
                     "DGL has no Windows CUDA wheels)")
a = ap.parse_args()

import torch  # noqa: E402

if a.dgl_stub:
    # import-time names only: every attribute resolves to an inert placeholder (a class for Capitalised names, a
    # function returning None otherwise). GLACIER's predict path never calls them; ICEBERG would fail loudly.
    def _stub_getattr(name):
        if name.startswith("__"):
            raise AttributeError(name)
        if name[:1].isupper():
            return type(name, (object,), {"__init__": lambda self, *x, **k: None})
        return lambda *x, **k: None

    for name in ["dgl", "dgl.nn", "dgl.function", "dgl.backend", "dgl.backend.pytorch"]:
        m = types.ModuleType(name)
        m.__path__ = []
        m.__getattr__ = _stub_getattr
        sys.modules[name] = m
    sys.modules["dgl"].NID = "_ID"

    class _MiniGraph:
        """the only DGL graph GLACIER's predict path builds: a CPU edge list for random-walk positional encodings
        (glacier/dataset.py get_pe_for_tensor -> nn_utils.random_walk_pe: edges / num_nodes / edata / ndata)."""
        def __init__(self, src, dst, num_nodes):
            self._s = torch.as_tensor(np.asarray(src), dtype=torch.long)
            self._d = torch.as_tensor(np.asarray(dst), dtype=torch.long)
            self._n = int(num_nodes)
            self.ndata, self.edata, self.device = {}, {}, torch.device("cpu")

        def edges(self):
            return self._s, self._d

        def num_nodes(self):
            return self._n

        def num_edges(self):
            return len(self._s)

    def _graph(data, num_nodes=None, **kw):
        s_, d_ = data
        n_ = num_nodes if num_nodes is not None else (int(max(np.max(s_), np.max(d_))) + 1 if len(s_) else 0)
        return _MiniGraph(s_, d_, n_)
    sys.modules["dgl"].graph = _graph
    sys.modules["dgl"].DGLGraph = _MiniGraph
    sys.modules["dgl"].nn = sys.modules["dgl.nn"]
    sys.modules["dgl"].function = sys.modules["dgl.function"]
    sys.modules["dgl"].backend = sys.modules["dgl.backend"]
    sys.modules["dgl.backend"].pytorch = sys.modules["dgl.backend.pytorch"]

import torch  # noqa: E402
import pathlib  # noqa: E402

if os.name == "nt":  # the checkpoints pickle PosixPath objects (hparams); load them as WindowsPath
    pathlib.PosixPath = pathlib.WindowsPath

torch.set_num_threads(a.threads)
sys.path.insert(0, a.mspred)
try:  # torch-scatter wheels for torch < 2.4 refuse int32 indices (DGL on Windows hands int32 edge ids): cast to int64
    import torch_scatter

    def _long_index(f):
        def g(src, index, *args, **kw):
            return f(src, index.long(), *args, **kw)
        return g
    for _n in ["scatter", "scatter_sum", "scatter_add", "scatter_mean", "scatter_max", "scatter_min", "scatter_mul"]:
        if hasattr(torch_scatter, _n):
            setattr(torch_scatter, _n, _long_index(getattr(torch_scatter, _n)))
except ImportError:
    pass
import ms_pred.common as common  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

T0 = time.time()
if a.model == "ice":
    from ms_pred.iceberg import joint_model as ice_joint  # noqa: E402
    model = ice_joint.JointModel.from_checkpoints(os.path.join(a.ckpt, "ice_msg_all", "gen", "best.ckpt"),
                                                   os.path.join(a.ckpt, "ice_msg_all", "inten_contr", "best.ckpt"))
    embed_instr = model.embed_instrument
else:
    from ms_pred.glacier import joint_model as gl_joint  # noqa: E402
    model = gl_joint.JointModel.load_from_checkpoint(os.path.join(a.ckpt, "glacier_msg", "best.ckpt"),
                                                     map_location="cpu")
    embed_instr = bool(getattr(model, "embed_instrument", False))
model.eval()
model = model.to(a.device)
t_load = time.time() - T0
print(f"loaded {a.model} in {t_load:.1f}s, embed_instrument={embed_instr}, device={a.device}", flush=True)

ADDUCTS = set(a.adducts.split(","))
INSTR = {"Orbitrap": "Orbitrap", "ESI-QFT": "Orbitrap", "QTOF": "QTOF", "LC-ESI-QTOF": "QTOF", "timsTOF": "QTOF"}
TOL = 0.01


def ce_bucket(ce):
    if not ce:
        return a.ce_default, 0
    v = float(np.mean(ce))
    return float(5.0 * round(v / 5.0)), 1


def norm_peaks(mz, it, maxmz=None):
    mz = np.asarray(mz, np.float64)
    it = np.asarray(it, np.float64)
    keep = it > 0
    if maxmz is not None:
        keep &= mz <= maxmz + 1.5
    mz, it = mz[keep], it[keep]
    o = np.argsort(mz)
    return mz[o], it[o]


def merge_pred(mz, it, how):
    """drop <= 0, merge identical m/z (4 dp) by `how` (max | sum), keep top 100."""
    keep = it > 0
    mz, it = np.round(mz[keep], 4), it[keep]
    if not len(mz):
        return mz, it
    u, inv = np.unique(mz, return_inverse=True)
    v = np.zeros(len(u))
    if how == "max":
        np.maximum.at(v, inv, it)
    else:
        np.add.at(v, inv, it)
    o = np.argsort(-v)[:100]
    o = o[np.argsort(u[o])]
    return u[o], v[o]


def match(m1, m2):
    """greedy one-to-one peak matching within TOL, closest pairs first. returns index pairs."""
    if not len(m1) or not len(m2):
        return [], []
    j = np.searchsorted(m2, m1)
    pairs = []
    for i in range(len(m1)):
        for jj in (j[i] - 1, j[i]):
            if 0 <= jj < len(m2) and abs(m2[jj] - m1[i]) <= TOL:
                pairs.append((abs(m2[jj] - m1[i]), i, jj))
    pairs.sort()
    u1, u2, I, J = set(), set(), [], []
    for _, i, jj in pairs:
        if i in u1 or jj in u2:
            continue
        u1.add(i); u2.add(jj); I.append(i); J.append(jj)
    return I, J


def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def weigh(p):
    p = p / p.sum()
    s = entropy(p)
    if s < 3.0:
        p = p ** (0.25 + 0.25 * s)
        p = p / p.sum()
    return p


def sims(qm, qi, pm, pi):
    """(entropy similarity [weighted, Li et al. 2021], cosine) between query and prediction."""
    if not len(pm) or not len(qm) or pi.sum() <= 0:
        return 0.0, 0.0
    I, J = match(qm, pm)
    a_ = weigh(qi.copy())
    b_ = weigh(pi.copy())
    # entropy similarity: 1 - (2 S_AB - S_A - S_B) / ln 4, AB = (A + B) / 2 with matched peaks merged
    ab = []
    ua, ub = set(I), set(J)
    for i, j in zip(I, J):
        ab.append(a_[i] + b_[j])
    ab += [a_[i] for i in range(len(a_)) if i not in ua]
    ab += [b_[j] for j in range(len(b_)) if j not in ub]
    ab = np.asarray(ab) / 2.0
    ent = 1.0 - (2 * entropy(ab) - entropy(a_) - entropy(b_)) / math.log(4)
    cos = float(np.dot(qi[I], pi[J]) / (np.linalg.norm(qi) * np.linalg.norm(pi))) if I else 0.0
    return float(max(0.0, min(1.0, ent))), cos


def predict(smis, ces, adducts, instrs, precs):
    with torch.no_grad():
        if a.model == "ice":
            out = model.predict_mol(smis, collision_eng=ces, precursor_mz=precs, adduct=adducts, threshold=0.0,
                                    device=a.device, max_nodes=100, instrument=instrs if embed_instr else None,
                                    binned_out=False, adduct_shift=False, canonical_root_smi=True)
        else:
            out = model.predict_mol(smis, collision_eng=ces, adduct=adducts, device=a.device,
                                    instrument=instrs if embed_instr else None)
    res = []
    for sp in out["spec"]:
        sp = sp.float().cpu().numpy()
        res.append(merge_pred(sp[:, 0].astype(np.float64), sp[:, 1].astype(np.float64),
                              "max" if a.model == "gl" else "sum"))
    return res


cands = json.load(open(a.cands))
Q = json.load(open(a.queries))
res = json.load(open(a.out)) if os.path.exists(a.out) else {}
meta = dict(model=a.model, device=a.device, threads=a.threads, n_pred=0, n_fail=0, t_pred=0.0, t_load=t_load,
            n_mols=0, n_cand=0, n_spec=0, n_cond=0, skipped_spec=0)
mids = [m for m in cands if m not in res and m in Q]
if a.max_mols:
    mids = mids[:a.max_mols]
last = time.time()
for k, mid in enumerate(mids):
    specs = []
    for si, s in enumerate(Q[mid]):
        if s.get("mode") not in (None, "positive") or s["adduct"] not in ADDUCTS:
            meta["skipped_spec"] += 1
            continue
        ce, known = ce_bucket(s.get("ce"))
        qm, qi = norm_peaks(s["mz"], s["it"], s["prec"])
        if not len(qm):
            continue
        specs.append((si, s["adduct"], INSTR.get(s.get("instr"), "Orbitrap"), ce, known, s["prec"], qm, qi))
    out = {}
    if specs:
        conds = sorted(set((ad, ins, ce) for _, ad, ins, ce, _, _, _, _ in specs))
        prec_of = {(ad, ins, ce): p for _, ad, ins, ce, _, p, _, _ in specs}
        smis = []
        for smi in cands[mid]:
            m = Chem.MolFromSmiles(smi)
            if m is None or m.GetNumAtoms() < 2:
                continue
            if any(at.GetSymbol() not in common.VALID_ELEMENTS for at in m.GetAtoms()):
                meta["skipped_elem"] = meta.get("skipped_elem", 0) + 1   # e.g. Hg / Sn: no mass table, no score
                continue
            smis.append((smi, Chem.MolToSmiles(m)))
        jobs = [(c, s0, s1) for c in conds for (s0, s1) in smis]
        pred = {}
        t1 = time.time()
        for b in range(0, len(jobs), a.bs):
            chunk = jobs[b:b + a.bs]
            try:
                sp = predict([j[2] for j in chunk], [j[0][2] for j in chunk], [j[0][0] for j in chunk],
                             [j[0][1] for j in chunk], [prec_of[j[0]] for j in chunk])
                for j, s_ in zip(chunk, sp):
                    pred[(j[0], j[1])] = s_
            except Exception as e:  # one bad molecule must not kill the batch: retry one by one
                for j in chunk:
                    try:
                        pred[(j[0], j[1])] = predict([j[2]], [j[0][2]], [j[0][0]], [j[0][1]], [prec_of[j[0]]])[0]
                    except Exception as e2:
                        meta["n_fail"] += 1
                        if meta["n_fail"] <= 5:
                            print("predict failed", mid, j[1][:60], repr(e2)[:200], flush=True)
                            if meta["n_fail"] == 1:
                                import traceback
                                traceback.print_exc()
        meta["t_pred"] += time.time() - t1
        meta["n_pred"] += len(jobs)
        meta["n_cond"] += len(conds)
        meta["n_spec"] += len(specs)
        for (s0, _) in smis:
            row = []
            for si, ad, ins, ce, known, _, qm, qi in specs:
                p = pred.get(((ad, ins, ce), s0))
                if p is None:
                    continue
                e_, c_ = sims(qm, qi, p[0], p[1])
                row.append([round(e_, 5), round(c_, 5), known, si])
            out[s0] = row
        meta["n_cand"] += len(smis)
    res[mid] = out
    meta["n_mols"] += 1
    if time.time() - last > 60 or k + 1 == len(mids):
        json.dump(res, open(a.out, "w"))
        json.dump(meta, open(a.out + ".meta.json", "w"), indent=1)
        last = time.time()
        rate = meta["n_pred"] / max(meta["t_pred"], 1e-9)
        print(f"  {k + 1}/{len(mids)} mols, {meta['n_pred']} predictions, {rate:.2f} pred/s, "
              f"{time.time() - T0:.0f}s", flush=True)
json.dump(res, open(a.out, "w"))
json.dump(meta, open(a.out + ".meta.json", "w"), indent=1)
print("DONE", json.dumps(meta), flush=True)
