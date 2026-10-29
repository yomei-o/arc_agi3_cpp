"""fw_sweep3.py (LB calibration of the rr form; copy of fw_sweep.py with another grid): offline lambda sweep of the forward-model (ICEBERG / GLACIER) same-formula re-ranking on the e2e set,
from the scores the v41 e2e run already computed (no new inference).

    python fw_sweep.py <work_v41 dir> <truth.tsv> <e2e_score.py dir> <ice pkg dir>
For each (lam_ice, lam_gl, score form, gate): re-order the v40 pool lists like v41 (fuse.rerank_multi over the top 25),
merge with direct_submission like the notebook, and score with the official identity (e2e_score.py).
"""
import csv, importlib.util, json, os, sys

W, TRUTH, E2E, ICEPKG = sys.argv[1:5]
sys.path.insert(0, ICEPKG); sys.path.insert(0, os.path.join(os.path.dirname(ICEPKG), "casmi26-glacier"))
import gl_fuse  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem.rdMolDescriptors import CalcMolFormula  # noqa: E402

RDLogger.DisableLog("rdApp.*")
spec = importlib.util.spec_from_file_location("e2e_score", os.path.join(E2E, "e2e_score.py"))


def rd(p):
    return {r[0]: r[1] for r in csv.reader(open(p, encoding="utf-8")) if r and r[0] != "molecule_id"}


pool = rd(os.path.join(os.path.dirname(W.rstrip("\\/")), "work_v40", "pool_candidates.csv"))
direct = rd(os.path.join(W, "direct_submission.csv"))
ice = json.load(open(os.path.join(W, "ice_work", "ice_out.json")))
gl = json.load(open(os.path.join(W, "gl_work", "gl_out.json")))
truth = {}
for l in open(TRUTH, encoding="utf-8"):
    t = l.rstrip("\r\n").split("\t")
    if len(t) >= 4:
        truth[t[0]] = t
form = {}


def fm(s):
    if s not in form:
        m = Chem.MolFromSmiles(s)
        form[s] = CalcMolFormula(m) if m is not None else "?" + s
    return form[s]


es = importlib.util.module_from_spec(spec)
_argv = sys.argv
sys.argv = [sys.argv[0]]
spec.loader.exec_module(es)
sys.argv = _argv
allsmi = set(t[3] for t in truth.values())
for v in list(pool.values()) + list(direct.values()):
    allsmi.update(s for s in v.split(";") if s)
K = es.keys_for(sorted(allsmi))
tkey = {m: (K[t[3]][0] or t[2]) for m, t in truth.items()}
sets = sorted({t[1] for t in truth.values()})


def mrr(lists):
    out = {s: [] for s in sets}
    for m, t in truth.items():
        L = lists.get(m, [])[:25]
        rr = 0.0
        for i, s in enumerate(L):
            k = K[s][0] if s in K else None
            if k is not None and k == tkey[m]:
                rr = 1.0 / (i + 1)
                break
        out[t[1]].append(rr)
    return {s: sum(v) / max(1, len(v)) for s, v in out.items()}


def run(li, lg, form_kind="rank", top_n=25):
    lists = {}
    for mid, d in direct.items():
        p = [s for s in pool.get(mid, "").split(";") if s]
        if p and (li or lg):
            fs = [fm(s) for s in p]
            if form_kind == "rank":
                sc = [float(len(p) - i) for i in range(len(p))]
            else:  # reciprocal rank: top positions far apart, tail compressed
                sc = [1.0 / (i + 1) for i in range(len(p))]
            o = gl_fuse.rerank_multi(p, p, sc, fs, [ice.get(mid, {}), gl.get(mid, {})], [li, lg], top_n=top_n)
            p = [p[i] for i in o]
        L, seen = [], set()
        for s in p + [s for s in d.split(";") if s]:
            if s not in seen:
                seen.add(s); L.append(s)
            if len(L) == 25:
                break
        lists[mid] = L
    return mrr(lists)


base = run(0, 0)
print("v40 (no forward model):", {k: round(v, 4) for k, v in base.items()})
# LB calibration grid (rr form = v43/v47): LB 0.5/1.0 0.356, 1.0/1.0 0.361, 2.0/1.0 0.343, 3.0/1.0 0.338
for li, lg in ((0.5, 1.0), (1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (1.0, 0.5), (1.0, 1.5), (1.0, 0), (0.75, 0.75),
               (1.5, 1.5), (0.75, 1.0)):
    r = run(li, lg, "rr")
    print(f"rr   ice {li:<4} gl {lg:<4}: " + "  ".join(f"{k} {v:.4f} ({v - base[k]:+.4f})" for k, v in r.items()), flush=True)
