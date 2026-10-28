"""v47 = v43 on a T4 (nvcc fw_score, gpu=1; CPU fallback). (v43 doc:) v43 from v40 (LB 0.345): own C++ GLACIER + ICEBERG (cpp/glacier fw_score, dataset yomeiotani/casmi26-fwmodels:
the host-approved public ms-pred weights + our own inference code) re-score the same-formula groups of the pool list.

Re-ranking = the public gl_fuse.rerank_multi rule re-implemented here: inside the top 25, every formula group with >= 2
members of which >= 2 are scored by a model is re-sorted by z(base) + lam_gl z(gl) + lam_ice z(ice) (z within the group,
sample std, missing -> 0, ties keep order) and put back into the same slots. Base score = reciprocal rank (fw_sweep.py:
rank-linear base lost 0.03 on e2e, reciprocal rank +0.009 hcqry / 0 hxqry at ice 0.5 / gl 1.0).
No Python RDKit and no GPU: fw_score is compiled here with g++ against rdkit-min, CPU threads.
    python make_v43_nb.py [tag] [lam_ice] [lam_gl] [ice_budget_s]
"""
import ast, json, os, sys
NBD = r"C:\prog\claude4\arc\arc3\casmi\notebook"
SP = r"C:\Users\yomei2\AppData\Local\Temp\claude\C--prog-claude4\7b6059ac-6d96-452f-ab4e-313fe77f5b89\scratchpad"
BUILD_SRC = open(r"C:\prog\claude4\arc\arc3\casmi\cpp\glacier\fw_build.py", encoding="utf-8").read()
BUILD_SRC = BUILD_SRC[BUILD_SRC.index("def build_fw_score"):BUILD_SRC.index('if __name__')]
TAG = sys.argv[1] if len(sys.argv) > 1 else "v47"
LAM_ICE = sys.argv[2] if len(sys.argv) > 2 else "0.5"
LAM_GL = sys.argv[3] if len(sys.argv) > 3 else "1.0"
ICE_BUDGET = sys.argv[4] if len(sys.argv) > 4 else "10800"
base = json.load(open(os.path.join(NBD, "casmi_match_v40.ipynb"), encoding="utf-8"))
meta = json.load(open(os.path.join(NBD, "kernel-metadata-v40.json"), encoding="utf-8"))

ANCHOR = 'pool_c = read_csv_dict(WORK / "pool_candidates.csv") if RANKER_OK else {}\n'
BLOCK = ANCHOR + '''
# ---- v43: own C++ GLACIER + ICEBERG re-scoring of same-formula groups in the pool list ---------------------------
# fw_score (own C++ inference of the host-approved public ms-pred GLACIER / ICEBERG weights, dataset casmi26-fwmodels)
# Any failure -> the v40 order (printed as FW DISABLED).
import glob as _glob, math as _math, os as _os, subprocess as _sp
from concurrent.futures import ThreadPoolExecutor
LAM_ICE, LAM_GL, FW_ICE_BUDGET = __LAM_ICE__, __LAM_GL__, __ICE_BUDGET__
__BUILD_SRC__

def _z(vals):
    ok = [v is not None and _math.isfinite(v) for v in vals]
    xs = [v for v, o in zip(vals, ok) if o]
    if len(xs) < 2:
        return [0.0] * len(vals)
    mu = sum(xs) / len(xs)
    sd = _math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))
    if not sd > 0:
        return [0.0] * len(vals)
    return [((v - mu) / sd) if o else 0.0 for v, o in zip(vals, ok)]

def _rerank(smis, forms, cols, lams, top_n=25):
    order = list(range(len(smis)))
    groups = {}
    for i in range(min(top_n, len(smis))):
        if forms[i]:
            groups.setdefault(forms[i], []).append(i)
    for f, idx in groups.items():
        if len(idx) < 2:
            continue
        act = [(c, l) for c, l in zip(cols, lams) if sum(c[i] is not None for i in idx) >= 2]
        if not act:
            continue
        fused = _z([1.0 / (i + 1) for i in idx])
        for c, l in act:
            zi = _z([c[i] for i in idx])
            fused = [a + l * b for a, b in zip(fused, zi)]
        if len(act) >= 2:
            fused = [round(x / 1e-9) for x in fused]
        new = [idx[k] for k in sorted(range(len(idx)), key=lambda k: (-fused[k], k))]
        for slot, src in zip(idx, new):
            order[slot] = src
    return order

t_fw = time.time()
try:
    if not RANKER_OK or not pool_c:
        raise RuntimeError("no pool list")
    hits = sorted(_glob.glob(_os.path.join(str(_PB_INPUT), "**", "fw_score.cpp"), recursive=True), key=len)
    if not hits:
        raise RuntimeError("casmi26-fwmodels not mounted")
    FW_SRC = Path(hits[0]).parent
    FW_DS = FW_SRC.parent
    test_df = pd.read_parquet(DATA / "test.parquet")
    def _f(v):
        return "" if v is None or (isinstance(v, float) and _math.isnan(v)) else str(v)
    with open(WORK / "fw_in.txt", "w", encoding="utf-8") as f:
        by = {}
        for r in test_df.itertuples(index=False):
            by.setdefault(str(r.molecule_id), []).append(r)
        for mid, v in pool_c.items():
            smis = [s for s in v.split(";") if s][:25]
            if len(smis) < 2:
                continue
            f.write(f"M {mid}\\n")
            for s in smis:
                f.write(f"C {s}\\n")
            for r in by.get(mid, []):
                ce = getattr(r, "collision_energy_ev", None)
                ce = ",".join(repr(float(x)) for x in ce) if ce is not None else ""
                pk = ",".join(f"{float(a)!r}:{float(b)!r}" for a, b in zip(r.ms2_mzs, r.ms2_normalized_intensities))
                f.write("S " + "\\t".join([_f(r.adduct), _f(r.ionization_mode), _f(r.instrument_type),
                                          _f(getattr(r, "collision_energy_orig_units", None)),
                                          _f(getattr(r, "collision_energy_orig", None)), ce,
                                          repr(float(r.precursor_mz)), pk]) + "\\n")
    import shutil as _sh
    _nvcc = _sh.which("nvcc") or ("/usr/local/cuda/bin/nvcc" if _os.path.exists("/usr/local/cuda/bin/nvcc") else None)
    _smi = _sh.which("nvidia-smi")
    _gpu = bool(_nvcc) and (_os.path.exists("/dev/nvidia0") or (
        _smi is not None and _sp.run([_smi, "-L"], capture_output=True, text=True).returncode == 0))
    print("GPU check: nvcc %s, /dev/nvidia0 %s -> gpu %s" % (_nvcc, _os.path.exists("/dev/nvidia0"), _gpu), flush=True)
    rc = build_fw_score(FW_SRC, RDKIT_MIN, WORK / "fw_obj", WORK / "fw_score", jobs=4, gpu="sm_75" if _gpu else None)
    if rc != 0 and _gpu:
        print("GPU fw_score build failed - CPU build", flush=True)
        _gpu = False
        rc = build_fw_score(FW_SRC, RDKIT_MIN, WORK / "fw_obj_cpu", WORK / "fw_score", jobs=4)
    if rc != 0:
        raise RuntimeError("fw_score build failed")
    print("fw_score built in %.0fs" % (time.time() - t_fw), flush=True)
    cmd = [str(WORK / "fw_score"), str(WORK / "fw_in.txt"), str(WORK / "fw_out.tsv"),
           "gl=" + str(FW_DS / "glacier_w.bin"), "ice=%s,%s" % (FW_DS / "ice_gen_w.bin", FW_DS / "ice_inten_w.bin"),
           "threads=4", "budget=%g" % FW_ICE_BUDGET, "groups=25"] + (["gpu=1"] if _gpu else [])
    r = _sp.run(cmd, capture_output=True, text=True, timeout=FW_ICE_BUDGET * 2 + 3600)
    print(r.stderr[-1500:], flush=True)
    if r.returncode != 0:
        raise RuntimeError("fw_score rc %d" % r.returncode)
    sc = {}
    for l in open(WORK / "fw_out.tsv", encoding="utf-8"):
        p = l.rstrip("\\n").split("\\t")
        if len(p) < 5:
            continue
        g = None if p[2] == "nan" else float(p[2])
        i = None if p[3] == "nan" else float(p[3])
        sc[(p[0], p[1])] = (g, i, p[4])
    moved = top1 = 0
    for mid, v in list(pool_c.items()):
        smis = [s for s in v.split(";") if s]
        if len(smis) < 2:
            continue
        forms = [sc.get((mid, s), (None, None, ""))[2] if i < 25 else "" for i, s in enumerate(smis)]
        gl = [sc.get((mid, s), (None, None, ""))[0] for s in smis]
        ic = [sc.get((mid, s), (None, None, ""))[1] for s in smis]
        o = _rerank(smis, forms, [ic, gl], [LAM_ICE, LAM_GL])
        if o != list(range(len(smis))):
            moved += 1
            top1 += o[0] != 0
            pool_c[mid] = ";".join(smis[i] for i in o)
    print("FW stage done in %.0fs: lists re-ordered %d, top-1 changed %d (lam ice %g, gl %g, ice budget %gs)"
          % (time.time() - t_fw, moved, top1, LAM_ICE, LAM_GL, FW_ICE_BUDGET), flush=True)
except Exception as e:
    print("FW DISABLED:", repr(e), "after %.0fs - keeping the v40 order" % (time.time() - t_fw), flush=True)
# ---- end v43 block -----------------------------------------------------------------------------------------------
'''
BLOCK = (BLOCK.replace("__LAM_ICE__", LAM_ICE).replace("__LAM_GL__", LAM_GL).replace("__ICE_BUDGET__", ICE_BUDGET)
         .replace("__BUILD_SRC__", BUILD_SRC))
nb = json.loads(json.dumps(base))
done = 0
for c in nb["cells"]:
    s = "".join(c["source"])
    if ANCHOR in s:
        s = s.replace(ANCHOR, BLOCK)
        done += 1
    c["source"] = s.splitlines(keepends=True)
assert done == 1, done
for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("\n".join(l for l in "".join(c["source"]).splitlines() if not l.startswith(("!", "%"))))
name = f"casmi_match_{TAG}.ipynb"
m = dict(meta, id=f"yomeiotani/casmi26-cpp-library-search-{TAG}", title=f"CASMI26 cpp library search {TAG}",
         code_file=name, dataset_sources=meta["dataset_sources"] + ["yomeiotani/casmi26-fwmodels"],
         enable_gpu=True, machine_shape="NvidiaTeslaT4")
for d in (NBD, os.path.join(SP, "kp_" + TAG)):
    os.makedirs(d, exist_ok=True)
    json.dump(nb, open(os.path.join(d, name), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
json.dump(m, open(os.path.join(NBD, f"kernel-metadata-{TAG}.json"), "w", encoding="utf-8"), indent=2)
json.dump(m, open(os.path.join(SP, "kp_" + TAG, "kernel-metadata.json"), "w", encoding="utf-8"), indent=2)
print(TAG, "lam ice", LAM_ICE, "gl", LAM_GL, "ice budget", ICE_BUDGET, m["id"])
