"""v41 from v40 (LB 0.345): ICEBERG + GLACIER forward models (host-approved public ms-pred checkpoints, MIT; Kaggle
packages ahmedberatozer/casmi26-iceberg, casmi26-glacier) re-score the same-formula groups of our pool list.

Public-lineage evidence (hn_STATUS.md, memory casmi-top-lineage-attribution): the forward models re-order isomers
within a formula group, z(ranker) + lam*z(forward), GLACIER on [M+H]+ spectra only, ICEBERG on [M+H]+/[M+Na]+.
Our pool list (pool_candidates.csv, 25 per molecule, no scores) is re-ordered with rank-based scores before the merge.
GPU kernel. Everything else = v40.    python make_v41_nb.py [tag] [lam_ice] [lam_gl]
"""
import ast, json, os, sys
NBD = r"C:\prog\claude4\arc\arc3\casmi\notebook"
SP = r"C:\Users\yomei2\AppData\Local\Temp\claude\C--prog-claude4\7b6059ac-6d96-452f-ab4e-313fe77f5b89\scratchpad"
KEYS_SRC = open(r"C:\prog\claude4\arc\arc3\casmi\py\ice_keys.py", encoding="utf-8").read()
TAG = sys.argv[1] if len(sys.argv) > 1 else "v41"
LAM_ICE = sys.argv[2] if len(sys.argv) > 2 else "0.5"
LAM_GL = sys.argv[3] if len(sys.argv) > 3 else "1.0"
base = json.load(open(os.path.join(NBD, "casmi_match_v40.ipynb"), encoding="utf-8"))
meta = json.load(open(os.path.join(NBD, "kernel-metadata-v40.json"), encoding="utf-8"))

ANCHOR = 'pool_c = read_csv_dict(WORK / "pool_candidates.csv") if RANKER_OK else {}\n'
BLOCK = ANCHOR + '''
# ---- v41: ICEBERG / GLACIER forward-model re-scoring of same-formula groups in the pool list -------------------
# Public ms-pred checkpoints (MassSpecGym-trained; approved by the host), run as the packages' own subprocess runners
# (their RDKit 2025.03 wheel site, GPU). Only members of same-formula groups with >= 2 members can move; everything
# else keeps its slot. Any failure -> the v40 order (printed as ICE DISABLED / ICE TIMEOUT).
import glob as _glob, json as _json
LAM_ICE, LAM_GL = __LAM_ICE__, __LAM_GL__
ICE_BUDGET = float(os.environ.get("ICE_BUDGET", "2400"))
GL_BUDGET = float(os.environ.get("GL_BUDGET", "2400"))
ICE_EXTRA = os.environ.get("ICE_EXTRA", "").split()
t_ice = time.time()
ice_stats = {}
try:
    if not RANKER_OK or not pool_c:
        raise RuntimeError("no pool list")
    def _find(name, env):
        p = os.environ.get(env)
        if p and os.path.exists(os.path.join(p, name)):
            return p
        hits = sorted(_glob.glob(os.path.join(str(_PB_INPUT), "**", name), recursive=True), key=len)
        if not hits:
            raise RuntimeError(name + " not found")
        return os.path.dirname(hits[0])
    ICE_PKG, GL_PKG = _find("ice_runner.py", "ICE_PKG"), _find("gl_runner.py", "GL_PKG")
    ICE_SITE = os.environ.get("ICE_SITE", "/tmp/ice_site")
    print("ice pkg:", ICE_PKG, " gl pkg:", GL_PKG, " site:", ICE_SITE, flush=True)
    sys.path.insert(0, ICE_PKG); sys.path.insert(0, GL_PKG)
    import fuse as ice_fuse, gl_fuse, ice_runner
    ice_runner.install_site(ICE_PKG, ICE_SITE, print)        # idempotent; RDKit 2025.03 + lightning wheels
    # formulas + InChIKey14 of every pool candidate, computed with the site's RDKit in a subprocess
    (WORK / "ice_keys.py").write_text(__KEYS_SRC__)
    all_smis = sorted({s for v in pool_c.values() for s in v.split(";") if s})
    env = dict(os.environ); env["PYTHONPATH"] = ICE_SITE + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, str(WORK / "ice_keys.py")], input=_json.dumps(all_smis), capture_output=True,
                       text=True, env=env, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError("ice_keys failed: " + r.stderr[-800:])
    fk = _json.loads(r.stdout)
    print("ice keys: %d candidate smiles, %d parsed in %.0fs" % (len(all_smis), sum(1 for v in fk.values() if v[0]),
                                                                time.time() - t_ice), flush=True)
    lists = {}
    for mid, v in pool_c.items():
        smis = [s for s in v.split(";") if s]
        lists[mid] = (smis, [fk.get(s, ["", ""])[0] or "?" + s for s in smis], [fk.get(s, ["", ""])[1] for s in smis])
    cands = {mid: ice_fuse.ice_candidates(smis, forms, top_n=25) for mid, (smis, forms, keys) in lists.items()}
    cands = {m: c for m, c in cands.items() if c}
    test_df = pd.read_parquet(DATA / "test.parquet")
    items = ice_fuse.build_ice_input(test_df, cands)
    items = [it for it in items if it["spectra"]]
    gl_items = [dict(it, spectra=[sp for sp in it["spectra"] if sp.get("adduct") == "[M+H]+"]) for it in items]
    gl_items = [it for it in gl_items if it["spectra"]]
    print("ice input: %d molecules with same-formula groups, %d covered (pos [M+H]+/[M+Na]+), %d with [M+H]+ for GL, "
          "%d candidate predictions" % (len(cands), len(items), len(gl_items), sum(len(it["cands"]) for it in items)),
          flush=True)
    ice = ice_fuse.run_ice(ICE_PKG, items, workdir=str(WORK / "ice_work"), device="cuda", budget_s=ICE_BUDGET,
                           site=ICE_SITE, extra_args=ICE_EXTRA) if items else {}
    gl = gl_fuse.run_gl(GL_PKG, gl_items, workdir=str(WORK / "gl_work"), device="cuda", budget_s=GL_BUDGET,
                        site=ICE_SITE, wheels=os.path.join(ICE_PKG, "wheels"), extra_args=ICE_EXTRA) if gl_items else {}
    n_ice = sum(1 for m in ice if any(v is not None for v in ice[m].values()))
    n_gl = sum(1 for m in gl if any(v is not None for v in gl[m].values()))
    moved = top1 = 0
    for mid, (smis, forms, keys) in lists.items():
        if not smis:
            continue
        # reciprocal-rank scores protect the top of the list (fw_sweep.py on e2e: rank-linear scores let the forward
        # models overturn top-1 in 200/800 molecules and lost 0.03; reciprocal rank +0.009 hcqry / 0 hxqry)
        scores = [1.0 / (i + 1) for i in range(len(smis))]
        order = gl_fuse.rerank_multi(smis, keys, scores, forms, [ice.get(mid, {}), gl.get(mid, {})],
                                     [LAM_ICE, LAM_GL], top_n=25)
        if order != list(range(len(smis))):
            moved += 1
            top1 += order[0] != 0
            pool_c[mid] = ";".join(smis[i] for i in order)
    ice_stats = dict(n_ice=n_ice, n_gl=n_gl, moved=moved, top1=top1)
    print("ICE/GL stage done in %.0fs: ice scores for %d molecules, gl for %d, lists re-ordered %d, top-1 changed %d "
          "(lam ice %g, gl %g)" % (time.time() - t_ice, n_ice, n_gl, moved, top1, LAM_ICE, LAM_GL), flush=True)
except Exception as e:
    print("ICE DISABLED:", repr(e), "after %.0fs - keeping the v40 order" % (time.time() - t_ice), flush=True)
# ---- end v41 block -----------------------------------------------------------------------------------------------
'''
BLOCK = BLOCK.replace("__LAM_ICE__", LAM_ICE).replace("__LAM_GL__", LAM_GL).replace("__KEYS_SRC__", repr(KEYS_SRC))

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
         code_file=name, enable_gpu=True, machine_shape="NvidiaTeslaT4",
         # the Python 3.12 image the public ICEBERG/GLACIER notebooks pin (their RDKit 2025.03 wheel is cp312)
         docker_image="gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
         dataset_sources=meta["dataset_sources"] + ["ahmedberatozer/casmi26-iceberg", "ahmedberatozer/casmi26-glacier"])
for d in (NBD, os.path.join(SP, "kp_" + TAG)):
    os.makedirs(d, exist_ok=True)
    json.dump(nb, open(os.path.join(d, name), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
json.dump(m, open(os.path.join(NBD, f"kernel-metadata-{TAG}.json"), "w", encoding="utf-8"), indent=2)
json.dump(m, open(os.path.join(SP, "kp_" + TAG, "kernel-metadata.json"), "w", encoding="utf-8"), indent=2)
print(TAG, "lam", LAM_ICE, LAM_GL, m["id"], "gpu", m["enable_gpu"])
