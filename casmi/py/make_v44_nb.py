"""v44 from v40 (LB 0.345): own C++ GLACIER added INSIDE the fused score (iceberg_STATUS.md "add_w0.5_mh": hxqry +0.015
[+0.006, +0.024], sibling no loss), not the same-formula re-rank (our +-10 ppm lists are one formula).

The ranker dumps its fused scores of the top 60 (ppdet=pp_dump.txt: name rank score isPool pop candname, before the
pool-popularity step). For the pool candidates there (names -> SMILES via pool730_mols.txt), fw_score predicts GLACIER
spectra ([M+H]+ spectra only, the runner's gl_ex_mean); then  s' = z(score + 0.15 pop) + w z(gl)  over the dumped
candidates (missing gl -> 0) and the pool candidates of the final list are re-ordered among their own slots by s'
(generated and PubChem entries keep their slots).
    python make_v44_nb.py [tag] [w]
"""
import ast, json, os, sys
NBD = r"C:\prog\claude4\arc\arc3\casmi\notebook"
SP = r"C:\Users\yomei2\AppData\Local\Temp\claude\C--prog-claude4\7b6059ac-6d96-452f-ab4e-313fe77f5b89\scratchpad"
BUILD_SRC = open(r"C:\prog\claude4\arc\arc3\casmi\cpp\glacier\fw_build.py", encoding="utf-8").read()
BUILD_SRC = BUILD_SRC[BUILD_SRC.index("def build_fw_score"):BUILD_SRC.index('if __name__')]
TAG = sys.argv[1] if len(sys.argv) > 1 else "v44"
W_GL = sys.argv[2] if len(sys.argv) > 2 else "0.5"
base = json.load(open(os.path.join(NBD, "casmi_match_v40.ipynb"), encoding="utf-8"))
meta = json.load(open(os.path.join(NBD, "kernel-metadata-v40.json"), encoding="utf-8"))

OPT_ANCHOR = 'print("v28 options:", V27_OPT, flush=True)\n'
OPT_NEW = 'V27_OPT += ",ppdet=pp_dump.txt"  # v44: fused scores of the top 60 for the GLACIER term\n' + OPT_ANCHOR
ANCHOR = 'pool_c = read_csv_dict(WORK / "pool_candidates.csv") if RANKER_OK else {}\n'
BLOCK = ANCHOR + '''
# ---- v44: own C++ GLACIER as a fused-score term (s' = z(score + 0.15 pop) + W_GL z(gl)) ---------------------------
import glob as _glob, math as _math, os as _os, subprocess as _sp
from concurrent.futures import ThreadPoolExecutor
W_GL = __W_GL__
__BUILD_SRC__

def _z(vals):
    ok = [v is not None and _math.isfinite(v) for v in vals]
    xs = [v for v, o in zip(vals, ok) if o]
    if len(xs) < 2:
        return [0.0] * len(vals)
    mu = sum(xs) / len(xs)
    sd = _math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))
    if not sd > 0:
        return [0.0] * len(vals)
    return [((v - mu) / sd) if o else 0.0 for v, o in zip(vals, ok)]

t_fw = time.time()
try:
    if not RANKER_OK or not pool_c:
        raise RuntimeError("no pool list")
    pp = {}
    for l in open(WORK / "pp_dump.txt", encoding="utf-8"):
        p = l.rstrip("\\n").split("\\t")
        if len(p) >= 6:
            pp.setdefault(p[0], []).append((int(p[1]), float(p[2]), int(p[3]), float(p[4]), p[5]))
    need = {e[4] for v in pp.values() for e in v if e[2] == 1}
    name2smi = {}
    with open(POOL730_DIR / "pool730_mols.txt", encoding="utf-8") as f:
        for l in f:
            t = l.find("\\t")
            if t > 0 and l[:t] in need:
                name2smi[l[:t]] = l[t + 1:].rstrip("\\r\\n")
    print("pp dump: %d queries, %d pool names, %d mapped to SMILES" % (len(pp), len(need), len(name2smi)), flush=True)
    hits = sorted(_glob.glob(_os.path.join(str(_PB_INPUT), "**", "fw_score.cpp"), recursive=True), key=len)
    if not hits:
        raise RuntimeError("casmi26-fwmodels not mounted")
    FW_SRC = Path(hits[0]).parent
    FW_DS = FW_SRC.parent
    test_df = pd.read_parquet(DATA / "test.parquet")
    by = {}
    for r in test_df.itertuples(index=False):
        by.setdefault(str(r.molecule_id), []).append(r)
    def _f(v):
        return "" if v is None or (isinstance(v, float) and _math.isnan(v)) else str(v)
    with open(WORK / "fw_in.txt", "w", encoding="utf-8") as f:
        for mid, v in pp.items():
            smis = [name2smi[e[4]] for e in v if e[2] == 1 and e[4] in name2smi][:40]
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
    rc = build_fw_score(FW_SRC, RDKIT_MIN, WORK / "fw_obj", WORK / "fw_score", jobs=4)
    if rc != 0:
        raise RuntimeError("fw_score build failed")
    print("fw_score built in %.0fs" % (time.time() - t_fw), flush=True)
    r = _sp.run([str(WORK / "fw_score"), str(WORK / "fw_in.txt"), str(WORK / "fw_out.tsv"),
                 "gl=" + str(FW_DS / "glacier_w.bin"), "threads=4", "budget=5400"],
                capture_output=True, text=True, timeout=9000)
    print(r.stderr[-1500:], flush=True)
    if r.returncode != 0:
        raise RuntimeError("fw_score rc %d" % r.returncode)
    gl = {}
    for l in open(WORK / "fw_out.tsv", encoding="utf-8"):
        p = l.rstrip("\\n").split("\\t")
        if len(p) >= 3 and p[2] != "nan":
            gl[(p[0], p[1])] = float(p[2])
    moved = top1 = nscored = 0
    for mid, v in pp.items():
        if mid not in pool_c:
            continue
        smi = [name2smi.get(e[4]) if e[2] == 1 else None for e in v]
        g = [gl.get((mid, s)) if s else None for s in smi]
        if sum(x is not None for x in g) < 2:
            continue
        nscored += 1
        zs = _z([e[1] + 0.15 * e[3] for e in v])
        zg = _z(g)
        sp = {s: a + W_GL * b for s, a, b in zip(smi, zs, zg) if s}
        L = [s for s in pool_c[mid].split(";") if s]
        slots = [i for i, s in enumerate(L) if s in sp]
        if len(slots) < 2:
            continue
        items = sorted((L[i] for i in slots), key=lambda s: -sp[s])
        new = list(L)
        for i, s in zip(slots, items):
            new[i] = s
        if new != L:
            moved += 1
            top1 += new[0] != L[0]
            pool_c[mid] = ";".join(new)
    print("FW stage done in %.0fs: GLACIER term w %g, queries scored %d, lists re-ordered %d, top-1 changed %d"
          % (time.time() - t_fw, W_GL, nscored, moved, top1), flush=True)
except Exception as e:
    print("FW DISABLED:", repr(e), "after %.0fs - keeping the v40 order" % (time.time() - t_fw), flush=True)
# ---- end v44 block -----------------------------------------------------------------------------------------------
'''
BLOCK = BLOCK.replace("__W_GL__", W_GL).replace("__BUILD_SRC__", BUILD_SRC)
nb = json.loads(json.dumps(base))
done = [0, 0]
for c in nb["cells"]:
    s = "".join(c["source"])
    if OPT_ANCHOR in s:
        s = s.replace(OPT_ANCHOR, OPT_NEW); done[0] += 1
    if ANCHOR in s:
        s = s.replace(ANCHOR, BLOCK); done[1] += 1
    c["source"] = s.splitlines(keepends=True)
assert done == [1, 1], done
for c in nb["cells"]:
    if c["cell_type"] == "code":
        ast.parse("\n".join(l for l in "".join(c["source"]).splitlines() if not l.startswith(("!", "%"))))
name = f"casmi_match_{TAG}.ipynb"
m = dict(meta, id=f"yomeiotani/casmi26-cpp-library-search-{TAG}", title=f"CASMI26 cpp library search {TAG}",
         code_file=name, dataset_sources=meta["dataset_sources"] + ["yomeiotani/casmi26-fwmodels"])
for d in (NBD, os.path.join(SP, "kp_" + TAG)):
    os.makedirs(d, exist_ok=True)
    json.dump(nb, open(os.path.join(d, name), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
json.dump(m, open(os.path.join(NBD, f"kernel-metadata-{TAG}.json"), "w", encoding="utf-8"), indent=2)
json.dump(m, open(os.path.join(SP, "kp_" + TAG, "kernel-metadata.json"), "w", encoding="utf-8"), indent=2)
print(TAG, "w_gl", W_GL, m["id"])
