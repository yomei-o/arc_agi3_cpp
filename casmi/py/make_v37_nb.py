"""make_v37_nb: notebook/casmi_match_v37nofp.ipynb from casmi_match_v36nofp.ipynb (v37_STATUS.md).

v37nofp = v36nofp (own FPNet + 1.0 z, CNN pool term cnnpool 2 / cnninst 0, attention, Boltzmann merge, time guards,
prebuilt hook) + FT pool term (phys_v37pc, pretrain_STATUS.md 8): our own masked-peak-pretrained transformer
fine-tuned on fp3c16k (ft_all_ep5.bin, TFM1, 221 MB), CPU inference inside the ranker, + 2 z(FT Tanimoto) in the pool
fusion, gated by cnninst=0 like the CNN (only queries with a timsTOF spectrum).
  - ,ftmodel=<POOL730_DIR>/ft_all_ep5.bin,ftpool=2 only when phys_v37pc built and the file exists; else "FT MISSING"
    (file) / "FT DISABLED" (source) and the run is v36nofp.
  - phys_fp compile: phys_v37pc first; if that compile fails, phys_v36pc is compiled instead (v36 run) rather than
    dropping to the v24 fallback.
  - time: ftbudget=<s> is passed to the ranker = min(40 min, ranker time left - 90 min); if the FT inference passes it
    the ranker drops the term ("FT TIMEOUT") and finishes as v36. Under 5 min of budget the FT options are not passed
    ("FT SKIPPED"). The ranker's own deadline (t0 + 480 min) is unchanged.
  - ranker retries on a fast failure: without FT, then without FT and CNN, then the plain library list.
Needs in the datasets: casmi26-rdkit-min rdkit-min/phys_v37pc.cpp (no new headers: rev/tfm_model.hpp, rev/fp3_v32.h are
already there); the pool730/ folder: ft_all_ep5.bin.
usage: python make_v37_nb.py   (paths are relative to this file)
"""
import json
from pathlib import Path

nbd = Path(__file__).resolve().parent.parent / "notebook"
nb = json.loads((nbd / "casmi_match_v36nofp.ipynb").read_text(encoding="utf-8"))


def edit(ci, old, new, check=True):
    src = "".join(nb["cells"][ci]["source"])
    assert src.count(old) == 1, (ci, old)
    src = src.replace(old, new)
    if check:
        compile(src, "cell%d" % ci, "exec")
    nb["cells"][ci]["source"] = src.splitlines(keepends=True)


# ---- cell 7: phys_v37pc as the phys_fp source, fall back to phys_v36pc if its compile fails
edit(7, '''    print("phys_v36pc.cpp present:", V36, "" if V36 else "- CNN DISABLED, running as v35", flush=True)
''', '''    print("phys_v36pc.cpp present:", V36, "" if V36 else "- CNN DISABLED, running as v35", flush=True)
    V37 = V36 and (RDKIT_MIN / "phys_v37pc.cpp").exists()  # v37: + FT pool term (v37_STATUS.md)
    print("phys_v37pc.cpp present:", V37, "" if V37 else "- FT DISABLED, running as v36", flush=True)
''')
edit(7, '''    srcs_fp = ["phys_v36pc.cpp" if V36 else "phys_v35pc.cpp" if V35 else "phys_v33pc.cpp" if V33 else "phys_v30pc.cpp"] + srcs[1:]
    r = subprocess.run(["g++", "-O3", "-std=c++20", "-pthread", "-w",
                        "-I", str(RDKIT_MIN), "-I", str(RDKIT_MIN / "boost-min"),
                        "-I", str(AUTOGRAD_DIR.parent),
                        "-o", str(WORK / "phys_fp")] + srcs_fp,
                       capture_output=True, text=True, cwd=str(RDKIT_MIN), timeout=150 * 60)
    print("phys_fp compile rc:", r.returncode, "in %.0fs" % (time.time() - t4), flush=True)
    if r.returncode != 0:
        print(r.stderr[-4000:]); raise RuntimeError("phys_fp compile failed")
''', '''    _main_old = "phys_v36pc.cpp" if V36 else "phys_v35pc.cpp" if V35 else "phys_v33pc.cpp" if V33 else "phys_v30pc.cpp"
    for _main in (["phys_v37pc.cpp"] if V37 else []) + [_main_old]:
        srcs_fp = [_main] + srcs[1:]
        r = subprocess.run(["g++", "-O3", "-std=c++20", "-pthread", "-w",
                            "-I", str(RDKIT_MIN), "-I", str(RDKIT_MIN / "boost-min"),
                            "-I", str(AUTOGRAD_DIR.parent),
                            "-o", str(WORK / "phys_fp")] + srcs_fp,
                           capture_output=True, text=True, cwd=str(RDKIT_MIN), timeout=150 * 60)
        print("phys_fp compile (%s) rc:" % _main, r.returncode, "in %.0fs" % (time.time() - t4), flush=True)
        if r.returncode == 0:
            break
        print(r.stderr[-4000:])
        if _main == "phys_v37pc.cpp":
            V37 = False
            print("FT DISABLED: phys_v37pc compile failed - compiling %s" % _main_old, flush=True)
    if r.returncode != 0:
        raise RuntimeError("phys_fp compile failed")
''')

# ---- cell 8: FT options, FT time budget, retries without FT / CNN, FT log lines
edit(8, '''                print("CNN DISABLED (no phys_v36pc or no cnn_prod_c.bin)", flush=True)
''', '''                print("CNN DISABLED (no phys_v36pc or no cnn_prod_c.bin)", flush=True)
            # v37: + 2 z(FT Tanimoto): own masked-peak-pretrained transformer fine-tuned on fp3c16k (pretrain_STATUS.md 8;
            # timsTOF holdouts +0.004..+0.011 over cnn 2), same cnninst=0 gate; ftbudget is added at the ranker call
            FTF = POOL730_DIR / "ft_all_ep5.bin"
            print("ft model:", FTF, FTF.exists(), flush=True)
            if not globals().get("V37"):
                print("FT DISABLED (no phys_v37pc): running as v36", flush=True)
            elif not FTF.exists():
                print("FT MISSING (no ft_all_ep5.bin): running as v36", flush=True)
            elif ",cnnmodel=" in V27_OPT:
                V27_OPT += ",ftmodel=" + str(FTF) + ",ftpool=2"
            else:
                V27_OPT += ",ftmodel=" + str(FTF) + ",ftpool=2,cnninst=0"
''')
edit(8, '''_opts = [V27_OPT]
if ",cnnmodel=" in V27_OPT:   # retry once without the CNN term if the first run fails (not on timeout)
  _opts.append(V27_OPT.split(",cnnmodel=")[0] + "".join("," + o for o in V27_OPT.split(",cnnmodel=")[1].split(",")[1:]
                                                      if not o.startswith(("cnnpool=", "cnninst="))))
for _ai, _opt in enumerate(_opts):
 RANKER_TIMEOUT = t0 + RANKER_END_MIN * 60 - time.time()
 print("ranker attempt %d, timeout %.0f s, options %s" % (_ai + 1, RANKER_TIMEOUT, _opt[-160:]), flush=True)
 try:
  if RANKER_TIMEOUT < 60:
    raise subprocess.TimeoutExpired("phys_rank", 0)
''', '''

def _drop_opts(opt, keys):
  return "".join("," + o for o in opt.split(",") if o and not o.startswith(keys))


# retries on a fast failure (not on timeout): without the FT term, then without FT and CNN
_opts = [V27_OPT]
if ",ftmodel=" in V27_OPT:
  _opts.append(_drop_opts(V27_OPT, ("ftmodel=", "ftpool=")))
if ",cnnmodel=" in V27_OPT:
  _opts.append(_drop_opts(V27_OPT, ("ftmodel=", "ftpool=", "cnnmodel=", "cnnpool=", "cnninst=")))
FT_BUDGET_MAX = 40 * 60    # FT inference: ~4 min on the build machine for ~1,200 spectra; 1.5-2x on Kaggle
FT_RANKER_REST = 90 * 60   # time kept for the rest of the ranker (~20 min on Kaggle for v33..v36)
for _ai, _opt in enumerate(_opts):
 RANKER_TIMEOUT = t0 + RANKER_END_MIN * 60 - time.time()
 if ",ftmodel=" in _opt:
  _ftb = int(min(FT_BUDGET_MAX, RANKER_TIMEOUT - FT_RANKER_REST))
  if _ftb >= 300:
   _opt += ",ftbudget=%d" % _ftb
  else:
   print("FT SKIPPED: only %.0f s left for the ranker - running as v36" % RANKER_TIMEOUT, flush=True)
   _opt = _drop_opts(_opt, ("ftmodel=", "ftpool="))
 print("ranker attempt %d, timeout %.0f s, options %s" % (_ai + 1, RANKER_TIMEOUT, _opt[-200:]), flush=True)
 try:
  if RANKER_TIMEOUT < 60:
    raise subprocess.TimeoutExpired("phys_rank", 0)
''')
edit(8, '''    if any(k in line for k in ("cnn", "attn", "alloc=boltz", "fpnet", "reverse model", "fp3")):''',
     '''    if any(k in line for k in ("cnn", "attn", "alloc=boltz", "fpnet", "reverse model", "fp3", "ft model", "ft term",
                               "FT TIMEOUT")):''')
edit(8, '''        "retrying without the CNN term" if _ai + 1 < len(_opts) else "submission = plain library list", flush=True)''',
     '''        "retrying without the %s term" % ("FT" if ",ftmodel=" in _opt else "CNN") if _ai + 1 < len(_opts)
        else "submission = plain library list", flush=True)''')

out = nbd / "casmi_match_v37nofp.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
meta = json.loads((nbd / "kernel-metadata-v36nofp.json").read_text(encoding="utf-8"))
meta.update(id="yomeiotani/casmi26-cpp-library-search-v37nofp", title="CASMI26 cpp library search v37nofp",
            code_file="casmi_match_v37nofp.ipynb")
(nbd / "kernel-metadata-v37nofp.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
for c in nb["cells"]:
    if c["cell_type"] == "code":
        compile("".join(c["source"]), "cell", "exec")
print("wrote", out, "and kernel-metadata-v37nofp.json")
