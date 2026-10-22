"""make_v36_nb: notebook/casmi_match_v36nofp.ipynb from casmi_match_v35nofp.ipynb (v36_STATUS.md).

v36nofp = v35nofp (conservative Boltzmann merge bpi 0.1, attention attpool 2, fp3 window model, prebuilt hook)
  + OWN FPNet (fpnet_own_f16.bin: FPNet architecture trained on train.parquet with our own C++/CUDA code,
    fpnet_own_STATUS.md) as + 1.0 z(fpnet). The public fpnet_full1 weights are never read: if the own file is
    missing the notebook prints "OWN FPNET MISSING" and runs without the term.
  + time guards: every FPNet-stage subprocess (cmake configure / build, fpnet_ggml, fpnet_pool) runs under one
    stage budget (FPNET_BUDGET, 75 min); on timeout the process (and its process group / new sessions on Linux)
    is killed, "FPNET TIMEOUT" is printed and the run continues without the term. The pool fingerprint file
    (933 MB, read by fpnet_pool with ~1e5 random seeks) is first copied sequentially to local /tmp (chunked,
    under the same deadline). The main ranker gets a timeout that ends RANKER_END_MIN after the notebook's t0;
    if it fails or times out the submission is the plain library list (match.cpp) - "RANKER FAILED".
  + CNN pool term (phys_v36pc, pool36_STATUS.md): ,cnnmodel=<POOL730_DIR>/cnn_prod_c.bin,cnnpool=2,cnninst=0
    (cnninst=0: only queries with a timsTOF spectrum get the term; the hidden test is all timsTOF).
Needs in the datasets: casmi26-rdkit-min rdkit-min/phys_v36pc.cpp, rdkit-min/rev/cnn_model.hpp + cnn_layers.hpp,
fpnet/fpnet_own_f16.bin (and NOT fpnet/fpnet_full1_f16.bin); casmi26-pool-expanded pool730/cnn_prod_c.bin.
usage: python make_v36_nb.py   (paths are relative to this file)
"""
import json
from pathlib import Path

nbd = Path(__file__).resolve().parent.parent / "notebook"
nb = json.loads((nbd / "casmi_match_v35nofp.ipynb").read_text(encoding="utf-8"))

CELL7 = r'''# v36nofp: OWN FPNet. fpnet_own_f16.bin = the FPNet architecture (same 10226 output bits, same file format as the
# public fpnet_full1) trained from train.parquet with our own C++/CUDA training code (fpnet_own_STATUS.md); the
# public weights are not used anywhere and there is NO fallback to them. Fused as + 1.0 z(fpnet) (as v22..v33).
# Holdout (wlib-trained own model, pool ranking): hcqry +0.013, hxqry +0.012, hasqry +0.000.
# Time guards: v33 was refused at 570 min > 540 because its FPNet stage alone took 30,622 s on Kaggle (37 min on
# other days). Every FPNet subprocess now runs under one stage budget; on timeout it is killed and the run goes on
# without the term ("FPNET TIMEOUT"). The 933 MB pool fingerprint file is copied sequentially to local disk first
# (fpnet_pool reads ~1e5 scattered rows of it; on a network-backed input mount that is the suspected slow part).
import shutil, signal
USE_FPNET = True
FPNET_BUDGET = 75 * 60      # seconds for the whole FPNet stage (build + logits + pool scores)
FP_ARGS = []
FPNET_OK = False


def _fp_sids():
    s = set()
    if hasattr(os, "getsid") and os.path.isdir("/proc"):
        for d in os.listdir("/proc"):
            if d.isdigit():
                try:
                    s.add(os.getsid(int(d)))
                except Exception:
                    pass
    return s


def _fp_kill_new_sessions(before):
    """Kill every process in a session that did not exist before the call (the timed-out child was started with
    start_new_session=True, so it and its children - make, cc1plus, ... - are in a new session)."""
    if not (hasattr(os, "getsid") and os.path.isdir("/proc")):
        return 0
    me = os.getsid(0)
    n = 0
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            sid = os.getsid(int(d))
            if sid not in before and sid != me:
                os.kill(int(d), signal.SIGKILL); n += 1
        except Exception:
            pass
    return n


def _fp_run(cmd, deadline, what, **kw):
    rem = deadline - time.time()
    if rem < 10:
        raise TimeoutError("%s: no stage budget left" % what)
    before = _fp_sids()
    t = time.time()
    try:
        r = subprocess.run(cmd, timeout=rem, start_new_session=True, **kw)
    except subprocess.TimeoutExpired:
        k = _fp_kill_new_sessions(before)
        raise TimeoutError("%s killed after %.0fs (%d leftover processes killed)" % (what, time.time() - t, k))
    print("[time] %s: %.0fs rc %s" % (what, time.time() - t, r.returncode), flush=True)
    return r


def _fp_copy(src, dst, deadline, what):
    t = time.time()
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while True:
            if time.time() > deadline:
                raise TimeoutError("%s: stage budget exceeded after %.0fs" % (what, time.time() - t))
            b = fi.read(64 << 20)
            if not b:
                break
            fo.write(b)
    print("[time] %s: %.0fs (%.0f MB)" % (what, time.time() - t, Path(dst).stat().st_size / 1e6), flush=True)


t3 = time.time()
FP_DEADLINE = t3 + FPNET_BUDGET
FP_LOCAL = None
try:
    if USE_FPNET:
        FPNET_SRC = RDKIT_MIN.parent / "fpnet"
        FP_W = FPNET_SRC / "fpnet_own_f16.bin"   # OWN weights only - never fpnet_full1_f16.bin
        print("own fpnet weights:", FP_W, FP_W.exists(), flush=True)
        if not FP_W.exists():
            print("OWN FPNET MISSING: running without the FPNet term", flush=True)
            raise RuntimeError("own fpnet weights missing")
        # avoid a full rglob over /kaggle/input (took 11 min once): look at most 4 levels deep
        FP_POOL = None
        for pat in ("*/pool730_fp10226.bin", "*/*/pool730_fp10226.bin", "*/*/*/pool730_fp10226.bin",
                    "*/*/*/*/pool730_fp10226.bin"):
            hit = sorted(Path("/kaggle/input").glob(pat))
            if hit:
                FP_POOL = hit[0]; break
        print("fpnet src:", FPNET_SRC, FPNET_SRC.exists(), " pool fp:", FP_POOL, flush=True)
        if FP_POOL is None:
            raise RuntimeError("fpnet pool fingerprints missing")
        FB = WORK / "fpnet_build"
        r = _fp_run(["cmake", "-S", str(FPNET_SRC), "-B", str(FB), "-DCMAKE_BUILD_TYPE=Release",
                     "-DGGML_NATIVE=OFF"], FP_DEADLINE, "fpnet cmake configure", capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-3000:]); raise RuntimeError("cmake configure failed")
        r = _fp_run(["cmake", "--build", str(FB), "-j", "4"], FP_DEADLINE, "fpnet cmake build",
                    capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-3000:]); raise RuntimeError("cmake build failed")
        print("fpnet built in %.0fs" % (time.time() - t3), flush=True)

        # molecule logits for every test molecule (1 forward pass per spectrum + 1 per polarity group)
        r = _fp_run([str(FB / "fpnet_ggml"), str(FP_W), str(WORK), "test", str(WORK / "fpnet_test"), "4"],
                    FP_DEADLINE, "fpnet_ggml (own weights)", capture_output=True, text=True)
        print(r.stderr[-600:], flush=True)
        if r.returncode != 0: raise RuntimeError("fpnet_ggml failed")

        # local sequential copy of the 933 MB pool fingerprints (fpnet_pool seeks ~1e5 scattered rows)
        FP_USE = FP_POOL
        try:
            FP_LOCAL = Path(os.environ.get("FPNET_LOCAL_DIR", "/tmp")) / "pool730_fp10226.bin"
            if shutil.disk_usage(FP_LOCAL.parent).free > 3 * FP_POOL.stat().st_size:
                _fp_copy(FP_POOL, FP_LOCAL, FP_DEADLINE, "pool fp copy to local disk")
                ok = Path(str(FP_POOL) + ".ok")
                if ok.exists():
                    shutil.copyfile(ok, str(FP_LOCAL) + ".ok")
                FP_USE = FP_LOCAL
        except TimeoutError:
            raise
        except Exception as e:
            print("pool fp copy skipped:", repr(e), flush=True)

        # cand_fp . logits for every pool730 candidate within 10 ppm (same window/adduct rule as phys_ana)
        r = _fp_run([str(FB / "fpnet_pool"), str(WORK), "test", str(WORK / "fpnet_test_mol.bin"), "pool730",
                     str(FP_USE), "10", str(WORK / "fpnet_test.txt")], FP_DEADLINE, "fpnet_pool",
                    capture_output=True, text=True)
        print(r.stderr[-600:], flush=True)
        if r.returncode != 0: raise RuntimeError("fpnet_pool failed")
        if not (WORK / "fpnet_test.txt").exists() or (WORK / "fpnet_test.txt").stat().st_size == 0:
            raise RuntimeError("fpnet_test.txt empty")
        FPNET_OK = True
        print("OWN FPNET stage done in %.0fs: + 1.0 z(fpnet) with %s" % (time.time() - t3, FP_W.name), flush=True)
except TimeoutError as e:
    print("FPNET TIMEOUT:", e, "- running without the FPNet term", flush=True)
except Exception as e:
    print("FPNET DISABLED:", repr(e), flush=True)
finally:
    if FP_LOCAL is not None:
        for p in (FP_LOCAL, Path(str(FP_LOCAL) + ".ok")):
            try:
                p.unlink()
            except Exception:
                pass

# phys_fp = phys_v36pc (+ optional [fpnet_scores.txt] [wf]); built to a temp name and only swapped in on success
try:
    t4 = time.time()
    V33 = (RDKIT_MIN / "phys_v33pc.cpp").exists()  # v33: Boltzmann merge of main + PubChem lists (pc5_STATUS.md)
    print("phys_v33pc.cpp present:", V33, "" if V33 else "- BOLTZ DISABLED, running as v30", flush=True)
    V35 = (RDKIT_MIN / "phys_v35pc.cpp").exists()  # v35: v33 merge + v34 attention engine (v35_STATUS.md)
    print("phys_v35pc.cpp present:", V35, "" if V35 else "- ATTN DISABLED, running as v33", flush=True)
    V36 = V35 and (RDKIT_MIN / "phys_v36pc.cpp").exists() and (RDKIT_MIN / "rev" / "cnn_model.hpp").exists()
    print("phys_v36pc.cpp present:", V36, "" if V36 else "- CNN DISABLED, running as v35", flush=True)
    srcs_fp = ["phys_v36pc.cpp" if V36 else "phys_v35pc.cpp" if V35 else "phys_v33pc.cpp" if V33 else "phys_v30pc.cpp"] + srcs[1:]
    r = subprocess.run(["g++", "-O3", "-std=c++20", "-pthread", "-w",
                        "-I", str(RDKIT_MIN), "-I", str(RDKIT_MIN / "boost-min"),
                        "-I", str(AUTOGRAD_DIR.parent),
                        "-o", str(WORK / "phys_fp")] + srcs_fp,
                       capture_output=True, text=True, cwd=str(RDKIT_MIN), timeout=150 * 60)
    print("phys_fp compile rc:", r.returncode, "in %.0fs" % (time.time() - t4), flush=True)
    if r.returncode != 0:
        print(r.stderr[-4000:]); raise RuntimeError("phys_fp compile failed")
    os.replace(WORK / "phys_fp", WORK / "phys_rank")
    FP_ARGS = ["fpnet_test.txt", "1.0"] if FPNET_OK else ["", "0"]
    print("fpnet term:", "+ 1.0 z(fpnet) (own weights)" if FPNET_OK else "none", flush=True)
    print("fpnet stage done in %.0fs" % (time.time() - t3), flush=True)
except Exception as e:
    print("PHYS_FP DISABLED:", repr(e), flush=True)
    FP_ARGS = []
'''

nb["cells"][7]["source"] = CELL7.splitlines(keepends=True)
compile(CELL7, "cell7", "exec")


def edit(ci, old, new, check=True):
    src = "".join(nb["cells"][ci]["source"])
    assert src.count(old) == 1, (ci, old)
    src = src.replace(old, new)
    if check:
        compile(src, "cell%d" % ci, "exec")
    nb["cells"][ci]["source"] = src.splitlines(keepends=True)


edit(8, '            V27_OPT += ",pcfp=" + str(FP3) + ",attn=1,attpool=2,alloc=boltz,bT1=3,bT2=0.5,bpi=0.1,ba=0.5"\n',
     '            V27_OPT += ",pcfp=" + str(FP3) + ",attn=1,attpool=2,alloc=boltz,bT1=3,bT2=0.5,bpi=0.1,ba=0.5"\n'
     '            # v36: + 2 z(CNN Tanimoto) in the pool fusion (pool36_STATUS.md: timsTOF holdouts +0.011..+0.021),\n'
     '            # only for queries with a timsTOF spectrum (cnninst=0; the hidden test is all timsTOF)\n'
     '            CNNF = POOL730_DIR / "cnn_prod_c.bin"\n'
     '            print("cnn model:", CNNF, CNNF.exists(), flush=True)\n'
     '            if globals().get("V36") and CNNF.exists():\n'
     '                V27_OPT += ",cnnmodel=" + str(CNNF) + ",cnnpool=2,cnninst=0"\n'
     '            else:\n'
     '                print("CNN DISABLED (no phys_v36pc or no cnn_prod_c.bin)", flush=True)\n')

# main ranker: timeout (ends RANKER_END_MIN after t0), one retry without the CNN term if the first run fails fast
# (not on timeout), and the plain library list as the last fallback
edit(8, '''t2 = time.time()
r = subprocess.run([str(WORK / "phys_rank"), "eval",''',
     '''t2 = time.time()
RANKER_END_MIN = 480  # the ranker must end by t0 + 480 min (Kaggle limit 540 min incl. setup and the merge below)
RANKER_OK = False
_opts = [V27_OPT]
if ",cnnmodel=" in V27_OPT:   # retry once without the CNN term if the first run fails (not on timeout)
  _opts.append(V27_OPT.split(",cnnmodel=")[0] + "".join("," + o for o in V27_OPT.split(",cnnmodel=")[1].split(",")[1:]
                                                      if not o.startswith(("cnnpool=", "cnninst="))))
for _ai, _opt in enumerate(_opts):
 RANKER_TIMEOUT = t0 + RANKER_END_MIN * 60 - time.time()
 print("ranker attempt %d, timeout %.0f s, options %s" % (_ai + 1, RANKER_TIMEOUT, _opt[-160:]), flush=True)
 try:
  if RANKER_TIMEOUT < 60:
    raise subprocess.TimeoutExpired("phys_rank", 0)
  r = subprocess.run([str(WORK / "phys_rank"), "eval",''', check=False)
edit(8, 'grid=0" + V27_OPT]', 'grid=0" + _opt]', check=False)
edit(8, '''                   capture_output=True, text=True)
print(r.stdout[-2000:], flush=True)
if r.returncode != 0:
    print(r.stderr[-3000:])
    raise SystemExit(1)
print("phys_rank ran in %.0fs" % (time.time() - t2), flush=True)
''', '''                   capture_output=True, text=True, timeout=RANKER_TIMEOUT)
  (WORK / "phys_rank_stdout.txt").write_text(r.stdout, encoding="utf-8")
  for line in r.stdout.splitlines():   # the configuration lines worth checking in the log
    if any(k in line for k in ("cnn", "attn", "alloc=boltz", "fpnet", "reverse model", "fp3")):
      print("  |", line[:300], flush=True)
  print(r.stdout[-2000:], flush=True)
  if r.returncode != 0:
    print(r.stderr[-3000:])
    raise RuntimeError("phys_rank rc %d" % r.returncode)
  RANKER_OK = (WORK / "pool_candidates.csv").exists()
  print("phys_rank ran in %.0fs" % (time.time() - t2), flush=True)
  break
 except subprocess.TimeoutExpired as e:
  print("RANKER FAILED: timeout after %.0fs - submission = plain library list" % (time.time() - t2), flush=True)
  break
 except Exception as e:
  print("RANKER FAILED:", repr(e), "after %.0fs -" % (time.time() - t2),
        "retrying without the CNN term" if _ai + 1 < len(_opts) else "submission = plain library list", flush=True)
''')
edit(8, 'pool_c = read_csv_dict(WORK / "pool_candidates.csv")\n',
     'pool_c = read_csv_dict(WORK / "pool_candidates.csv") if RANKER_OK else {}\n')

# the earlier comment cell says v33nofp: keep. Write.
out = nbd / "casmi_match_v36nofp.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
meta = json.loads((nbd / "kernel-metadata-v35nofp.json").read_text(encoding="utf-8"))
meta.update(id="yomeiotani/casmi26-cpp-library-search-v36nofp", title="CASMI26 cpp library search v36nofp",
            code_file="casmi_match_v36nofp.ipynb")
(nbd / "kernel-metadata-v36nofp.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
for c in nb["cells"]:
    if c["cell_type"] == "code":
        compile("".join(c["source"]), "cell", "exec")
print("wrote", out, "and kernel-metadata-v36nofp.json")
