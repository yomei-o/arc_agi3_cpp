# CASMI26 cpp build: compile the rdkit-min C++ tree ONCE and link every program the submission
# notebooks compile, so they can copy binaries instead of compiling (use_prebuilt.py).
#
# Each target is built from object files, but its manifest entry records the EXACT one-shot g++
# command the submission notebook runs (normalised: <DS> = dataset root, <WORK> = working dir,
# <OUT> = output). Separate compilation + a plain link is what g++ itself does for a multi-source
# command (no LTO), with the objects in the same order. The submission side uses a binary only when
# its own command, the main .cpp sha256 and the tree hash of rdkit-min/ + autograd/ all match.
#
# Expects MATCH_SOURCES (list of match.cpp texts) from the cell above (may be empty).
PB_INPUT = Path(os.environ.get("PB_INPUT", "/kaggle/input"))
PB_WORK = Path(os.environ.get("PB_WORK", "/kaggle/working"))
PB_TMP = Path(os.environ.get("PB_TMP", "/tmp/pb_build"))
PB_JOBS = int(os.environ.get("PB_JOBS", str(os.cpu_count() or 4)))
PB_SKIP_FPNET = os.environ.get("PB_SKIP_FPNET", "") == "1"
PB_ONLY = [s for s in os.environ.get("PB_ONLY", "").split(",") if s]  # test hook: build only these mains
PB_TEST_CFLAGS = os.environ.get("PB_TEST_CFLAGS", "").split()  # test hook (mingw needs a define); unset on Kaggle
EXE = ".exe" if os.name == "nt" else ""
from concurrent.futures import ThreadPoolExecutor

t0 = time.time()
BIN = PB_WORK / "bin"
BIN.mkdir(parents=True, exist_ok=True)
OBJ = PB_TMP / "obj"
OBJ.mkdir(parents=True, exist_ok=True)
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    LOG.append(s)
    print(s, flush=True)


def find_input(rel):
    # at most 4 directory levels below the input root, never a recursive walk of all of it
    for k in range(1, 5):
        hits = sorted(PB_INPUT.glob("/".join(["*"] * k + [rel])))
        if hits:
            return hits[0]
    return None


hit = find_input("rdkit-min/pool_candidate_rank.cpp")
if hit is None:
    raise SystemExit("rdkit-min/pool_candidate_rank.cpp not found under " + str(PB_INPUT))
RDKIT_MIN = hit.parent
DS_ROOT = RDKIT_MIN.parent
AUTOGRAD_DIR = DS_ROOT / "autograd"
log("dataset root:", DS_ROOT)
TREE_DIRS = ["rdkit-min", "autograd"]
t1 = time.time()
TREE_SHA, TREE_N, TREE_BYTES = pb_tree_hash(DS_ROOT, TREE_DIRS)
log("TREE HASH rdkit-min+autograd: %s  (%d files, %.1f MB, %.1fs)"
    % (TREE_SHA, TREE_N, TREE_BYTES / 1e6, time.time() - t1))

# the same source list, in the same order, as the submission notebook (cell 5: srcs[1:])
COMMON = [str(AUTOGRAD_DIR / "autograd.cpp"), str(AUTOGRAD_DIR / "nn_ops.cpp")]
for sub in ("RDGeneral", "DataStructs", "Geometry", "GraphMol",
            "GraphMol/CIPLabeler", "GraphMol/CIPLabeler/configs",
            "GraphMol/CIPLabeler/rules", "GraphMol/SmilesParse"):
    COMMON += [str(p) for p in sorted((RDKIT_MIN / sub).glob("*.cpp"))]
FLAGS = ["-O3", "-std=c++20", "-pthread", "-w",
         "-I", str(RDKIT_MIN), "-I", str(RDKIT_MIN / "boost-min"), "-I", str(AUTOGRAD_DIR.parent)]

MAINS = ["pool_candidate_rank.cpp", "phys_ana.cpp", "phys_probe.cpp"]
MAINS += sorted(p.name for p in RDKIT_MIN.glob("phys_v*.cpp") if p.name not in MAINS)
MAINS = [m for m in MAINS if (RDKIT_MIN / m).exists() and (not PB_ONLY or m in PB_ONLY)]
log("common sources:", len(COMMON), " mains:", " ".join(MAINS))


def run(cmd, cwd=None):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, cwd=cwd)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def compile_obj(job):
    src, obj = job
    t = time.time()
    rc, msg = run(["g++"] + FLAGS + PB_TEST_CFLAGS + ["-c", src, "-o", str(obj)], cwd=str(RDKIT_MIN))
    return src, obj, rc, msg, time.time() - t


jobs = [(s, OBJ / ("c%03d_%s.o" % (i, Path(s).stem))) for i, s in enumerate(COMMON)]
jobs += [(m, OBJ / ("m_%s.o" % Path(m).stem)) for m in MAINS]
t1 = time.time()
failed = {}
with ThreadPoolExecutor(PB_JOBS) as ex:
    for src, obj, rc, msg, dt in ex.map(compile_obj, jobs):
        if rc != 0:
            failed[src] = msg[-3000:]
            log("COMPILE FAIL", src, "\n", msg[-3000:])
log("compiled %d objects (%d failed) in %.0fs with %d jobs" % (len(jobs), len(failed), time.time() - t1, PB_JOBS))
common_fail = [s for s in COMMON if s in failed]
if common_fail:
    raise SystemExit("common sources failed to compile: %s" % common_fail)
COMMON_OBJS = [str(o) for s, o in jobs[:len(COMMON)]]

# reference no-argument exit code: the submission side reruns it on the copied binary as a smoke
# test (catches a binary that cannot start there: missing libs, illegal instruction)
def noarg_rc(path):
    d = PB_TMP / "smoke" / Path(path).name
    d.mkdir(parents=True, exist_ok=True)
    try:
        return subprocess.run([str(path)], capture_output=True, cwd=str(d), timeout=20,
                              stdin=subprocess.DEVNULL).returncode
    except subprocess.TimeoutExpired:
        return None


entries = []
for m in MAINS:
    if m in failed:
        continue
    name = Path(m).stem
    out = BIN / (name + EXE)
    link = ["g++", "-O3", "-std=c++20", "-pthread", "-o", str(out), str(OBJ / ("m_%s.o" % name))] + COMMON_OBJS
    rc, msg = run(link)
    if rc != 0:
        log("LINK FAIL", name, msg[-2000:])
        continue
    ref = ["g++"] + FLAGS + ["-o", str(PB_WORK / name), m] + COMMON  # what the notebook runs, cwd=RDKIT_MIN
    entries.append({
        "name": name, "kind": "gcc", "file": out.name,
        "key": pb_norm_cmd([m], str(RDKIT_MIN), DS_ROOT, PB_WORK)[0],
        "main_sha256": pb_sha_file(RDKIT_MIN / m),
        "cmd": pb_norm_cmd(ref, str(RDKIT_MIN), DS_ROOT, PB_WORK),
        "tree_dirs": TREE_DIRS, "tree_exclude": [], "tree_sha256": TREE_SHA,
        "bin_sha256": pb_sha_file(out), "bin_bytes": out.stat().st_size,
        "noarg_rc": noarg_rc(out),
        "built_with": {"objects": ["g++"] + FLAGS + ["-c", "<src>", "-o", "<obj>"], "cwd": str(RDKIT_MIN),
                       "link": ["g++", "-O3", "-std=c++20", "-pthread", "-o", "<OUT>", "<main.o>", "<common objs in source order>"]},
    })
    log("linked", name, "%.1f MB" % (out.stat().st_size / 1e6), "noarg rc", entries[-1]["noarg_rc"])

# match.cpp: one small file, written by the submission notebook from its CPP_SOURCE string
seen = set()
for text in globals().get("MATCH_SOURCES", []):
    # written exactly as the notebook writes it (write_text, utf-8), then hashed as a file
    d = PB_TMP / ("match_%d" % len(seen))
    d.mkdir(parents=True, exist_ok=True)
    (d / "match.cpp").write_text(text, encoding="utf-8")
    sha = pb_sha_file(d / "match.cpp")
    if sha in seen:
        continue
    seen.add(sha)
    out = BIN / ("match_" + sha[:12] + EXE)
    flags = ["-O3", "-std=c++17", "-pthread", "-w"]
    rc, msg = run(["g++"] + flags + ["-o", str(out), str(d / "match.cpp")])
    if rc != 0:
        log("MATCH COMPILE FAIL", sha[:12], msg[-2000:])
        continue
    entries.append({
        "name": "match_" + sha[:12], "kind": "gcc", "file": out.name, "key": "<WORK>/match.cpp",
        "main_sha256": sha,
        "cmd": ["g++"] + flags + ["-o", "<OUT>", "<WORK>/match.cpp"],
        "tree_dirs": [], "tree_exclude": [], "tree_sha256": "",
        "bin_sha256": pb_sha_file(out), "bin_bytes": out.stat().st_size, "noarg_rc": noarg_rc(out),
        "built_with": {"cmd": ["g++"] + flags + ["-o", "<OUT>", "<tmp>/match.cpp"]},
    })
    log("built match", sha[:12])

# FPNet ggml tools (cmake), same configure line as the notebook
FPNET_SRC = DS_ROOT / "fpnet"
if (FPNET_SRC / "CMakeLists.txt").exists() and not PB_SKIP_FPNET:
    t1 = time.time()
    FB = PB_TMP / "fpnet_build"
    conf = ["cmake", "-S", str(FPNET_SRC), "-B", str(FB), "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF"]
    rc, msg = run(conf)
    if rc == 0:
        rc, msg = run(["cmake", "--build", str(FB), "-j", str(PB_JOBS)])
    if rc != 0:
        log("FPNET BUILD FAIL", msg[-3000:])
    else:
        fsha, fn, fb = pb_tree_hash(DS_ROOT, ["fpnet"], [".bin"])
        log("TREE HASH fpnet (without *.bin): %s  (%d files)" % (fsha, fn))
        for tool in ("fpnet_ggml", "fpnet_pool"):
            src = FB / (tool + EXE)
            if not src.exists():
                log("FPNET missing output", tool)
                continue
            out = BIN / (tool + EXE)
            shutil.copyfile(src, out)
            os.chmod(out, 0o755)
            entries.append({
                "name": tool, "kind": "cmake", "group": "fpnet", "file": out.name, "key": "<DS>/fpnet",
                "main_sha256": "", "cmd": pb_norm_cmd(conf, None, DS_ROOT, PB_WORK),
                "tree_dirs": ["fpnet"], "tree_exclude": [".bin"], "tree_sha256": fsha,
                "bin_sha256": pb_sha_file(out), "bin_bytes": out.stat().st_size, "noarg_rc": noarg_rc(out),
                "built_with": {"configure": conf, "build": ["cmake", "--build", "<B>", "-j", str(PB_JOBS)]},
            })
        log("fpnet built in %.0fs" % (time.time() - t1))
else:
    log("fpnet: skipped")

manifest = {
    "format": PB_FORMAT, "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "dataset_root": str(DS_ROOT), "work": str(PB_WORK),
    "tree_sha256": TREE_SHA, "tree_files": TREE_N, "tree_bytes": TREE_BYTES,
    "gxx": run(["g++", "--version"])[1].splitlines()[0] if shutil.which("g++") else "",
    "platform": sys.platform, "binaries": entries,
}
(BIN / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
(BIN / "build_log.txt").write_text("\n".join(LOG), encoding="utf-8")
shutil.rmtree(PB_TMP, ignore_errors=True)
print("\n==== PREBUILT SUMMARY ====")
print("tree sha256:", TREE_SHA)
for e in entries:
    print("  %-22s %-7s %8.1f MB  main %s  noarg rc %s" % (e["name"], e["kind"], e["bin_bytes"] / 1e6,
                                                         e["main_sha256"][:12] or "-", e["noarg_rc"]))
print("failed mains:", [m for m in MAINS if m in failed] or "none")
print("total %.0fs, output %s" % (time.time() - t0, BIN))
