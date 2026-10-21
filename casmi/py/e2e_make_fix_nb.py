"""Make casmi_match_v33nofp_fix.ipynb from casmi_match_v33nofp.ipynb (e2e_STATUS.md, fix 1).

Fix 1 (runtime, not score): the notebook compiles the whole rdkit-min + autograd tree three times, single-threaded
(Kaggle v30 log: 1261 s + 1180 s + 1230 s = 61 min; v31 was refused at 665 min > 540). fast_gpp compiles every
source once into object files with 4 parallel jobs (same flags), reuses the shared objects for the three
executables and links. Output binaries are built from the same sources with the same flags.

    python e2e_make_fix_nb.py <src.ipynb> <dst.ipynb>
"""
import json, sys

FAST = r'''
# e2e fix 1: object-wise parallel compile with shared objects (same flags as the one-shot g++ calls)
import hashlib as _hl
from concurrent.futures import ThreadPoolExecutor as _TPE

def fast_gpp(cmd, cwd=None, capture_output=True, text=True, jobs=4):
    srcs = [a for a in cmd[1:] if a.endswith(".cpp")]
    out = cmd[cmd.index("-o") + 1]
    flags = [a for i, a in enumerate(cmd[1:], 1) if not a.endswith(".cpp") and a != "-o" and cmd[i - 1] != "-o"]
    objdir = WORK / "obj"; objdir.mkdir(exist_ok=True)
    def one(s):
        sp = Path(s) if Path(s).is_absolute() else Path(cwd or ".") / s
        h = _hl.md5((" ".join(flags) + "|" + str(sp.resolve())).encode()).hexdigest()[:12]
        o = objdir / (sp.stem + "_" + h + ".o")
        if not o.exists():
            r = subprocess.run(cmd[:1] + flags + ["-c", str(sp), "-o", str(o) + ".tmp"], cwd=cwd,
                               capture_output=True, text=True)
            if r.returncode != 0:
                return None, r.stderr
            os.replace(str(o) + ".tmp", o)
        return str(o), ""
    with _TPE(jobs) as ex:
        res = list(ex.map(one, srcs))
    errs = [e for o, e in res if o is None]
    if errs:
        return subprocess.CompletedProcess(cmd, 1, "", "\n".join(errs)[-6000:])
    return subprocess.run(cmd[:1] + [f for f in flags if f == "-pthread"] + ["-o", out] + [o for o, _ in res],
                          cwd=cwd, capture_output=True, text=True)
'''

nb = json.load(open(sys.argv[1], encoding="utf-8"))
n_sub = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    if 'r = subprocess.run(["g++", "-O3", "-std=c++17"' in s and "match.cpp" in s:
        s = s.replace('print("compiled in %.0fs" % (time.time() - t0), flush=True)',
                      'print("compiled in %.0fs" % (time.time() - t0), flush=True)\n' + FAST)
        n_sub += 1
    k = s.count('r = subprocess.run(["g++", "-O3", "-std=c++20"')
    if k:
        s = s.replace('r = subprocess.run(["g++", "-O3", "-std=c++20"', 'r = fast_gpp(["g++", "-O3", "-std=c++20"')
        n_sub += k
    c["source"] = s.splitlines(keepends=True)
assert n_sub == 4, n_sub   # helper + pool_candidate_rank + phys_rank + phys_fp
json.dump(nb, open(sys.argv[2], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote", sys.argv[2], "substitutions", n_sub)
