"""e2e: run the production notebook's cells end-to-end on the fake competition input (build machine).

    python e2e_run.py <notebook.ipynb> [stage]     stage: all (default) | nb | eval

Input tree C:\\prog\\casmi\\e2e\\input (file symlinks): comp/ (e2e_make.py output), rdkitmin/ (= ds_rdkit, with the
holdout-safe models under the production names: fwd_mlp_rank_ylib.bin -> rdkit-min/fwd_mlp_rank_v2.bin,
rev2_z.bin -> pool730/rev2_all_v4k.bin; fpnet/ left out), pubchem/ (pcds), poolx/ (empty pool_head.bin: only
symlinked by the notebook, never read). Paths /kaggle/input and /kaggle/working are rewritten; the only code change
is the Windows .exe suffix in the os.replace of phys_fp. Multi-source g++ calls are compiled object-wise with 4 jobs
(same flags) and the rdkit objects are shared between the three compiles. The process is pinned to 4 cores
(children inherit the affinity).

stage eval: the same phys_rank.exe in `eval` mode on the holdout binaries (hcqry / hxqry, library zlib = the
same spectra as the fake train.parquet), with the notebook's exact argv and options, SMILES output.
"""
import ctypes, hashlib, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

E2E = Path(r"C:\prog\casmi\e2e"); INP = E2E / "input"; WORK = Path(os.environ.get("E2E_WORK", str(E2E / "work"))); CAS = Path(r"C:\prog\casmi")
DS = CAS / "ds_rdkit"
os.environ["PATH"] = r"C:\prog\w64devkit\bin;" + os.environ["PATH"]
ctypes.windll.kernel32.SetProcessAffinityMask(ctypes.windll.kernel32.GetCurrentProcess(), int(os.environ.get("E2E_MASK", "0xF0"), 16))
nbpath = Path(sys.argv[1]); stage = sys.argv[2] if len(sys.argv) > 2 else "all"
WORK.mkdir(parents=True, exist_ok=True)


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() and Path(os.readlink(dst)) != Path(src):  # v36: re-point overridden models
        dst.unlink()
    if not dst.exists():
        os.symlink(src, dst)


OVR = {Path("rdkit-min/fwd_mlp_rank_v2.bin"): CAS / "fwd_mlp_rank_ylib.bin",
       Path("pool730/rev2_all_v4k.bin"): CAS / "rev2_z.bin",
       # v36 (v36_STATUS.md): holdout-safe substitutes for the new production models
       Path("pool730/rev3_all_fp3c16k_big.bin"): CAS / "rev3_fp3c16k_big.bin"}   # fp3 trained on wlib
EXTRA = {Path("pool730/cnn_prod_c.bin"): CAS / "cnn_a_c.bin",                    # CNN trained on wlib
         Path("fpnet/fpnet_own_f16.bin"): CAS / "fpown" / "fpown_w_f16.bin",       # own FPNet trained on wlib
         Path("rdkit-min/phys_v36pc.cpp"): CAS / "v36" / "ds_add" / "rdkit-min" / "phys_v36pc.cpp",
         Path("rdkit-min/rev/cnn_model.hpp"): CAS / "v36" / "ds_add" / "rdkit-min" / "rev" / "cnn_model.hpp",
         Path("rdkit-min/rev/cnn_layers.hpp"): CAS / "v36" / "ds_add" / "rdkit-min" / "rev" / "cnn_layers.hpp",
         # v37 (v37_STATUS.md): FT pool term, holdout-safe fine-tuned transformer (pt_w + ft8 on wlib, epoch 5)
         Path("pool730/ft_all_ep5.bin"): CAS / "ft8_ep5.bin",
         Path("rdkit-min/phys_v37pc.cpp"): CAS / "v37" / "ds_add" / "rdkit-min" / "phys_v37pc.cpp"}
import shutil
for p in DS.rglob("*"):
    rel = p.relative_to(DS)
    if not p.is_file():
        continue
    if rel.parts[0] == "fpnet":   # sources copied (cmake), weights never: the public fpnet_full1 is left out
        if p.suffix != ".bin" and not (INP / "rdkitmin" / rel).exists():
            (INP / "rdkitmin" / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, INP / "rdkitmin" / rel)
        continue
    link(OVR.get(rel, p), INP / "rdkitmin" / rel)
for rel, src in EXTRA.items():   # copies, not links: g++ resolves "" includes next to the resolved path
    if src.exists():
        dst = INP / "rdkitmin" / rel
        if dst.is_symlink():
            dst.unlink()
        if src.suffix in (".cpp", ".hpp", ".h"):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        else:
            link(src, dst)
for n in ("pool730_fp10226.bin", "pool730_fp10226.bin.ok"):
    link(CAS / "ds_fpnet_pool" / n, INP / "fpnetpool" / n)
os.environ.setdefault("FPNET_LOCAL_DIR", str(WORK))   # the notebook's local copy of the pool fp file (Kaggle: /tmp)
for p in (CAS / "pcds").glob("pc_*.bin"):
    link(p, INP / "pubchem" / p.name)
(INP / "poolx").mkdir(parents=True, exist_ok=True)
for n in ("pool_head.bin", "pool_mols.txt"):
    (INP / "poolx" / n).touch()

# ---- parallel g++ with shared objects
_run = subprocess.run
OBJ = WORK / "obj"; OBJ.mkdir(exist_ok=True)


def gpp(cmd, cwd=None, **kw):
    srcs = [a for a in cmd[1:] if a.endswith(".cpp")]
    if len(srcs) <= 3:
        return _run(cmd[:1] + ["-D_POSIX_THREAD_SAFE_FUNCTIONS"] + cmd[1:], cwd=cwd, **kw)
    out = cmd[cmd.index("-o") + 1]
    flags = [a for i, a in enumerate(cmd[1:], 1) if not a.endswith(".cpp") and a != "-o" and cmd[i - 1] != "-o"]
    flags += ["-D_POSIX_THREAD_SAFE_FUNCTIONS"]  # mingw: localtime_r (RDLog.cpp); Windows-only compat
    t0 = time.time()

    def one(s):
        sp = Path(s) if Path(s).is_absolute() else Path(cwd or ".") / s
        h = hashlib.md5((" ".join(flags) + "|" + str(sp.resolve()) + "|" + str(sp.stat().st_mtime)).encode()).hexdigest()
        o = OBJ / (sp.stem + "_" + h[:10] + ".o")
        if not o.exists():
            r = _run(["g++"] + flags + ["-c", str(sp), "-o", str(o) + ".tmp"], cwd=cwd, capture_output=True, text=True)
            if r.returncode != 0:
                return None, r.stderr
            os.replace(str(o) + ".tmp", o)
        return str(o), ""

    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(one, srcs))
    errs = [e for o, e in res if o is None]
    if errs:
        return subprocess.CompletedProcess(cmd, 1, "", "\n".join(errs)[-6000:])
    r = _run(["g++", "-pthread", "-o", out] + [o for o, _ in res], cwd=cwd, capture_output=True, text=True)
    print(f"  [gpp] {Path(out).name}: {len(srcs)} sources, {time.time() - t0:.0f}s, link rc {r.returncode}", flush=True)
    return r


def cmk(cmd, *a, **kw):
    # Windows: MSVC + Ninja for the fpnet cmake (the .bat runs vcvars64 first); Kaggle uses its default gcc/make
    if "-S" in cmd:
        cmd = list(cmd) + ["-G", "Ninja", "-DCMAKE_C_COMPILER=cl", "-DCMAKE_CXX_COMPILER=cl"]
    return _run(cmd, *a, **kw)


subprocess.run = lambda cmd, *a, **kw: (gpp(cmd, **kw) if isinstance(cmd, list) and cmd and cmd[0] == "g++" else
                                        cmk(cmd, *a, **kw) if isinstance(cmd, list) and cmd and cmd[0] == "cmake" else
                                        _run(cmd, *a, **kw))

if stage in ("all", "nb"):
    nb = json.load(open(nbpath, encoding="utf-8"))
    code = []
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            code.append("".join(c["source"]))
    src = "\n\n".join(code)
    src = src.replace('"/kaggle/input"', repr(str(INP))).replace('"/kaggle/working"', repr(str(WORK)))
    src = src.replace("/kaggle/input/**/train.parquet", str(INP).replace("\\", "/") + "/**/train.parquet")
    left = [l for l in src.splitlines() if "/kaggle/" in l.split("#")[0] and "SystemExit" not in l]
    assert not left, left
    src = src.replace('os.replace(WORK / "phys_fp", WORK / "phys_rank")',
                      'os.replace(WORK / "phys_fp.exe", WORK / "phys_rank.exe")')
    (WORK / "nb_run.py").write_text(src, encoding="utf-8")
    t0 = time.time()
    g = {"__name__": "__main__"}
    exec(compile(src, str(WORK / "nb_run.py"), "exec"), g)
    print(f"NOTEBOOK DONE in {time.time() - t0:.0f}s", flush=True)

if stage in ("all", "eval"):
    for n in ("hcqry", "hxqry", "zlib"):
        for ext in ("_head.bin", "_peaks.bin", "_mols.txt"):
            link(CAS / (n + ext), WORK / (n + ext))
    pool730 = INP / "rdkitmin" / "pool730"
    opts = ("gen=1,cap=300,prior=gen2_prior_c.txt,gw=-2:0:-1:4:0:1.5,thr=4,grid=0,libcap=1,ppop=0.15,ppgate=0.9,ppopf="
            + str(pool730 / "pool730_pop.bin") + ",pc=1,pcv=2,pcrank=fused,relhi=6,relp=3,sp=6,pcdir=" + str(INP / "pubchem")
            + ",alloc=boltz,bT1=2,bT2=1,bpi=0.2,ba=0.25,pcrev=tan")
    for q in ("hcqry", "hxqry"):
        for fmt in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["smiles"]):
            t0 = time.time()
            out = f"eval_{q}_{fmt or 'names'}.csv"
            r = _run([str(WORK / "phys_rank.exe"), "eval", str(WORK), q, "pool730", "10", "phys_size_fqry.txt", out,
                      "fwd_mlp_rank_v2.bin", "0.25", fmt, "rev2_all_v4k.bin", "1.0", "zlib", "8", "32", "24", "", "4", "3",
                      "8000", "", "", "0", opts], capture_output=True, text=True)
            (WORK / f"eval_{q}_{fmt or 'names'}.log").write_text(r.stdout + "\n" + r.stderr[-3000:], encoding="utf-8")
            print(f"EVAL {q} {fmt or 'names'} rc {r.returncode} in {time.time() - t0:.0f}s", flush=True)
            print(r.stdout[-1500:], flush=True)
