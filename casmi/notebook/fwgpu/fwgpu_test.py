# fwgpu_test.py: Kaggle T4 check of the GPU fw_score build (nvcc + cuBLAS): build, then run the same small input
# (40 e2e queries, 1117 candidates) with gpu=1 and on the CPU, compare the gl / ice columns and the speed.
import glob, os, subprocess, sys, time
from pathlib import Path
src = Path(sorted(glob.glob("/kaggle/input/**/fw_score.cpp", recursive=True), key=len)[0]).parent
ds = src.parent
rdk = next(Path(p).parent for p in sorted(glob.glob("/kaggle/input/**/RDGeneral/Invariant.h", recursive=True), key=len)).parent
print("src", src, "rdkit-min", rdk, flush=True)
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout, flush=True)
print(subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout[-200:], flush=True)
sys.path.insert(0, str(src))
from fw_build import build_fw_score
W = Path("/kaggle/working")
small = sorted(glob.glob("/kaggle/input/**/fw_small.txt", recursive=True))[0]
t = time.time()
rc = build_fw_score(src, rdk, W / "obj", W / "fw_score_gpu", jobs=4, gpu="sm_75")
print("gpu build rc", rc, "%.0fs" % (time.time() - t), flush=True)
args = ["gl=" + str(ds / "glacier_w.bin"), "ice=%s,%s" % (ds / "ice_gen_w.bin", ds / "ice_inten_w.bin"), "threads=4"]
res = {}
for tag, extra in (("gpu", ["gpu=1"]), ("cpu", [])):
    t = time.time()
    r = subprocess.run([str(W / "fw_score_gpu"), small, str(W / f"out_{tag}.tsv")] + args + extra + ["budget=3000"],
                       capture_output=True, text=True)
    print(tag, "rc", r.returncode, "%.1fs" % (time.time() - t), r.stderr[-800:], flush=True)
    d = {}
    for l in open(W / f"out_{tag}.tsv"):
        p = l.rstrip("\n").split("\t")
        if len(p) >= 4:
            d[(p[0], p[1])] = (p[2], p[3])
    res[tag] = d
ks = [k for k in res["gpu"] if k in res["cpu"]]
for j, nm in ((0, "gl"), (1, "ice")):
    diffs = [abs(float(res["gpu"][k][j]) - float(res["cpu"][k][j])) for k in ks
             if res["gpu"][k][j] != "nan" and res["cpu"][k][j] != "nan"]
    print(nm, "common", len(diffs), "max|diff|", max(diffs) if diffs else None, flush=True)
