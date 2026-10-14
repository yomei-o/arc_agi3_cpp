# FPNet integration for casmi_match.ipynb  (prepared 2026-10-14, NOT yet in the ipynb, NOT pushed)
#
# Two edits to the notebook:
#   (A) a NEW cell inserted between cell 5 (compiles phys_ana as WORK/phys_rank) and cell 6 (runs it):
#       builds ggml + fpnet_ggml + fpnet_pool with cmake, scores the test molecules' pool candidates
#       with FPNet (cand_fp . molecule logits), and rebuilds WORK/phys_rank from phys_fp.cpp
#       (= phys_ana.cpp + optional "+ wf * z(fpnet)" on the final fused score).
#       Any failure leaves FP_ARGS = [] and the old phys_ana binary in place -> notebook behaves as v21.
#   (B) cell 6: append FP_ARGS to the phys_rank argument list (one-line change, shown at the bottom).
#
# Datasets needed:
#   yomeiotani/casmi26-rdkit-min (new version) with
#       fpnet/  : CMakeLists.txt fpnet_ggml.cpp fpnet_pool.cpp to_f16.py fpnet_full1_f16.bin ggml/ (subset)
#       rdkit-min/phys_fp.cpp
#   pool fingerprints (pool730_fp10226.bin + pool730_fp10226.bin.ok, 933 MB raw / ~424 MB zipped):
#       separate dataset recommended, e.g. yomeiotani/casmi26-fpnet-pool (staged at build:C:\prog\casmi\ds_fpnet_pool)
#
# ======================= (A) new cell, insert after cell 5 =======================
# v22: FPNet (ahmedberatozer casmi26-fpnet-full1 weights, ported to C++/ggml):
# spectrum -> 10226 fingerprint-bit logits per molecule, candidate score = cand_fp . logits,
# fused as + 1.0 * z(fpnet) on top of the v21 learned fusion.
# CAVEAT: fpnet_full1 was trained on all of train (holdout numbers leaked), weight 1.0 is hand-set.
FP_ARGS = []
try:
    t3 = time.time()
    FPNET_SRC = RDKIT_MIN.parent / "fpnet"
    FP_W = FPNET_SRC / "fpnet_full1_f16.bin"
    # avoid a full rglob over /kaggle/input (took 11 min once): look at most 4 levels deep
    FP_POOL = None
    for pat in ("*/pool730_fp10226.bin", "*/*/pool730_fp10226.bin", "*/*/*/pool730_fp10226.bin",
                "*/*/*/*/pool730_fp10226.bin"):
        hit = sorted(Path("/kaggle/input").glob(pat))
        if hit:
            FP_POOL = hit[0]; break
    print("fpnet src:", FPNET_SRC, FPNET_SRC.exists(), " weights:", FP_W.exists(), " pool fp:", FP_POOL, flush=True)
    if FP_POOL is None or not FP_W.exists():
        raise RuntimeError("fpnet inputs missing")
    FB = WORK / "fpnet_build"
    r = subprocess.run(["cmake", "-S", str(FPNET_SRC), "-B", str(FB), "-DCMAKE_BUILD_TYPE=Release",
                        "-DGGML_NATIVE=OFF"], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-3000:]); raise RuntimeError("cmake configure failed")
    r = subprocess.run(["cmake", "--build", str(FB), "-j", "4"], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:]); raise RuntimeError("cmake build failed")
    print("fpnet built in %.0fs" % (time.time() - t3), flush=True)

    # molecule logits for every test molecule (1 forward pass per spectrum + 1 per polarity group)
    t4 = time.time()
    r = subprocess.run([str(FB / "fpnet_ggml"), str(FP_W), str(WORK), "test", str(WORK / "fpnet_test"), "4"],
                       capture_output=True, text=True)
    print(r.stderr[-600:], flush=True)
    if r.returncode != 0: raise RuntimeError("fpnet_ggml failed")
    print("fpnet_ggml ran in %.0fs" % (time.time() - t4), flush=True)

    # cand_fp . logits for every pool730 candidate within 10 ppm (same window/adduct rule as phys_ana)
    t4 = time.time()
    r = subprocess.run([str(FB / "fpnet_pool"), str(WORK), "test", str(WORK / "fpnet_test_mol.bin"), "pool730",
                        str(FP_POOL), "10", str(WORK / "fpnet_test.txt")], capture_output=True, text=True)
    print(r.stderr[-600:], flush=True)
    if r.returncode != 0: raise RuntimeError("fpnet_pool failed")
    print("fpnet_pool ran in %.0fs" % (time.time() - t4), flush=True)

    # phys_fp = phys_ana + optional [fpnet_scores.txt] [wf]; same compile line as cell 5, built to a
    # temp name and only swapped in on success
    t4 = time.time()
    srcs_fp = ["phys_fp.cpp"] + srcs[1:]
    r = subprocess.run(["g++", "-O3", "-std=c++20", "-pthread", "-w",
                        "-I", str(RDKIT_MIN), "-I", str(RDKIT_MIN / "boost-min"),
                        "-I", str(AUTOGRAD_DIR.parent),
                        "-o", str(WORK / "phys_fp")] + srcs_fp,
                       capture_output=True, text=True, cwd=str(RDKIT_MIN))
    print("phys_fp compile rc:", r.returncode, "in %.0fs" % (time.time() - t4), flush=True)
    if r.returncode != 0:
        print(r.stderr[-4000:]); raise RuntimeError("phys_fp compile failed")
    os.replace(WORK / "phys_fp", WORK / "phys_rank")
    FP_ARGS = ["fpnet_test.txt", "1.0"]
    print("fpnet stage done in %.0fs" % (time.time() - t3), flush=True)
except Exception as e:
    print("FPNET DISABLED:", repr(e), flush=True)
    FP_ARGS = []

# ======================= (B) cell 6: the phys_rank call becomes =======================
# r = subprocess.run([str(WORK / "phys_rank"), "eval", str(WORK), "test", "pool730", "10",
#                     "phys2_fqry.txt", "pool_candidates.csv", "fwd_mlp_rank_v2.bin", "0.25", "smiles",
#                     "rev_all3072.bin", "1.0", "train", "8", "32", "24", "", "4", "3", "8000",
#                     "fuse_w21.txt"] + FP_ARGS,
#                    capture_output=True, text=True)
# (phys_fp prints "fpnet scores: N lines, M queries" and
#  "output + 1 z(fpnet): candidates with score .., without .., queries without any .., MRR@25 .." in stdout;
#  the MRR there is 0 on the real test set since it has no truth.)
