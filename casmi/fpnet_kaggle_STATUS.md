# FPNet Kaggle integration — status 2026-10-14 15:50 (prepared, NOT pushed / NOT versioned)

## Done (all tested on the build machine)
1. Pool scoring tool `cpp/fpnet/fpnet_pool.cpp` (no ggml, plain C++17):
   `fpnet_pool <dir> <qry_stem> <mol_logits.bin> <pool_stem> <pool_fp.bin> <ppm> <out.txt>`
   -> "<molecule_id>\t<pool name>\t<cand_fp . logits>"; same window/adduct rule as phys_ana (ADDUCT_SHIFT,
   tag%16==15 -> tag/16+1, [2M+H]+/[2M+Na]+ halve). Reads only the needed fp rows (seek).
   Cross-check on cqry vs score.py's fpnet_cqry.txt: identical candidate sets (35,635 pairs), max |diff| 0.0027 (print rounding).
2. `cpp/fpnet/fpnet_ggml.cpp`: accepts FPN2 weights (2-D matrices f16, 1-D tensors f32; `to_f16.py`) -> 99.5 MB instead of 199 MB.
   f16 vs f32 molecule logits (20 test mols): max |diff| 0.007 on a +-150 range, top-100 bits identical.
   PITFALL found: converting EVERYTHING to f16 breaks the model (SinEmb `*.inv` frequencies up to 628 -> ~200 rad phase
   error at m/z 1000; logits off by up to 94, top-100 bit overlap 0.51). 1-D tensors must stay f32 — the converter does that now.
3. `cpp/rdkit-min/phys_fp.cpp` = phys_ana.cpp + optional args 22/23: `[fpnet_scores.txt] [wf=1]`
   (path relative to <dir> like the other files). After the learned-weights fusion it adds wf*z(fpnet) per query;
   a candidate without a score gets the query mean (z=0). wf=0 or no arg -> exactly phys_ana's code path.
   Private build: build:C:\prog\casmi\fpnet\physfp\ (build_fp.bat, own build dir). Staged as ds_rdkit\rdkit-min\phys_fp.cpp.
4. End-to-end on visible `test` + library `alllib` (= notebook's `train`), cell-6 args, pinned to 4 cores (start /affinity F):
   | step | time (4 cores) |
   |---|---|
   | cmake build ggml+fpnet (MSVC, -j8) | 16 s (GCC/MinGW Makefiles build of the staged tree also OK; outputs match MSVC) |
   | fpnet_ggml, 400 mols / 1710 forward passes | 94 s (f16 file), 88 s (f32) — 55 ms/pass |
   | fpnet_pool, 58,496 candidates | 1.2 s |
   | phys_fp wf=1 | 2 min 57 s (same as wf=0; the fpnet term costs nothing) |
   Output: 400 rows, 0 empty lists, 369 full 25, min 2 candidates. Every candidate had an fpnet score (0 missing).
   wf=1 vs wf=0 on visible test: top-1 changed for 4/400 molecules, mean top-25 overlap 0.81, 7 lists identical.
   (Visible test is a placeholder without truth -> no MRR here.)
5. Dataset staging (build machine):
   - C:\prog\casmi\ds_rdkit\fpnet\ (105 MB, 123 files): CMakeLists.txt, fpnet_ggml.cpp, fpnet_pool.cpp, to_f16.py,
     fpnet_full1_f16.bin (99.5 MB; deflate ratio 0.92 -> ~92 MB upload), ggml\ subset (CMakeLists, cmake\, include\, src\*.c/h/cpp + src\ggml-cpu\ only, 1.2 MB).
   - C:\prog\casmi\ds_rdkit\rdkit-min\phys_fp.cpp
   - C:\prog\casmi\ds_fpnet_pool\ (new separate dataset, id yomeiotani/casmi26-fpnet-pool in dataset-metadata.json):
     pool730_fp10226.bin (933 MB, deflate ratio 0.455 -> ~424 MB upload) + .ok. Separate because it is static and
     would otherwise be re-uploaded with every code version of casmi26-rdkit-min.
6. Notebook cell code: notebook\fpnet_cells.py — (A) new cell after cell 5: cmake build (-DGGML_NATIVE=OFF), fpnet_ggml,
   fpnet_pool, compile phys_fp -> replaces WORK/phys_rank; wrapped in try/except: any failure -> FP_ARGS=[] and the v21
   phys_ana binary stays (notebook unaffected). (B) cell 6: append `+ FP_ARGS` to the phys_rank arg list.
   Pool fp file found with a <=4-level glob (no full rglob).

## Remaining (user decision)
1. wf=1.0 is hand-set and unvalidated: fpnet_full1 was trained on all of train, so holdout MRR is leaked. Decide wf
   (or fit it with fuse_fit on something leak-free); first LB submission is the real test.
2. Upload: `kaggle datasets create -p C:\prog\casmi\ds_fpnet_pool` (new) and `kaggle datasets version -p C:\prog\casmi\ds_rdkit`;
   add both datasets to kernel-metadata.json dataset_sources; paste fpnet_cells.py (A) as a new cell, apply (B); push.
3. Not tested on actual Linux/Kaggle: GCC build was tested only via w64devkit (MinGW). Kaggle CPU is slower; expect
   fpnet_ggml ~2-3 min. ggml uses OpenMP on Linux by default (libgomp present on Kaggle).
4. Optional cleanup: WORK/fpnet_build, fpnet_test_mol.bin (16 MB) end up in kernel output.

## Files
- local: cpp\fpnet\{fpnet_pool.cpp, fpnet_ggml.cpp, to_f16.py, CMakeLists.txt}, cpp\rdkit-min\phys_fp.cpp, notebook\fpnet_cells.py
- build: C:\prog\casmi\fpnet\{kbuild (MSVC), kbuild_gcc, physfp\, run_e2e.bat, run_e2e2.bat, physfp_test_wf{0,1}.log, KAGGLE_STATUS.txt},
  C:\prog\casmi\fpnet_test.txt, C:\prog\casmi\pool_candidates_fp{0,1}.csv, C:\prog\casmi\fpnet\gg_test32_mol.bin
