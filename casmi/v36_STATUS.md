# v36nofp — own FPNet + production CNN pool term + time guards (status file)

2026-10-22 morning. v36nofp = v35nofp (conservative Boltzmann merge bT1 3 / bT2 0.5 / bpi 0.1 / ba 0.5, attention
attpool 2, fp3 window model, prebuilt hook) + **own FPNet** (+1.0 z) + **CNN pool term** (+2 z, timsTOF queries only)
+ **time guards**. Every model is trained by our own code on train.parquet (prize-eligible). Nothing uploaded /
pushed / submitted.

## Why v33 hit 570 min (measured, Kaggle log of kernel v33 v1)
`kaggle kernels output yomeiotani/casmi26-cpp-library-search-v33` (log only):
| step | Kaggle v33 | build machine (e2e, 800 molecules, 4 cores) |
|---|---|---|
| fpnet cmake configure + build | 371 s | 8 s + 8 s (MSVC/Ninja) |
| fpnet_ggml (logits) | 211 s (400 mol, 1710 passes, 119 ms/pass) | 122 s (800 mol, 3010 passes, 41 ms/pass) |
| **fpnet_pool** | **29,381 s** (58,496 candidates) | **1.8 s** (90,864 candidates) |
| phys_fp compile | 658 s | 36 s (object-wise, 4 jobs) |
| ranker | 1,047 s | 1,270 s |
**The whole overrun is fpnet_pool**: it reads one 1.3 KB row of the 933 MB `pool730_fp10226.bin` per candidate with
`fseek` (rows in pool order, i.e. scattered). On the Kaggle input mount that cost **0.50 s per row read**
(29,381 s / 58,496); locally it is 30 µs. The run-to-run spread (37 min other days vs 8.5 h) fits a network-backed,
lazily fetched dataset mount whose per-request latency depends on load/cache state, not CPU (ggml and the compiles
were normal that day). Fix: copy the file **sequentially** to local disk (/tmp) first (64 MB chunks; 1 s locally),
run fpnet_pool on the copy, delete it after. Plus the guards below in case the copy itself is slow.

## Notebook (`notebook/casmi_match_v36nofp.ipynb`, `kernel-metadata-v36nofp.json`; generator `py/make_v36_nb.py`)
id yomeiotani/casmi26-cpp-library-search-v36nofp, title "CASMI26 cpp library search v36nofp", same dataset_sources and
kernel_sources (casmi26-cpp-build) as v35nofp. Changes vs v35nofp:
- Cell 7 (FPNet stage) rewritten:
  - `FP_W = FPNET_SRC / "fpnet_own_f16.bin"` only. Missing -> prints `OWN FPNET MISSING` and runs without the term
    (never looks at fpnet_full1_f16.bin).
  - `_fp_run`: every FPNet subprocess (cmake configure, cmake --build, fpnet_ggml, fpnet_pool) under one stage deadline
    `FPNET_BUDGET = 75 min`, `start_new_session=True`; on timeout the child and every process in a session created
    during the call are SIGKILLed (make / cc1plus grandchildren too) -> `FPNET TIMEOUT: ...`, run continues without
    the term. `[time] <step>: Ns` lines for each step. Pool fp copy to `/tmp` (FPNET_LOCAL_DIR) under the same deadline.
  - phys_fp compile is now outside the FPNet try: a failed / timed-out FPNet stage no longer drops phys_v36pc (in v22..v35
    any FPNet exception also skipped the phys_fp compile -> v24 fallback). Compile timeout 150 min.
  - Source choice phys_v36pc (if phys_v36pc.cpp and rev/cnn_model.hpp exist) > v35 > v33 > v30.
- Cell 8: `,cnnmodel=<POOL730_DIR>/cnn_prod_c.bin,cnnpool=2,cnninst=0` (only if phys_v36pc built and the file exists,
  else `CNN DISABLED`). Ranker: `timeout = t0 + 480 min - now` (t0 = notebook start in cell 3); a fast failure
  (rc != 0) is retried once without the CNN options; timeout or second failure -> `RANKER FAILED ...` and the submission
  is the plain library list (match.cpp). Full ranker stdout -> `/kaggle/working/phys_rank_stdout.txt`, and the
  configuration lines (cnn / attn / alloc=boltz / fpnet / reverse model / fp3) are echoed as `  | ...`.
- Time guard test on Linux (WSL, `py/v36_timeout_test.py`, helpers exec'd from the notebook cell): a child forking two
  grandchildren under a 13 s budget -> TimeoutError at 13.0 s, "2 leftover processes killed", no `sleep` left; normal
  command passes; expired copy -> TimeoutError. PASS.
- Worst case: pre-ranker steps (match, export, compiles; ~2 h on a slow Kaggle day without prebuilt) + FPNet <= 75 min,
  ranker ends by t0 + 480 min whatever happened before -> total <= ~8 h + merge + Kaggle overhead < 540 min.

## phys_v36pc: `cnninst=k` (one condition)
`cnninst=0`: a query gets CNN p (and so the cnnpool term) only if one of its spectra has instrument tag 0 (timsTOF);
other queries keep the v35 fusion. Log line `cnninst 0: CNN term on N of M queries`. Default -1 = unchanged.
Verified (`cpp/v36/chkinst.bat`, MSVC rebuild): hcqry (no timsTOF spectrum) with `cnnmodel=cnn_a_c.bin,cnnpool=2,cnninst=0`
vs p36c_0 (cnnpool 0 = v35_0): **fc /b no differences** on the plain and the inserted CSV; "CNN term on 0 of 400 queries".
In the e2e run (hc + hx mixed): "CNN term on 400 of 800 queries" (= the hx timsTOF queries).

## Dataset staging (build machine `C:\prog\casmi\v36\ds_add\`, `cpp/v36/kcheck36.bat`) — NOT uploaded
| file | size | to |
|---|---|---|
| `rdkit-min\phys_v36pc.cpp` | 144,233 | casmi26-rdkit-min rdkit-min/ |
| `rdkit-min\rev\cnn_model.hpp`, `cnn_layers.hpp` | 18,400 / 14,402 | casmi26-rdkit-min rdkit-min/rev/ (they include parallel.hpp, rev2_model.hpp: already there) |
| `fpnet\fpnet_own_f16.bin` | 99,517,113 (md5 c44d7511…) | casmi26-rdkit-min fpnet/ |
| `pool730\cnn_prod_c.bin` | 262,673,248 (md5 d92a7274…) | the pool730/ folder holding pool730_head.bin (where rev3_all_fp3c16k_big.bin went) |
| **DELETE** `fpnet/fpnet_full1_f16.bin` | | from casmi26-rdkit-min (`DELETE_FROM_casmi26-rdkit-min.txt`) |
`g++ -std=c++20 -fsyntax-only` with the Kaggle include layout: **rc 0** (`C:\prog\casmi\v36\kcheck.log`). The full g++
(w64devkit) compile + link of phys_v36pc also ran inside the e2e notebook run.
After the upload: re-run casmi26-cpp-build (tree hash changes; it globs phys_v*.cpp so phys_v36pc is included; the
fpnet tree hash excludes *.bin so swapping the weights does not invalidate the fpnet prebuilt).

## End-to-end (e2e harness, `C:\prog\casmi\e2e`, fake competition = hcqry + hxqry, 800 molecules, library = zlib)
Holdout-safe substitutes under the production names (`py/e2e_run.py`): fp3 rev3_fp3c16k_big (wlib) for
rev3_all_fp3c16k_big (the earlier input tree linked the production fp3 — fixed: links are re-pointed), cnn_a_c.bin
(wlib) for cnn_prod_c.bin, fpown_w_f16.bin (wlib) for fpnet_own_f16.bin, fpnet sources copied (no public weights),
MSVC+Ninja for the fpnet cmake (`cpp/v36/e2e36.bat` runs vcvars64). Runs: `cpp/v36/e2e_v36.bat`, `e2e_v35.bat`.
Log check (v36): `own fpnet weights ... True`, `OWN FPNET stage done ... fpnet_own_f16.bin`, `fpnet term: + 1.0 z(fpnet)
(own weights)`, `| attn: index 57219915 fragment + 57219852 loss postings`, `| pubchem v33: alloc=boltz ... T1 3 T2 0.5 pi
0.1 ba 0.5`, `| cnn model ...cnn_prod_c.bin: 4096 bits, 1163 spectra in 44.6s, cnnpool 2`, `| cnninst 0: CNN term on 400
of 800 queries`, `| output + 1 z(fpnet): candidates with score 90864`, `final: rows 800 full lists 798`.

### Score (official identity, e2e_score.py; paired bootstrap e2e_pair.py, 2000 resamples)
| run | hcqry | hxqry | all 800 |
|---|---|---|---|
| v33nofp (e2e_STATUS) | 0.5576 | 0.8613 | 0.7094 |
| v35nofp | 0.5700 | 0.8714 | 0.7207 |
| **v36nofp** | **0.5804** [0.539, 0.619] | **0.8908** [0.867, 0.912] | **0.7356** |
| v36 − v35 | +0.0105 [−0.004, +0.024] (61 better / 27 worse) | **+0.0194 [+0.009, +0.031]** (29 / 6) | **+0.0149 [+0.006, +0.024]** |
| v35 − v33 | +0.0124 [−0.005, +0.031] | +0.0101 [−0.001, +0.022] | +0.0113 [+0.001, +0.021] |
ABLATION_PLACEHOLDER

### Wall time per stage (build machine, 4 cores; v36 / v35)
| stage | v36 | v35 |
|---|---|---|
| match compile + export + library search | 71 s | ~70 s |
| pool_candidate_rank + phys_rank compiles | 112 s + 14 s (first try; cached 1 s on the rerun) | 130 s + 15 s |
| FPNet stage (configure 8, build 8, ggml 113-122, copy 1, pool 2) | 116-142 s | — |
| phys_fp compile (phys_v36pc / v35pc, object-wise) | 36 s | 38 s |
| ranker | 1,270 s | 1,279 s |
| notebook total | 1,495 s | 1,540 s |
The CNN adds 45 s (1,163 timsTOF spectra), the FPNet term ~2 min here; on Kaggle expect ggml ~4-8 min, CNN ~1-2 min,
fpnet_pool on the local copy seconds (+ the sequential copy time of 933 MB from the mount).

## Files
- `py/make_v36_nb.py` (generator), `notebook/casmi_match_v36nofp.ipynb`, `notebook/kernel-metadata-v36nofp.json`.
- `cpp/rdkit-min/phys_v36pc.cpp` (+ cnninst), `cpp/v36/chkinst.bat`, `kcheck36.bat`, `e2e36.bat`, `e2e_v36.bat`,
  `e2e_v35.bat`, `abl36.bat`, `myprocs.ps1`.
- `py/e2e_run.py` (v36 substitutes, copies for EXTRA sources, cmake via MSVC/Ninja, E2E_MASK), `py/e2e_pair.py`,
  `py/e2e_rerank.py` (ranker-only ablations + notebook merge; merge check `py/e2e_remerge_check.py`: 0 / 800 differ),
  `py/v36_timeout_test.py`.
- Build machine: `C:\prog\casmi\e2e\work_v36`, `work_v35` (submission.csv, pool_candidates.csv, phys_rank_stdout.txt),
  logs `C:\prog\casmi\night\e2e_v36.log`, `e2e_v35.log`, `abl36.log`, `v36_chkinst.log`; first v36 try (failed phys_fp
  include via symlink, a harness artefact, exercised the PHYS_FP fallback) `C:\prog\casmi\v36\e2e_v36_try1.log`.

## Push steps (user)
1. Upload `C:\prog\casmi\v36\ds_add` contents to the datasets as in the table, delete fpnet/fpnet_full1_f16.bin.
2. Re-run casmi26-cpp-build; check `failed mains: none` and that phys_v36pc is in the manifest.
3. Push the v36nofp folder (notebook + kernel-metadata-v36nofp.json renamed kernel-metadata.json). Log lines to check:
   `PREBUILT ... used`, `own fpnet weights: ... True`, `[time] pool fp copy to local disk`, `[time] fpnet_pool` (seconds,
   not hours), `OWN FPNET stage done`, `phys_v36pc.cpp present: True`, `fpnet term: + 1.0 z(fpnet) (own weights)`,
   `| cnninst 0: CNN term on 400 of 400 queries`, `| attn: index`, `| pubchem v33: alloc=boltz ... pi 0.1`,
   `phys_rank ran in`, `final: rows 400`. Kaggle one kernel at a time.
