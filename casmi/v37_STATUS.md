# v37nofp — v36nofp + FT pool term (own pretrained transformer) (status file)

2026-10-22 afternoon. v37nofp = v36nofp (own FPNet + 1.0 z, CNN pool term cnnpool 2 / cnninst 0, attention attpool 2,
Boltzmann merge bT1 3 / bT2 0.5 / bpi 0.1 / ba 0.5, time guards, prebuilt hook) **+ 2 z(FT)**: our own masked-peak-pretrained
transformer fine-tuned on fp3c16k (pretrain_STATUS.md §8), CPU inference inside the ranker, same cnninst=0 gate as the CNN.
Every model is trained by our own code on train.parquet. Nothing uploaded / pushed / submitted.

## phys_v37pc (`cpp/rdkit-min/phys_v37pc.cpp` = phys_v36pc + options, all default off)
- `ftmodel=<TFM1>`: loads the fine-tuned model with the dataset's existing `rev/tfm_model.hpp` (same file as cpp/rev's
  apart from the fpnetPhase switch, default off = the code path pt_train_cpu2 / mollogits use; md5 e5be3e53… on the build
  include dir and in ds_rdkit) — **no new headers**. Per query: spectra with peaks (as mollogits), peaks/tag prepared
  with the existing tfmPrep/tfmTag (= pt_train prepPeaks/decodeTag), batches of 32 sorted by peak count, **p = sigmoid(mean
  logit over the query's spectra)** (what pool37ft.py simulated; not the mean of sigmoids). Candidate score = continuous
  Tanimoto sum_bits p / (n + sum p - sum_bits p) over the candidate's `fp3::features` that are in the model's own
  vocabulary (fp3c16k carried in the TFM1 file, rE >= 300, rF = FP3_VERSION checked).
- `ftpool=w`: + w z(ft) over the query's pool + generated candidates in the main fusion, next to cnnpool.
- `cnninst=k` gates FT as well (only queries with a spectrum of instrument k; inference skipped for the others).
- `ftshuf=k` (query q scored with q+k's p, control), `ftdump=file` (per-query mean logits, fpnet `_mol.bin` layout +
  names in file.txt), `ftbudget=s` (inference abandoned after s seconds -> `FT TIMEOUT ... no FT term`, run continues as v36).
- `cfdump` gets a 10th column ft only when ftmodel is given.
- Log lines: `ft model <path>: d512 L6 H8 HH2048, 16384 fp3 bits (fp3 v1), N spectra in Xs (.. ms/spectrum, T threads),
  ftpool 2, ftshuf 0` and `ft term on N of M queries (cnninst gate)`.

## Verification (build machine, `cpp/v37/`: build_v37.bat (MSVC), run37.bat, chk37.bat a|b, v37_check.py)
| check | result |
|---|---|
| default = v36: hbqry, chkb options (cnnpool 2, no ftmodel), `fc /b` vs v36\out\p36b_cnn2 | **no differences** on .csv, _ins.csv, _det.txt, _pp.txt |
| FT logits in the ranker (ftdump) vs GPU `pt_train mollogits` (ft8_ep5_hbqry_mol.bin), 400 x 16384 | max abs dlogit 6.5e-5, mean 3.8e-6, max abs dp 2.8e-6 |
| **C++ point vs pool37ft.py**: hbqry, ft8_ep5.bin, cnnpool 2 + ftpool 2 (pool36 config) | pool **0.8129** / final **0.8100** = simulation 0.8129 / 0.8100 exactly (cnn 2 alone 0.8058 / 0.8023) |
| ft inference speed, MSVC /arch:AVX2, 3 threads, a 2nd ranker in parallel | 1,156 spectra in 185 s (160 ms/spectrum) |
| ft inference speed, **g++ -O3 (notebook flags, w64devkit), 4 threads** (e2e) | **1,163 spectra in 58.9 s (50.7 ms/spectrum)** |

## Notebook (`notebook/casmi_match_v37nofp.ipynb`, `kernel-metadata-v37nofp.json`; generator `py/make_v37_nb.py`)
id yomeiotani/casmi26-cpp-library-search-v37nofp, title "CASMI26 cpp library search v37nofp", same dataset_sources /
kernel_sources as v36nofp. Changes vs v36nofp (all v36 guards kept):
- cell 7: `V37 = V36 and phys_v37pc.cpp exists` (`phys_v37pc.cpp present: ...`, else `FT DISABLED, running as v36`).
  phys_fp is compiled from phys_v37pc; **if that compile fails, phys_v36pc is compiled** (`FT DISABLED: phys_v37pc compile
  failed`) instead of falling to the v24 path.
- cell 8: `,ftmodel=<POOL730_DIR>/ft_all_ep5.bin,ftpool=2` when V37 and the file exists, else `FT MISSING (no
  ft_all_ep5.bin): running as v36` (or `FT DISABLED (no phys_v37pc)`). Log echo also shows `ft model`, `ft term`, `FT TIMEOUT`.
- Time: at the ranker call `ftbudget = min(40 min, ranker time left - 90 min)` is appended; under 5 min of budget the FT
  options are dropped (`FT SKIPPED`). The ranker deadline (t0 + 480 min) is unchanged. Measured FT cost on the build machine
  is ~1 min (above), so on Kaggle expect ~1-3 min, far below the 4-8 min feared from the MSVC number; the 40-min budget only
  matters if Kaggle is pathologically slow, and then the ranker drops the term and still finishes as v36.
- Ranker retries on a fast failure: without FT, then without FT and CNN, then the plain library list (as v36).

## End-to-end (e2e harness, hcqry + hxqry, 800 molecules, library zlib, 4 cores; `cpp/v37/e2e_v37.bat`)
Stand-ins as v36 (rev3_fp3c16k_big, cnn_a_c, fpown_w_f16, ...) + **ft8_ep5.bin** (pt_w + ft8 on wlib = holdout-safe) as
`pool730/ft_all_ep5.bin` (`py/e2e_run.py` EXTRA). Log (`C:\prog\casmi\night\e2e_v37.log`): `phys_fp compile (phys_v37pc.cpp)
rc: 0 in 36s`, `OWN FPNET stage done in 126s`, `| cnninst 0: CNN term on 400 of 800 queries`, `| ft model ...ft_all_ep5.bin:
... 1163 spectra in 58.9s`, `| ft term on 400 of 800 queries (cnninst gate)`, `ranker attempt 1 ... ftbudget=2400`,
`phys_rank ran in 1153s`, `final: rows 800 full lists 798`, notebook total 1,505 s (v36 1,495 s).

Official identity (e2e_score.py; paired bootstrap e2e_pair.py vs v36nofp's e2e submission, 2000 resamples):
| run | hcqry | hxqry | all 800 |
|---|---|---|---|
| v36nofp | 0.5804 | 0.8908 | 0.7356 |
| **v37nofp** | **0.5804** [0.539, 0.619] | **0.8943** [0.872, 0.915] | **0.7374** |
| v37 − v36 | +0.0000 (identical: cnninst, no timsTOF in hc) | +0.0035 [−0.003, +0.010] (9 better / 3 worse) | +0.0017 [−0.001, +0.005] |
Reading: on top of v36 (own FPNet + CNN already in the fusion, production merge bT1 3 / bpi 0.1) the FT term adds a small,
not significant +0.0035 on the timsTOF set — about half of the pool37ft estimate over cnn 2 (+0.0065 on hxqry, which had no
FPNet term and the bT1 2 / bpi 0.2 merge). FPNet, CNN and FT are three spectrum->fingerprint models and overlap. No set gets
worse; cost ~1 min. With the usual 0.3-0.5x holdout-to-LB transfer expect +0.001..+0.003 on the LB: a low-risk addition,
not a big step.

## Staging (`C:\prog\casmi\v37\ds_add\`, `cpp/v37/kcheck37.bat`) — NOT uploaded
| file | to |
|---|---|
| `rdkit-min\phys_v37pc.cpp` | casmi26-rdkit-min rdkit-min/ |
| `pool730\ft_all_ep5.bin` (production: pt_all + ft_all on alllib, epoch 5, 221,665,360 B, md5 264b1264…, written 14:27 by pt_prod) | the pool730/ folder holding pool730_head.bin / cnn_prod_c.bin |
Production model check (`py/tfm_hdr.py`): same TFM1 config (d512 L6 H8 ff4 HH2048 NB16384 maxPeaks 64) and the **same fp3c16k
vocabulary** (rE 300 rF 1, vocab md5 a15a6010…) as ft8_ep5.bin, weights finite; staged copy md5 = source md5.
CPU smoke test (`pt_train_cpu2 mollogits model=ft_all_ep5.bin qry=hxqry threads=4`, same forward code as phys_v37pc;
`py/mol_cmp.py`): 400 x 16384 logits, 0 non-finite, mean logit −14.7 (ft8_ep5 −13.8), mean sum p 389 (372), per-molecule
correlation of p with ft8_ep5 median 0.94 (min 0.62). (Not a score: ft_all saw the hxqry molecules.)
No new rev/ headers. **Prerequisite: v36\ds_add uploaded** (rev/cnn_model.hpp, cnn_layers.hpp, cnn_prod_c.bin,
fpnet_own_f16.bin, delete fpnet_full1_f16.bin) — phys_v37pc includes rev/cnn_model.hpp like phys_v36pc.
`g++ -std=c++20 -fsyntax-only` with the Kaggle include layout (ds_rdkit + v36 ds_add headers + phys_v37pc): **rc 0**
(`C:\prog\casmi\v37\kcheck.log`); the full g++ compile + link ran inside the e2e notebook run.
After the upload: re-run casmi26-cpp-build (tree hash changes; it globs phys_v*.cpp so phys_v37pc gets a prebuilt).

## Push steps (user)
1. Upload v36\ds_add (if not done) and v37\ds_add as in the tables.
2. Re-run casmi26-cpp-build; check `failed mains: none` and phys_v37pc in the manifest.
3. Push the v37nofp folder (notebook + kernel-metadata-v37nofp.json renamed kernel-metadata.json). Log lines to check:
   `phys_v37pc.cpp present: True`, `phys_fp compile (phys_v37pc.cpp) rc: 0`, `ft model: .../ft_all_ep5.bin True`,
   `ranker attempt 1 ... ftbudget=2400`, `| ft model ... spectra in Ns`, `| ft term on 400 of 400 queries (cnninst gate)`,
   `| cnninst 0: CNN term on 400 of 400 queries`, `fpnet term: + 1.0 z(fpnet) (own weights)`, `final: rows 400`.
