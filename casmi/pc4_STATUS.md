# pc4 — own fp3 reverse model for the PubChem window (phys_v32pc, v32 notebook)

2026-10-20. Motivation (fp3_STATUS.md): window MRR (Tanimoto scoring, clean models) hcqry / hxqry:
rev2_w (REV2 4096-bit, same recipe as production rev2_all_v4k) 0.254 / 0.299; **rev3_fp3c16k_big
(own 16384-bit fp3 vocabulary, 4096x2, 20 ep) 0.327 / 0.326**. Contrastive fine-tuning did nothing.
Goal: score the PubChem window candidates with the fp3 model in production.

## Files
- `cpp/rev3/prod_fp3.bat` (build machine `C:\prog\casmi\fwdmodel\rev3b\prod_fp3.bat`): production model
  `C:\prog\casmi\rev3_all_fp3c16k_big.bin` = rev3_fp3c16k_big recipe (fp3c16k vocabulary, H=4096,4096,
  20 epochs, hold=20 -> Platt calibration on the 5% held-out molecules) on **alllib** (all of train,
  275,810 molecules / 2.54M spectra; rev2_all_v4k used alllib with hold=0 — hold=20 here only so the
  trainer fits the calibration; Tanimoto scoring does not use it). Bits `C:\prog\casmi\alllib_fp3c16k.bin`
  (fp3_prep apply, 11 s). Logs `C:\prog\casmi\pc4\prod_progress.txt`, `prod_train.log`.
  Launched 15:12 detached (`night\launch.ps1 fp3_prod`, pid in `C:\prog\casmi\night\fp3_prod.pid`), ~35 min.
  Size: the wlib version is 549 MB (float32; 13016x4096 + 4096x4096 + 4096x16384 weights) -> expect ~550 MB.
- `cpp/rdkit-min/phys_v32pc.cpp` = phys_v30pc + argv[24] options
  - `pcfp=<model>` (relative to dir unless absolute): an fp3-vocabulary reverse model (rev3_train output;
    the file format is REV2's, so it loads with the unchanged `rev/rev2_model.hpp` — rev3_model.hpp only adds
    a training-time gradient hook and declares the same `namespace rev2` types, so it is NOT copied; including
    both would be a redefinition). Per query: mean p over the query's spectra (unclipped). PubChem window
    candidates: rev = continuous Tanimoto of fp3 bits (fp3::features -> vocabulary index), exactly
    cpp/rev3/win_eval.cpp. Pool candidates get the same fp3 score in `revPc` for the window statistics only
    (relz, z of the fused PubChem list's rev term); their main-fusion rev term stays REV2 (poolrev).
    Requires pc=1 and a REV2 model; otherwise ignored.
  - `pcfptop=N` (default 0 = all): fp3 only for the top N window candidates by the REV2 pcrev score.
  - log line: `pcfp <path>: fp3 reverse model, 16384 bits (fp3 v1); load+forward Xs (Y ms/query); window top N`
  - absent pcfp = phys_v30pc (every changed branch is guarded by pcFp; verified by run, below).
- `cpp/rdkit-min/rev/fp3_v32.h` = copy of cpp/rev3/fp3.h (includes the sibling `fp2.h`, identical to rev/fp2.h).
  On the build machine it is also in `C:\prog\casmi\fwdmodel\v27\rev\` (MSVC include path of bat_pc2/build.bat).
- `bat_pc4/run4.bat` = bat_pc3/run.bat with outputs in `C:\prog\casmi\pc4\`; `laneA..D.bat` the holdout runs;
  `kcheck.bat` the Kaggle staging + syntax check. Build machine copies in `C:\prog\casmi\fwdmodel\pc4\`.
  Build: `C:\prog\casmi\fwdmodel\pc2\build.bat phys_v32pc` (source in fwdmodel\pc2).
- `notebook/casmi_match_v32.ipynb` = v30 + `srcs_fp = ["phys_v32pc.cpp"] + ...` + `,pcfp=<POOL730_DIR>/rev3_all_fp3c16k_big.bin`
  (only if the file exists; else prints PCFP DISABLED and runs as v30). `notebook/kernel-metadata-v32.json`
  (id yomeiotani/casmi26-cpp-library-search-v32, dataset_sources as v30).
- Kaggle staging (NOT uploaded): `C:\prog\casmi\pc4\ds_add\rdkit-min\phys_v32pc.cpp`, `...\rdkit-min\rev\fp3_v32.h`,
  and (after training) `C:\prog\casmi\pc4\ds_add\pool730\rev3_all_fp3c16k_big.bin` -> add to dataset
  casmi26-rdkit-min (rdkit-min/, rdkit-min/rev/, pool730/). `g++ -std=c++20 -fsyntax-only` with the Kaggle
  include layout: **rc 0** (`C:\prog\casmi\pc4\kcheck.log`).

## Verification runs (launched 15:17, detached, 2 threads each, ylib / fwd_mlp_rank_ylib / rev2_y as pc2/pc3)
pcfp model for the holdouts = `rev3_fp3c16k_big.bin` (wlib: clean for hcqry/hxqry; hasqry query spectra excluded).
| lane | runs | baseline |
|---|---|---|
| A | hc32_1 (v32, rm=1), then hc32nofp_1 (v32 without pcfp, rm=1) + `fc /b` vs pc3\hcT1 -> pc4\eq_csv.txt, eq_ins.txt | pc3\hcT1 (v30 pcrev=tan rm=1) |
| B | hc32_0 (v32) | pc3\hcT0 |
| C | has30_0 (v30 pcrev=tan, hasqry) | – |
| D | has32_0 (v32, hasqry) | has30_0 |
Logs `C:\prog\casmi\night\pc4_lane?.log` ("done <tag>", "ALLDONE"); results = last lines of `C:\prog\casmi\pc4\<tag>.log`
(`MRR@25` plain, `pubchem slots: ... plain -> inserted`), timing line `pubchem: X thread-s (Y ms/query)`.
v30 reference (pc3): hcT1 rm=1 inserted **0.2152** (plain 0.1514; logit form hcD1 0.2014); hcT0 inserted **0.5255** (plain 0.5328);
pubchem 2.7 s/query (thread time, v30).

### Results (2026-10-20 ~15:50)
Production model: **`C:\prog\casmi\rev3_all_fp3c16k_big.bin`, 549,224,504 bytes** (2185 s; calibration 13,859 held-out
molecules, BCE/mol 809.5 -> Platt 755.6, bits F1>=0.5 3406/16384). Copied to `C:\prog\casmi\pc4\ds_add\pool730\`.
Sanity (`pc4\prod_wineval_hcqry.log`, leaky: alllib contains hcqry): window Tanimoto MRR_in 0.592, shuffled 0.032 -> loads and scores fine.

Holdout (MRR@25 by mrr.py on the final inserted list; "PubChem-list MRR" = truth's 1/rank inside the 50-long
fused PubChem list from the pcdet dump, `py/pc_det_rank.py`):
| set | v30 logit (hcD1) | v30 pcrev=tan | **v32 pcfp** | Δ v32−v30tan |
|---|---|---|---|---|
| hcqry rm=1, inserted | 0.2014 | 0.2152 (hcT1) | **0.2173** (hc32_1) | +0.002 |
| hcqry rm=1, truth among the 50 listed | 124 | 149 | **159** | +10 |
| hcqry rm=1, PubChem-list MRR@25 / top1 / top10 | 0.236 / 81 / 118 | 0.283 / 97 / 142 | **0.296 / 101 / 147** | +0.013 |
| hcqry (answer in pool), inserted | – | 0.5255 (hcT0) | 0.5256 (hc32_0) | +0.000 |
| hasqry (sibling), inserted | – | 0.8374 (has30_0) | 0.8362 (has32_0) | −0.001 |
Plain (no insertion) is identical by construction (0.1514 / 0.5328 / 0.8393).
Reading: the fp3 model improves the PubChem list itself (+10 truths listed, +0.013 list MRR on answer-removed hcqry), but
the final score moves little (+0.002) because the list is re-ranked by the hand fusion (pcrank=fused: phys/MLP/analog
+ z(rev)) and only a few gated slots are inserted. Check pcrank=rev (lanes E/F below) to see whether the fp3 gain
survives when the list order is the rev score itself.

Timing (thread-seconds, MSVC build, CPU): fp3 model load+forward **34.5 s per run = 86 ms/query** (400 queries);
PubChem stage v30 1.25 s/query (hasqry, window 6.9k) -> v32 2.19 s/query; hcqry 2.70 (v30) -> 2.31 s/query (v32;
noise from machine load — the fp3 bits replace REV2's fp2 bits for window candidates, so the extra cost is only the
non-ECFP fp3 kinds). Extra ≈ 0.14 ms per window candidate -> a 13k window ≈ +1.8 s/query single thread; 400 queries on
Kaggle's 4 threads ≈ +3 min. **No cap needed: default pcfptop=0 (every window candidate) chosen**; pcfptop=N is there if
the hidden set is much larger.

**Default = v30, verified**: v32 without pcfp on hcqry rm=1 (hc32nofp_1) vs pc3\hcT1: `fc /b` of both the plain csv and
the inserted csv: no differences (`pc4\eq_csv.txt`, `eq_ins.txt`). Same-load timing on that pair: PubChem stage
1.31 s/query (no pcfp) vs 2.31 s/query (hc32_1, pcfp) -> **+1.0 s/query for a 6.7k window (0.15 ms/candidate)**.

pcrank=rev (list ordered by z(rev)+0.15 pop only), hcqry rm=1:
| | v30 tan (hc30r_1) | v32 pcfp (hc32r_1) |
|---|---|---|
| inserted MRR@25 | 0.2096 | 0.2100 |
| PubChem-list MRR@25 / top1 / top10 | 0.245 / 80 / 139 | **0.281 / 92 / 146** |
-> with the rev order the fp3 gain in the list is +0.035 (matches the window yardstick), but the final score still
moves +0.000. pcrank=fused stays the better configuration (0.2173).

**Main finding**: the fp3 model makes the PubChem list better (fused: 0.283 -> 0.296, rev: 0.245 -> 0.281; truth
among the 50 listed 149 -> 159 of 400) but the final MRR is capped by the slot insertion (gated fixed slots
lo 4:8:12:16:20 / hi 2:4:6:8:10, promotion to rank 1 only at relz>3 or S>6&pop>=5): on hcqry rm=1 the list has the
truth at its rank 1 for 101/400 queries, yet the inserted MRR is 0.217. Next lever = the insertion rule (relz / S
thresholds were tuned on REV2 scores — the fp3 Tanimoto z has a different spread; re-tune slots/promotion with the
pcdet dumps, e.g. with pc2_grid.py-style simulation on hc32_1_det.txt / hc32_0_det.txt / has32_0_det.txt).
Expected LB effect of v32 as is: small positive (+0.002 on answer-removed, ±0.001 with the answer present).

RESULTS_PLACEHOLDER

## Resume
1. `ssh ... "type C:\prog\casmi\pc4\prod_progress.txt"` — "all done" -> `dir C:\prog\casmi\rev3_all_fp3c16k_big.bin`,
   `findstr calibration C:\prog\casmi\pc4\prod_train.log`; copy it to `C:\prog\casmi\pc4\ds_add\pool730\`.
   If the job died (check `Get-Process -Id (Get-Content C:\prog\casmi\night\fp3_prod.pid)`): relaunch
   `powershell -ExecutionPolicy Bypass -File C:\prog\casmi\night\launch.ps1 fp3_prod C:\prog\casmi\fwdmodel\rev3b\prod_fp3.bat`
   (skips finished steps).
2. Holdout results: `findstr /c:"MRR@25" /c:"pubchem" C:\prog\casmi\pc4\*.log`, equality `type C:\prog\casmi\pc4\eq_*.txt`
   ("no differences encountered" expected).
3. Kaggle (user): upload ds_add files into casmi26-rdkit-min, then push notebook v32 with kernel-metadata-v32.json.
