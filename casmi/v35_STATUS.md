# v35 — Boltzmann merge (v33) + attention engine (v34), FPNet-free

2026-10-21. v35 = phys_v35pc with `pcfp=<fp3 model>,attn=1,pcrank=rev,attpool=2,alloc=boltz,bT1=2,bT2=0.5,bpi=0.2,ba=0.5`
(+ the fixed v30 options pcrev=tan, ppop 0.15 / ppgate 0.9). No FPNet anywhere. Nothing uploaded / pushed / submitted.

## Code
- `cpp/rdkit-min/phys_v35pc.cpp` = 3-way merge (`git merge-file`) of phys_v33pc and phys_v34pc over phys_v32pc; the only
  conflict was the header comment. The hunks are disjoint: v34 changes the PubChem window scores (pcfp -> z(fp3) +
  attw z(att)), the list order (pcrank=rev) and the main fusion (attpool); v33 only changes the final interleave
  (`alloc=boltz`, from the main list's final scores — which include attpool — and the PubChem list's `fused` key, which
  under pcrank=rev is z(rev) + 0.15 pop = the list order). LF line endings.
- `cpp/v35/`: `build_v35.bat` (MSVC, build machine `C:\prog\casmi\v35\`), `run35.bat` (= attn/run34.bat, outputs
  `C:\prog\casmi\v35\out\<tag>.*`), `laneA/B/C.bat` (verification + dumps), `laneF1/F2/F3.bat` (final config),
  `kcheck35.bat` (Kaggle staging + syntax check).
- `py/v35_tune.py`: re-tune of the merge on the v35 dumps with the exact simulator of `py/boltz_merge.py` (grid T1 x T2 x
  pi x a, side gate, 2-fold, paired bootstrap vs v33 / v34). `py/make_v35_nb.py`: notebook generator.

## Verification (hcqry rm=1, `fc /b` on both the plain and the inserted CSV: "no differences" for all six)
| run | options | reference |
|---|---|---|
| vdef_1 | `rm=1,pcrev=tan,pcfp=rev3_fp3c16k_big.bin` (no new option) | pc4\hc32_1 (phys_v32pc) — identical |
| vbo_1 | `rm=1,pcrev=tan,alloc=boltz,bT1=2,bT2=1,bpi=0.2,ba=0.25` | pc5\hc33r_1 (phys_v33pc) — identical |
| vatt_1 | `rm=1,pcrev=tan,pcfp=...,attn=1,pcrank=rev,attpool=2` | attn\pc\hc34rp2_1 (phys_v34pc) — identical |
(`C:\prog\casmi\v35\out\eq_{def,bo,att}_{csv,ins}.txt`)

## Measurement
Holdout-safe models as attn_STATUS / pc5: ylib library, fwd_mlp_rank_ylib, rev2_y (hxqry: zlib + rev2_z), pcfp =
rev3_fp3c16k_big.bin (trained on wlib), no FPNet. MRR@25 by mrr.py on the written CSVs. Mixture = 0.5 hasqry + 0.42
hcqry + 0.08 hcqry rm=1 (holdout2 yardstick); side gate = no loss (>= -0.0005) on hasqry / hbsqry / hxqry vs v34.
v33 references: hc33r_0 / hc33r_1 (C++), the other sets simulated from the v30 dumps with the v33 setting (the simulator
reproduces C++ exactly: hcT0 0.5250 = hc33r_0, has 0.8390, hx 0.8613, ha rm1 0.2483, hbs 0.8583 = pc5's table).
v34 = the attn runs' own inserted CSV (fixed slots); v35 candidates = the simulator on those runs' dumps.

### attpool (v34 slots, final list; v35-merge rows below are on the same dumps)
| attpool | hcqry | hc rm=1 | hasqry | mix (v34) | best-merge mix |
|---|---|---|---|---|---|
| 1.5 | 0.5316 | 0.2439 | 0.8447 | 0.6652 | 0.6685 |
| **2** | **0.5319** | 0.2435 | **0.8462** | **0.6660** | **0.6697** |
| 2.5 | 0.5233 | 0.2438 | 0.8454 | 0.6620 | 0.6658 |
attpool 2 stays (1.5/2.5 runs only for hc / hc rm1 / has; haqry rm1, hbsqry, hxqry at attpool 2).

### Merge re-tune at attpool 2 (600 configs: T1 {1,1.5,2,3,4} x T2 {0.5,..,3} x pi {0.05..0.5} x a {0,0.25,0.5,1}, relz form)
| config | mix | hcqry | hc rm=1 | ha rm=1 | hasqry | hbsqry | hxqry |
|---|---|---|---|---|---|---|---|
| v33 (v30 dumps, REV2 list) | — | 0.5250 | 0.2467 | 0.2483 | 0.8390 | 0.8583 | 0.8613 |
| v34 attpool 2 (slots) | 0.6660 | 0.5319 | 0.2435 | 0.2736 | 0.8462 | 0.8781 | 0.8730 |
| v33 setting 2/1/0.2/0.25 on the attn scale | 0.6687 | 0.5394 | 0.2368 | **0.2323** | 0.8464 | 0.8795 | 0.8731 |
| best mix 3/0.5/0.1/0.5 (conservative) | 0.6697 | 0.5408 | 0.2396 | 0.2533 | 0.8467 | 0.8783 | 0.8731 |
| **chosen 2/0.5/0.2/0.5** | 0.6691 | 0.5357 | **0.2648** | **0.2788** | 0.8457 | 0.8780 | 0.8731 |
| 3/1/0.3/0.5 (fails side gate: hbs −0.0017) | 0.6698 | 0.5369 | 0.2631 | 0.2772 | 0.8465 | 0.8764 | 0.8731 |
Reading: v33's setting does NOT transfer to the attention scale (haqry rm=1 −0.041 vs v34, CI [−0.059, −0.024]); the
attention list is sharper, so the PubChem temperature has to come down (T2 1 -> 0.5) and the relz slope up (0.25 -> 0.5).
2-fold (seeded split, fit by mixture with side gate on one half): the pure-mixture fits pick the conservative pi=0.1
setting on both halves (held-out mix +0.002/+0.003) but it loses haqry rm=1 in both halves (−0.016 / −0.024). The chosen
setting was picked on full data as the best mixture among configs that pass the side gate AND gain on both answer-removed
sets (same logic as pc5's choice, i.e. it bets on a non-negligible f_out); its per-half deltas vs v34 are stable:
mix +0.0031 / +0.0029, hc rm1 +0.031 / +0.011, ha rm1 +0.007 / +0.004, hcqry +0.002 / +0.006, sides within ±0.0005.
(Selection on the full data with the extra hc1/ha1 condition is a mild optimism; the per-half consistency is the check.)

### Paired bootstrap 95% CI (2000 resamples) — chosen v35 (simulated on the attpool-2 dumps)
| set | v35 − v34 | v35 − v33 |
|---|---|---|
| hcqry | +0.0038 [−0.0021, +0.0104] | +0.0107 [−0.0067, +0.0277] |
| hcqry rm=1 | **+0.0213 [+0.0059, +0.0379]** | +0.0181 [+0.0003, +0.0362] |
| haqry rm=1 | +0.0052 [−0.0129, +0.0245] | **+0.0305 [+0.0080, +0.0524]** |
| hasqry | −0.0005 [−0.0042, +0.0034] | +0.0067 [−0.0027, +0.0164] |
| hbsqry | −0.0001 [−0.0038, +0.0037] | **+0.0198 [+0.0075, +0.0328]** |
| hxqry | +0.0001 [+0.0000, +0.0003] | **+0.0118 [+0.0007, +0.0236]** |
| mixture | +0.0031 [−0.0002, +0.0065] | **+0.0093 [+0.0002, +0.0180]** |
Gate (sibling / xinstr) passed vs both. Most of v35's gain over v33 on the sibling/xinstr sets comes from v34's
attention (main-list attpool + PubChem list), not the merge; the merge adds +0.021 on hc rm1 over v34's slots.

FINAL_CPP_PLACEHOLDER

## Kaggle readiness (nothing uploaded)
- Staged on the build machine `C:\prog\casmi\v35\ds_add\` (`cpp/v35/kcheck35.bat`):
  - `rdkit-min\phys_v35pc.cpp`, `rdkit-min\rev\fp3_v32.h` -> dataset **casmi26-rdkit-min** (rdkit-min/, rdkit-min/rev/;
    fp3_v32.h is the same file v32/v33/v34 staged — skip if already uploaded).
  - `pool730\rev3_all_fp3c16k_big.bin` (549,224,504 B, copy of `C:\prog\casmi\rev3_all_fp3c16k_big.bin`, the production
    fp3 model trained on all of train, pc4) -> dataset **casmi26-pool-expanded**, folder pool730/ (next to
    pool730_head.bin; the notebook looks for it at `POOL730_DIR / "rev3_all_fp3c16k_big.bin"`).
  - `g++ -std=c++20 -fsyntax-only` with the Kaggle include layout (ds_rdkit + staged files): **rc 0** (`C:\prog\casmi\v35\kcheck.log`).
- Notebook `notebook/casmi_match_v35nofp.ipynb` (generated by `py/make_v35_nb.py` from casmi_match_v33nofp_pb.ipynb; FPNet
  off, prebuilt hook kept) + `notebook/kernel-metadata-v35nofp.json` (id yomeiotani/casmi26-cpp-library-search-v35nofp,
  same dataset_sources and kernel_sources as v33nofp_pb). Cell 7 compiles phys_v35pc.cpp if present (else phys_v33pc /
  phys_v30pc as before). Cell 8, when phys_v35pc and the fp3 model are both present: pcrank=fused -> pcrank=rev and
  `,pcfp=<POOL730_DIR>/rev3_all_fp3c16k_big.bin,attn=1,attpool=2,alloc=boltz,bT1=2,bT2=0.5,bpi=0.2,ba=0.5`; otherwise
  prints `ATTN DISABLED` and appends v33's merge (= v33 output, verified above). Option-string logic dry-run for all four
  presence combinations; all cells ast-parse.
- Build kernel: `py/prebuilt_build.py` links every `phys_v*.cpp` found in the dataset's rdkit-min/ (glob), so phys_v35pc
  needs no list change. **Re-run casmi26-cpp-build after the dataset upload** (the tree hash changes -> every prebuilt
  would miss and fall back to compiling, ~13 min+).
- Log lines to check: `phys_v35pc.cpp present: True`, `fp3 model: ... True`, `phys_fp compile rc: 0` (or PREBUILT used),
  `v28 options: ...pcrank=rev...pcfp=...attn=1,attpool=2,alloc=boltz,bT1=2,bT2=0.5,bpi=0.2,ba=0.5,pcrev=tan`, in the
  phys_rank stdout `attn: index ... postings`, `pubchem v33: alloc=boltz ... T1 2 T2 0.5 pi 0.2 ba 0.5`.
- Runtime estimate: on the build machine (2 threads, 400 hcqry rm=1 queries, shared machine) a whole run took 16 min for
  v33 (vbo_1), 20 min with pcfp (vdef_1), 29 min with attn+attpool (vatt_1): enumeration wall 905 -> 1650 s, PubChem
  522 -> 1109 thread-s. Kaggle (4 threads, 400 molecules / 1,213 spectra = ~1.25x the holdout spectra, slower cores):
  **+8 to +15 min** over v33nofp's phys_rank stage; index build ~5 s, model load ~5-20 s. Far below the 540 min limit
  (caveat: v31 hit 665 min when three kernels shared Kaggle's queue — run v35 alone).
- Memory: peak working set on the build machine 6.2-6.3 GB (v32/v33 runs) vs 7.9-8.0 GB (attn runs) at 2 threads:
  +1.8 GB inverted index (2 x 57.3M postings) + 0.55 GB fp3 model + ~80 MB per thread accumulators -> **~+2.5 GB on
  Kaggle (4 threads)**, total ~9 GB for the phys_rank process, within the 30 GB of a CPU notebook.

## Resume
- Re-run tuning: dumps `C:\prog\casmi\v35\out\{hcap*,hasap*,haap2_1,hbsap2_0,hxap2_0}_{pp,det}.txt` + `_ins.csv`, the
  `<qry>_mols.txt`, v33 reference CSVs (pc5\hc33r_0/1_ins.csv as v33_hc0/hc1.csv; the rest simulated from pc4\has30_0,
  pc5\hx30_0, ha30_1, hbs30_0) in one folder -> `V35PICK="T1=2 T2=0.5 pi=0.2 a=0.5" python casmi/py/v35_tune.py <dir> 2 15 25`.
- LB steps (user decision): upload ds_add to the two datasets, re-run casmi26-cpp-build, push casmi_match_v35nofp
  (folder with the notebook + kernel-metadata-v35nofp.json renamed kernel-metadata.json), check the log lines above.
- If the probe gives a small f_out (< 0.1), the conservative merge `bT1=3,bT2=0.5,bpi=0.1,ba=0.5` is the safer pick
  (change BOLTZ35 in py/make_v35_nb.py and regenerate): hcqry +0.009 vs v34, but ha rm1 −0.020.
