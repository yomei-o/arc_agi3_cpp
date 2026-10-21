# pool36 — CNN / attention / fp3 as POOL-ranking terms on top of v35 (状態ファイル)

2026-10-21 夕方. Why: the probe (RESUME top) says only ~2–4 % of hidden answers are outside pool730, so the LB is decided by
ranking the ~100 pool candidates per query. Question: what do (1) w·z(CNN Tanimoto) and (2) the attention term at other
weights (attpool) add to the **pool list** of v35 (answer in pool, rm=0)? FPNet off; holdout-safe models as v35_STATUS
(ylib / zlib libraries, fwd_mlp_rank_ylib, rev2_y / rev2_z, pcfp rev3_fp3c16k_big trained on wlib); CNN = cnn_a_c.bin
(trained on wlib, REV2-style 4096 v4k vocabulary carried in the model file). Nothing uploaded / pushed to Kaggle / submitted.

## Code
- `cpp/rdkit-min/phys_v36pc.cpp` = phys_v35pc + options (all default off -> v35 behaviour):
  - `cnnmodel=<CNN1 file>`: CPU inference of the CNN (cpp/cnn/cnn_model.hpp, copied to the build include dir
    `C:\prog\casmi\fwdmodel\v27\rev\` next to rev2_model.hpp) on the query's spectra, mean p per bit; per candidate the
    continuous Tanimoto sum_bits p / (n + sum p - sum_bits p) over the candidate's bits in the **CNN file's own
    vocabulary** (fp2::Vocab from the model tail; independent of the REV2 model's vocabulary).
  - `cnnpool=w`: + w·z(cnn) over the query's pool + generated candidates in the main fusion (same place/form as attpool).
  - `cnnshuf=k`: query q scored with query q+k's CNN p (shuffled control).
  - `cfdump=file`: every candidate `name idx score isPool pop candname cnn att fp3` (score = final v35 main fusion).
- `cpp/v36/`: build_v36.bat, run36.bat (= run35 with v36 paths, outputs `C:\prog\casmi\v36\out\`), lane1-3.bat, shufx/shufb.bat.
- `py/pool36.py`: offline grid. score' = score + w z(cnn) + (a−2) z(att) + f z(fp3) → top 60 → pool-popularity re-fill and
  the v35 Boltzmann merge reproduced with py/boltz_merge.py (T1 2, T2 0.5, pi 0.2, relz a 0.5). Two lists: **pool** = main
  list top 25 (no PubChem), **final** = after the merge. The w=0 reconstruction is checked against the C++ CSVs.

## Verification
- phys_v36pc with `cnnmodel=cnn_a_c.bin,cfdump=...` and cnnpool 0 (= v35 config) vs the v35 runs, `fc /b`: **no differences**
  for hxqry (plain CSV, inserted CSV, pp dump vs v35x_0), hasqry (ins vs v35s_0), hcqry (ins vs v35_0), hcqry rm=1 (plain +
  ins vs v35_1). CNN inference inside phys: 1,163 spectra in 98 s at 2 threads (~85 ms/spectrum/2 threads).
- Offline reconstruction (`pool36.py`, w=0 from the C++ pp dump): pool list and final list equal the C++ CSVs for
  400/400 queries on all six runs. Re-sorting the every-candidate dump itself (6-digit scores) moves hcqry rm=0 by
  +0.0004 (near ties) and nothing elsewhere; deltas below are vs the C++ lists.
- C++ check of one non-zero point (hbqry, cnnpool=2, run p36b_cnn2) and the shuffled-query controls (cnnshuf=7 on hxqry
  and hbqry): see "Shuffled-query control" below.

## Results (rm=0 = answer in pool unless noted; MRR@25 via boltz_merge's rr = mrr.py names; paired bootstrap 2000)
Raw tables: `casmi/pool36/grid1.txt` (hx, hb, hbs), `grid2.txt` (has, hc0, hc1). "pool" = main list top 25 (no PubChem),
"final" = after the v35 Boltzmann merge. PERM-control = CNN values permuted within the query (destroys the
spectrum-candidate link, keeps the distribution).

### (1) + w z(CNN Tanimoto), final list, delta vs v35 [95% CI]
| set (v35 final) | w 0.5 | w 1 | **w 2** | w 3 | w 4 | PERM w 1 | PERM w 2 |
|---|---|---|---|---|---|---|---|
| **hxqry** (0.8731) | +0.0053 [+0.0015,+0.0102] | +0.0076 [+0.0011,+0.0148] | **+0.0109 [+0.0020,+0.0207]** | +0.0126 [+0.0026,+0.0236] | +0.0088 | +0.0037 | −0.0022 |
| **hbqry** (0.7811) | +0.0044 [−0.0018,+0.0118] | +0.0139 [+0.0050,+0.0238] | **+0.0212 [+0.0091,+0.0336]** | +0.0223 [+0.0090,+0.0359] | +0.0151 | −0.0080 | −0.0420 |
| hbsqry (0.8781) | +0.0057 [−0.0004,+0.0130] | +0.0109 [+0.0013,+0.0207] | **+0.0126 [+0.0008,+0.0251]** | +0.0122 | +0.0155 | −0.0054 | −0.0227 |
| hasqry (0.8457) | −0.0015 [−0.0060,+0.0014] | −0.0016 [−0.0086,+0.0052] | −0.0045 [−0.0125,+0.0029] | −0.0037 | −0.0060 | −0.0170 | −0.0305 |
| hcqry (0.5361) | +0.0022 [−0.0035,+0.0085] | −0.0074 [−0.0170,+0.0018] | −0.0130 [−0.0279,+0.0021] | −0.0146 | −0.0201 [−0.0391,−0.0007] | −0.0171 | −0.0520 |
| hcqry rm=1 (0.2648) | −0.0003 | +0.0036 [−0.0005,+0.0083] | +0.0064 [−0.0032,+0.0166] | +0.0078 | +0.0071 | −0.0016 | −0.0165 |
Pool-list deltas are the same within ±0.002 (hbqry pool: +0.0189 [+0.0072,+0.0312] at w 2).

**Reading.** The CNN is a real pool-ranking term on the three **timsTOF** sets (hxqry, hbqry, hbsqry: +0.011 / +0.021 /
+0.013 at w 2, all CIs above 0; the permutation control is ≤ 0 and strongly negative on hb/hbs → the gain is
spectrum-specific, not a candidate-size/ distribution artefact). It **loses on hcqry / hasqry** (−0.013 / −0.005 at w 2).
**hcqry and hasqry contain no timsTOF spectrum at all** (head tags: instrument 1/15/4/3/2 and unknown; hbqry 1153/1156
timsTOF), and hcqry has the train-metadata quirks of e2e_STATUS. The hidden test is timsTOF (Enveda) → the
timsTOF sets are the relevant ones. The CNN's hxqry window advantage (cnn_STATUS: 0.546 vs 0.299) was also a timsTOF
result. Weight: 2–3 flat on hx/hb, w 2 is the hand-set choice (w 3 is not better on hbs/hc and is the edge of the plateau).
Mixture yardstick of v35 (0.5 has + 0.42 hc + 0.08 hc1): w 0.5 +0.0002, w 2 −0.0072 — the old yardstick is all
non-timsTOF and says no; the timsTOF sets say yes.

### (2) attention weight attpool (v35 = 2), final list delta vs attpool 2
| set | attpool 0 | 1 | 3 | 4 |
|---|---|---|---|---|
| hxqry | −0.0117 [−0.0234,−0.0006] | −0.0014 | +0.0024 [−0.0024,+0.0076] | +0.0051 [−0.0020,+0.0125] |
| hbqry | −0.0057 | −0.0024 | −0.0068 [−0.0157,+0.0024] | −0.0141 [−0.0279,−0.0003] |
| hbsqry | −0.0216 [−0.0343,−0.0089] | −0.0078 | +0.0044 [−0.0015,+0.0118] | +0.0038 |
| hasqry | −0.0063 | −0.0064 | −0.0035 [−0.0076,−0.0004] | −0.0071 |
| hcqry | −0.0074 | +0.0025 | −0.0107 [−0.0203,−0.0022] | −0.0197 |
| hcqry rm=1 | −0.0018 | +0.0011 | +0.0014 | +0.0033 |
**attpool 2 stays** — the attention term is worth keeping (0 loses on hx −0.012, hbs −0.022), 3–4 helps hx/hbs a little but
costs hbqry and hcqry. Combined with the CNN (final, delta vs v35):
| | hxqry | hbqry | hbsqry | hasqry | hcqry | hc rm=1 |
|---|---|---|---|---|---|---|
| attpool 2 + cnn 2 | +0.0109 | **+0.0212** | +0.0126 | −0.0045 | −0.0130 | +0.0064 |
| attpool 1 + cnn 2 | +0.0089 | +0.0169 | +0.0078 | −0.0046 | −0.0057 | +0.0040 |
| attpool 3 + cnn 2 | +0.0074 | +0.0130 | +0.0131 | −0.0029 | −0.0149 | +0.0096 |
| attpool 1 + cnn 1 | +0.0040 | +0.0137 | +0.0017 | −0.0038 | −0.0010 | −0.0011 |
→ no reason to move attpool when the CNN is added; **attpool 2 + cnn 2** is the best timsTOF point.

### (3) + f z(fp3 Tanimoto) (same 16k fp3 big model as the PubChem window), final delta
| set | f 0.5 | f 1 | f 2 | cnn 1 + fp3 1 |
|---|---|---|---|---|
| hxqry | +0.0026 | +0.0026 | +0.0016 | +0.0060 |
| hbqry | −0.0003 | +0.0012 | −0.0036 | +0.0117 |
| hbsqry | +0.0016 | +0.0046 | +0.0040 | +0.0120 |
| hasqry | −0.0006 | −0.0048 [−0.0106,−0.0006] | −0.0078 | −0.0054 |
| hcqry | −0.0066 [−0.0136,−0.0006] | −0.0079 | −0.0166 | −0.0099 |
| hc rm=1 | −0.0007 | −0.0005 | +0.0011 | +0.0056 |
fp3 adds ~nothing on the pool (the REV2 term z(rev) is already in the fusion and fp3 is the same kind of MLP); cnn + fp3
is worse than cnn alone everywhere → **do not add fp3 to the pool fusion**. (4) z(physics within top-N) was not done
(physics is not in the dump; it is already the z(phys) base term).

### Shuffled-query control (cnnshuf=7) and C++ check of w 2
Launched 17:46 (a first launch at 17:22 failed on a shell-quoting slip, nothing ran), detached, 2 threads each:
`pool36_shufx` (hxqry, cnnshuf=7, cnnpool 0 -> `v36\out\p36x_0_shuf_cf.txt`), `pool36_shufb` (hbqry -> `p36b_0_shuf_cf.txt`),
`pool36_chkb` (hbqry with cnnpool=2 in C++ -> `v36\out\p36b_cnn2_ins.csv`; ~21 min each, expected ~18:07).
To finish: copy `p36x_0_shuf_cf.txt`, `p36b_0_shuf_cf.txt` next to the other dumps and re-run pool36.py (rows
"cnn w SHUF-query control"); for the C++ check compare the simulated final list at cnn 2 with `p36b_cnn2_ins.csv`
(expected MRR 0.8023 final / 0.8058 plain, `findstr MRR v36\out\p36b_cnn2.log`).
**Shuffled-query control (C++ cnnshuf=7: query q scored with query q+7's CNN p, same candidates; `pool36/grid3.txt`)**, final delta vs v35:
| set | SHUF w 1 | SHUF w 2 | SHUF w 3 | real CNN w 2 |
|---|---|---|---|---|
| hxqry | −0.0009 | −0.0047 [−0.0144,+0.0035] | −0.0135 [−0.0255,−0.0021] | **+0.0109 [+0.0020,+0.0207]** |
| hbqry | −0.0078 | −0.0231 [−0.0387,−0.0082] | −0.0508 | **+0.0212 [+0.0091,+0.0336]** |
→ another molecule's CNN prediction hurts; the gain is query-specific.
**C++ check of cnnpool=2 NOT done**: chkb.bat had a printf-escape bug (``, `
` in the path), it ran nothing. Fixed
`cpp/v36/chkb.bat` is on the build machine; run `powershell -ExecutionPolicy Bypass -File C:\prog\casmi
ight\launch.ps1 pool36_chkb C:\prog\casmi36\chkb.bat`
(~21 min) and compare `findstr MRR v36\out\p36b_cnn2.log` / its `_ins.csv` with the simulation (final 0.8023, pool 0.8058).
The simulator is exact at w=0 on all six runs and the C++ term is the same zscore add as attpool, so a match is expected,
but float-vs-6-digit rounding can flip a few near ties.

## Recommendation
- For a timsTOF test the measured pool-ranking gain of **+2 z(CNN)** (cnn_a_c, CPU ~30–85 ms/spectrum) is +0.011 to
  +0.021 MRR on the three timsTOF holdouts, larger than anything else measured today on the pool. If the LB is ~0.34
  because hidden pool rankings are hard, expect a smaller absolute gain (holdout→LB transfer has been 0.3–0.5x).
- Safer variant for mixed instruments: apply the CNN term only to queries with a timsTOF spectrum (instrument tag
  0) — on these holdouts that equals "w 2 on hx/hb/hbs, 0 on hc/has" (not separately implemented; one `if` in the fusion).
- Kaggle: cnn_a_c.bin is trained on wlib (holdout-safe). Production needs a CNN trained on all of train (the night job
  cnn_night may produce one) + cnn_model.hpp/cnn_layers.hpp in the dataset's rdkit-min/rev/ + `cnnmodel=...,cnnpool=2`.

## Resume
- Runs: build machine `C:\prog\casmi\v36\out\` (p36{x,b,bs,s,c}_0, p36c_1 + `_cf.txt` every-candidate dumps), logs
  `C:\prog\casmi\night\pool36_*.log`. Re-run the grid: copy `<tag>*` and `<qry>_mols.txt` into a folder ->
  `python casmi/py/pool36.py <dir> <dir>\out p36x_0:hxqry,p36b_0:hbqry,p36bs_0:hbsqry` (`POOL36_TIMS=1` = subset column
  of timsTOF queries; `<tag>_shuf_cf.txt` next to a dump adds the shuffled-query rows).
- Build: copy cpp/cnn/cnn_model.hpp + cnn_layers.hpp to `C:\prog\casmi\fwdmodel\v27\rev\`, `cpp/v36/build_v36.bat`.
