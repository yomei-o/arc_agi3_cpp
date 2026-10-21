# formula — candidate fragment formulas explain the query peaks ("output side", formula_PLAN.md)

2026-10-21. Design: formula_PLAN.md (output side first). Window yardstick: fp3_STATUS.md. Bar to beat: fp3 + attention
(attn_STATUS.md) window MRR_in hcqry 0.454 / hxqry 0.635 / haqry 0.418. All own C++ (CPU, MSVC); Python only to summarise.

## Short answer
1. **The subformula score clears the +0.02 bar over fp3 + attention on both sets** — expl_E@20ppm at weight 0.5
   (weight chosen on haqry): hcqry **+0.071**, hxqry **+0.033**, haqry +0.041 (top 1000 re-ranked; shuffled-query
   control alone 0.003-0.009, and as an addend it *costs* 0.03-0.10).
2. **But it is not new information: the physics model's own likelihood (phys_ll, the existing term of the pool fusion)
   used the same way does as well or better** — base + 0.5 z(phys_ll): hcqry +0.066, hxqry +0.047, haqry +0.050. Inside
   the top N the formula features correlate r = 0.7-0.9 with phys_ll, and **added on top of phys_ll they give
   +0.000 to +0.005 (ha / hc) and +0.008 to +0.013 (hx)**, about one SE, and negative on haqry at haqry's best
   phys weight. The key question ("does the subformula score add information beyond fp3 + attention?") — yes beyond
   fp3 + attention, **no (not measurably) beyond the physics term that already exists**.
3. The actionable finding is therefore that **the PubChem window order of the best FPNet-free pipeline
   (v34 `pcrank=rev` = z(fp3) + z(attn) + 0.15 pop) does not use the physics likelihood at all**, and adding it
   inside the window's top N is worth +0.05-0.07 window MRR_in without popularity and **+0.026 / +0.047 / +0.040
   (ha / hc / hx) with the pipeline's 0.15 pop** (weight 0.25, chosen on haqry). Integration plan below (for
   phys_v35pc; not edited here).

## What was built (cpp/formula/)
- `win_dump.cpp` (CPU build of attn_eval's scoring path): z_win(fp3 rev3_fp3c16k_big) + z_win(attention ment k10 T0.1
  over fp3c16k, attn_eval's mpen=4 neighbour cache, library wlib) for every window candidate, the shuffled attention
  (q <- q+7) and popularity; dumps each query's top 1000 by z(fp3) + z(attn) to `C:\prog\casmi\formula\<qry>_top.bin`.
  **Reproduces attn_STATUS exactly**: fp3 + attn 0.4543 / 0.6354 / 0.4181 (hc / hx / ha), fp3 0.3267 / 0.3262 / 0.2804.
- `phys_core.h`: verbatim copy of phys_size.cpp lines 55-397 (= phys_v33pc's bond classes, `enumerate()`, Params,
  `predict()`, `loglik()`; the two files differ by one comment). Parameters phys_size_fqry.txt.
- `formula_eval.cpp`: for each query x top-N candidate: SMILES (w_<qry>_wmols.txt) -> `enumerate()` (intact, every
  non-ring single cut, pairs of the 12 most labile cuts, two-bond ring openings) -> `predict()` per query spectrum
  (H shifts k = -2..2 with the fitted k weights, class barriers exp(-E), proton-affinity share, adduct / polarity).
  Every predicted ion is a sub-formula of the candidate (exact monoisotopic mass of the cut sub-graph +- k H, minus the
  electron), so ions with the same exact mass (1e-5 Da) are merged into one **formula with summed weight W_F**.
  Query peaks (>= 0.5 % of the base peak, m/z < precursor - 0.5, so the precursor that every isomer explains is out)
  are annotated with the candidate formulas within tolerance (5 / 10 / 20 ppm with a 0.001 Da floor, and 0.01 Da).
  Features (mean over the molecule's spectra):
  `expl_raw` intensity-weighted explained fraction; `expl_sqrt` same with sqrt intensity; `expl_E` energy-weighted:
  each explained peak counts W/(W + W_median) of its best formula (a peak explained only by a high-barrier fragment
  counts little); `miss` fraction of the predicted weight of the 10 strongest formulas (inside the observed m/z range)
  with no peak = the penalty; `nexpl` fraction of the 20 strongest peaks explained (count); `expl_raw - lam miss`;
  and the control **`phys_ll`** = phys_v33pc's `physOf()` (background-mixture log-likelihood, tol 0.01 Da, eps =
  exp(logEps), range = precursor), plus phys_ll at 0.005 / 0.003 Da. Everything also with the spectra of query q+7
  (shuffled-query control). Features cached (`formula\<qry>_feat_N1000.bin`), so re-evaluation takes seconds.
  Evaluation as win_eval: MRR@25 with pessimistic ties, MRR_in over truth-in-window queries, MRR_all over all 400
  (truth outside the top N = 0); paired d vs the base with SE and queries up / down.
- Speed: 76k candidates (hcqry top 200) in 98 s on 4 threads, 337k-400k (top 1000) in 280-300 s — ~5 ms per candidate
  including both spectra sides and all features; phys_ll alone is a fraction of that.
- `py/formula_sum.py <log> [N]`: per feature the best weight + its shuffled control, and the a/b grid.

## Truth recall in the top N of fp3 + attention (of truth-in-window queries)
| N | 25 | 50 | 100 | 200 | 300 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|---|---|---|
| hcqry (339 in window) | 0.779 | 0.838 | 0.870 | 0.903 | 0.917 | 0.920 | 0.935 | 0.959 |
| hxqry (348) | 0.876 | 0.908 | 0.925 | 0.954 | 0.966 | 0.977 | 0.991 | 0.994 |
| haqry (345) | 0.765 | 0.803 | 0.852 | 0.887 | 0.899 | 0.910 | 0.922 | 0.942 |
N = 1000 was used for the main numbers (results at N = 200 are within ~0.005, table 2b); the gain does not depend on
reaching deep (the rescued truths mostly sit at ranks 26-200).

## Results (window MRR, re-rank of the top 1000; base = z(fp3) + z(attn) = 0.4181 / 0.4543 / 0.6354 MRR_in ha / hc / hx)
Logs `C:\prog\casmi\formula\fe_<qry>_N1000.log` (features + single-feature table), `fe2_<qry>.log` (a/b grid),
`fpop_<qry>.log` (with popularity).

### 2a. Each feature alone and added to the base (d = MRR_in difference vs base; ha / hc / hx)
| feature | alone (in the top N) | SHUF alone | base + 0.25 z | base + 0.5 z | base + 1 z | SHUF: base + 0.5 z |
|---|---|---|---|---|---|---|
| expl_raw@10ppm | 0.097 / 0.153 / 0.184 | 0.006 / 0.004 / 0.001 | +0.026 / +0.050 / +0.024 | +0.020 / +0.052 / +0.027 | +0.004 / +0.042 / +0.024 | -0.039 / -0.039 / -0.084 |
| expl_sqrt@10ppm | 0.100 / 0.152 / 0.195 | 0.006 / 0.004 / 0.001 | +0.029 / +0.051 / +0.031 | +0.024 / +0.048 / +0.039 | +0.011 / +0.039 / +0.027 | -0.034 / -0.036 / -0.072 |
| expl_E@10ppm | 0.127 / 0.207 / 0.275 | 0.008 / 0.008 / 0.003 | +0.029 / +0.058 / +0.024 | +0.034 / +0.059 / +0.033 | +0.014 / +0.036 / +0.022 | -0.038 / -0.040 / -0.113 |
| **expl_E@20ppm** | 0.116 / 0.201 / 0.260 | 0.008 / 0.009 / 0.003 | +0.040 / +0.070 / +0.025 | **+0.041 / +0.071 / +0.033** | +0.023 / +0.049 / +0.026 | -0.029 / -0.037 / -0.099 |
| expl_E@0.01Da | 0.110 / 0.192 / 0.245 | 0.006 / 0.009 / 0.005 | +0.042 / +0.068 / +0.022 | +0.038 / +0.066 / +0.032 | +0.025 / +0.051 / +0.019 | -0.030 / -0.030 / -0.080 |
| miss@10ppm (penalty alone) | 0.044 / 0.091 / 0.124 | 0.005 / 0.007 / 0.004 | +0.016 / +0.021 / +0.023 | -0.005 / +0.007 / +0.003 | -0.061 / -0.040 / -0.049 | -0.043 / -0.040 / -0.160 |
| nexpl@10ppm | 0.082 / 0.132 / 0.182 | 0.003 / 0.003 / 0.004 | +0.031 / +0.045 / +0.028 | +0.030 / +0.047 / +0.039 | +0.010 / +0.030 / +0.019 | -0.023 / -0.036 / -0.065 |
| expl_raw - 0.5 miss @20ppm | 0.107 / 0.179 / 0.232 | 0.006 / 0.007 / 0.004 | +0.030 / +0.056 / +0.028 | +0.028 / +0.059 / +0.039 | +0.008 / +0.048 / +0.025 | -0.029 / -0.039 / -0.073 |
| **phys_ll (control)** | 0.136 / 0.220 / 0.288 | 0.012 / 0.014 / 0.003 | +0.043 / +0.072 / +0.035 | **+0.050 / +0.066 / +0.047** | +0.036 / +0.058 / +0.051 | -0.028 / -0.039 / -0.069 |
| phys_ll @0.005 Da | 0.143 / 0.232 / 0.303 | 0.010 / 0.013 / 0.003 | +0.040 / +0.070 / +0.036 | +0.049 / +0.067 / +0.052 | +0.034 / +0.059 / +0.054 | -0.022 / -0.034 / -0.079 |
| phys_ll @0.003 Da | 0.144 / 0.227 / 0.298 | 0.009 / 0.012 / 0.003 | +0.037 / +0.059 / +0.035 | +0.040 / +0.058 / +0.051 | +0.021 / +0.042 / +0.053 | -0.023 / -0.033 / -0.079 |
SE of d is 0.008-0.015 (paired over ~340 queries). Other tolerances / variants (5 ppm, sqrt, -0.25/-1 miss) are in the
logs and are no better. Reading:
- Formula vs mass: 5 ppm < 10 ppm < 20 ppm ~ 0.01 Da for every feature, and phys_ll at 0.003 Da is worse than at 0.01 /
  0.005 Da: **the query spectra's mass accuracy does not reward a formula-tight tolerance** (in the timsTOF set hx,
  0.003 Da is as good as 0.01 Da; in the mixed-instrument COCONUT sets it loses 0.01).
- Energy weighting helps (expl_E > expl_raw by +0.01-0.02); the absent-fragment penalty alone is weak and hurts at
  weight >= 0.5; explained-minus-penalty is about expl_E.
- phys_ll is the strongest single re-ranker on ha and hx and ties expl_E on hc.

### 2b. Same at N = 200 (d of base + 0.5 z; ha / hc / hx)
expl_E@20ppm +0.039 / +0.065 / +0.032; nexpl@10ppm +0.031 / +0.045 / +0.038; phys_ll +0.049 / +0.067 / +0.043.

### 3. Is it just the old physics term? base + a z(phys_ll) + b z(formula); dLL = paired d vs base + a z(phys_ll)
(N = 1000; `fe2_<qry>.log`, grid a 0.25-1.5, b 0.125-1)
| | haqry | hcqry | hxqry |
|---|---|---|---|
| base + 0.25 ll | 0.4614 (+0.043) | 0.5267 (+0.072) | 0.6705 (+0.035) |
| base + 0.5 ll (best a on haqry) | **0.4677 (+0.050)** | 0.5206 (+0.066) | 0.6825 (+0.047) |
| base + 0.75 ll | 0.4606 | 0.5229 | 0.6869 (+0.052) |
| base + 1 ll | 0.4541 | 0.5123 | 0.6863 |
| 0.25 ll + 0.25 expl_E@20ppm, dLL | 0.4668, +0.005 (0.005) | 0.5293, +0.003 (0.006) | 0.6773, +0.007 (0.005) |
| 0.25 ll + 0.25 nexpl@10ppm, dLL | 0.4662, +0.005 (0.004) | 0.5309, +0.004 (0.006) | 0.6834, +0.013 (0.005) |
| 0.5 ll + 0.25 expl_E@20ppm, dLL | 0.4581, -0.010 (0.004) | 0.5229, +0.002 (0.003) | 0.6815, -0.001 (0.004) |
| 0.5 ll + 0.25 nexpl@10ppm, dLL | 0.4617, -0.006 (0.003) | 0.5235, +0.003 (0.005) | 0.6905, +0.008 (0.005) |
| 0.5 ll + 0.25 (expl_raw - 0.5 miss)@20ppm, dLL | 0.4573, -0.011 (0.005) | 0.5245, +0.004 (0.004) | 0.6891, +0.007 (0.005) |
| shuffled formula: 0.5 ll + 0.25 SHUF expl_E@20ppm, dLL | -0.014 | -0.005 | -0.015 |
The best total on haqry is phys_ll alone (a = 0.5); with formula added the best is 0.4668-0.4662 (a 0.25, b 0.25) =
the same. On hcqry nothing beats ll alone by more than 0.004; on hxqry nexpl adds +0.008-0.013 (1.5-2.5 SE, one set,
grid-picked). Mean within-query correlation with phys_ll over the top 1000: expl_E@0.01Da 0.85 / 0.83 / 0.91,
expl_E@10ppm 0.72 / 0.69 / 0.84, nexpl@10ppm 0.57 / 0.56 / 0.70, miss -0.35 / -0.35 / -0.53 (ha / hc / hx).

### 4. With the pipeline's popularity (base + 0.15 pop = 0.5352 / 0.5191 / 0.6276; d vs that; `fpop_<qry>.log`)
| | haqry | hcqry | hxqry |
|---|---|---|---|
| + 0.25 z(phys_ll) (best a on haqry) | **+0.026** (0.008) | **+0.047** (0.010) | **+0.040** (0.008) |
| + 0.5 z(phys_ll) | +0.023 | +0.047 | +0.057 |
| + 0.25 z(expl_E@20ppm) (formula only) | +0.025 | +0.046 | +0.027 |
| + 0.25 ll + 0.25 expl_E@20ppm | +0.028 | +0.052 | +0.053 |
| + 0.25 ll + 0.25 nexpl@10ppm (best total on haqry) | +0.031 | +0.053 | +0.052 |
-> with popularity in, the formula term on top of 0.25 ll adds +0.002-0.005 / +0.005-0.006 / +0.013 (ha / hc / hx) —
the same sign on all three sets, but small, ~0.5-1 SE, and below the +0.02 bar. Resulting window MRR_in with pop:
ha 0.535 -> 0.561 (ll) / 0.566 (ll + nexpl), hc 0.519 -> 0.566 / 0.572, hx 0.628 -> 0.667 / 0.680.

### 5. Parameter set control (phys2_fqry.txt = the older main-pool physics parameters, instead of phys_size_fqry.txt)
`go_p2.bat`, logs `fp2_<qry>.log`, features `<qry>_feat_N1000_p2.bin`. Same picture, d at N = 1000 (ha / hc / hx):
base + 0.5 z(phys_ll) +0.052 / +0.069 / +0.047; base + 0.5 z(expl_E@20ppm) +0.042 / +0.071 / +0.029; nexpl is
parameter-free (identical). Formula on top of ll: a 0.25 + 0.25 expl_E dLL +0.011 / +0.003 / +0.005; a 0.5 + 0.25 expl_E
-0.011 / +0.001 / +0.000; a 0.5 + 0.25 nexpl -0.004 / +0.004 / +0.010. The conclusion does not depend on the parameter set.

## How the subformula score differs from the physics likelihood (and why it adds so little)
Both read the same fragment table: `enumerate()` produces cut sub-graphs; a sub-graph's atoms fix its formula, and the
predicted m/z is the exact mass of that formula +- k H. Matching a peak to a fragment mass within 0.01 Da (phys_ll)
and annotating the peak with a candidate sub-formula within 10 ppm (this note) are the same operation at different
tolerances — **the "formula" was always in the physics model, implicitly, as an exact mass**. The differences are only
in the scoring functional:
- phys_ll = sum over peaks of sqrt-intensity x log((1 - eps) m(peak) / total predicted + background): a peak counts in
  proportion to the share of the candidate's *total* predicted intensity that it receives. The normalisation by the
  total is an implicit penalty for fragments predicted but not seen (a candidate that spreads intensity over many
  unobserved fragments loses on every explained peak) and keeps "more fragments = more explained" in check.
- the explained fraction counts coverage (max over formulas, no normalisation): an isomer with more cleavable bonds
  explains more by chance; the energy weighting (expl_E) and the explicit miss term partly put back what the
  likelihood's normalisation does, and the better they do it the more they correlate with phys_ll (r 0.85-0.91 at
  0.01 Da).
- the formula version allows a tighter tolerance — which does not help with these spectra (table 2a).
So the measured result is the expected one: inside a window of same-formula isomers the discriminating information is
*which sub-graphs exist*, and phys_ll already uses it. The formula *annotation* would add information only through
something phys_ll does not model: e.g. peaks whose sub-formula is impossible for a candidate under *any* cut
(a hard constraint, not a soft likelihood), multi-step rearrangement products (`k` beyond +-2, ring cleavages with
H transfer — QM ipso / H-transfer pathways, HOME_QM.md), or formula-level priors learned from the library
(input side of formula_PLAN.md). The output-side score as specified here is a re-parameterisation of the physics term.

## Integration plan (phys_v35pc; do not edit — the other agent owns it)
Not the formula score (no measurable gain over phys_ll) but **the physics likelihood in the PubChem window order**,
which v34's best config (`pcrank=rev`) skips (`full = pcV != 2 || pcRankFused` = false -> physOf not called; the order
is z_win(fp3 + attn) + 0.15 pop). Proposal for a v35 option `pcphys=a[:M]` (default off = v35 unchanged):
1. After `all` is sorted by `pre` = z_win(score) + pcw pop and before truncation to `pcPre`, take the top M
   (default 300 = pcPre; window yardstick: N = 200 and 1000 give the same gain) candidates, parse each, `physOf(*m)`
   (the existing lambda: enumerate + predict + loglik with P0 over the query's spectra), z-score phys over these M
   (NaN / unparsable -> min), and re-sort them by `pre + a * z_M(phys)`, **a = 0.25** (chosen on haqry with
   pop; 0.5 if pop is off). Keep `rev`, `pcS`, relz, promotion and slot logic computed from the unchanged REV score
   (they gate on calibrated z(score) + pop; changing them would need re-tuning), only the order of the listed picks
   changes. The SMILES parse and `canon()` of the dedupe loop can be shared with the physics pass (parse once).
2. Optional second term `+ 0.25 z_M(nexpl@10ppm)` (+0.005 / +0.006 / +0.013 window MRR with pop, ~1 SE): only
   if a pipeline A/B on hcqry rm=1 shows it; code = `featOne()` of formula_eval.cpp (F_NEXPL at 10 ppm), ~40 lines.
3. Cost (estimate from formula_eval's ~5 ms per candidate for all features on both spectrum sides): physOf ~1-3 ms per candidate per spectrum -> 300 candidates ~0.5-1 s per query on one thread (PubChem stage
   was 2.9-3.7 s/query in v34), memory nil. M = 100 would keep most of the gain (N = 100: phys_ll +0.07 on hc at 0.25).
4. Validation, in this order: (a) v35 with the option off == v35 (fc /b on CSVs, as done for v34); (b) hcqry rm=1 with
   `attn=1,pcrank=rev,attpool=2,pcphys=0.25` vs without: inserted MRR and pc_det_rank.py list MRR (expect list MRR
   +0.03-0.05 from the window numbers); (c) hcqry and hasqry with the answer in the pool (expect ~0: the PubChem list
   only fills slots); (d) holdout2 mix. If the list gains but the inserted MRR does not, the cap is the slot rule
   (attn_STATUS lever (a)), not the order.
Expected LB effect: as for attention, the window gain is diluted by the insertion rule — attention's +0.13 window
MRR became +0.019 final on hc rm=1; by the same ratio +0.05 window -> ~+0.007 final on answer-outside-pool queries.

## Resume
- Rebuild: `ssh ... C:\prog\casmi\formula\build.bat` (copies rev2_model.hpp / backend.hpp / parallel.hpp from ..\attn;
  rdkit-min headers from fwdmodel\rdkit-min, libs from fwdmodel\ana\build\rdkitmin).
- Top-N dumps exist for hcqry / hxqry / haqry (`formula\<qry>_top.bin`, N = 1000); features cached
  (`formula\<qry>_feat_N1000.bin`, phys_size params; `_p2.bin` phys2 params). Re-evaluate in seconds:
  `formula\formula_eval.exe qry=hcqry N=1000 "evalN=200,1000" [popgrid=1]`; summarise `python casmi/py/formula_sum.py <log> [N]`.
- Full run from scratch: `formula\run_all.bat <N> "<evalN list>"` (or detached via `night\launch.ps1 <name> formula\go1000.bat`).
- Ideas not done: hard "impossible sub-formula" constraint per candidate; k beyond +-2 / rearrangement products from
  the QM pathways; the input side (formula tokens into the reverse model).
