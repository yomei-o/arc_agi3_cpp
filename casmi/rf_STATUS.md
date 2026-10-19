# rf — rank-level fusion with a second engine (status)

2026-10-19. Task: the top public pipelines gain about +0.015 LB from fusing two engines' lists with
`1/(3+r1) + 0.6/(3+r2)` (one vote per identity) plus "engine-2 top-1 at rank 2" (E2R2) (gap_STATUS #3).
Try the same thing with hand-set rules (no learned weights), using only lists we already compute,
and measure on the holdout2 components (holdout2_STATUS: yardstick = 0.5·hasqry + 0.42·hcqry + 0.08·hcqry-rm,
components reported separately, no loss on sibling/xinstr allowed).

## Conclusion

- **Plain RRF as the public pipelines do it LOSES with every second list we have.** With the library list
  (match.cpp), `1/(3+r1)+0.6/(3+r2)` gives sibling +0.013 but denovo hcqry −0.097, xinstr hxqry −0.058,
  timsTOF denovo hbqry −0.174; mixture −0.037 [−0.047, −0.028]. The library list is wrong by construction
  whenever the answer has no spectrum in the library (denovo, answer-removed) or only spectra from another
  instrument (xinstr), and RRF gives it half the vote in every case. Lighter weights (k1 w0.3) are still
  negative or zero on every non-sibling component (hc −0.010, ha −0.009, hb −0.007, hx −0.000; mixture +0.000).
- **Single-engine lists cut out of our own ranker (REV2 alone, physics alone, MLP alone, library/analog terms
  alone) are not a second engine.** They are the parts the main list already fuses, each much weaker
  (hcqry: main 0.533, REV2 0.354, physics 0.420, MLP 0.376, library/analog terms 0.448). RRF with any of them is
  about 0 at best (k1 w0.3: mixture within ±0.001; the one significant cell is physics k1 w0.3 on hbqry +0.005
  [+0.002, +0.009], not seen on the other components) and clearly negative at w ≥ 0.6 (hasqry −0.01 to −0.14).
- **What passes: E2R2 with the library list, gated on the library's own confidence.** The library top-1 goes to
  rank 2 only when match.cpp's best score ≥ 1.0 (best_score.csv; entropy similarity + 0.25 same-instrument
  bonus − mass prior, so ≥ 1.0 means about "same instrument, similarity ≥ 0.75"):

| component | base | gated E2R2 (θ=1.0) | Δ [95% paired bootstrap] | gate open |
|---|---|---|---|---|
| hasqry (sibling, COCONUT, any instr) | 0.8544 | 0.8708 | **+0.0164** [+0.009, +0.025] | 281/400 |
| hbsqry (sibling, timsTOF) | 0.8584 | 0.8618 | +0.0034 [+0.000, +0.007] | 212/400 |
| hxqry (xinstr, timsTOF) | 0.8615 | 0.8615 | **+0.0000** [0, 0] | 52/400 |
| hcqry (denovo, popularity-matched) | 0.5328 | 0.5323 | −0.0005 [−0.002, +0.000] | 68/400 |
| hcqry-rm (answer removed) | 0.1514 | 0.1513 | −0.0001 [−0.000, +0.000] | 68/400 |
| haqry (denovo, COCONUT) | 0.5773 | 0.5752 | −0.0021 [−0.004, −0.000] | 67/400 |
| hbqry (denovo, timsTOF) | 0.7795 | 0.7793 | −0.0002 [−0.001, +0.000] | 70/400 |
| **mixture S** (0.5 has + 0.42 hc + 0.08 hc-rm) | 0.6631 | 0.6711 | **+0.0080** [+0.0040, +0.0125] | |
| mixture X (hxqry in place of hasqry) | 0.6667 | 0.6664 | −0.0002 | |

  Passes the gate (no loss on sibling or xinstr). The gain is entirely from the same-instrument sibling
  case; on xinstr the gate rarely opens and when it does the library top-1 is never right (0/52), and the move
  costs nothing because rank 2 of the main list is rarely the answer there. **Expected LB effect is
  therefore +0.008 × ~0.47 ≈ +0.004 if the hidden test's library-findable molecules look like hasqry,
  and 0 if they look like hxqry** — within LB noise either way (one molecule ≈ ±0.004–0.008).
  Our library-only v6 scored 0.121 on the LB, so some hidden molecules are library-findable.
- θ was picked from {0.9, 1.0, 1.1, 1.2} on these holdouts (hand-set, but a 4-value choice is still a choice):
  0.9 gives a larger mixture gain (+0.011) but costs hxqry −0.0013 and haqry −0.003; 1.1/1.2 are smaller but
  just as safe (+0.0067/+0.0049, xinstr 0). 1.0 is the lowest value with xinstr exactly 0.
- Gated "library top-1 at rank 1" (the old v13 rule): mixture +0.011 but hxqry −0.009 and hbqry −0.007 at θ=1.0
  → fails the gate. Gated RRF (k1 w1.0, θ ≥ 1.1): mixture +0.005 but small losses on every non-sibling
  component → E2R2 is better.
- Depth: our main list 40 deep vs 25 deep (production) gives the same numbers within 0.002 (D25 rows).

## Notebook

`notebook/casmi_match_v29rf.ipynb` = casmi_match.ipynb (v28a) with a post-step in cell 7 (only cell 7 changed):
`RF_MODE = "e2r2"`, `RF_GATE = 1.0` (`"rrf"` with RF_K/RF_W and `"off"` = the v28a merge exactly are kept as
switches). Standard library only, no new inputs: it reads pool_candidates.csv, direct_submission.csv and
best_score.csv (all already written), and works on any pool_candidates.csv (e.g. the LambdaMART ranker's).
One place per identity: InChIKey14 looked up from the SMILES through pool730_mols.txt / train_mols.txt (both
already in /kaggle/working), else the SMILES with stereo marks removed (generated candidates). **Not** the
official tautomer-insensitive identity (needs RDKit); tautomer twins can still take two places. Prints
"v29rf: library gate open N, lists changed by the fusion M".
All code cells pass `ast.parse`. Not run on Kaggle, not pushed, not submitted.

Checked end to end (`py/rf_nbtest.py`): cell 7's rf_* functions, executed on the holdout lists converted to
production SMILES (pool730 names → SMILES, library list as match.cpp writes it, 25 deep), reproduce the rf_eval
numbers exactly (hasqry +0.0164, hcqry −0.0005, hcqry-rm −0.0001, hxqry 0, haqry −0.0021, hbqry −0.0002,
hbsqry +0.0034).

The C++ was **not** needed for the chosen rule. `cpp/rdkit-min/phys_v29rf.cpp` (phys_v28pc + `outn=N` deeper
out.csv and `alt=prefix` single-engine lists; both off = phys_v28pc output) exists for the measurements only; the
notebook still builds phys_v28pc.

## Setup (build machine, C:\prog\casmi\rf)

- Main list: `bat_rf/run.bat` = phys_v29rf eval, v28a configuration (phys_size_fqry.txt, libcap, generation,
  REV2, ppop 0.15 gated at 0.9; FPNet off; holdout-safe fwd_mlp_rank_ylib.bin + rev2_y.bin, rev2_z.bin for hxqry
  on zlib), names, 40 deep, + `<tag>_{rev,phys,mlp,ana}.csv`. Main list alone: has 0.839 (= pc2 0.839),
  hc 0.533, hc-rm 0.151, hx 0.862, ha 0.577, hb 0.780, hbs 0.858.
- Library list: `bat_rf/match.bat` = the notebook's match.cpp (byte-identical to cpp/match.cpp, w64devkit g++),
  cell-4 arguments but 40 deep, on ylib (zlib for hxqry). Alone: hasqry 0.869, hbsqry 0.814, hxqry 0.451,
  denovo 0 (the answer is not in ylib).
- `py/rf_eval.py` — every rule on every component, paired bootstrap (2000), mixtures; table in
  `rf_results/eval2.tsv`. Base = main list then library list, one place per InChIKey14 (= the notebook merge with
  identity dedup).
- `bat_rf/all1.bat`, `all2.bat` — the runs (about 4 min per phys run at 6 threads, match 2 cores).
