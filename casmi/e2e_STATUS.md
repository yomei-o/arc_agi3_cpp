# e2e — production path end-to-end on a holdout with known answers (status file)

2026-10-21 afternoon. Motivation: all local evaluation ran the C++ ranker in `eval` mode on holdout binaries; the
production path (parquet -> export_bin -> match.cpp -> ranker -> Python merge -> submission.csv) had never been
scored on molecules with known answers.

## Setup (build machine, `C:\prog\casmi\e2e`)
- `py/e2e_make.py C:\prog\casmi e2e\input\comp` — fake competition input from **hcqry (denovo) + hxqry (xinstr)**:
  - `train.parquet` = train minus every spectrum of the ha/hb/hc denovo molecules, minus the has/hbs sibling query
    spectra, minus the hx molecules' timsTOF spectra = **exactly zlib** (2,509,738 rows = zlib's spectrum count;
    all 972 + 1163 query spectra and 2151 sibling spectra matched byte-exactly, 0 unmatched).
  - `test.parquet` with the real test schema (12 columns, same arrow types), raw peaks, 2135 spectra / 800
    molecules, random `m_xxxxxx` ids; `sample_submission.csv`; `truth.tsv` (id, set, IK14, truth SMILES).
- `py/e2e_run.py <nb> [all|nb|eval]` — executes the code cells of `casmi_match_v33nofp.ipynb` with
  /kaggle/input and /kaggle/working rewritten to `e2e\input` / `e2e\work`. Input tree = file symlinks to
  `ds_rdkit` (the casmi26-rdkit-min staging) with the holdout-safe models under the production file names
  (`fwd_mlp_rank_ylib.bin` -> `rdkit-min/fwd_mlp_rank_v2.bin`, `rev2_z.bin` -> `pool730/rev2_all_v4k.bin`), PubChem
  tables from `pcds`, FPNet off (as v33nofp). Only code change: `.exe` in the `os.replace(phys_fp -> phys_rank)`.
  g++ (w64devkit) multi-source compiles are done object-wise with 4 jobs (same flags). Process pinned to 4 cores.
  Then the same `phys_rank.exe` in `eval` mode on the hcqry/hxqry binaries with library zlib and the notebook's
  exact argv/options (SMILES output).
- `py/e2e_score.py score|names|inspect` — official identity (stereo stripped -> RDKit canonical tautomer ->
  InChIKey[:14]), MRR@25 with bootstrap CI; `inspect` = sanity counts of a submission.

## Conclusion (first)
**The production path is not where the LB gap comes from.** Run end-to-end on 800 holdout molecules with known
answers, the notebook's submission.csv scores the same as (hxqry) or slightly better than (hcqry, +0.007) the
`eval`-mode CSV of the same executable, options, models and library. 779 / 800 lists are byte-identical; the 21
that differ come from eval-only truth-aware code (below), not from export / match / test mode / merge. No unparsable
SMILES, no missing rows, no placeholders, the official identity of the truth is found in the written SMILES.
The gap to the public pipelines (0.338 vs ~0.45) has to be explained by the hidden test vs our holdouts (candidate
coverage / molecule mix), not by a pipeline bug. Only a runtime risk (61 min of redundant single-threaded compiles)
was found and fixed in a copy.

## 1. End-to-end vs eval mode (official identity, n = 400 each, 95% bootstrap CI)
| run | hcqry official | hcqry plain IK14 | hxqry official | hxqry plain IK14 |
|---|---|---|---|---|
| **e2e submission.csv** (notebook v33nofp, all cells) | **0.5576** [0.518, 0.598] | 0.5292 | **0.8613** [0.834, 0.885] | 0.8613 |
| e2e pool_candidates.csv (ranker output alone) | 0.5576 | 0.5292 | 0.8613 | 0.8613 |
| same phys_rank.exe in `eval` mode on hcqry/hxqry bins, lib zlib, notebook argv, SMILES out | 0.5507 | 0.5218 | 0.8613 | 0.8613 |
| (reference) pc5 MSVC eval, names scored by mrr.py (hc: ylib) | — | 0.5250 | — | 0.8613 |
| e2e direct_submission.csv (match.cpp library list alone) | 0.0475 | 0.0000 | 0.4131 | 0.4131 |

- The merge adds nothing on top of the ranker (submission = pool list except where it is short: 9 empty + 13 short of 800 rows).
- Ranker log, e2e vs eval: generation products 603/query vs (1147 + 78)/2, kept 106.8 vs (146.3 + 67.7)/2,
  PubChem gate open 502 / 800 vs 300 + 200 — the channels that the visible test never exercises (gate 0 / 400 and
  0 popularity moves on Kaggle, because the visible molecules are in train) do run in the production path.
- The 21 differing lists (12 hc, 9 hx; `e2e_diff.py`): eval mode knows the truth and drops PubChem / generated copies of
  an in-pool truth ("the answer's pool entry (canonicalized differently) stays the only copy", phys_v33pc ~l.1577,
  1896). That also changes the PubChem list that enters the Boltzmann merge, so the whole interleave moves.
  Production keeps those copies, e.g. truth rank e2e 1 vs eval 14 (m_ea9cc9), 3 vs absent, 4 vs absent, 16 vs absent,
  2 vs 10, 2 vs 5, 1 vs 2; the other 14 have the same first truth rank; no row is worse in e2e. → **eval mode is slightly pessimistic for the
  PubChem channel** (hcqry −0.007); harmless for the LB, but holdout deltas of PubChem changes are a bit conservative.
- Holdout artifact: 9 hcqry queries get an empty pool list and 34 / 400 hcqry truths are outside the candidate set,
  mostly train-metadata spectra the hidden timsTOF test cannot have (precursor with 2 decimals, e.g. 557.45 for
  [M+H]+ 557.4565 = 12 ppm; `[M]+` FAB spectra; 'Ion Trap' 687.0 for M 650.33). hcqry (any instrument) caps lower than
  a timsTOF test would.

## 2. Real Kaggle outputs (visible test; `e2e_score.py inspect`, `C:\prog\casmi\e2e\inspect.log`)
| file | rows | <25 | unparsable | multi-comp. | charged | 'CCO' | official dup slots (rows) | repeated lists |
|---|---|---|---|---|---|---|---|---|
| kv30 submission (v30, LB 0.338) | 400/400 | 0 | 0 | 8 | 185 | 0 | 33 (26) | none |
| kv30 direct_submission | 400 | 0 | 0 | 0 | 79 | 0 | 16 (13) | none |
| k29out (v28a), kv31b (v31) | same as kv30 (fc /b: 14–18 differing bytes) | | | | | | | |
| kprobe1 (probe) | 400 | 0 | 0 | 0 | 770 | 0 | 19 (18) | none |
| e2e submission (800 holdout) | 800 | 2 (3, 18) | 0 | 8 | 494 | 0 | 227 (117) | none |
- Coverage 400/400 molecule_ids, no extras. Multi-component = pool/train salts (`...C(=O)O.Cl`, a two-drug mixture);
  charged = N-oxides / nitro / sulfoxides written as charge-separated forms (normal) and, in the probe, PubChem cations.
- v28a / v30 / v31 are the same list on the visible test because their differences (pool popularity, PubChem gate,
  LambdaMART on ...) are gated on libSim < 0.9, which the visible (= train) molecules never pass.

## 3. Code review of the production pieces
| # | item | status | impact on the test |
|---|---|---|---|
| a | adduct shifts (phys_v33pc / rev2 ADDUCT_SHIFT): [M+H]+, [M-H]-, [M+Na]+, [M+NH4]+, [M+K]+, [M+Cl]- (34.969402), formate (44.998201), 2M | verified correct | — |
| b | [M+H-H2O]+ shift −17.002740 should be −17.003289 (electron mass on the wrong side) | bug | 0: not in the test vocabulary (7 adducts: [M+H]+ 959, [M-H]- 193, formate 31, Na 22, NH4 4, K 2, Cl 2) |
| c | tag = adduct·16 + instr with −1 for unknown: [M+H]+ with an unknown instrument → tag −1 → "adduct unknown" (match.cpp: no adduct filter, instr 15); ranker falls back to the [M+H]+ shift (right by coincidence). Unknown adduct + known instrument → [M+H]+ shift even in negative mode | bug | 0: test is 100% timsTOF with known adducts |
| d | merge: `order = p_list + d_list`, exact-string dedup; `tight`, `best`, `K, THETA` are read but unused (dead since v19) | ok | library list only fills short pool lists |
| e | official-identity dedup (v28's RDKit step) is absent from v28b..v35 notebooks | regression | measured 0: e2e dedup+refill changes 117 / 800 rows, MRR hc 0.5576 → 0.5576, hx 0.8613 → 0.8613 (dups sit below the truth) |
| f | written SMILES (pool / generated rdkit-min canonical / PubChem): 0 unparsable, 0 without an official key among 19,971 e2e candidates; stereo kept on pool/PubChem, absent on generated — official identity strips it | ok | — |
| g | Platt: holdout rev2_y / rev2_z carry Platt calibration, production rev2_all_v4k does not ("Platt calibration in the model: 1" vs 0). Used only by pcrev=cal / poolrev=cal; v30/v33 use tan / logit | no effect (checked in code) | — |
| h | eval-only truth-aware skips (section 1) | asymmetry | production ≥ eval |
| i | **runtime**: three full single-threaded compiles of rdkit-min (Kaggle v30 log: pool_candidate_rank 1261 s, phys_rank 1180 s, phys_fp 1230 s = 61 min; pool_candidate_rank and phys_ana are never used when phys_v33pc builds). v31 was refused at 665 min > 540 | risk | fix 1 |

## Fixes (copies only; production notebooks/sources untouched)
- **Fix 1** `notebook/casmi_match_v33nofp_fix.ipynb` (made by `py/e2e_make_fix_nb.py`): `fast_gpp` compiles each source
  once into objects with 4 jobs (same flags) and reuses them for the three executables. Same sources and flags, so
  the same program. Build machine: 96 sources in 113 s with 4 jobs, the next two executables 14 s and 32 s.
  E2E check of the fix notebook (`work_fix\`, run_fix.log, 884 s total vs 915 s with the e2e harness's own
  parallel compile): pool_candidates.csv **byte-identical** to the unfixed run, submission hc 0.5576 / hx 0.8613
  (same). On Kaggle (4 CPUs) this should cut the ~61 min of compiles to roughly 10–15 min.
- Side finding (j): **match.cpp's library list is not deterministic in tie order**. Two runs on the same data gave
  different direct_submission.csv files (same MRR 0.0475 / 0.4131): `best` is an `unordered_map` reused per thread
  (`clear()` keeps the bucket count, so iteration order depends on which molecules the thread did before) and
  `std::sort` is not stable on equal scores. Affects only library fills of short rows (submission differed in ~150
  bytes, same MRR). Fix if wanted: sort by (score desc, molecule index asc).
- (b), (c) and (e) are not worth a submission (0 impact on this test); if ever touched: shift −17.003289; store adduct and
  instrument in separate fields (or tag = (a+1)·16 + (i+1)); dedup step = v28's cell.

## Files
- Scripts: `py/e2e_schema.py`, `py/e2e_make.py`, `py/e2e_run.py`, `py/e2e_score.py`, `py/e2e_cmp.py`, `py/e2e_diff.py`,
  `py/e2e_dedup.py`, `py/e2e_empty.py`, `py/e2e_make_fix_nb.py`.
- Build machine `C:\prog\casmi\e2e\`: input\comp (fake train/test/sample/truth), work\ (notebook run: submission.csv,
  pool_candidates.csv, direct_submission.csv, eval_hcqry_smiles.csv / eval_hxqry_smiles.csv + logs, nb_run.py =
  the executed code), e2e_*.csv copies, run.log, score_e2e.log, inspect.log, cmp.log, work_fix\ + run_fix.log.
- Re-run: `python e2e_make.py C:\prog\casmi C:\prog\casmi\e2e\input\comp` (2 min), then
  `python -u e2e_run.py <notebook> all` (15 min notebook + 12 min eval at 4 cores), then
  `python e2e_score.py score input\comp\truth.tsv work\submission.csv`. A new notebook version can be checked the
  same way before pushing (FPNet-free versions only; the FPNet stage needs cmake + ggml).
