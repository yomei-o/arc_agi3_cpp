# Structure generation from spectral analogs — status (2026-10-14, stopped: build machine handed to FPNet)

Build machine dir: C:\prog\casmi\fwdmodel\gen\ (private). New files in C:\prog\casmi: pg_{t,c}{A,B}_{head.bin,mols.txt}, gen_f_{t,c}{A,B}.txt. Nothing shared was modified.

## Files (fwdmodel\gen)
- ana_dump.cpp/.exe: standalone analog search (= phys_ana stage1+2 variant 4: modified cosine, >=6 matched peaks, |dPrec|<=250, K=8000), top 200 analog molecules/query -> ad_cqry.txt, ad_tqry.txt (~1 min each, 20 threads)
- gen_recall.py (+gr_sum.py): Phase 1 recall -> gr_cqry_80b.jsonl, gr_tqry_80.jsonl
- gen_cands.py: generated candidates (key14, smiles, mass, bestSim, bestRank, minSteps, nSupport, ops), cap 1000/query -> gc_cqry.txt, gc_tqry.txt
- build_genpool.py, run_gen_eval.bat: pools A = pool730 + gen, B = pool730 - holdout answers + gen; runs phys_ana (ana\build, read-only use) -> gen_f_*.txt, pa_*.log
- gen_fuse.py: conditional logit (8 production features + 6 gen features), train cqry -> test tqry and reverse. gen_fuse3.log = the leak-free run (gen_fuse.log / gen_fuse2.log had a pool-membership leak, ignore)

## Phase 1 (K=80 analogs, T = +-OH, +-Me, +-Ac, +-Hex, +-dHex, +-Pen(pyr/fur), +-Api, +-GlcA, +-Mal, +-SO3, +-prenyl, +-H2, +-H2O, +-N-oxide; one- and two-step)
| | recall 1-step | recall 1+2-step | cands/query median / mean |
|---|---|---|---|
| cqry (400) | 0.282 | 0.360 | 41 / 714 |
| tqry (176) | 0.494 | 0.562 | 64 / 188 |
Answer among the first 25 generated candidates (ordered by analog sim): cqry 0.287, tqry 0.477.

## Phase 2 (pool-augmentation eval, leak-free gen_fuse3.log)
| test | in pool: no gen | in pool: +gen | not in pool: no gen | not in pool: +gen | 50/50 mix: no gen -> +gen |
|---|---|---|---|---|---|
| tqry (train cqry) | 0.781 | 0.774 | 0 | 0.327 | 0.390 -> 0.550 |
| cqry (train tqry) | 0.658 | 0.615 | 0 | 0.187 | 0.329 -> 0.401 |
Naive merge (production weights, no gen features): in pool falls to 0.62 / 0.47 -> gen features are required.
Absolute in-pool numbers are higher than production because this fit puts a large weight on inLib (holdout artifact); compare differences only.

## Next steps
1. Re-fit with fuse_fit-like regularisation / without inLib to check the gain holds; add an op-type prior (one-step vs two-step, per-op frequency) and a mass-window check (B pool only kept 99/176 and 129/400 answers vs Phase 1 recall 0.56/0.36: cap 1000 + the ±10 ppm pool window vs the 0.003 Da Δ tolerance).
2. C++ port phys_gen.cpp (copy of phys_ana.cpp in fwdmodel\gen, own CMakeLists): after the analog search, generate candidates with RWMol edits (rdkit-min has GraphMol + SmilesParse + SmilesWrite), dedupe by stereo-blind ECFP hash / canonical SMILES, append to the query's candidate list with gen features, score with the existing models.
3. Kaggle runtime estimate: generation in Python ~7.5 s/query (cqry; tqry much less) -> ~13 min for 400 queries on 4 cores; C++ is probably 5-10x faster. Scoring 3.3x more candidates: + ~9 min on 4 cores. Total +15-25 min, well within budget.
