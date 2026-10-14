# rev2 (reverse model v2) status — 2026-10-14

## Result (fused = phys_ana formula z(phys)+0.25 z(mlp)+wr z(rev)+8 lib^32+24 anaP, ana var4 p3)
Eval = rev2_train.exe recomputing only the rev column of phys_ana feature dumps f21_cqry/f21_tqry.txt
(pool730 candidates). Reproduces phys_ana exactly: rev_x3072 -> cqry 0.5500 / tqry 0.6124.

| model | rev alone cqry/tqry (pool730 cands) | fused wr=1 cqry/tqry | best wr |
|---|---|---|---|
| (no rev, wr=0) | - | 0.5366 / 0.6228 | |
| rev_x3072.bin (current) | 0.3715 / 0.3402 | 0.5500 / 0.6124 | |
| A rev2_eh.bin (ECFP2048, same arch, 95% data, 12 ep) | 0.3713 / 0.3164 | 0.5488 / 0.6103 | |
| A + Platt (OOF) | 0.4120 / 0.3841 | 0.5546 / 0.6172 | |
| **B rev2_vh.bin (vocab 4096: ECFP r0-3 + FCFP r0-3, collision-free)** | 0.3959 / 0.3435 | **0.5640 / 0.6345** | wr=1.5: 0.5666 / 0.6398 |
| B, logit clipped to p in [0.01,0.99] | 0.3869 / 0.3230 | 0.5655 / 0.6411 | wr=1.5: 0.5704 / 0.6429 |
| B + Platt | 0.4267 / 0.4098 | 0.5645 / 0.6333 | wr=1.5: 0.5668 / 0.6433 |
| B + rev_x3072 ensemble (z-sum) | 0.4153 / 0.3783 | 0.5632 / 0.6323 | |

Findings
- The gain comes from the target bits: collision-free vocabulary fingerprints (+0.014 cqry, +0.022 tqry at wr=1).
- Platt calibration raises rev-alone MRR a lot (+0.03-0.07) but adds ~0 in the fusion. Dropping bits with F1<0.5 hurts.
- Continuous Tanimoto score is the best rev-alone score (B: 0.442/0.405) but not better fused.
- Ensembling with the old model: no gain.
- With the current model the rev term HURT tqry in the full fusion (0.6228 at wr=0 -> 0.6124 at wr=1); with B it helps.
- Training: ~32 s/epoch alone on the 3090 (7 ms/step, B=512); A/B ran in parallel, ~19 min each for 12 epochs.

## Best model / how to use
- C:\prog\casmi\rev2_vh.bin (REV2 format: vocab + Platt embedded). NOT drop-in for phys_ana (different format & bits).
- phys_ana needs an adapted copy: replace rev::Model with rev2::Model (rev2_model.hpp: featurize(m.fc,...), m.load),
  and the candidate fingerprint ecfp::compute(*m,2,revNB) with fp2 vocab bits
  (fp2::Vocab v; v.rE=m.rE; v.rF=m.rF; v.hashes=m.vocab; v.build(); v.bits(*mol)). Suggested wr 1-1.5, logit clip p to [0.01,0.99].
- Not done yet: the adapted phys_ana2 and a real phys_ana run (dump-based eval is exact for the fixed candidate set).

## Files
- sources: local C:\prog\claude4\arc\arc3\casmi\cpp\rev\{fp2.h, fp2_prep.cpp, rev2_model.hpp, rev2_train.cpp}; remote C:\prog\casmi\fwdmodel\rev2\ (+ b2.bat build, A.log, B.log)
- data: C:\prog\casmi\v4k.bin (vocab), xlib_v4k.bin, fcand_mols.txt, fcand_ecfp.bin, fcand_v4k.bin
- commands: `rev2_train.exe train fp=v4k out=X.bin epochs=12 hold=20` ; `rev2_train.exe eval models=rev2_vh.bin:v4k`
  (cwd C:\prog\casmi; defaults dir=C:\prog\casmi eval=cqry:f21_cqry.txt,tqry:f21_tqry.txt cand=fcand)
  `fp2_prep.exe build C:\prog\casmi xlib v4k.bin 3 3 4096 100` ; `fp2_prep.exe apply C:\prog\casmi <stem> v4k.bin`

## Next steps
1. phys_ana2 (adapted copy) + one real run to confirm 0.564/0.635.
2. Bigger vocab (8192, minCount ~300), full data (hold=0), 0.05 Da bins / NL 500, wider/deeper — each ~7 min training alone.
3. Learned fusion (fuse_fit) with the new rev column; rev-alone on the 4.37M pool needs fp2_prep apply on pool_mols (not done).
