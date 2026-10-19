# lmart2 — LambdaMART v29 trained on the holdout2 sets, gated (status file)

Written 2026-10-19. Base: v28a (phys_v28pc: v27a + pool popularity ppop=0.15, ppgate=0.9). Goal: a learned ranker that learns
both classes (sibling/xinstr = the answer's other spectra are in the library, denovo = they are not, rm = the answer is not
in the pool) and cannot lose on the sibling side. That side is where v21 and v26 failed.

## Conclusion first
- **One configuration passes the gate, but only just.** Measured by grouped 5-fold CV over 9 sets (FPNet off, ppop on in both arms):
  - Yardstick mixture (0.5 hasqry + 0.42 hcqry + 0.08 hcqry rm): **+0.0050 (paired SE 0.0036)**. That is about 1.4 SE, and I picked this config out of roughly 40 CV configs, so the real gain is probably smaller.
  - All-sets mixture (0.5 mean S + 0.42 mean D + 0.08 mean R): +0.016.
  - Sibling/xinstr sets: hasqry **−0.0012 ± 0.0029**, hbsqry +0.028 ± 0.007, hxqry +0.010 ± 0.004.
  - Denovo sets: haqry +0.018, hcqry +0.016, hbqry +0.041.
  - **rm sets get worse:** haqry rm −0.015 ± 0.005, hcqry rm −0.015 ± 0.005, hbqry rm +0.001.
- The chosen model = `fl5np`: mix S:0.5,D:0.42,R:0.08, **libfloor 0.5**, no pool-popularity features, no merr, no query-meta features, no raw scores. Inference gate libSim < 0.9 (queries with a library match ≥ 0.9 keep the hand score), blend 1.
- What made it pass:
  1. **The gate.** Without it, hasqry loses −0.04: on the 304/400 queries whose answer has lib ≥ 0.9, the ranker is worse than the hand term 8·lib^32.
  2. **libfloor 0.5.** It hides the "has library spectra but they don't match → not the answer" shortcut. With it, the lib features drop out of the top-20 gains.
  3. **Dropping the pop features.** With them, denovo gains more (haqry +0.035), but hasqry gets worse. The hand ppop post-step stays.
- **Label-permutation controls collapse** (mixture −0.19, all sets −0.34 at gate 0.9). So the training path has no answer leak.
- The C++ path matches the Python simulation exactly: phys_v29lm with the fold-0 model on hcqry gives the same per-query reciprocal rank on all 80 held-out queries (0.5349 = 0.5349, 0 differences).
- Honest read: the big, consistent gains are on the ordinary-molecule sets (hb/hbs/hx, enveda-180 timsTOF, popularity 1.3), and those are probably the closest to the hidden test. The yardstick gain itself is within noise, and the rm component loses. Only the LB can decide. If you submit it, compare against v28a (same FPNet and ppop).

## Files
- `cpp/rdkit-min/phys_v29lm.cpp`: generated from phys_v28pc.cpp by scratchpad `mkv29.py` (patch only). It adds merr, query meta, the dump extras (9 digits; + pool pop column) and the ranker at argv[25] (relative to dir or absolute). It changes nothing without argv[25] (the hand CSVs = pc2's v28a numbers: hcqry 0.533, hasqry 0.839).
- `cpp/rdkit-min/lm_feat2.h`: 54 features, namespace lmf2. The hand score = the v28a fusion before FPNet/ppop. Opt libFloor.
- `cpp/rdkit-min/lmart_model.h`: now accepts optional `opt <key> <value>` lines before `base` (backward compatible with lmart_prod.txt).
- `cpp/lmart_v29.cpp`: the trainer, generated from lmart.cpp. New keys: libfloor, gate/blend (recorded in the model), `oofs=` (per-row OOF scores), `trgate=`.
- `py/lm29_cv.py` runs the 5 folds in parallel (thr 2 each). `py/lm29_eval.py` simulates phys_v29lm's output (scale, gate, blend, ppop) from OOF scores, then prints per-set MRR, paired SE, mixtures and PASS/fail. `py/lm29_make_nb.py` builds the notebook copy.
- `bat_lm29/`: build.bat (MSVC), run.bat (one holdout, v28a config + dump), dumpA/B.bat, cv_batch1-3.bat.
- Build machine: dumps `C:\prog\casmi\lm29\<set>_f.txt` (has0 hbs0 hx0 ha0 hc0 hb0 ha1 hc1 hb1; 0 = normal, 1 = rm=1), OOF `cv_<tag>_<k>.txt`, **production model `C:\prog\casmi\lm29\lmart_v29_prod.txt`** (407 trees, 285 KB, opt libfloor 0.5 / gate 0.9 / blend 1), Kaggle add-on files staged in `C:\prog\casmi\lm29\ds_add\` (ds_rdkit is NOT modified).

## Setup
- Holdout-safe models: fwd_mlp_rank_ylib.bin, REV2 rev2_y.bin (hxqry: rev2_z.bin, lib zlib), lib ylib. v28a = REV2 in the rev slot (no rev v1), phys_size_fqry.txt, libcap=1, gen cap 300 + prior + gw, ppop 0.15 / ppgate 0.9. No PubChem, FPNet off (it leaks).
- Hand (v28a, FPNet off) by mrr.py on the C++ CSV = Python: has 0.839, hbs 0.858, hx 0.862, ha 0.577, hc 0.533, hb 0.780, rm ha 0.143 / hc 0.151 / hb 0.058.
- Folds: FNV hash of the molecule name, so the rm variant sits in the same fold as its molecule. Early stopping on a 20% mix-weighted validation split inside the training folds (es 150, lr 0.05, 15 leaves, metric mrr).
- Inference formula (per query, libmax < gate): out = mean(hand) + sd(hand)·((1−blend)·z(hand) + blend·z(rank + 1e-4·z(hand))). The 1e-4 term only breaks the trees' exact ties.
- **FPNet in production**: the ranker output sits on the hand fusion's per-query scale (same mean and sd). The existing `+ 1.0·z(fpnet)` is then added unchanged, so FPNet has the same relative weight as in v28a. ppop then works on the result as in v28a. Caution: with the ranker, ppop moves many more list places (hbsqry 7278 vs ~1500 with hand), because the z-spread of the top-60 differs. The CV includes ppop in both arms, so the effect is already in the numbers above.

## Results (grouped 5-fold CV, OOF, paired differences vs hand, ppop on both arms)
Mixture = yardstick 0.5 has + 0.42 hc + 0.08 hc-rm. "all" = 0.5 mean(S) + 0.42 mean(D) + 0.08 mean(R).

| variant (gate 0.9, blend 1 unless noted) | has | hbs | hx | ha | hc | hb | ha rm | hc rm | hb rm | mix | all | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base (pop feats, no floor), no gate | −0.040 | +0.068 | +0.054 | +0.128 | +0.092 | +0.127 | −0.013 | −0.013 | +0.004 | +0.017 | +0.062 | fail |
| base | −0.014 | +0.023 | −0.002 | +0.064 | +0.040 | +0.050 | −0.009 | −0.012 | +0.003 | +0.008 | +0.022 | fail |
| nopop | −0.013 | +0.025 | +0.006 | +0.033 | +0.021 | +0.059 | −0.018 | −0.018 | +0.000 | +0.001 | +0.018 | fail |
| fl5 (floor 0.5, pop feats) | −0.008 | +0.027 | +0.004 | +0.035 | +0.022 | +0.034 | −0.013 | −0.013 | +0.005 | +0.004 | +0.016 | fail |
| **fl5np (chosen)** | **−0.001** | **+0.028** | **+0.010** | **+0.018** | **+0.016** | **+0.041** | **−0.015** | **−0.015** | **+0.001** | **+0.005** | **+0.016** | **PASS** |
| fl5np blend 0.5 | −0.002 | +0.016 | +0.006 | +0.010 | +0.011 | +0.023 | −0.010 | −0.006 | +0.001 | +0.003 | +0.009 | PASS |
| fl5np gate 0.8 | −0.001 | +0.014 | +0.005 | +0.009 | +0.007 | +0.016 | −0.009 | −0.010 | +0.000 | +0.002 | +0.007 | PASS |
| fl8np (floor 0.8) | +0.000 | +0.029 | +0.011 | +0.010 | +0.000 | +0.035 | −0.020 | −0.023 | +0.001 | −0.002 | +0.012 | fail (mix) |
| fl5np, mix S .7/D .25/R .05 | −0.004 | +0.030 | +0.011 | +0.021 | +0.015 | +0.035 | −0.019 | −0.021 | +0.001 | +0.003 | +0.015 | fail |
| fl5np, mix S .5/D .35/R .15 | −0.002 | +0.031 | +0.009 | +0.020 | +0.006 | +0.039 | −0.000 | −0.005 | +0.004 | +0.001 | +0.016 | PASS |
| fl5np, mix S .45/D .3/R .25 | −0.004 | +0.031 | +0.003 | +0.012 | +0.002 | +0.028 | +0.005 | −0.001 | +0.005 | −0.001 | +0.011 | fail |
| tg9 (train only on libmax<0.9) | −0.007 | +0.021 | +0.009 | +0.021 | +0.015 | +0.042 | −0.016 | −0.019 | +0.003 | +0.001 | +0.014 | fail |
| tg9, mix R .15, blend 0.5 | −0.002 | +0.011 | +0.002 | +0.015 | +0.016 | +0.025 | −0.005 | −0.007 | +0.002 | +0.005 | +0.009 | PASS |
| permutation control (fl5np) | −0.075 | −0.470 | −0.365 | −0.362 | −0.344 | −0.600 | −0.104 | −0.115 | −0.049 | −0.191 | −0.342 | collapse |

Paired SE: has 0.003–0.004, hbs 0.006–0.007, hx 0.004, D 0.007–0.009, rm 0.004–0.005; yardstick mixture 0.0036.
GATE rule used: no sibling/xinstr set below −1 SE, and the yardstick mixture above hand.
Without any gate, every variant loses 0.02–0.06 on hasqry. That is the v21/v26 failure mode, and it is visible here.

Where hasqry loses (base, no gate): 304/400 answers have lib ≥ 0.9 (−0.034 of the −0.040), 23 have lib 0–0.5 (−0.009), 35 have lib 0.5–0.9 (−0.002).

Feature gains (fl5np): rf_hand 0.51, rk_hand 0.18, z_hand 0.15, gap_hand 0.04, then gap_rev / a_mcos / isGen / ana_m_max / spec3 ≤ 0.007 each. lib features are not in the top 20. The ranker mostly re-orders around the hand fusion: rank-normalised hand plus small corrections from the analog variants and REV2/phys gaps.

## Kaggle integration (nothing uploaded, main notebook untouched)
1. Dataset casmi26-rdkit-min (staging C:\prog\casmi\ds_rdkit). Add the files from `C:\prog\casmi\lm29\ds_add\`:
   - `rdkit-min/phys_v29lm.cpp`, `rdkit-min/lm_feat2.h`, `rdkit-min/lmart_model.h` (replaces the old one; backward compatible)
   - `pool730/lmart_v29_prod.txt`
2. Notebook: `notebook/casmi_match_v29lm.ipynb` (made by `py/lm29_make_nb.py` from casmi_match.ipynb = v28a). It changes only:
   - cell 6: `srcs_fp = ["phys_v29lm.cpp"] + srcs[1:]`
   - cell 7: `LM_ARGS = [str(POOL730_DIR / "lmart_v29_prod.txt")]` when FP_ARGS and the file exists, appended after the argv[24] option string. argv[0..24] are unchanged from v28a, including FP_ARGS = fpnet_test.txt 1.0 and the ppop options.
3. Log lines to check:
   - `v29 lmart: ['/kaggle/input/.../pool730/lmart_v29_prod.txt']`. If you see `OFF (...)`, the model is missing: v28a output.
   - `phys_fp compile rc: 0`, `fpnet stage done`
   - `output uses lmart /kaggle/.../lmart_v29_prod.txt (407 trees, gate libSim < 0.9, blend 1, libfloor 0.5): ranker used for N / 400 queries`. Expect N < 400. Queries with a library match ≥ 0.9 keep the hand score.
   - then `output + 1 z(fpnet): ...` and `pool popularity: weight 0.15, top 60 re-ordered, ...` as in v28a.
   - If the model fails to load, the program prints `cannot load lmart model` and exits 1, and the notebook stops (SystemExit). That is visible, never silent.
4. Compile check: `g++ -std=c++20 -fsyntax-only -w -I <syn dir> -I ds_rdkit/rdkit-min -I ds_rdkit/rdkit-min/boost-min -I ds_rdkit phys_v29lm.cpp` with w64devkit gcc 14 passes (the build machine copy of the new headers comes first, as they will be in the dataset). The MSVC build runs on the holdouts. The absolute model path and the model's opt lines were checked on hbsqry: the log printed gate 0.9 / blend 1 / libfloor 0.5.
5. Runtime: none to speak of (407 trees × ~200 candidates × 400 queries).
6. The rank-fusion agent's post-step reads pool_candidates.csv. The format is unchanged (molecule_id,smiles with 25 `;`-joined SMILES).

## Not checked / risks
- Production feature generators (fwd_mlp_rank_v2, rev2_all_v4k, full train library, FPNet added afterwards) were not measured with this model. The features are mostly rank/z of the hand fusion (scale-free). For v26, holdout2 found that the production-model shift did not hurt.
- The rm component loses −0.015 on two of three sets. If the hidden test has many more out-of-pool answers than 8%, the gain shrinks. The mix S.5/D.35/R.15 variant (rm ±0, mix +0.001, all +0.016) is the safer alternative there. Train it with `mix=S:0.5,D:0.35,R:0.15`, everything else the same.
- The yardstick gain (+0.005) is at the noise level and smaller than the LB SE (~0.03). One submission cannot confirm it. A clear LB loss (≤ −0.02) would again point to "learned fusion does not transfer".
