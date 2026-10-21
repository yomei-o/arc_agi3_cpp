# cnn — reverse model with translation symmetry on the mass axis (状態ファイル)

2026-10-21 開始。Idea from the user's series (kangaeru-ai: locality = small kernels, translation symmetry = weight sharing, time evolution = stacked layers).
Goal: a spectrum -> fingerprint-bit model (REV2 4096 vocabulary first, fp3 later) whose structure carries the symmetry of MS/MS, compared with REV2 (rev2_w) on the window yardstick (fp3_STATUS.md, clean hcqry / hxqry) at equal training budget, and ensembled with rev2_w to test whether the symmetry adds information.

## 1. Why (theory)
- **The symmetry.** A substructure leaves the same trace wherever it sits in the molecule: the same *neutral loss* (precursor − fragment: H2O 18.011, CO 27.995, CO2 43.990, hexose 162.053, rhamnose 146.058, ...) and the same low-mass *fragment ions* (tropylium 91.055, C-ring/tropylium series ...). A loss of 162.053 means "glycoside" whether the precursor is 450 or 780 → on the neutral-loss axis the pattern is the same at any precursor mass; and *between fragments* the same differences recur (sequential losses: fragment − H2O, fragment − CO ...), i.e. the spectrum's pattern is (approximately) invariant under translation along the mass axis.
- **What the REV2 MLP lacks.** REV2 sees fixed 0.1 Da bins of fragment m/z (0–1000) and neutral loss (0–300). Every bin has its own weights: "two peaks 18.011 apart" must be learned separately at every position (no weight sharing), and bin neighbours are unrelated inputs (no locality: 85.028 and 85.032 are the same physics but can fall in different bins). The relations *between* peaks are only reachable through the hidden layers.
- **The algebra.** A conv layer is the most general linear map that commutes with translations (weight sharing = symmetry), small kernels = locality (a mass difference is a local relation on a fine grid), pooling = coarse-graining / renormalization (fine 0.01 Da structure → 1 Da → 16 Da), stacked layers = composition of local rules (a ladder of losses). Global pooling after the stack gives exactly translation-invariant features ("this loss pattern occurs somewhere").
- **Where absolute position still matters.** The *value* of a loss (162 vs 146) is information the invariant readout must not lose: on the NL axis it is the position itself. A pure global pool only keeps "which pattern detectors fired", not where. So two readouts: `flat=0` = global max+mean pool only (the pure symmetric model), `flat=1` = global pool + the flattened top map (position at 16 Da cells + whatever the channels encode = CNN + dense head, the image-classifier recipe). Comparing them measures what translation invariance alone buys.
- **Pair (difference) channel.** NL = precursor − m/z treats the precursor as a special peak. Fragment − fragment differences are the same algebra for fragment-level losses (a fragment losing water). So the pair signal |m/z_i − m/z_j| over the top-K peaks (K=60, weight √(I_i I_j)) is a third axis, and **NL and pair share one conv stack** (same weights: same chemistry whether the heavier partner is the precursor or a fragment; separate stems because the intensity statistics differ). The pair axis is translation-invariant by construction (autocorrelation of the spectrum).
- **Mass defect = the fine grid.** Inside a 1 Da cell the fine position (0.01 Da grid, F=100) encodes the mass defect, which separates compositions at the same nominal mass (CO 27.995 vs C2H4 28.031 vs N2 28.006; hexose 162.053 vs others). `cell = floor(v + 0.3)` so defects −0.3 … +0.7 stay in their nominal cell.

## 2. Architecture (cpp/cnn/cnn_model.hpp)
Per spectrum (top 300 peaks by intensity, w = √(I/I_max), as REV2):
| signal | axis | cells (1 Da) | stem (first pyramid level) | stack |
|---|---|---|---|---|
| frag | m/z | Lf = 1024 | sparse patchify: out[cell][c] = b + Σ_items w·Ws[fine][c] (= conv at 0.01 Da, kernel = stride = 100) | own |
| nl | precursor − m/z | Ln = 512 | own table | **shared** |
| pair | m/z_i − m/z_j (top 60, ≥0.5 Da) | Lp = 512 | own table | **shared** |
- stem C0 = 32 → ReLU → 4 × [conv k=5, ReLU, maxpool 2] with channels 64, 64, 128, 128 → top maps 64 / 32 / 32 cells × 128 (16 Da per cell; receptive field ≈ 60 Da at the 1 Da level).
- Readout: for each signal global max + mean (3 × 256) [+ flattened top maps 8192 + 4096 + 4096 when flat=1] + side features (adduct one-hot 10, instrument 5, precursor/1000) → dense 3072 (ReLU, dropout 0.4) → 4096 logits. Same BCE, same targets, same held-out split (name hash mod 20) and Platt calibration as REV2 → same file tail (vocab, cal), so every win_eval scoring variant applies.
- Size: flat=1 65.6M params (53M in the first dense layer), flat=0 15.3M. REV2 rev2_w: 62M.
- The stem is evaluated sparsely (items sorted into CSR by output cell; CSC by fine index for dW), the convs as im2col + cuBLAS GEMM, channel-last [N][L][C].

## 3. Implementation and verification
Files (all new, cpp/cnn/; rev/ headers unchanged, copied next to the sources on the build machine):
- `cnn_layers.hpp` — Conv1d (fwd/bwd: im2col, dW = colᵀ·dY, dcol = dY·Wᵀ, col2im gather without atomics), ReLU, maxpool2, global max+mean pool, SparseStem, own Param/Adam and GEMM wrappers templated on `CNN_REAL` (float; double for the check).
- `cnn_model.hpp` — featurize, Stack, Dense head, model forward/backward, file format `CNN1`.
- `cnn_gradcheck.cpp` — central finite differences on every parameter tensor (≤60 entries each, all for small tensors) of a tiny model (Lf 16, Ln 8, F 4, chans 4/5, k 3, head 6, 7 bits, 3 spectra), both readouts.
- `cnn_train.cpp` — train / calib / cputime / speed. `cnn_win_eval.cpp` — copy of rev3/win_eval.cpp that also loads CNN1 files (REV2 + CNN in one run, z-sum ensemble with ens=1).
- `build.bat`, `day_cnn.bat` (the experiment set below).

**Gradient check (required) — PASS.**
- CPU, double (`cnn_gradcheck_cpu.exe`, eps 1e-6): all 16 tensors × both readouts, seeds 7, 11 (g++) and 7, 23 (MSVC): max |analytic − numeric| ≤ 1.1e-8, max relative 1.5e-8, 0 kinks.
- CUDA, float (`cnn_gradcheck.exe`, eps 3e-3): loss 21.96107239 / 29.65427387 vs CPU float 21.96107434 / 29.65426939 vs CPU double 21.96107442 / 29.65427209 at the same weights → the GPU kernels compute the same function; analytic gradients identical between CPU and GPU float (e.g. −3.61323, 0.349545). The float run reports 1–2 "mismatches" of ~2.5 % per readout: the same points pass in double (max diff 2.8e-9), so they are float finite-difference error (loss ~30, eps 3e-3), not bugs. Lesson: a float gradcheck alone could not have separated these — build the check in double.
- Smoke test on wlib (600 steps, 34 s): held-out loss 257, window MRR_in(tanimoto) hcqry 0.096 already; save/load/calibrate/eval path works.

**Speed.** GPU (3090, B=512): forward 18 ms + backward 30 ms + Adam 2.5 ms + upload 2.5 ms = 54 ms/step; the host side (featurize + counting sorts, 21 ms) runs in the prefetch thread (first version sorted on the GPU thread: 177 ms/step). 12 epochs ≈ 4,645 × 12 steps ≈ 55 min (rev2_w: 532 s).
**CPU inference (Kaggle-like, cl /O2 /arch:AVX2, plain-loop GEMM, 4 threads): 43 ms/spectrum** (flat=1). Well inside "a few hundred ms"; Eigen GEMM would cut it further.

## 4. Experiments and comparison plan (day_cnn.bat, launched 09:33 as `cnn_day`, pid 1628)
Yardstick: `cnn_win_eval` on clean hcqry (400, window 5,470) and hxqry (400, window 12,447), all variants (logit / cal / tanimoto, + 0.15 pop, shuffled control). Baseline rev2_w: tanimoto MRR_in 0.254 / 0.299, cal 0.226 / 0.229.
| # | model | question |
|---|---|---|
| 1 | rev2_w_s32 (REV2, seed 32) | control: what does *any* second model add in an ensemble |
| 2 | cnn_a (flat=1, 12 ep) | CNN vs MLP at equal epochs (equal data passes); + ens with rev2_w |
| 3 | cnn_g (flat=0, 12 ep) | pure translation-invariant readout; + ens with rev2_w |
| 4 | ens rev2_w + rev2_w_s32 | the control ensemble |
| 5 | cnn_np (pairK=0, 12 ep) | value of the pair-difference channel |
| 6 | cnn_t2 (2 ep ≈ rev2_w wall clock) | equal GPU-time budget |
**Decision rule for "the symmetry adds information":** ens(rev2_w + cnn) − ens(rev2_w + rev2_w_s32) > ~0.01 MRR_in on both hcqry and hxqry (seed ensembles of the MLP gave ~0 before: rev2_STATUS "ensembling with the old model: no gain"). CNN alone ≥ rev2_w would be a bonus, not required.
Budget: "equal training budget" = equal epochs (12 passes over the same 2.38M wlib spectra, same B, lr schedule, dropout, held-out split); the GPU time is ~6× (55 vs 9 min), so cnn_t2 gives the equal-time point.

## 5. Results
### 5.1 cnn_a (flat=1, 12 epochs, v4k) vs rev2_w — clean window MRR_in (2026-10-21 11:37)
| | hcqry logit | cal | **tanimoto** | z(tan)+0.15pop | hxqry logit | cal | **tanimoto** | z(tan)+0.15pop |
|---|---|---|---|---|---|---|---|---|
| rev2_w (REV2 MLP) | 0.173 | 0.226 | 0.254 | 0.357 | 0.090 | 0.229 | 0.299 | 0.300 |
| cnn_a epoch 7 (uncalibrated checkpoint) | 0.206 | – | 0.289 | 0.389 | | | | |
| **cnn_a** | 0.203 | 0.230 | **0.294** | 0.391 | 0.364 | 0.503 | **0.546** | 0.544 |
| z-sum rev2_w + cnn_a | 0.187 | 0.232 | 0.297 | 0.400 | 0.195 | 0.384 | 0.495 | 0.490 |
| fp3 MLPs for reference (fp3_STATUS): rev3_fp3c16k / _big | | | 0.303 / 0.327 | | | | 0.270 / 0.326 | |
Shuffled-label control: 0.021 / 0.0002 (same as rev2_w) → the scores are spectrum-specific.
- **The CNN beats the MLP on the same data, targets and epochs: hcqry +0.040 (0.254 → 0.294), hxqry +0.247 (0.299 → 0.546)**, and on hxqry also the 16k-bit fp3 MLPs (0.326). On hcqry it is level with fp3c16k although it predicts only the 4096 v4k bits.
- The equal-weight z-sum ensemble with rev2_w adds +0.003 on hcqry and *loses* 0.05 on hxqry (the weaker model gets equal weight) → the CNN already contains what rev2_w knows there; a weighted ensemble is the next question (control rev2_w + rev2_w_s32 pending).
- Overfitting: held-out loss/spectrum 207 → 195.4 (epoch 4) → 201.6 (epoch 12) while the window MRR still rose from epoch 7 (0.289) to 12 (0.294); early stopping on held-out loss would have stopped at 4–6. Regularized run (cnn_awd) is in the night job.
- Calibration issue: Platt fit made held-out BCE worse (157 → 175, mean a 7733): the CNN's held-out logits are so confident on some bits that the ridge (λ=1) cannot hold a → the 'cal' variant is under-performing; tanimoto (raw probabilities) is the variant to use. Fix: stronger ridge or clip logits before fitting (todo).
- CPU (4 threads, Kaggle-like): **30 ms/spectrum**.

**Why so much better on hxqry? — leakage check (cpp/cnn/leak_check.cpp)**: hxqry is *less* analog-rich in wlib than hcqry, so the gain is not memorized near-duplicates:
| | NN Tanimoto (v4k) median | =1.00 | ≥0.90 | ≥0.80 | identical spectra in wlib |
|---|---|---|---|---|---|
| hxqry (timsTOF, ordinary molecules) | 0.713 | 0.0 % | 6.3 % | 23.3 % | 1 / 1163 |
| hcqry (COCONUT natural products) | 0.884 | 20.6 % | 46.0 % | 70.2 % | 21 / 972 |
Reading: hcqry truths often have a near-identical library analog (the MLP's bin lookup already exploits that, small room for the CNN); hxqry truths are novel scaffolds made of common building blocks (amides, aryl, heterocycles — the Enveda-library style), exactly where position-free detectors of losses/fragments/differences should generalize and fixed-bin weights cannot. This is the theory's prediction, but one test set: confirm on another clean ordinary-molecule holdout before believing +0.25, and check that the gain survives the full pipeline (window → final list) and the LB.
Also note REV2 logit scoring is unusually poor on hxqry (0.090 vs cal 0.229), the CNN's is not (0.364).

### 5.1b Does the symmetry add information? — ensembles, no FPNet anywhere (window MRR_in, tanimoto; cal in the logs)
`cnn_win_eval ... ensw=` = weighted z-sum of the per-window z-scored scores (eval_ctl2_*, eval_w3_* in C:\prog\casmi\cnn\).
| models (weights) | hcqry | hxqry |
|---|---|---|
| rev2_w | 0.254 | 0.299 |
| rev2_w_s32 (same MLP, other seed) | 0.271 | 0.310 |
| rev3_fp3c16k_big (fp3 16k MLP) | 0.327 | 0.326 |
| cnn_a | 0.294 | **0.546** |
| **control** rev2_w + rev2_w_s32 (1:1) | 0.263 | 0.314 |
| two MLPs rev2_w + fp3big (1:1) | 0.331 | 0.363 |
| rev2_w + cnn_a (1:1) | 0.297 | 0.495 |
| **fp3big + cnn_a (1:1)** | **0.354** | 0.559 |
| fp3big + cnn_a (0.5:1) | 0.352 | **0.576** |
| rev2_w + fp3big + cnn_a (1:1:1) | 0.348 | 0.519 |
| rev2_w + fp3big + cnn_a (0.25:0.25:1) | 0.339 | 0.549 |
**Answer: yes.** A second MLP adds +0.004 (fp3big + rev2_w, hcqry) or ~0 over the better member (seed control); the CNN added to the best MLP adds **+0.027 on hcqry** (0.327 → 0.354) and +0.03 on hxqry over the CNN alone (0.546 → 0.576 at 0.5:1). The CNN errs on different queries than the MLPs — it is not a re-parametrized copy. Best robust choice on both sets: **fp3big + cnn_a, weights 0.5–1 : 1** (0.352–0.354 / 0.559–0.576); rev2_w adds nothing once fp3big is in.
Caveat: weights were chosen on these two test sets (coarse grid of 8); the 1:1 point needs no tuning.

### 5.2 remaining day runs (cnn_g, control ensemble, cnn_np, cnn_t2): `type C:\prog\casmi\cnn\SUMMARY.txt`
### 5.3 night runs (night_cnn.bat, `cnn_night`, waits for the day job): cnn_f16 (CNN on fp3c16k 16k bits) + ensembles with rev3_fp3c16k_big / rev2_w (no FPNet anywhere), cnn_awd (wd 0.05, drop 0.5), cnn_a33 (2nd seed: CNN+CNN ensemble) → `type C:\prog\casmi\cnn\NIGHT_SUMMARY.txt`

## 6. Resume steps
1. `ssh ... "type C:\prog\casmi\cnn\progress.txt"`; job alive: `powershell -c "Get-Process -Id (Get-Content C:\prog\casmi\night\cnn_day.pid)"`; stop only it: `taskkill /pid <pid> /t /f`.
2. Read `C:\prog\casmi\cnn\SUMMARY.txt` (training curves, calibration, eval tables). Fill §5.
3. Rebuild after source changes: copy cpp/cnn/* and cpp/rev/{backend.hpp,parallel.hpp,rev2_model.hpp} to `C:\prog\casmi\fwdmodel\cnn\`, run `build.bat` there.
4. Single eval: `C:\prog\casmi\fwdmodel\cnn\cnn_win_eval.exe dir=C:\prog\casmi qry=hcqry win=w_hcqry models=rev2_w.bin:v4k,cnn_a.bin:v4k ens=1 threads=6`.
