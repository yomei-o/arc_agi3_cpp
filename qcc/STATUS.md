# Google Verifiable Quantum Circuit Compilation (Kaggle, deadline 2027-01-20)

Rules/verifiers/paper on the build machine: `C:\prog\qcc` (verify_submission_invsqrt.py, nr_reference_circuit.py,
SCORING_README.md, verifiable_circuit_compilation.pdf = arXiv:2610.23832). kickmix 1.0.0 from PyPI has a Windows wheel
(`C:\prog\qcc\venv`). The verifier only parallelizes with fork, so on Windows it runs single-process (~1 h for the
200M shots of Benchmark I); verify_fast_invsqrt.py (threads) does not help (GIL).

## Benchmark I: phase by 1/sqrt(x), 22-bit x, score Q x (Toffoli + T/4), reference 2.08e6 (Q 260, C 7,988)

Own design (no Newton step): normalize x by leading zero pairs -> M in [1,4); table of quadratic segments
addressed by the top 9 bits of x_norm (384 reachable entries); y = c0 - T(t (c1 - T(c2 t))) with t = next 13 bits;
denormalize Y 2^s; per-bit Z_POW; uncompute. Precision WY=30 / W1=30 / W2=28 (guard bits: each truncated row
biases by up to 1 ulp): worst relative error 2^-26.1 over every x (inv_model.py, exhaustive).

| version | change | Toffoli (static) | Q | est. score |
|---|---|---|---|---|
| v1 inv_circuit.py | unary flags as in the reference, full 512 table | 3,321 | 178 | 5.4e5 |
| v2 inv_circuit2.py | binary s (barrel rotate), lookup over reachable range, narrow scratch | 2,772 | 162 | 4.9e5 |
| v3 inv_circuit3.py | registers allocated only while needed | 2,772 | 152 | 4.59e5 |

Circuits match inv_model bit-exactly on ~3,000 sampled x (`--check`). Score estimate: score_est.py (verifier's
rotation-cost rule on the exact model error). v1 cost by stage: Horner 1,920 (both directions), denormalize 620,
lookup 551, normalize 230. kickmix adders hold ~n temporary carry qubits, which sets the Q peak (Horner).
Official verification of v1 running.

Next ideas: single-bit normalization with a parity-addressed table (256 entries instead of 384, ~-86 Toffoli);
narrower scratch with short-offset adders (Q -9, Toffoli +104); fewer phase rotations by rounding.

## First submission (2026-11-02)
sub1 = own invsqrt v3 + reference dqi + reference dfthcsos (zip 70 MB built on the build machine; the CLI submit
returned 400, `KaggleApi.competition_submit` with the file named submission.zip worked). Expected GM ~(4.5)^(1/3).
Leaderboard: score = geometric mean of cost_ref / cost over the three tasks. Public notebooks plateau at ~23.9 (they
stack each other's circuits: invsqrt ~20x at 62-101 qubits, dqi ~64x at 883-892 qubits / 535k Toffolis, chemistry
~10x at lambda 47, 428 qubits, 16k Toffolis); the top private entries are at ~28.

## Benchmark III (dfthcsos) analysis
- Reference reproduced on the build machine (C:\prog\qcc\verifiers, score 3.3556e9).
- chem_cost.py: T_total / Q_total from (R, B, C, brot, J) with the verifier's formulas; reproduces the reference
  exactly and the public best (9, 143, 40) at brot 8, J 5 (Q 428). brot <= 8 halves the inner-QROAM temporaries
  (606 -> 303) because the verifier picks the QROAM size below beta (N - 1). The public ANGLES circuits cost ~316
  Toffolis per batch against ~1361 for a plain QROM.
- chem_spectrum.py: Eckart-Young bound on the shifted two-electron target supermatrix: error_2 <= 20 mHa needs at
  least 290 rank-one terms (R*C), 330 for 10 mHa; negative-eigenvalue floor 0.3 mHa.
- Group rank: each group spans at most min(B, C) independent X, so R * min(B, C) >= ~300.
- Starter trainer (JAX CPU) runs at ~0.1 s/epoch for (9, 143, 40); seeded from the truncated reference it starts
  at 8.7 Ha error -> needs a physics init (eigen-matrices grouped, per-group INDSCAL).
- Benchmark I v1 passed the full official verification (200M shots, 2865 s single process): score 635,569.
- Physics init (chem_init.py): eigen-matrices of the target grouped 9 x 40 + per-group INDSCAL with mu*lambda_H2
  penalty: error_2 492 mHa, lambda_H2 847 (without the penalty 833 mHa but the trainer blew lambda to 2.3e5).
  Ideal lambda_H2 of the exact eigen-matrices is 566, the reference's whole Lambda is 308: lambda depends on the
  gauge (orthogonal mixing of all terms), not only on the eigen-decomposition.
- Trainer from that init, (9,143,40), 3000 epochs: error 43.6 mHa, lambda_eff 316 (still falling) vs the truncated
  reference seed 4.8 Ha after 200 epochs.
- Overnight (night_chem1.bat, build machine C:\prog\casmi\night\night_chem1.log): (9,143,40) 40k epochs, then
  inits + 40k epochs for (20,30,15) and (30,15,10) to test whether a short angle table is feasible.
- ANGLE deltas of the reference are dense (0.1% zero words at brot 12, 1.3% at brot 8, every 5-word delta row
  nonzero), so pruned unary iteration gives nothing on generic factorizations (chem_angle_sparsity.py,
  chem_angle_coarse.py); sparsity or shared rows would have to be designed into the factorization.
- Shared-basis THC (R = 1, B >= 300 by the rank bound): ANGLES drop to ~3.7k Toffolis (L = 376) but the inner state
  prep over R*C*B = 90k entries costs 6.8k; best T*Q 6.15e6 vs public 6.95e6 (-12%). The cost map's optimum is
  many small groups (e.g. R 30, B 15, C 10: T*Q ~3.3e6 with plain QROM) - feasible only if 15 vectors per group
  can carry the accuracy; tonight's (30,15,10) and (20,30,15) runs test that.
- Angle quantization of the reference factorization (chem_brot_sweep.py, starter brot_sweep), log-depth tree:
  error_2 16.1 mHa unquantized, 18.6 at brot 12, 40.0 at 10, 155 at 8 (lambda_eff 54.5); the path tree is about
  2x worse. brot 8 (needed for the halved state-prep temporaries) therefore needs quantization-aware training: only
  the vectors u are quantized, the weights w stay continuous -> next: quantize u, re-fit w (and W_rho_c).
- Quantization at brot 8 of the trained (9,143,40) (3000 epochs, unquantized 43.6 mHa): error_2 238 mHa.
  chem_qrefit.py: re-fitting only the continuous weights w with the quantized u fixed: 238 -> 166 mHa (400 steps,
  still slowly falling). Joint straight-through (STE) training of the angles + w made it worse (300 mHa; with a
  10x smaller angle step 246 mHa): the 1/128 rounding grid makes STE oscillate. Next: drive the unquantized error far
  lower first, then w-refit; or discrete coordinate descent on the angle words.
- w-only refit run longer (2000 steps): plateaus at 163 mHa (lambda_H2 573 -> 499). The quantized vectors themselves
  must change: discrete search on the angle words (coordinate descent / annealing per row) is the next step.
- Overnight (9,143,40) at epoch 4000: unquantized error 43.1 mHa, lambda_eff 133.8 (falling fast; public 47).

## 2026-11-04
- Overnight: (9,143,40) 40k epochs from the physics init: unquantized error_2 34.1 mHa (plateau; lower lr 2e-4 did not
  move it). (20,30,15) plateau 322 mHa, (30,15,10) 896 mHa: small groups cannot carry the accuracy (negative result;
  each group needs ~100+ vectors).
- First submission scored 1.65362 (= (4.5)^(1/3), as expected).
- Discrete coordinate descent on the brot-8 words (chem_qcd.py): every single-word move is rejected (one word
  = pi/128 overshoots the valley); all gain came from re-fitting w (185 -> 122 mHa). Quantization noise is not
  removable word by word.
- **Gauge (chem_gauge.py)**: per group an orthogonal C x C mixing of the leaves changes only lambda_H2 (error,
  E_SOS, words untouched). On the 40k factorization: lambda_H2 733.5 -> 352.7, official Lambda ~745 -> 364.9,
  lambda_eff ~76 -> 53.40 (public best 47.04), error_2 unchanged 36.29 mHa at brot 12.
  Note: H2_I in the npz stores w_B = W + sum_b w (verify.native_inputs); the starter kit's evaluator reports a
  different Lambda, so chem_eval.py now uses the official dfthcsos.verify.
- Consequence: lambda can be recovered after training, so train for the error only (rho 1e-6) -> day2_chem2.
- Public chemistry method (read the notebook text only, user-approved; code written here independently): compact
  seed (pivoted Cholesky leaves + balanced k-means on orbital profiles + OMP vectors), long L-BFGS (A100 hours),
  progressive rounding with L-BFGS stages, gauge + constrained lambda refit, one-body word polish, compressed loaders.
- GPU pipeline on the build machine's RTX 3090 (system Python, torch 2.6 cu124), all own code:
  chem_gpu.py (seed: 754.8 mHa at (9,143,40), 17.1 orbitals/group; L-BFGS fit: 29.0 / 26.7 / 25.9 mHa after
  10k / 20k / 30k iterations - beats the CPU trainer's 34 after 40k epochs), chem_round.py (progressive rounding),
  chem_lamfit.py (offsets-only lambda_eff refit: 1273 -> 57.7 at unchanged error_2; LS offset init),
  chem_build.py (one-body words polished by coordinate descent: error_1 55 -> 13.2 mHa; plain QROM circuits;
  official verify). End-to-end FAIL only on error_2 (test table at 190 mHa).
- Plain QROM: J = 13 at brot 8 is best (cap 125): (9,143,40) ~5.9x at lambda_eff 50; (9,143,60) ~5.5x but
  R*C = 540 gives slack for the rounding noise -> switched the fit to (9,143,60) (day2_gpu2).
- (9,143,60) GPU L-BFGS: 24.0 / 20.8 / 19.7 / 19.2 / 18.9 / 18.88 mHa over 6 x 8k iterations (stall). Not an fp32
  issue (chem_prec.py: fp32 and fp64 agree to 0.1 uHa). Adam with a cosine lr (chem_adam.py, ~9 ms/step) escapes:
  18.88 -> 18.23 in 4k steps; 150k-step run launched, then the rounding pipeline (pipe_c60b).
- invsqrt: truncation-bias compensation does not allow narrower registers (WY 28: 2^-24.1 with or without it).

## 2026-11-04 midday handoff
- adam_60a (9,143,60): Adam 150k -> error_2 11.81 mHa unquantized.
- pipe_c60b (40 stages x 1500) running: error_2 12.69 mHa at 81% frozen (12:34). Result in C:\prog\casmi\night\pipe_c60b.log, build in C:\prog\qcc\build_c60b.
- Night chain night2_chain.bat (pid 6960): waits for c60b, then Adam 400k (lr 2e-4) -> adam_60b, then pipeline 60 x 2000 -> build_c60c.
- If verify PASS: qcc_submit.py own_v3.kmb sub1\dqi.kmb build_c60x "<msg>" (--dry first). Not auto-submitted.
