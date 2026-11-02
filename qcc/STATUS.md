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
