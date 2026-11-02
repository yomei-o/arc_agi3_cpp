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
