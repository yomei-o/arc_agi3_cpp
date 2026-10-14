# fp10226 reverse-MLP experiment (2026-10-14, DONE)
Question: does the rev MLP (spectrum -> fp) improve if it predicts FPNet's 10226-bit fingerprint instead of v4k?
Answer: NO in the fusion. Rev-alone improves slightly, fused MRR@25 is worse than v4k.

Setup (leak-free: trained on xlib, hold=20, same arch H=3072,3072 / hyperparams as v4k run B; only targets differ)
- No code change needed: rev2_train reads sparse targets <lib>_<fp>.bin; vocab file f10k.bin = header only (n=0).
- Build machine C:\prog\casmi: f10k.bin, xlib_f10k.bin (275238 mols, 0 missing, ~952 on-bits/mol), fcand_f10k.bin (53374, 0 missing),
  made from pool730_fp10226.bin by fwdmodel\rev2\mk_f10k.py (np.unpackbits). Model rev2_f10k.bin. Logs fwdmodel\rev2\F10K.log, F10K_ens.log.
- Training: 749 s total (~62 s/epoch, 2x v4k due to 10226-wide output layer).

| model | rev alone (logit) cqry/tqry | fused wr=1 | fused wr=1.5 |
|---|---|---|---|
| no rev | - | 0.5366 / 0.6228 | |
| rev_x3072 (current) | 0.3715 / 0.3402 | 0.5500 / 0.6124 | |
| v4k (rev2_vh) | 0.3959 / 0.3435 | 0.5640 / 0.6345 | 0.5666 / 0.6398 |
| f10k (rev2_f10k) | 0.4072 / 0.3551 | 0.5531 / 0.6191 | 0.5459 / 0.6102 |
| f10k Platt (cal) | 0.4363 / 0.4218 | 0.5552 / 0.6239 | 0.5481 / 0.6248 |
| f10k tanimoto | 0.4054 / 0.4456 | 0.5421 / 0.6332 | 0.5391 / 0.6325 |
| v4k+f10k z-sum ensemble (logit) | 0.4505 / 0.4137 | 0.5594 / 0.6252 | 0.5596 / 0.6311 |

Interpretation: FPNet's bit definition alone does not explain its strength for our MLP; with these targets the MLP
is a slightly better standalone retriever but adds less complementary signal to the fusion than v4k. Points toward
the Transformer architecture (or its training) rather than the bit set. Caveat: single seed, 400/176 queries (~0.01 noise).
