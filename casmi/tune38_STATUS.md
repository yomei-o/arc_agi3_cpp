# tune38 — hand weights re-tuned on the timsTOF holdouts (2026-10-22)

Why: the hidden test is all timsTOF and ~98% of answers are in the pool; timsTOF-specific terms reached the LB more
than their holdout gain (v35nofp -> v36nofp: e2e +0.015, LB +0.027). Learned rankers failed 3 times on the LB, so only
a few hand weights are tuned, coarse values, both-halves rule.

## Pieces
- `cpp/tune38/phys_v38td.cpp` = phys_v37pc + `tdump=` (every candidate, every fusion term, 9 digits; eval only).
  Runs: `cpp/tune38/run38.bat`, `lanes38.bat`, `fp38.bat`; outputs build machine `C:\prog\casmi\tune38\out\t38{x,b,bs,s,c}*`
  (hxqry, hbqry, hbsqry, hasqry, hcqry; holdout-safe models, v37 production config).
- `py/tune38_core.py` exact offline re-scoring (fusion -> top 60 -> pool-popularity re-fill -> Boltzmann merge).
  Check `py/tune38_check.py`: all 5 sets match the C++ final MRR (hx 0.8942, hb 0.8270, hbs 0.9153, has 0.8430, hc 0.5493).
- `py/tune38_search.py`: coordinate descent, yardstick = mean(hx, hb, hbs), change accepted only if both md5 halves
  improve; cross-fit and random-label control. Log `tune38_out/search.log`, weights `tune38_out/tuned.json`.
- `py/tune38_variants.py`: which change carries the gain. `py/make_v38_nb.py`: notebook from tuned.json.

## Result (v37 -> tuned, paired bootstrap)
| set | v37 | tuned | delta [95% CI] |
|---|---|---|---|
| hxqry | 0.8942 | 0.9330 | +0.0388 [+0.023, +0.054] |
| hbqry | 0.8270 | 0.8769 | +0.0499 [+0.034, +0.067] |
| hbsqry | 0.9153 | 0.9442 | +0.0289 [+0.013, +0.045] |
| hasqry (no timsTOF) | 0.8430 | 0.7947 | −0.0483 [−0.069, −0.030] |
| hcqry (no timsTOF) | 0.5493 | 0.5543 | +0.005 [−0.014, +0.025] |
Cross-fit out-of-fold gain +0.032; random-label control: tuned config +0.0011.
Tuned: phys 1, MLP 1 (0.25), REV2 0.5 (1.0), attention 2.5 (2), CNN 1.5 (2), FT 2, **lib^32 0 (8), analog 0 (24)**,
g5 3 (1.5), own FPNet 0.75 (1.0), pool popularity 0.05 (0.15).

**The gain is almost entirely from dropping the hard library-identity (8 lib^32) and analog (24) terms**:
| variant vs v37 | hx | hb | hbs | has |
|---|---|---|---|---|
| v37 with lib32 0, ana 0 only | +0.023 | +0.041 | +0.017 | −0.041 |
| tuned but lib/ana back (8, 24) | +0.002 | +0.010 | +0.002 | +0.003 |
| tuned with half lib/ana (4, 12) | +0.006 | +0.023 | +0.006 | −0.004 |
Reading: on timsTOF the soft attention retrieval (top-10 softmax) does the library's job better than the hard
identity term; on the non-timsTOF sibling set the hard term still matters. Risk: v21 (which learned to push library
matches down) lost −0.041 on the LB; v20 (which added these terms) gained, but together with a bug fix.

## e2e (v38nofp notebook, holdout-safe stand-ins)
hcqry 0.5810 (v37 0.5804), hxqry 0.9320 (v37 0.8943).

## LB
v38nofp (kernel yomeiotani/casmi26-cpp-library-search-v38nofp v1) pushed 2026-10-22 ~16:10, submitted after checks.
If it loses vs v37nofp: next is the half-lib variant (lib32 4, ana 12) with the other tuned weights.
