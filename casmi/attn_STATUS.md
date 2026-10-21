# attn — memory-based "second engine" (attention = one Hopfield retrieval step)

2026-10-21. Idea (user's physics/AI series, kangaeru-ai: attention = one Hopfield retrieval step, temperature =
how sharp the memory is): for a query spectrum, retrieve the most similar library spectra, weight the retrieved
molecules by w = softmax(sim / T) over the top k, and predict the query molecule's fingerprint as the
attention-weighted average of their fingerprints, p_b = sum_j w_j x_jb (optionally shrunk toward the library bit
frequency, (1-lam) p + lam f). Candidates are scored by the continuous Tanimoto of p vs their bits, exactly the
formula of cpp/rev3/win_eval.cpp. T -> 0 with k = 1 = "Tanimoto to the best library analog"; larger T blends analogs.

## Code
- `cpp/attn/attn_eval.cpp` (CUDA build like win_eval: `cpp/attn/build.bat`, build machine `C:\prog\casmi\attn\`):
  window yardstick of fp3_STATUS (same window caches `w_<qry>_win.bin / _wpop.bin / _<vocab>.bin`, same MRR@25
  with pessimistic ties, MRR_in / MRR_all), REV2 / fp3 models predicted and Tanimoto-scored as in win_eval
  (numbers reproduce fp3_STATUS: rev2_w 0.2544 / 0.2992, rev3_fp3c16k_big 0.3267 / 0.3262 on hcqry / hxqry).
  - retrieval: inverted peak index (0.01 Da bins, |dm| <= tol = 0.01) over EVERY library spectrum (no
    precursor prefilter), polarity-matched; peaks below precursor - 1, top 40 by intensity.
    Similarities: `cos` sqrt-intensity cosine (sum-normalised a: sum sqrt(a_i b_j)); `ent` entropy similarity
    (Li 2021 weighting; sim = sum_matched [(a+b)ln(a+b) - a ln a - b ln b]/ln 4 — the unmatched-peak terms of
    1-(2S_AB-S_A-S_B)/ln4 cancel exactly, so it is a sum of a pair kernel and fits the inverted index);
    `mcos` / `ment` shift-aware versions (per query peak the better of the fragment match and the neutral-loss
    match precursor - m/z, the latter only when the precursors differ by > tol). Per query peak only the best
    posting of each library spectrum counts; sim clipped at 1; `mpen=4`: sim x min(1, matched peaks / 4).
    Per library molecule: max over its spectra and the query's spectra. Top 200 kept (`nbr=`), cached in
    `C:\prog\casmi\attn_<qry>_<lib>_v2_t<tol>_p<pk>_d<maxdm>_m<mpen>_nbr.bin`.
  - grid sims x k {1,10,50,200} x T {0.01,0.03,0.1,0.3} x lam {0,0.2}, then for the best config (and `refcfg=`):
    attn alone, SHUFFLED (query q predicted from query q+7's neighbours), + 0.15 pop, and z-sum ensembles
    `z(model) + w z(attn)` (w 0.25/0.5/1/2) with each model and with both, each with its shuffled-attn control.
  - `diag=1`: how close the retrieved molecules are to the truth (binary Tanimoto); `notwin=1`: pessimistic
    control that drops retrieved molecules that are fp3 twins of the truth (Tanimoto >= 0.999: stereo / tautomer /
    salt forms that survived the InChIKey14 exclusion).
  - `cpp/attn/run.bat <qry> <tag> "opt=v" ...` = the standard run (rev2_w + rev3_fp3c16k_big, vocabs v4k + fp3c16k,
    pop, 6 threads) -> `C:\prog\casmi\attn\<qry>_<tag>.log`; `py/attn_sum.py` summarises logs.
- First version (sims summed over all pairs within tol, loss and fragment matches added) saturated at sim = 1 for
  most queries (pair inflation; hxqry mcos p10 = 1.000) — fixed in v2 (per-peak best match, max of fragment / loss).
  v1 numbers (hc1.log / hx1.log) are superseded.
- Library = **wlib** (train minus every spectrum of haqry/hbqry/hcqry/hxqry molecules, minus hasqry/hbsqry query
  spectra): clean for hcqry, hxqry and haqry (haqry's windows were used to select the fp3 vocabulary, labels unused).

## Results (window yardstick, MRR@25; MRR_in / MRR_all; mpen=4; best config = ment k=10 T=0.1 lam=0 on all three sets)
No FPNet anywhere here (window yardstick = own models only; host ruling 2026-10-21: public FPNet weights cannot be
in final submissions -> these are the FPNet-free numbers). Logs `C:\prog\casmi\attn\{hcqry,hxqry,haqry}_m4.log`
(+ `_m4nt` = notwin control).

| window MRR_in / MRR_all | hcqry (COCONUT, pop-matched, win 5.5k) | hxqry (timsTOF ordinary, win 12.4k) | haqry (COCONUT >=1, famous, win ~7k) |
|---|---|---|---|
| popularity only | 0.145 / 0.123 | 0.010 / 0.009 | 0.331 / 0.285 |
| REV2 rev2_w (Tanimoto) | 0.254 / 0.216 | 0.299 / 0.260 | 0.234 / 0.202 |
| fp3 rev3_fp3c16k_big | 0.327 / 0.277 | 0.326 / 0.284 | 0.280 / 0.242 |
| z(REV2) + z(fp3) | 0.331 / 0.280 | 0.363 / 0.316 | 0.303 / 0.261 |
| **attn alone, fp3c16k bits** (ment k10 T0.1) | **0.472 / 0.400** | **0.612 / 0.532** | **0.422 / 0.364** |
| attn alone, v4k bits | 0.404 / 0.342 | 0.591 / 0.514 (T0.3) | 0.353 / 0.305 |
| attn SHUFFLED (q <- q+7) | 0.036 | 0.001 | 0.020 |
| **z(fp3) + 1 z(attn)** | **0.454 / 0.385** (+0.127) | **0.635 / 0.553** (+0.309) | **0.418 / 0.361** (+0.138) |
| z(fp3) + 2 z(attn) | 0.460 / 0.390 | 0.639 / 0.556 | 0.415 / 0.358 |
| z(REV2) + 1 z(attn) | 0.451 / 0.382 | 0.651 / 0.566 | 0.411 / 0.355 |
| z(REV2) + z(fp3) + 2 z(attn) | 0.456 / 0.387 | 0.655 / 0.570 | 0.423 / 0.365 |
| SHUF control of z(fp3) + 1 z(attn) | 0.225 | 0.131 | 0.179 |
| fp3 + 0.15 pop | 0.390 / 0.331 | 0.312 / 0.272 | 0.459 / 0.396 |
| attn + 0.15 pop | 0.514 / 0.435 | 0.599 / 0.521 | 0.548 / 0.473 |
| **z(fp3) + 1 z(attn) + 0.15 pop** | **0.519 / 0.440** (+0.129 vs fp3+pop) | 0.628 / 0.546 (+0.315) | 0.535 / 0.462 (+0.077) |
| notwin control: attn alone / fp3 + 1 attn / + pop | 0.465 / 0.453 / 0.514 (26 q affected) | identical (no twins) | 0.420 / 0.418 / 0.535 (22 q) |

Grid (fp3c16k bits, MRR_in; hc / hx / ha): similarity at k10 T0.1: cos 0.345 / 0.428 / 0.316, ent 0.368 / 0.503 / 0.334,
mcos 0.416 / 0.484 / 0.377, **ment 0.472 / 0.612 / 0.422** (shift-aware entropy best everywhere).
k (ment, T0.1): k=1 (T->0, best analog only) 0.388 / 0.414 / 0.339; k10 0.472 / 0.612 / 0.422; k50 0.458 / 0.593 / 0.392;
k200 0.429 / 0.527 / 0.358; k200 T0.3 0.290 / 0.340 / 0.229 (too soft = the average library molecule).
T 0.03-0.1 best; T 0.01 = near k=1. Shrinkage lam 0.2 toward the bit frequency: -0.003 to -0.010 everywhere (useless:
z-scoring over the window already removes the common part). The same config (ment, k10, T0.1, lam0) is best on all three
sets, so the grid choice is not a per-set fit. v4k bits are weaker than fp3c16k (more informative vocabulary helps the
memory too). mpen=4 (scale by matched peaks / 4) vs 0: +0.015 (hc) / +0.006 (hx) for attn alone.

Neighbour diagnostics (diag=1, fp3 bits, ment): mean binary Tanimoto of the top-1 retrieved molecule to the truth 0.61 (hc) /
0.47 (hx) / 0.54 (ha); best of the top 10: 0.73 / 0.64 / 0.68. Exact fp3 twins of the truth in the top 10: 19 / 0 / 13
queries — removing every twin (notwin) costs only 0.007 / 0 / 0.003, so the gain is not identity leakage through
stereo / tautomer duplicates.

**Answer to the key question**: yes — the retrieval engine carries information the trained reverse models do not have.
It is stronger alone than either model (hc +0.15, hx +0.29 over fp3), the ensemble with fp3 adds +0.13 / +0.31 / +0.14,
and the shuffled control collapses (attn 0.001-0.036; ensembles with shuffled attn fall BELOW the model alone). On hxqry
(timsTOF, ordinary molecules, popularity useless) the gain is largest: these are enveda-180-like molecules whose close
analogs (same scaffold series) are in the library; the memory recalls them directly, the MLP only on average.
On hcqry the fp3 model adds ~nothing on top of attention (0.472 alone vs 0.454-0.460 in z-sum) — at w=1 the model term
costs 0.02 there, but helps on hx (+0.02) and with popularity on hc (+0.005). Caveat: holdouts remove the molecule but keep
its analog series; if the hidden set's molecules come from series absent in train, the gain shrinks toward k=1-type values.

## Integration (step 2): `cpp/rdkit-min/phys_v34pc.cpp`
PHYS_SECTION

## Resume
RESUME_SECTION
