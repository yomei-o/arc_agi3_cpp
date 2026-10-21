# pool36 — CNN / attention / fp3 as POOL-ranking terms on top of v35 (状態ファイル)

2026-10-21 夕方. Why: the probe (RESUME top) says only ~2–4 % of hidden answers are outside pool730, so the LB is decided by
ranking the ~100 pool candidates per query. Question: what do (1) w·z(CNN Tanimoto) and (2) the attention term at other
weights (attpool) add to the **pool list** of v35 (answer in pool, rm=0)? FPNet off; holdout-safe models as v35_STATUS
(ylib / zlib libraries, fwd_mlp_rank_ylib, rev2_y / rev2_z, pcfp rev3_fp3c16k_big trained on wlib); CNN = cnn_a_c.bin
(trained on wlib, REV2-style 4096 v4k vocabulary carried in the model file). Nothing uploaded / pushed to Kaggle / submitted.

## Code
- `cpp/rdkit-min/phys_v36pc.cpp` = phys_v35pc + options (all default off -> v35 behaviour):
  - `cnnmodel=<CNN1 file>`: CPU inference of the CNN (cpp/cnn/cnn_model.hpp, copied to the build include dir
    `C:\prog\casmi\fwdmodel\v27\rev\` next to rev2_model.hpp) on the query's spectra, mean p per bit; per candidate the
    continuous Tanimoto sum_bits p / (n + sum p - sum_bits p) over the candidate's bits in the **CNN file's own
    vocabulary** (fp2::Vocab from the model tail; independent of the REV2 model's vocabulary).
  - `cnnpool=w`: + w·z(cnn) over the query's pool + generated candidates in the main fusion (same place/form as attpool).
  - `cnnshuf=k`: query q scored with query q+k's CNN p (shuffled control).
  - `cfdump=file`: every candidate `name idx score isPool pop candname cnn att fp3` (score = final v35 main fusion).
- `cpp/v36/`: build_v36.bat, run36.bat (= run35 with v36 paths, outputs `C:\prog\casmi\v36\out\`), lane1-3.bat, shufx/shufb.bat.
- `py/pool36.py`: offline grid. score' = score + w z(cnn) + (a−2) z(att) + f z(fp3) → top 60 → pool-popularity re-fill and
  the v35 Boltzmann merge reproduced with py/boltz_merge.py (T1 2, T2 0.5, pi 0.2, relz a 0.5). Two lists: **pool** = main
  list top 25 (no PubChem), **final** = after the merge. The w=0 reconstruction is checked against the C++ CSVs.

## Results
(pending)
