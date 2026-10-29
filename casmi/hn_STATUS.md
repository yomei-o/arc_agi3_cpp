# hn — 本番と同じ領域の物差し: 天然物 × timsTOF の denovo ホールドアウト（2026-10-26）

## なぜ
train.parquet を出所 × 装置で数えた（2026-10-26）:
| 出所 | 装置 | スペクトル | 分子 | COCONUT にある分子 |
|---|---|---|---|---|
| enveda-180 | timsTOF | 1,153,785 | 182,941 | **51**（他ライブラリにもある分子は 2,288） |
| enveda-np-examples | timsTOF | 1,184 | 250 | 249（全部が他ライブラリにもある、平均 227 本の超有名分子） |
| gnps / mona / riken / massbank / pluskal など | Orbitrap, QTOF, … | ~1.0M | ~90k | 多い |

隠しテスト = 天然物（正解の ~98% が pool730 にある、probe_STATUS）× timsTOF。train でこの組み合わせは np-examples の 250 分子だけ。
手元の物差しは片方ずつしか再現していない: hbqry（合成化合物 × timsTOF、denovo）0.83、hcqry（天然物 × Orbitrap/QTOF、denovo）0.55、LB 0.344。
→ np-examples を denovo（分子のスペクトルを全部ライブラリとモデル学習から抜く）で評価する組 **hnqry** を作り、holdout-safe モデルを nlib で学習し直す。
有名分子（PubChem 知名度 9.3）なので **知名度の項（ppop, PubChem pop）は切って測る**。ライブラリ系の項（lib^32, analog, attention）は denovo では
他ライブラリの同分子スペクトルが無いので「アナログ」としてしか効かない = 隠しテストの「train に無い分子」の部分に相当。

## ファイル（ビルド機 C:\prog\casmi）
- `np_keys.txt`（np_keys.py: inchikey14 \t smiles、250 行）。
- `hnqry_*`: `make_holdout.py C:\prog\casmi --mode denovo --floor=0.002 --keys np_keys.txt --min-spec 1 --n 250 --mz=0,100000 --seed 2606 --no-lib --prefix hnlib,hnqry`
  → **642 スペクトル / 250 分子**（分子あたり最大 3 本 = MAX_Q、質量の制限なし: `--mz` オプションを make_holdout.py に追加）。hnlib_* は使わない。
- `nlib_*` = wlib − hnqry 分子の全スペクトル（54,944 本）: `build_ylib.py C:\prog\casmi wlib nlib hnqry` → 2,448,301 本 / 273,965 分子。
  wlib 起点なので nlib 学習モデルは hx/hb/hc/has/hbs でも安全。
- 夜間ジョブ（launch.ps1、2026-10-26 17:00 起動、進行は `hn\cpu_progress.txt` / `hn\gpu_progress.txt`、ログ `night\hn_cpu.log` / `hn_gpu.log`）:
  - `cpp/hn/night_hn_cpu.bat`（hn_cpu, PID 22796, 6 スレッド）: nlib_v4k / nlib_fp3c16k（fp3_prep apply）/ nlib_fp10226（fp10226.py）→ `rev2_n.bin`（REV2）
    → `rev3_n_fp3c16k_big.bin`（fp3 big、20 ep）→ `fwd_mlp_nlib.bin` / `fwd_mlp_rank_nlib.bin`（順方向 MLP）。
  - `cpp/hn/night_hn_gpu.bat`（hn_gpu, PID 2664）: bits.done を待って `cnn_n_c.bin`（cnn_calib train、12 ep）→ `fpown_n.bin` → `fpown\fpown_n_f16.bin`（自前 FPNet、5 ep）。
  FT は作らない（v37: FT は LB で効果なし）。

## 結果（2026-10-27 朝、`cpp/hn/hn_eval.bat`、final list = _ins.csv、知名度なし）
| 変種 | hnqry MRR@25 | LB の対応 |
|---|---|---|
| 基準（v36 重み、FPNet 内積 1.0） | **0.655** | 0.344 |
| ライブラリ ×0 (0, 0) | 0.677 (+0.022) | 0.303 (−0.041) **逆** |
| ライブラリ ×1.5 / ×2 | 0.642 / 0.631 | 0.328 / 0.319（向きは同じ） |
| FPNet 1.5 | 0.658 (+0.003) | 0.339 (−0.005) |
| FPNet Tanimoto 形 1.0 | 0.667 (+0.012) | 0.345 (+0.001) |
| 基準 + 知名度（ppop 0.15） | 0.768 (+0.11) | （有名分子なので当然） |
読み: (1) 水準 0.655 は LB 0.344 から遠い（denovo なので in-library の部分がなく、有名分子なので easy）。(2) ライブラリの項の向きは他のホールドアウトと同じく
LB と逆 → LB の「ライブラリの項が効く」は、隠しテストの 2 割前後が train にスペクトルのある分子（sibling/xinstr 型）であることを意味する。
hnqry は「天然物 × timsTOF × train に無い分子」の部分だけを測る物差し。(3) Tanimoto 形 +0.012 は LB +0.001 と向きは合う。
**結論: この領域でも重みの調整に伸びしろはない。壁は別（下の「上位の内訳」）。**

## 上位の公開ノートブックの内訳（2026-10-27、`casmi-26-from-spectra-to-structures` 第 14 節、メモリ casmi-top-lineage-attribution）
0.341 → 0.455 の内訳: **ICEBERG + GLACIER（MassSpecGym で学習された公開の順方向モデル、候補のスペクトルを予測して上位 60 件を再採点）≈ +0.06〜0.08**、
PubChem の documentation prior（0.15·(ln(1+substances)+ln(1+PubMed))）+ 深い shortlist ≈ +0.013、第 2 エンジン（PubChemLite/ChEBI 候補の LightGBM、
模擬ランカー行で学習）≈ +0.015〜0.03。彼らの fold-safe FPNet は未見の天然物で top-1 33〜42% → 指紋モデルは差の原因ではない。
隠しテストに近い公開パネル = enveda-np-examples の 250 分子（= hnqry と同じ分子）。公開 LB は test の 31%（約 124 分子、誤差 ±0.02）。

## PubChem チャネルの強さの較正（2026-10-27、`py/pc_sweep.py`、`cpp/hn/rm1_hc.bat` = hcqry rm=1 のダンプ t38c_rm1）
Boltzmann merge の PubChem 側（pi = 混合比 bpi、a = relz の係数 ba、T2）をオフラインで総当たり。正解をプールから抜いた hcqry（rm1 = 隠しテストの
「プール外」部分の代理）の利得 vs プール内 4 集合（hc, hx, hb, hbs）の損:
| pi / a / T2 | rm1 | in-pool 平均 | hcqry | hbqry |
|---|---|---|---|---|
| 0.1 / 0.5 / 0.5（v36〜v40 本番） | 0.201 | 0.797 | 0.549 | 0.827 |
| 0.2 / 0.5 / 0.5 | 0.227 | 0.795 | 0.546 | 0.826 |
| 0.3 / 0.5 / 0.5 | 0.254 | 0.792 | 0.536 | 0.825 |
| 0.5 / 1.0 / 0.5 | 0.315 | 0.779 | 0.505 | 0.811 |
プール外の割合 f の仮定で最適が変わる: f=0.1 → pi 0.2（混合 +0.002）、f=0.15 → pi 0.3（+0.004）、f≥0.2 → pi 0.5 a 1.0（+0.008）。
読み: 私たちの PubChem チャネルは「プール外の正解」を 0.20〜0.31 でしか拾えず（上位の公開版は昇格で +0.024〜0.037）、強めても LB の期待利得は
+0.01 以下で LB の誤差（±0.02）より小さい。昇格そのもの（relp/sp）は relz の単位が違い一度も発火しない（0/400）。v42 以降で pi 0.3 を同梱候補。

## PubChem 1 位昇格の較正（2026-10-27、`py/pc_promo.py`）
本番（alloc=boltz）では v2 の昇格ルールは無効（Boltzmann がスロット/昇格を置き換える）。Boltzmann の後段に「PubChem の 1 位候補を
1 位へ」（relz > relp or (S > sp and pop_top >= popp)、gate libSim < 0.9）を足してオフラインで総当たり。
最良（relp 3, sp 3, popp 7）でもプール外 10/15/20% の仮定で +0.0007 / +0.0017 / +0.0027。最も攻めた設定（sp 3, popp 3）で rm1 の昇格
77 回中 37 回正解、プール内では 40+19+34+15 回昇格して全部外れ。**私たちの PubChem 1 位の精度が低いので昇格は効かない** → 上位の +0.024〜0.037
は昇格ルールではなく PubChem 側の採点（全データ学習の FPNet A+B + 知名度 0.15 で 1 億件の窓）の強さから来ている。負の結果。

## 順方向モデル（GLACIER / ICEBERG）の自前 C++ 版（2026-10-27、`cpp/glacier/`）
- GLACIER: `gl_model.hpp` + `gl_feat.hpp`（RDKit C++ の特徴量、Gasteiger は rdkit-min に追加）。Python と段ごとに一致（特徴量 3e-8、
  切断点・断片 全一致、出力 6e-7）。AVX2 で 0.08 s/予測。
- ICEBERG: `ice_model.hpp`。断片生成（3 段の原子除去、ハッシュで重複除去、上位 100）が一致（断片集合 99〜100/100、確率 1e-6）、強度の生値 1e-7。
  GGNN（512、6 ステップ）が支配的で 0.2〜1.0 s/予測（CPU 1 コアでほぼ上限）。
- `fw_score.cpp`: 公開 runner と同じ採点（前処理・CE バケット・合算 CE・エントロピー類似度）。runner とのスコア一致: GLACIER r 0.9988 /
  順序 97.4%、ICEBERG r 0.9937 / 順序 94.9%（残差は float32 の和の順序 → 4 桁丸めと上位 100 の境目）。
- e2e のオフライン λ 掃引（`fw_sweep.py`）: 順位線形のスコアに足すと悪化（−0.03）、逆数順位なら ICE 0.5 + GL 1.0 で hc +0.009 / hx 0。
- Kaggle: v41（Python runner）は既定環境で RDKit wheel（cp312）が入らず無効、v41b は py3.12 イメージ固定で prebuilt バイナリが GLIBC 2.38 で動かず失敗、
  v41c（同イメージ + ソースからコンパイル）実行中。データセット案 `casmi26-fwmodels`（重み + src、ビルド機 `C:\prog\casmi\ds_fwm`）。

## 評価の手順（記録）
tune38/run38.bat と同じ eval（phys_v38td、v36 相当の設定）を hnqry で: lib=nlib、mlp=fwd_mlp_rank_nlib.bin、rev2=rev2_n.bin、
pcfp=rev3_n_fp3c16k_big.bin、cnnmodel=cnn_n_c.bin、FPNet スコア = fpnet_ggml(fpown_n_f16) → fpnet_pool（dot と tan）。知名度は ppopf を外し pc=0 で。
見るもの: (1) hnqry の MRR が LB（0.34）に近い領域か。(2) 既知の LB 順位 v36 (0.344) > v37 (0.343) > v39c (0.339) > v39a (0.328) > v39b (0.319) > v38 (0.303)
を再現するか（ライブラリ ×0/×1/×1.5/×2、FPNet 1.0/1.5）。再現すれば、この物差しで重みとモデルの変更（ピーク前処理、timsTOF 風の増強、窓内対照損失）を回す。

## 2026-10-27 LB results (forward models)
| run | change vs v40 (0.345) | LB | e2e hcqry / hxqry |
|---|---|---|---|
| v42a | pool popularity 0.15 ungated | 0.347 | - |
| v41c | public Python ICEBERG 0.5 / GLACIER 1.0, same-formula rr re-rank | **0.355** | - |
| v43 | own C++ ICE+GL, same form | submitted 16:30, pending | - |
| v44 | own C++ GLACIER as +0.5 z(gl) inside the fused pool score | 0.348 | 0.5955 / 0.8973 (v40 0.5823 / 0.8918) |
First LB evidence that forward models help (+0.010). GPU ICEBERG (gpu_ggnn.cu) committed: 127 pred/s at 4 threads, parity 1e-6.
Note: ice_model.hpp now includes ice_gpu.h -> ship it in the fwmodels dataset src/ on the next version.
v44: e2e +0.013 hcqry became only +0.003 on LB (holdout gain does not transfer 1:1); the same-formula ICE-carrying form (v41c) is the one that moved the LB.

## 2026-10-28
- v43 (own C++ ICE+GL, same-formula rr re-rank) LB **0.356** (best; v41c public Python 0.355) -> own inference replaces the public runners.
  v43 log: ICEBERG hit the 10800 s CPU budget, 4746 / 18515 predictions skipped -> v45 = v43 with budget 18000 (running).
- e2e offline sweep of the fused-score form with ICEBERG (`cpp/glacier/fw_sweep2.py`, GPU fw_score over work_v44/fw_in.txt:
  45k GL + 48k ICE predictions in 1080 s on the build machine). Reproduces v44 exactly (gl 0.5: 0.5955 / 0.8973).
  | w_gl | w_ice | hcqry | hxqry |
  |---|---|---|---|
  | 0.5 | 0 | +0.013 | +0.006 |
  | 0 | 1.0 | +0.020 | +0.000 |
  | 0.5 | 0.25 | +0.023 | +0.005 |
  | **0.5** | **0.5** | **+0.022** | **+0.006** |
  | 1.0 | 0.5 | +0.018 | +0.009 |
  | 1.0 | 1.0 | +0.022 | +0.007 |
  (se about 0.008 hc / 0.003 hx). Caveat: e2e ranked v44 above v43, the LB the other way round.
- Kaggle T4 GPU: fw_build.py gpu=sm_75 (nvcc 12.8) builds; 2597 ICE predictions 198 s GPU vs 2122 s CPU, ice max|diff| 1e-6.
- v46 = v40 + 0.5 z(gl) + 0.5 z(ice) inside the fused score, T4 (running).
- **v46b (GPU worked: all 24,716 ICE predictions, 2017 s on T4) LB 0.338 - below v40 0.345.** The fused-score form
  (v44 0.348, v46b 0.338) loses on the LB although e2e put it first (+0.022 hcqry). The same-formula rr group form
  (v41c 0.355, v43 0.356) is the one that transfers. Build on make_v43_nb.py; judge its variants on the LB only.
  v45 (v43 + full ICE coverage) and v46 (CPU fallback) pending.

## 2026-10-29
- Overnight LB: v45 (v43 + full ICE coverage) 0.356 = v43 -> coverage is not the lever; v46 (fused, CPU) 0.347;
  **v47 (v43 form, GPU, ice 1.0 / gl 1.0) 0.361 (best)** -> the ICEBERG weight is.
- Running: v48a ice 2.0 / gl 1.0, v48b ice 3.0 / gl 1.0 (weight bracket); v49 = v47 + PubChem rank-1 promotion
  (pcdet stats, relz > 3 or (S > 6 and pop >= 5), libSim < 0.9) as an LB probe of the step rejected offline.
- LB: v48a (ice 2.0 / gl 1.0) 0.343, v48b (ice 3.0) 0.338 -> ICE weight peaks near 1.0 (0.5: 0.356, 1.0: 0.361).
  v49 (v47 + PubChem rank-1 promotion) 0.361 = v47; the visible-test run promoted 0 of 400 queries, hidden unknown -> no
  LB effect at relp 3 / sp 6 / popp 5.
- Running: v50a ice 1.0 / gl 0.5, v50b ice 1.0 / gl 1.5 (GL direction).
- rr-form offline calibration (`cpp/glacier/fw_sweep3.py`, e2e work_v41 Python scores) vs LB:
  | ice / gl | e2e hcqry | e2e hxqry | LB |
  |---|---|---|---|
  | 0.5 / 1.0 | +0.0086 | 0 | 0.356 |
  | 1.0 / 1.0 | +0.0017 | -0.0014 | 0.361 |
  | 2.0 / 1.0 | -0.0089 | -0.0034 | 0.343 |
  | 3.0 / 1.0 | -0.0142 | -0.0162 | 0.338 |
  | 1.0 / 0.5 | +0.0088 | -0.0007 | v50a |
  | 1.0 / 1.5 | -0.0099 | -0.0026 | v50b |
  | 0.75 / 0.75 | +0.0101 | -0.0003 | |
  Direction agrees (heavy weights hurt) but the optimum is shifted (e2e 0.5, LB 1.0): e2e cannot fine-tune weights.
- Truth position in the fused top 60 (e2e work_v44): hcqry rank 1: 167, 2-5: 131, 6-25: 48, 26-60: 10, absent 44;
  hxqry 325 / 70 / 4 / 1 / 0 -> going deeper than 25 is worth at most ~2.5%; the lever is the order of the top 5.
- Public 0.468 forward-model path (subagent read of nbtxt/lb-0-468 FROZEN): the FINAL same-formula rerank uses
  **GLACIER alone** (ICEBERG only as a fallback where GLACIER has no [M+H]+ scores; their note: both 0.402, GL only 0.417),
  base = z of the metric-identity RRF score 1/(3+r_main) + 0.6/(3+r_eng2) (not a rank of one list), window 40 then cut
  to 25, z within formula group (ddof 1), GL_LAM 1.0; also reranks the PubChem list (z f.z + z ICE + z GL, top 25);
  molecule score = mean entropy similarity over spectra. Stage order: main list -> PubChem rerank/merge/promotion ->
  RRF with engine 2 (40) -> GLACIER rerank -> engine-2 top-1 to rank 2 -> neural RRF 0.25.
- v51a = v47 with ice 0 / gl 1.0 (GLACIER alone): pushed, auto-submit after 09:05 JST 10-30 (pw_v51a.bat, wait_0905.ps1).
