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

## 評価の手順（記録）
tune38/run38.bat と同じ eval（phys_v38td、v36 相当の設定）を hnqry で: lib=nlib、mlp=fwd_mlp_rank_nlib.bin、rev2=rev2_n.bin、
pcfp=rev3_n_fp3c16k_big.bin、cnnmodel=cnn_n_c.bin、FPNet スコア = fpnet_ggml(fpown_n_f16) → fpnet_pool（dot と tan）。知名度は ppopf を外し pc=0 で。
見るもの: (1) hnqry の MRR が LB（0.34）に近い領域か。(2) 既知の LB 順位 v36 (0.344) > v37 (0.343) > v39c (0.339) > v39a (0.328) > v39b (0.319) > v38 (0.303)
を再現するか（ライブラリ ×0/×1/×1.5/×2、FPNet 1.0/1.5）。再現すれば、この物差しで重みとモデルの変更（ピーク前処理、timsTOF 風の増強、窓内対照損失）を回す。
