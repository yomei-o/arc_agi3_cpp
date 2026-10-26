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

## 評価（翌朝）
tune38/run38.bat と同じ eval（phys_v38td、v36 相当の設定）を hnqry で: lib=nlib、mlp=fwd_mlp_rank_nlib.bin、rev2=rev2_n.bin、
pcfp=rev3_n_fp3c16k_big.bin、cnnmodel=cnn_n_c.bin、FPNet スコア = fpnet_ggml(fpown_n_f16) → fpnet_pool（dot と tan）。知名度は ppopf を外し pc=0 で。
見るもの: (1) hnqry の MRR が LB（0.34）に近い領域か。(2) 既知の LB 順位 v36 (0.344) > v37 (0.343) > v39c (0.339) > v39a (0.328) > v39b (0.319) > v38 (0.303)
を再現するか（ライブラリ ×0/×1/×1.5/×2、FPNet 1.0/1.5）。再現すれば、この物差しで重みとモデルの変更（ピーク前処理、timsTOF 風の増強、窓内対照損失）を回す。
