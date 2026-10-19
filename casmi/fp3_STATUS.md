# fp3 — ranking the answer INSIDE the PubChem mass window (状態ファイル)

2026-10-19 開始。動機（RESUME.md「2026-10-19 夕方の方針転換」）: 隠しテストの正解の 6〜7 割はプール外 → 点は PubChem の ±10 ppm 窓（~1.3 万件の異性体）の中で正解を何位にできるかで決まる。今の自前部品（REV2）は窓内 MRR ~0.05、知名度は普通の分子では効かない。
目標: 自前の spectrum → fingerprint → 候補採点で窓内 MRR 0.05 → ≥ 0.15。

## 物差し（window MRR）
- `cpp/rev3/win_build.cpp`: クエリ集合の窓キャッシュ。phys_pc / phys_v28pc と同じ窓（スペクトルごとのアダクト、2M は 1/2、±10 ppm、マージ）で PubChem 表（pc_mass/off/smi/pop）から全化合物を読み、**非立体カノニカル SMILES（MolToSmiles 非異性体、再パース 1 回 = phys_v28pc の canon）でまとめる**（立体異性体の CID が正解を押し下げないように; pop = その CID の最大）。正解 = キーが一致する候補（互変異性体・電荷形の違いは不一致 = やや悲観的）。
  出力 `C:\prog\casmi\w_<qry>_win.bin`（クエリごとの候補 uid リスト + 正解位置）、`w_<qry>_wmols.txt`（uid → SMILES）、`w_<qry>_wpop.bin`。
- `cpp/rev3/fp3_prep.cpp apply`: 任意の語彙（fp2 = v4k など、fp3）で候補ビット `w_<qry>_<tag>.bin`。**v4k で fp2_prep と bit 単位で一致を確認**（fcand_mols.txt 53,374 分子、fc /b 差なし）。
- `cpp/rev3/win_eval.cpp`（CUDA）: モデルの予測（分子ごとにスペクトル平均）で窓の全候補を採点、MRR@25（同点は悲観的）。
  MRR_in = 正解が窓にあるクエリで平均、MRR_all = 全クエリで平均（窓にない = 0）。変種: logit 和、logit [0.01,0.99] クリップ、Platt 較正 logit 和（cal）、cal×held-out F1（cal_wq）、連続 Tanimoto。各々 + 0.15·pop（z 化後）、知名度だけ、**ラベル置換の対照（クエリ q を q+7 の予測で採点）**。
- 評価集合: hcqry（知名度を合わせた COCONUT、denovo）、hxqry（timsTOF 普通の分子、xinstr）。ビット選択用の窓は haqry（評価しない holdout、ラベルは使わない）。
- 学習ライブラリ **wlib** = train − (haqry, hbqry, hcqry, hxqry の全スペクトル) − (hasqry, hbsqry のクエリスペクトル)（`build_ylib.py ... alllib wlib haqry,hbqry,hcqry,hxqry hasqry,hbsqry`）。ylib は hxqry の分子（他装置スペクトル）を含み、zlib は hx の他装置スペクトルを含むので、**hxqry では rev2_y / rev2_z は漏れあり**、wlib で学び直した rev2_w が両方で clean な基準。

## 新しいビット（fp3.h, FP3_VERSION 1、全部自前定義）
kind 0/1 = fp2 の ECFP/FCFP r0–3（同じハッシュ）、2 = 原子対（粗い原子型 × 位相距離 1–12）、3 = 環の置換パターン（環原子の元素 + 環外置換基クラスの巡回列を回転・反転で正準化、細/粗の 2 種）、4 = 環系（縮環ブロックの環サイズ・ヘテロ原子数・芳香環数、環系の数）、5 = 個数（ECFP r0/r1 環境の重複度 ≥2,3,4,6、元素数・ドナー・アクセプター・環数の閾値）、6 = 直線パス 2–6 結合（元素 + 芳香族、結合次数、向き正準化）。
語彙の選び方（`fp3_prep count` → `select`）: wlib での文書頻度 ≥ 100 の特徴のうち、**窓の中での情報量 I = 窓ごとの「その特徴を持つ候補の割合」の 2 値エントロピーの平均**（haqry の窓 400 個 × 各 3000 候補）が大きい順。fp3a8k = 純粋に I 順 8192、fp3b8k = v4k 全部 + I 順で 8192 まで、fp3c16k = v4k + I 順で 16384。

## 結果

### 1. 物差し（基準値、2026-10-19 16:08）
窓: **hcqry 正解が窓にある 339/400 (0.848)、窓の平均 5,470 候補（非立体キーでまとめた後; 生の CID は 6,677）**、全クエリで 2,085,735 種類。win_build 363 s（8 スレッド）、v4k ビット 364 s（2 スレッド）、win_eval は採点 1.3 s。

hcqry、rev2_y.bin（ylib で学習 = hcqry の分子を見ていない）:
| 採点 | MRR_in | MRR_all | top1 | top25 | top100 | 中央順位 | 対照（ラベル置換）MRR_in |
|---|---|---|---|---|---|---|---|
| 知名度だけ | 0.145 | 0.123 | 0.083 | 0.401 | 0.634 | 41 | – |
| logit 和（phys_pc / v28pc の窓内の並べ方） | 0.185 | 0.157 | 0.106 | 0.481 | 0.646 | 28 | 0.016 |
| logit [0.01,0.99] | 0.184 | 0.156 | | | | 30 | 0.014 |
| Platt 較正 logit 和（cal） | 0.222 | 0.188 | 0.133 | 0.555 | 0.705 | 16 | 0.018 |
| cal × held-out F1 | 0.221 | 0.188 | | | | 16 | 0.018 |
| **連続 Tanimoto** | **0.266** | **0.225** | 0.165 | 0.605 | 0.764 | 12 | 0.024 |
| z(logit) + 0.15 pop | 0.297 | 0.251 | 0.195 | 0.637 | 0.764 | 8 | |
| z(cal) + 0.15 pop | 0.333 | 0.282 | | | | 6 | |
| **z(Tanimoto) + 0.15 pop** | **0.367** | **0.311** | 0.260 | 0.711 | 0.835 | 5 | |

**読み方（重要、測り方の訂正）**: 窓の中の REV2 は「MRR 0.047」（pcslot_STATUS の tqry 正解除去、rev2_vh、logit 和、CID 単位）よりずっと強い: 知名度を合わせた hcqry で logit 和 0.185、Tanimoto 0.266。対照は 0.016〜0.024（偶然 ~0.0002 より大きいのは「天然物らしい候補」を選ぶ一般的な好みのため）。
0.047 との差の候補: (a) 集合の違い（tqry は有名な天然物で、配糖体・フラボノイドなど窓内に近い異性体が多い）、(b) CID 単位 vs 非立体キー単位、(c) モデル（rev2_vh/xlib vs rev2_y/ylib）。→ tqry に同じ物差しを当てて切り分け中（下）。
**切り分け（16:22、tqry に同じ物差し）**: tqry 正解が窓に 173/176、窓の平均 **10,777** 候補（hcqry の 2 倍）。
| tqry（176） | logit | cal | Tanimoto | z(cal)+0.15pop | 知名度だけ |
|---|---|---|---|---|---|
| rev2_vh（xlib = tqry の分子なし、clean） | **0.058** | 0.133 | 0.171 | 0.551 | 0.537 |
| rev2_y（ylib は tqry の分子を含む = 漏れ） | 0.518 | 0.626 | 0.638 | 0.799 | |
→ 0.047 は再現（0.058、差は非立体キーでのまとめ）。**hcqry の 0.185 との差は集合の違い**（tqry = 有名な天然物、窓が 2 倍、近い異性体が多い）で、物差しの誤りではない。
**漏れの大きさ**: 同じ分子のスペクトルを学習で見ていると窓内 MRR は 0.06 → 0.52（tqry）。
hxqry（timsTOF の普通の分子、窓の平均 **12,447**、正解が窓に 348/400 = 0.870）では **知名度は無力（MRR_in 0.010）**、rev2_y（hx 分子のスペクトルを見ている）logit 0.382 / cal 0.634 / Tanimoto 0.666、rev2_z（hx 分子の他装置スペクトルだけ見ている）0.205 / 0.450 / 0.484 → どちらも漏れ。**clean な基準は wlib で学び直した rev2_w（今夜のジョブ）**。
→ 普通の分子で clean な数字は今のところ hcqry だけ: **logit 0.185 / cal 0.222 / Tanimoto 0.266**。有名な天然物（tqry、clean）では 0.058 / 0.133 / 0.171。どちらでも **logit 和 → 較正・Tanimoto で 1.2〜3 倍**。

すぐ使える知見: **窓内の並べ方を logit 和から連続 Tanimoto（または Platt 較正）に替えるだけで MRR_in +0.08（logit 0.185 → Tanimoto 0.266）**、知名度 0.15 込みでも +0.07（0.297 → 0.367）。phys_v28pc の PubChem v2 チャネルの前段（`z_window(REV2)+0.15 pop`、REV2 = logit 和）にそのまま効くはず。

## 今夜のジョブ（ビルド機、WMI で切り離し、launch.ps1）
| 名前 | PID | 中身 | 出力 |
|---|---|---|---|
| fp3_night | 22232 | `C:\prog\casmi\fwdmodel\rev3\night_fp3.bat`（= cpp/rev3/night_fp3.bat）: 窓（hxqry, haqry; hcqry は済）→ wlib → v4k ビット → 基準 eval → **rev2_w**（wlib, v4k, 12 ep, hold 20）→ fp3 語彙 3 種（count ≥100 on wlib, select on haqry 窓）→ ビット → rev3_fp3b8k / fp3a8k / fp3c16k（12 ep）→ 全 eval（ens=1）→ fp3c16k を 20 ep・4096×2（_big）→ eval → SUMMARY | `C:\prog\casmi\fp3\progress.txt`, `*.log`, **`C:\prog\casmi\fp3\SUMMARY.txt`**, モデル `C:\prog\casmi\rev2_w.bin, rev3_fp3*.bin` |
| fp3_ctr | 28044 | `C:\prog\casmi\fwdmodel\rev3b\night_ctr.bat`（= cpp/rev3/night_ctr.bat）: fp3_night の "all done" を待つ（最大 10 時間）→ wlib から 30,000 分子（py/fp3_dq.py）→ 各分子の窓からランダム ~64 件の PubChem 化合物（win_build sampleK=64）= 窓内の偽物 → **窓内対照学習**（rev3_train ctr=: 学習スペクトルごとに softmax(0.2·logit 和) で正解 vs 偽物の CE を BCE に加える）: rev2_w から 3 ep の微調整 lam 0.5 / 2、対照 = 同じ微調整で ctr なし、fp3b8k の微調整、v4k でゼロから 12 ep → eval | `C:\prog\casmi\fp3\ctr_progress.txt`, `ctr_*.log`, **`C:\prog\casmi\fp3\CTR_SUMMARY.txt`** |
確認: `powershell -c "Get-Process -Id 22232,28044"`、止めるとき `taskkill /pid <pid> /t /f`（自分の PID だけ）。CPU 8 スレッド + GPU。
ツールは 2 か所: `fwdmodel\rev3\`（fp3_night が使う; ctr なし版）、`fwdmodel\rev3b\`（ctr/init/sampleK 版 = 今のソース）。ソースは cpp/rev3/、ヘッダ fp2.h / rev2_model.hpp / backend.hpp / parallel.hpp は cpp/rev/ から build.bat の隣にコピーしてビルド（`build.bat` は自分のディレクトリでビルド）。

## 再開メモ（次にやること）
1. `type C:\prog\casmi\fp3\SUMMARY.txt` と `CTR_SUMMARY.txt` を読む。見る行: hcqry / hxqry の rev2_w（clean 基準）と rev3_fp3* の logit / cal / tanimoto の MRR_in、SHUFFLED が ~0.02 以下であること、ens（z 和）。
   成功の判定: 普通の分子（hcqry・hxqry とも）で rev2_w より MRR_in が上がるか。目標は Tanimoto/cal で ≥ 0.15 は hcqry では既に超えている → 目標を「hxqry clean（窓 1.2 万）で ≥ 0.15〜0.2」「tqry clean（rev2_vh 系）で 0.17 → 0.25」に読み替える。
2. tqry（有名な天然物、窓 1.1 万）でも clean に測るには、tqry の分子を除いたライブラリが必要（xlib は除いているが ylib/wlib は含む）: wlib から tqry・cqry の分子も抜いた vlib で学ぶか、rev2_vh 系（xlib）で fp3 を学び直す。
3. 組み込み（すぐ効くはず、学習不要）: phys_v28pc の PubChem v2 チャネル前段 `z_window(REV2)+0.15 pop` の REV2 を **logit 和 → 連続 Tanimoto（または Platt 較正 logit 和）** に、窓の候補を **非立体キーでまとめる**。hcqry で z(.)+0.15 pop が 0.297 → 0.367、tqry clean で z(cal)+0.15 pop 0.430 → 0.551。cpp/rdkit-min/phys_v28pc.cpp を直接変えず、コピー（phys_v30pc.cpp など）に `pcrev=tanimoto|cal` を足す。LB への見積もり: f_out × Δ(窓内 MRR)、Δ ≈ +0.07〜0.12（ただし本番の最終リストは窓の上位を本体の融合で並べ直すので、そのまま全部は乗らない）。
4. ctr（窓内対照学習）が効けば: 偽物を「ランダム」から「今のモデルで上位の異性体（hard negative）」に替える（win_build で全窓を作り、rev2_w の上位 64 を選ぶモードを足す）。
5. hxqry の popularity は 0.01 → 本番が enveda 型の普通の分子なら知名度の重み 0.15 は害になりうる（hcqry では +0.1）。LB プローブ（probe_STATUS.md）の結果と合わせて判断。
