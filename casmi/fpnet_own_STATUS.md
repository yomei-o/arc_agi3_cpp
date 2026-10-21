# fpnet_own — FPNet-architecture weights trained with OUR code (状態ファイル)

2026-10-21 開始。動機: 主催者回答（RESUME.md 2026-10-21 のルール）で、第三者の重みは学習コードがある場合のみ賞の対象。
公開 FPNet（ahmedberatozer fpnet-full1）は学習コード非公開 → 最終 2 本は FPNet なし。その穴（LB 約 +0.02）を、
**同じ構造・同じ出力ビットの重みを自前コードで train.parquet から学習**し、`fpnet_full1_f16.bin` と同じ形式のファイル
（ドロップイン差し替え）で埋める。

## 出所（provenance、最終の適合確認用）— 重要
- **学習データ**: train.parquet のスペクトルだけ（wlib = 評価用、alllib = 本番、どちらも train.parquet 由来の C:\prog\casmi\*_head/peaks.bin）。
- **ラベル**: 自前スクリプト `cpp/fpown/fp10226.py`（RDKit の Morgan/FeatureMorgan/RDKit path/AtomPair/TopTorsion/MACCS 生成器を直接呼ぶ）で計算した 10226 ビット。
  ビット位置の選択表 `fp_bits.npy`（16551 → 10226 の添字リスト）は推論ファイルの出力ヘッドが何を意味するかの定義（データ）で、重みではない。
- **学習コード**: 自前の C++/CUDA `cpp/rev/tfm_model.hpp`（手書きの順伝播・逆伝播、gradcheck 済み）+ `cpp/rev/tfm_train.cpp`（`fpnet=1` モード）。
  エクスポータ `cpp/fpown/tfm2fpnet.cpp`（テンソル一覧はモジュール定義から手で書いたもの、第三者の重みファイルは読まない）。
- **公開 FPNet の重み・出力は学習に一切使っていない**: 教師（蒸留）にも、初期値にも、ハイパーパラメータ選び（ロジットを合わせる等）にも使わない（ユーザー確認済み、2026-10-21）。
  公開ファイルに触れたのは (a) テンソル名・形の一覧を表示した 1 回（形式確認のみ、`lsfpn.py`）だけ。ggml 推論ツールは notebook と同じソース（ds_rdkit\fpnet）を
  自前のファイルの読み込みにだけ使用。
- 推論側（fpnet_ggml.cpp / fpnet_pool.cpp）は以前から私たちの C++ 移植（ggml、MIT）。

## 1. 推論ファイルの仕様（fpnet_ggml.cpp から確定）
- ファイル: `"FPN1"`(f32) または `"FPN2"`(2 次元テンソルだけ f16、1 次元は f32) | i32 nbits=10226, d=512, layers=8, heads=8, n_tensors=117 |
  テンソルごとに i32 名前長, 名前, i32 ndim, i32 dims[ndim]（torch の順 [out,in]）, データ。fpnet_ggml は名前で引くので順序は不問だが、公開ファイルと同じ順に書く。
- テンソル（順）: mz_emb.inv(256), nl_emb.inv(256), int_emb.weight(512,2), int_emb.bias, pk.weight(512,1536), pk.bias, prec_emb.inv(256), ad.weight(60,512),
  ce_emb.inv(64), gl.weight(512,644), gl.bias, blocks.{0..7}.{n1.weight,n1.bias,qkv.weight(1536,512),qkv.bias,o.weight,o.bias,n2.weight,n2.bias,
  ff.0.weight(2048,512),ff.0.bias,ff.3.weight(512,2048),ff.3.bias}, norm.weight, norm.bias, head.0.weight(2048,1024), head.0.bias, head.3.weight(10226,2048),
  head.3.bias, elem_head.0.weight(256,1024), elem_head.0.bias, elem_head.2.weight(10,256), elem_head.2.bias（elem_head は ggml 未使用 → 0 で書く）。
- 前処理（prep_peaks）: m/z ≤ prec+2 かつ > 0、I/Imax ≥ 1e-3、強度上位 160、m/z 順、強度は sqrt(I/Imax)。空スペクトル → 0 のピーク 1 個。
- ピークトークン = pk([SinEmb(mz) 512 | SinEmb(max(prec−mz,0)) 512 | int_emb([it, it²]) 512])。SinEmb(x) = [sin(x·inv), cos(x·inv)]、inv = 2π/wav、
  wav = 10^((hi−lo)·i/(n−1)+lo)、lo −2、hi 3.3（ce_emb は lo 0 hi 2.2）。ggml は角度を **float で x·inv** として sinf。
- グローバルトークン = gl([SinEmb(prec) 512 | ce_emb(ce/10+1) 128 | ce_known | merged | mode | log1p(prec)/10]) + ad[adduct]。推論では ce=0, ce_known=0,
  merged = 1/4（単独）/ min(n,8)/4（マージ）、mode = アダクトの極性（不明は +1）、アダクト不明 → 行 59（<unk>）。私たちの 10 アダクト → 行 {0,6,4,1,5,9,8,2,46,47}。
- 8 層 pre-LN（LN eps 1e-5）: x += o(MHA(n1 x))、x += ff.3(GELU_erf(ff.0(n2 x)))。qkv の出力列は [3][H][hd]。マスクなし（1 スペクトル 1 グラフ）。
- 最終 LN → emb = [グローバルトークン, ピークトークン平均] → head.3(GELU(head.0(emb))) = 10226 ロジット。
- 分子ロジット = 0.5·(スペクトルごとの平均 + 極性ごとのマージスペクトルの平均)。候補スコア（fpnet_pool）= cand_fp · logits。

## 2. 学習コード: C++/CUDA（tfm_train の拡張）— PyTorch は使わない
選んだ理由: 既存の自前 Transformer（tfm_model.hpp）が既に同じ関数族（pre-LN、GELU erf、SinEmb、[global, mean] ヘッド、キーのパディングマスク）で、
差はグローバル入力とアダクト/装置の埋め込みだけ → **線形写像で厳密に FPNet 形式へ変換できる**（下）。新規の逆伝播は不要で、gradcheck 済みのコードをそのまま使える。

`tfm_train train ... fpnet=1` で変わる点（既定 0 は従来どおり）:
- 既定の形 d 512 / L 8 / H 8 / ff 4 / HH 2048 / maxPeaks 160（FPNet と同じ）。
- 装置入力なし（常に「不明」= 5、エクスポート時にアダクト表へ畳み込み）、アダクト不明は mode +1（fpnet_ggml と同じ入力）。
- **角度を ggml と同じ float の x·inv で計算**（tfm::fpnetPhase()）。これをしないと m/z 1000 で最も細かい周波数（inv 628）の位相が ~0.03 rad ずれ、
  ggml との差が最大 1.4e-2 になった（スモークテストで確認）。入れた後は 5.7e-6。
- `eval=none`（候補ファイル不要）、`threads=`（CPU 上限）、`dumpz` モード（スペクトルごとのロジットを fpnet_ggml の _spec.bin 形式で書く）。

エクスポート（tfm2fpnet.cpp、私たちの [in,out] 行優先 → torch [out,in]）:
- pk.weight[:, 0:1024] = wtok の SinEmb 行、pk.weight[:, 1024] / [:,1025] = wtok の it / it² 行、残り 0。int_emb = 恒等（weight の (0,0),(1,1) が 1、bias 0）。
- gl.weight[:, 0:512] = wg の SinEmb(prec) 行、mode 列 = wg の mode 行、log1p 列 = wg の log1p 行、ce/ce_known/merged 列 = 0（CE は私たちのデータにない、マージは単独と同じ扱い）。
- ad.weight[行] = ead[アダクト] + ein[5]、それ以外の 50 行（<unk> 含む）= ead[10] + ein[5]。inv = float(2π·iwav)（学習時の角度と同一ビット）。
- f16 化は自前（最近接偶数丸め）、`cpp/fpnet/to_f16.py` の出力とバイト単位で同一を確認。

## 3. 検証（スモークテスト、`cpp/fpown/smoke.bat`、wlib で 3 分学習した重み）
- ラベル: 自前 fp10226.py で pool730 の無作為 300 行を再計算 → **300/300 ビット単位で一致**（pool730_fp10226.bin と同じビット）。
- 往復: FPN1 書き出し → 読み戻し max|diff| 0、FPN2 は f16 丸めのみ（max rel 4.9e-4）、117 テンソル・サイズ一致。**FPN2 のサイズ 99,517,113 B = 公開 fpnet_full1_f16.bin と同じ**。
- 順伝播の一致（hcqry 先頭 5 分子・13 スペクトル、ロジット範囲 −11〜+9、notebook と同じソースからビルドした fpnet_ggml vs 学習器の forward、TF32 off）:
  f32 ファイル **max|diff| 5.7e-6**（平均 2.4e-7）、f16 ファイル **max|diff| 9.6e-4**（平均 7e-5、重みの f16 丸め分）。
- 注意: ビルド機の古い `C:\prog\casmi\fpnet_src\cpp\build\fpnet_ggml.exe`（10-14 15:20）は FPN1 しか読めない（FPN2 で "bad magic"）、`C:\prog\casmi\fpnet\to_f16.py` も古版（1 次元まで f16）。
  使うのは `C:\prog\casmi\fpown\gg\build\fpnet_ggml.exe`（ds_rdkit\fpnet = notebook のソースからビルド、`cpp/fpown/ggbuild.bat`）と `cpp/fpnet/to_f16.py`。

## 4. 学習ジョブ（ビルド機、launch.ps1 で切り離し）
| 名前 | PID | 中身 | 出力 |
|---|---|---|---|
| fpown_train | 20064（wlib 10:13–13:14 済、alllib 13:15〜） | `C:\prog\casmi\fpown\fpown_train.bat`（= cpp/fpown/fpown_train.bat）: (1) wlib 5 ep（hold 20 = 5% の分子で val BCE）→ export、(2) alllib 6 ep（hold 0）→ export → ds_add に配置。B 128, lr 3e-4 cosine, warm 2000, wd 0.05, drop 0.1, clip 1, BCE のみ | ログ `C:\prog\casmi\night\fpown_train.log`、進行 `C:\prog\casmi\fpown\progress.txt`、モデル `C:\prog\casmi\fpown_w_ep*.bin / fpown_w.bin / fpown_all.bin`、`C:\prog\casmi\fpown\fpown_w_f16.bin`、**`C:\prog\casmi\fpown\ds_add\fpnet\fpnet_own_f16.bin`** |
速度: スモークで B 64 が 15.5 step/s（GPU は他エージェントと共有、使用率 ~90%）→ wlib 1 エポック ~40 分の見込み。途中で落ちたら同じ bat をもう一度起動すれば `resume=1` で続きから（_ck.bin）。

## 5. 評価（holdout で安全 = wlib モデル）
(a) 窓内 MRR（`cpp/rev3/win_eval.cpp` に `<x>_mol.bin`（fpnet_ggml の分子ロジット）を読む経路を追加）: 窓候補のビット `C:\prog\casmi\w_{hcqry,hxqry}_fp10226.bin`（fp10226.py、計算中）。
(b) プールの順位: 本番の融合 + 1.0 z(fpnet)、hcqry/hasqry/hxqry、FPNet なしと比較（`bat_h2/h2_fpown.bat`、準備中）。
### 学習の経過（wlib、hold 20 の 2 万スペクトル）
| エポック | val BCE/スペクトル | micro-F1 (P / R) |
|---|---|---|
| 1 | 1636.7 | 0.596 (0.765 / 0.488) |
| 2 | 1554.7 | 0.634 (0.775 / 0.536) |
| 3 | **1530.6** | 0.654 (0.781 / 0.562) |
| 4 | 1541.1 | 0.664 (0.781 / 0.578) |
| 5（最終 fpown_w.bin） | 1555.2 | 0.666 (0.779 / 0.582) |
速度: 1 エポック 2,100〜2,270 s（35〜38 分、B 128、GPU 共有中）、5 エポック 3.0 時間。
val BCE は ep3 が最小（以後は確信過剰で少し悪化）だが、順位の指標（窓内 MRR、下）は ep5 まで単調に上がる → 本番も最後のエポックを使う。

### (a) 窓内 MRR（`cpp/fpown/eval_win.bat`、PubChem ±10 ppm 窓の全候補、正解が窓にあるクエリで MRR_in）
fpown は notebook と同じ fpnet_ggml で f16 ファイルから分子ロジットを出し、"logit" = fpnet_pool と同じ cand_fp·logits。基準 rev2_w（wlib、clean、fp3_STATUS）。
| holdout | モデル | logit | tanimoto | z(tanimoto)+0.15 pop | 対照（ラベル置換）logit |
|---|---|---|---|---|---|
| hcqry（窓 5,470） | rev2_w | 0.173 | 0.254 | 0.357 | 0.014 |
| | fpown ep1 | 0.238 | 0.265 | 0.342 | 0.032 |
| | fpown ep2 | 0.296 | 0.296 | 0.368 | 0.028 |
| | fpown ep3 | 0.329 | 0.324 | **0.401** | |
| | **fpown ep5（最終）** | **0.341** | **0.329** | 0.392 | |
| hxqry（窓 12,447） | rev2_w | 0.090（cal 0.229） | 0.299 | 0.300 | 0.000 |
| | fpown ep2 | 0.376 | 0.407 | 0.392 | 0.000 |
| | fpown ep3 | 0.474 | 0.525 | 0.494 | |
| | **fpown ep5（最終）** | **0.535** | **0.552** | **0.538** | |
→ 自前 REV2（rev2_w）を全変種で上回る。fpnet_pool の採点形（logit 和）で hcqry 0.17 → 0.34、hxqry 0.09 → 0.54。
fp3 の目標（「hxqry clean で窓内 ≥ 0.15〜0.2」）を大きく超える → **PubChem 窓の採点（phys_v28pc 系の pcrev）にもこのモデルを使う価値が高い**（未実装、次の候補）。

### (b) プールの順位（本番融合の holdout 版 + 1.0 z(fpnet)、`cpp/fpown/pool_eval.bat` → `bat_h2/h2_fpown.bat`）
融合 = v30 の holdout 版（`phys_v30pc`、pcrev=tan、REV2 = rev2_w、MLP = fwd_mlp_rank_ylib、ライブラリ wlib（hxqry は zlib = xinstr））。公開 FPNet の項だけ外したもの = 基準。
数字は PubChem 挿入後の本番リスト（_ins.csv）を mrr.py で採点（括弧内は挿入前）。対の差の SE は ~0.005–0.01。
| holdout | FPNet なし（基準） | + 1.0 z(fpown ep2) | + 1.0 z(**fpown ep5 = 最終**) | 差（ep5） |
|---|---|---|---|---|
| hcqry | 0.524 (0.531) | 0.537 (0.544) | **0.537** (0.544) | **+0.013** |
| hasqry | 0.836 (0.838) | 0.837 (0.839) | 0.836 (0.838) | +0.000 |
| hxqry | 0.857 (0.858) | 0.868 (0.870) | **0.869** (0.870) | **+0.012** |
重み wf（ep2）: wf 2.0 は hcqry 0.518（基準より −0.006）、hxqry 0.869 → **wf 1.0（公開 FPNet と同じ）のまま**。ep5 の wf グリッド（挿入後）: hcqry 0.5/1.0/1.5 = 0.542/0.537/0.536、hasqry 0.5/1.0 = 0.838/0.836、hxqry 0.5/1.0/1.5 = 0.866/0.869/0.871
（基準 0.524 / 0.836 / 0.857）。平均の上げ幅 wf 0.5 +0.010、wf 1.0 +0.008 でノイズ内の差 → 推奨は 1.0 のまま（0.5〜1.0 なら同等）。
窓内 MRR は ep2 → ep5 で大きく伸びた（hxqry 0.38 → 0.54）のに、プールの順位はほぼ同じ: プールの候補は ~100/クエリで他の項（物理・REV2・ライブラリ・アナログ）が
既に強く、FPNet 項の伸びが乗りにくい。hasqry（sibling、ライブラリに正解の別スペクトルあり）は天井付近で効かない。
（公開 FPNet の LB 寄与は約 +0.02。公開重みは train 全体で学習しているのでどの holdout でも漏れており、同じ物差しでは比べられない。）

## 6. 本番モデル（alllib = train 全体、hold 0、6 エポック）
**完了（2026-10-21 18:25）**: 6 エポック 18,653 s（後半は GPU 混雑で 1 エポック ~58 分）。`fpnet_own_f16.bin` 99,517,113 B（公開 f16 と同サイズ）、往復 OK、
notebook と同じソースの fpnet_ggml で読み込み・順伝播 OK（hcqry 5 分子）。train 全体で学習しているので holdout の点は漏れ = 測っていない。
- ジョブ fpown_train（PID 20064）の後半。開始 13:15、1 エポック ~37 分（19,840 step）→ **17:00 ごろ終了見込み**。終わると自動で
  `C:\prog\casmi\fpown_all.bin`（TFM1）→ `C:\prog\casmi\fpown\fpown_all_f32.bin` と **`C:\prog\casmi\fpown\ds_add\fpnet\fpnet_own_f16.bin`**（FPN2、99,517,113 B）、
  `C:\prog\casmi\fpown\progress.txt` に "all done"。
- 確認: `type C:\prog\casmi\fpown\progress.txt`、`powershell -c "Get-Content C:\prog\casmi\night\fpown_train.log -Tail 3"`、`powershell -c "Get-Process -Id 20064"`。
  落ちていたら `powershell -ExecutionPolicy Bypass -File C:\prog\casmi\night\launch.ps1 fpown_train C:\prog\casmi\fpown\fpown_train.bat`（wlib は済みなので飛ばし、alllib は _ck.bin から resume）。

## notebook での使い方（ドロップイン、まだ何もアップロード・push・提出していない）
notebook（casmi_match_v33*.ipynb など）の FPNet 段:
```
FPNET_SRC = RDKIT_MIN.parent / "fpnet"
FP_W = FPNET_SRC / "fpnet_full1_f16.bin"
```
→ ds_add\fpnet\fpnet_own_f16.bin をデータセットの `fpnet/` に入れて **`FP_W = FPNET_SRC / "fpnet_own_f16.bin"`**
（別データセットにするなら `FP_W = next(Path("/kaggle/input").rglob("fpnet_own_f16.bin"))`）。fpnet_ggml / fpnet_pool のソース、pool730_fp10226.bin、
融合の `+ 1.0 z(fpnet)` は変更なし。**適合のため、最終提出に使うデータセットからは fpnet_full1_f16.bin を外す**（読み込まないだけでなく同梱もしない方が明確）。
学習コードとして提出するもの: cpp/rev/tfm_model.hpp, tfm_train.cpp, backend.hpp, parallel.hpp（+ rev2_model.hpp, fp2.h, ecfp.h はビルドに必要）、cpp/fpown/（fp10226.py, tfm2fpnet.cpp, build.bat, fpown_train.bat）、fp_bits.npy。

## 再開メモ（次にやること）
1. 本番モデルの完了確認（6 の確認コマンド）。`fpnet_own_f16.bin` を fpnet_ggml で読めること: `C:\prog\casmi\fpown\gg\build\fpnet_ggml.exe C:\prog\casmi\fpown\ds_add\fpnet\fpnet_own_f16.bin C:\prog\casmi hcqry C:\prog\casmi\fpown\gg_all_hcqry 4 5`
   （train 全体で学習しているので holdout の点は漏れ = 意味なし、読めることだけ見る）。
2. notebook: v33nofp 系に FP_W を差し替えた版を作る（コーディネータ / ユーザー判断で提出）。
3. 伸びしろ: このモデルは PubChem 窓の中の選別で REV2 よりずっと強い（hxqry 窓内 0.54 vs 0.09〜0.30）→ PubChem チャネルの窓内採点（pcrev）に
   fpown のロジット（tanimoto 形）を使う版を試す（phys_v30pc の pcrev の入力を差し替え）。正解の 6〜7 割がプール外という見積もりなら、プール内の +0.012 より効く可能性。
4. 学習の改善候補（未）: エポック数（ep5 まで単調に改善 → 8〜10 も）、FPNet 型の対照損失（tfm_train の rank=、窓内の難しい負例）、マージスペクトルでの学習（merged 列は今 0）。
