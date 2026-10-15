# 自前 Transformer（spectrum → fingerprint, FPNet型）— 状態

2026-10-15。目的: FPNet と同型の Transformer を **自前 C++/CUDA（PyTorch なし）** で xlib（holdout 分子を除いた train）から学習し、
v4k MLP 逆向きモデルと **漏れなし** で比べる。

## 実装（新規ファイルのみ）
- `cpp/rev/tfm_model.hpp` — モデル本体。backend.hpp と同じく1ソースで CUDA（nvcc -DUSE_CUDA -DUSE_CUBLAS）/ CPU。
  `-DTFM_DOUBLE` で double 版（勾配チェック用）。
  - 正弦波埋め込み（FPNet SinEmb: 波長 10^((hi−lo)t+lo), lo=−2, hi=3.3, 位相は double で計算して float の桁落ちを防ぐ）
  - ピークトークン = Linear([SinEmb(mz), SinEmb(max(prec−mz,0)), it, it²])（FPNet の int_emb→pk の合成と同じ関数族）
  - グローバルトークン = Linear([SinEmb(prec), log1p(prec)/10, mode]) + adduct 埋め込み(11) + 装置埋め込み(6)
  - pre-LN Transformer ブロック × L（MHA + key padding mask、GELU(erf) FFN、残差 dropout）、最終 LN、
    [global, ピーク平均] → Linear(2d,HH) → GELU → dropout → Linear(HH, NB)、BCE with logits
  - 全層の forward と手書き backward。attention の QKᵀ / PV / 逆伝播は cuBLAS strided-batched GEMM（[B,H,N,hd]）、
    LayerNorm / softmax は warp-per-row カーネル（CPU では1行1タスク）。ホスト同期なしの launch。
  - AdamW（行列のみ weight decay）、warmup + cosine、勾配ノルム clip、TF32（任意）
  - バッチは「最大ピーク数+1」にパディング。エポック毎に shuffle → 64バッチ分ずつピーク数でソート → バッチ順 shuffle（決定的、再開可能）
  - 出力バイアスは学習分子のビット出現率の logit で初期化
  - 形式 TFM1（設定・重み・語彙・任意で Adam 状態）。`<out>_ck.bin` で途中再開（resume=1、stopAfter=分）
- `cpp/rev/tfm_train.cpp` — gradcheck / train / eval。評価は rev2_train と同じ（f21_cqry/tqry の rev 列だけ差し替えて
  z(phys)+0.25 z(mlp)+wr z(rev)+8 lib^32+24 anaP）。分子確率 = スペクトル平均の sigmoid（rev2 と同じ）。
  holdout は rev2 と同じ `nameHash(InChIKey14) % 20 == 0`（構造単位）。
- ビルド機: `C:\prog\casmi\fwdmodel\tfm\`（bt.bat: CUDA 版 tfm_train.exe と CPU double 版 tfm_check_cpu.exe）

## 勾配チェック（有限差分、極小構成 d=8 L=2 H=2 ff=2 NB=7, B=3 でピーク数 1/4/6 の可変パディング、dropout 0.2 / 0）
| ビルド | eps | 最悪相対誤差（テンソル内 max|num−an| / max|g|） |
|---|---|---|
| CPU double（drop 0.2） | 1e-6 | 7.2e-9 PASS |
| CPU double（drop 0） | 1e-6 | 6.3e-9 PASS |
| CUDA float（drop 0.2, TF32 off） | 3e-3 | 3.7e-4 PASS |
全36パラメータテンソル（埋め込み・各ブロック・ヘッド）。CPU と GPU で loss が一致（8.751818）。
bqkv の key 部分は真の勾配が 0（softmax の平行移動不変性）なので要素単位の相対誤差は意味なし。

## 速度（RTX 3090, TF32, B=256, maxPeaks 128, 平均 35.8 トークン/スペクトル, 6.0% が128ピークで打ち切り）
| 構成 | パラメータ | token/s | 1エポック（237万スペクトル, 9273 step） |
|---|---|---|---|
| d256 L4 H8 HH2048 | 13.1M | 約48万 | 184 s（MLP v4k は 32 s） |
| d512 L6 H8 HH2048 | 30.2M | 約15万 | 約 630 s（単独時）|
| d256 L4 + ランキング損失 K=32 | 13.1M | 約45万（単独時の推定） | 他ジョブと共有で 655 s |

## 実行一覧（すべて xlib, hold=20, v4k 4096ビット, AdamW wd 0.05, warmup 2000 + cosine, clip 200）
| run | 構成 | 学習 | 時間 | モデル（ビルド機 C:\prog\casmi） |
|---|---|---|---|---|
| A | d256 L4, drop 0.1, lr 3e-4, 12 ep | BCE | 37.7 分 | tfm_A.bin（+ tfm_A_ep1..12.bin） |
| B | d512 L6, drop 0.2, lr 2e-4, 6 ep | BCE | 113 分（C と GPU 共有） | tfm_B.bin（+ ep1..6） |
| C | d256 L4, drop 0.1, lr 3e-4, 8 ep | BCE + 4·ランキング（τ=4, K=32, ±15ppm） | 88 分（B と共有） | tfm_C.bin（+ ep1..8） |
ログ: ビルド機 C:\prog\casmiwdmodel	fm{A,B,C}.log。`*_ck.bin` は Adam 状態つき再開用。

## 結果（融合 MRR@25、cqry 400 / tqry 176。ノイズ目安 ±0.01）
フル融合 = z(phys)+0.25 z(mlp)+wr z(rev)+8 lib^32+24 anaP（phys_ana / rev2_train と同一）。

| モデル | rev 単独 (logit) | rev 単独 (tanimoto) | 融合 wr=1 | 融合 wr=1.5 | 融合 wr=3 |
|---|---|---|---|---|---|
| rev なし | – | – | 0.5366 / 0.6228 | | |
| 旧 rev_x3072（本番 v20） | 0.3715 / 0.3402 | | 0.5500 / 0.6124 | | |
| **v4k MLP (rev2_vh, 12 ep)** | 0.3959 / 0.3435 | | **0.5640 / 0.6345** | 0.5666 / 0.6398 | 0.5634 / 0.6331 |
| TFM A ep1 | 0.3708 / 0.3733 | | 0.5682 / 0.6331 | 0.5695 / 0.6367 | 0.5675 / 0.6470 |
| TFM A ep3 | 0.4182 / 0.4045 | | 0.5664 / 0.6393 | 0.5682 / 0.6393 | 0.5718 / 0.6550 |
| **TFM A ep12** | 0.4159 / 0.4349 | 0.4344 / 0.4472 | **0.5706 / 0.6405** | 0.5703 / 0.6434 | 0.5795 / 0.6552 |
| TFM B ep6 (d512 L6) | 0.4368 / 0.5103 | 0.4432 / 0.5271 | 0.5659 / 0.6412 | 0.5680 / 0.6445 | 0.5660 / 0.6563 |
| TFM C ep8 (+ランキング損失) | 0.4338 / 0.4873 | 0.4513 / 0.4995 | 0.5639 / 0.6385 | 0.5720 / 0.6428 | 0.5713 / 0.6580 |
| A+B+C 確率平均 | 0.4522 / 0.5068 | 0.4517 / 0.5235 | 0.5657 / 0.6394 | 0.5712 / 0.6473 | 0.5780 / 0.6556 |
| A ep4+ep8+ep12 確率平均 | 0.4212 / 0.4399 | | 0.5717 / 0.6391 | 0.5779 / 0.6412 | 0.5794 / 0.6576 |
| TFM A + 0.5 z(MLP) | | | 0.5652 / 0.6378 | 0.5771 / 0.6401 | 0.5722 / 0.6547 |

**診断: lib / ana 項を抜いた融合**（z(phys)+0.25 z(mlp)+wr z(rev)、ライブラリ類似・アナログに頼れないクエリの近似）
| rev 列 | wr=1 | wr=2 | wr=3 |
|---|---|---|---|
| なし（phys+mlp） | 0.4121 / 0.3745 | | |
| 旧 rev_x3072 | 0.4897 / 0.4913 | | |
| v4k MLP | 0.4970 / 0.4914 | 0.5035 / 0.4983 | 0.5096 / 0.4768 |
| TFM A | 0.5016 / 0.5356 | 0.5108 / 0.5497 | 0.5128 / 0.5470 |
| TFM B | 0.4954 / 0.5317 | 0.5071 / 0.5700 | 0.5078 / 0.5774 |
| TFM C | 0.5052 / 0.5385 | 0.5202 / 0.5485 | 0.5191 / 0.5543 |
| TFM C tanimoto | 0.5190 / 0.5495 | 0.5196 / 0.5639 | 0.5137 / 0.5581 |
| TFM B tanimoto | 0.4981 / 0.5730 | 0.4987 / 0.5769 | 0.5000 / 0.5745 |
| A+B+C 確率平均 | 0.5012 / 0.5531 | 0.5200 / 0.5599 | 0.5258 / 0.5641 |

## 学習曲線（held-out 20000 スペクトル: hold=20 の構造単位分割。BCE はスペクトルあたり4096ビットの和）
| ep | A val BCE / F1 | B val BCE / F1 | C val BCE / F1 |
|---|---|---|---|
| 1 | 195.8 / 0.537 | 189.2 / 0.561 | 196.1 / 0.538 |
| 2 | 190.9 / 0.572 | **185.4** / 0.597 | 190.4 / 0.567 |
| 3 | **190.6** / 0.585 | 189.3 / 0.612 | **187.4** / 0.583 |
| 4 | 192.1 / 0.594 | 195.7 / 0.622 | 188.2 / 0.592 |
| 5 | 191.6 / 0.600 | 202.8 / 0.625 | 189.7 / 0.598 |
| 6 | 193.1 / 0.603 | 205.8 / 0.626 | – |
| 8 | 197.4 / 0.609 | – | 193.5 / 0.604 |
| 12 | 202.6 / 0.614 | – | – |
val BCE は 2〜3 エポックで最小、その後は分子の暗記による過信で悪化（F1・rev 単独 MRR は上がり続ける）。
B（大きいモデル）は最小 BCE が最も低いが過学習も速い。C（ランキング損失）は BCE の悪化が遅い。

## 読み取り
1. **Transformer は MLP より単独ではっきり強い（漏れなし・同じビット・同じ holdout）**。rev 単独 tqry 0.344 → 0.435（A）
   → 0.487（C, ランキング損失）→ 0.510（B, d512 L6）、tanimoto スコアで 0.527。cqry も 0.396 → 0.437〜0.452。
   規模とランキング損失はどちらも単独性能を上げる。差は構造（ピーク＝トークン、m/z と中性損失の正弦波埋め込み、attention）と学習目的による。
2. **しかし今のフル融合ではほぼ頭打ち**: wr=1 で MLP 比 +0.007 / +0.006（A）、B・C・アンサンブルでも 0.566〜0.578 / 0.638〜0.66。
   ノイズ ±0.01 と同程度で、単独での大差（tqry +0.17）が融合に乗らない。MLP と z 和を取っても上積みなし。
3. **理由は ana / lib 項との重なり**: それらを抜いた診断では TFM の差がはっきり出る（tqry wr=1: MLP 0.491 → TFM 0.53〜0.55、
   wr=2〜3 で 0.55〜0.58 vs 0.50）。正解に近いアナログがライブラリにあるクエリでは anaP（重み24）が支配して rev 列の改善が効かない。
   → TFM が効くのは「アナログが弱いクエリ」。隠しテストでアナログがどれだけ効くか次第で、LB への効果は holdout より大きくも小さくもなりうる。
   v21 の教訓（隠しテストはライブラリに近いものが多い）からは、過大な期待は禁物。
4. 公開 FPNet 単独 0.750 / 0.907（漏れあり）と自前同型モデルの漏れなし 0.44 / 0.51 の差の大部分は、**漏れ（holdout 分子で学習済み）** によるものと推定
   （仮説。残る差は 10226 ビット、d512 L8、スペクトル併合ビュー、学習量）。
5. 融合の重みは手で決めた v20 のまま。TFM 列の最適 wr は 1.5〜3 で MLP（1〜1.5）より大きい。学習した融合（fuse_fit）で評価し直すのが筋。

## 再現コマンド（ビルド機, cwd C:\prog\casmi）
- ビルド: `C:\prog\casmiwdmodel	fmt.bat`（tfm_train.exe = 初版、tfm_eval.exe = 最新ソース tfm_train2.cpp のビルド：ランキング損失・MLP 組合せ・診断入り）
- 勾配チェック: `tfm_check_cpu.exe gradcheck drop=0.2`（double）/ `tfm_eval.exe gradcheck`（CUDA float）
- 学習: `tfm_eval.exe train out=tfm_C d=256 L=4 epochs=8 clip=200 rank=4 tau=4`（`resume=1` で `<out>_ck.bin` から再開、`stopAfter=分`）
- 評価: `tfm_eval.exe eval models=tfm_A.bin[,tfm_B.bin,...] [mlp=rev2_vh.bin] [hold=20]`

## 次の一手
1. **学習した融合**（fuse_fit）に TFM 列（logit と tanimoto）を加えて cqry/sqry で学習・tqry で確認。sqry 比重は v21 の教訓どおり高めに。
2. phys_ana への組込み（TFM1 読み込み + fp2 v4k 語彙ビット）→ pool730 全体で実走、Kaggle 用は推論だけ CPU（d256 L4 なら 1 スペクトル数 ms 見込み）。
3. 早期終了（2〜4 エポック）＋ ランキング損失 ＋ d512：B の ep2〜3 と C の組合せ。分子単位の dropout（同じ分子のスペクトルをまとめて落とす）や
   ピーク drop / m/z ジッタの増強で暗記を抑える。
4. FPNet と同じスペクトル併合ビュー（極性ごとに併合したスペクトルの logit を平均）を推論に追加。
