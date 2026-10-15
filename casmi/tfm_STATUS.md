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

## 速度（RTX 3090, TF32, B=256, maxPeaks 128）
- d=256 L=4 H=8 HH=2048（約13M パラメータ）: 約 48万 token/s、**1エポック（237万スペクトル, 9273 step）≈ 184 s**。
  MLP（v4k, 1エポック 32 s）の約6倍だが十分速い。

## 結果（融合 MRR@25、cqry 400 / tqry 176、ノイズ目安 ±0.01）
| モデル | rev 単独 | 融合 wr=1 | 融合 wr=1.5 | 融合 最良 wr |
|---|---|---|---|---|
| rev なし | – | 0.5366 / 0.6228 | | |
| v4k MLP (rev2_vh, 12 ep) | 0.3959 / 0.3435 | 0.5640 / 0.6345 | 0.5666 / 0.6398 | |
| TFM A ep1 | 0.3708 / 0.3733 | 0.5682 / 0.6331 | 0.5695 / 0.6367 | wr3: 0.5675 / 0.6470 |
| TFM A ep3 | 0.4182 / 0.4045 | 0.5664 / 0.6393 | 0.5682 / 0.6393 | wr3: 0.5718 / 0.6550 |
| TFM A ep6 | 0.4186 / 0.4320 | 0.5670 / 0.6403 | 0.5729 / 0.6447 | |
| **TFM A ep12（tfm_A.bin）** | **0.4159 / 0.4349** | **0.5706 / 0.6405** | 0.5703 / 0.6434 | wr3: 0.5795 / 0.6552 |
| TFM A ep12, tanimoto スコア | 0.4344 / 0.4472 | 0.5700 / 0.6449 | 0.5698 / 0.6563 | |
| TFM A ep4+ep8+ep12 確率平均 | 0.4212 / 0.4399 | 0.5717 / 0.6391 | 0.5779 / 0.6412 | wr3: 0.5794 / 0.6576 |
| TFM A + MLP（z和, a=0.5, logit） | | 0.5652 / 0.6378 | 0.5771 / 0.6401 | |
| TFM A tanimoto + 1.5 z(MLP) | | 0.5688 / 0.6406 | 0.5733 / 0.6489 | wr2.5: 0.5697 / 0.6643 |

run A = d256 L4 H8 HH2048, 13.1M パラメータ, B256, lr 3e-4 (warmup 2000, cosine), AdamW wd 0.05, drop 0.1, clip 200, 12エポック 37.7分。

学習曲線（run A、held-out 20000 スペクトル = hold=20 の構造単位分割）:
| ep | val BCE/スペクトル | micro-F1 | P / R |
|---|---|---|---|
| 1 | 195.8 | 0.537 | 0.777 / 0.411 |
| 2 | 190.9 | 0.572 | 0.772 / 0.454 |
| 3 | **190.6** | 0.585 | 0.767 / 0.472 |
| 4 | 192.1 | 0.594 | 0.764 / 0.486 |
| 6 | 193.1 | 0.603 | 0.762 / 0.499 |
| 8 | 197.4 | 0.609 | 0.760 / 0.508 |
| 12 | 202.6 | 0.614 | 0.757 / 0.516 |
val BCE は ep3 で最小、その後は過信（分子の暗記）で悪化。F1 と rev 単独 MRR は ep6 前後で頭打ち。

### 読み取り
- **Transformer は単独では MLP よりはっきり強い**: rev 単独 cqry 0.396→0.416、**tqry 0.344→0.435（+0.09）**。
  同じ v4k ビット・同じ xlib・同じ holdout なので差は構造（ピーク=トークン、正弦波 m/z・中性損失埋め込み、attention）による。
- しかし **融合ではほぼ頭打ち**: wr=1 で +0.007 / +0.006、最良 wr でも +0.013 / +0.015 程度（ノイズ ±0.01 と同程度）。
  MLP と z 和を取っても上積みなし → Transformer の情報は MLP の情報をほぼ包含しているが、phys / ana / lib 項とも重なる。
- 公開 FPNet 単独 0.750 / 0.907 との差（漏れなし 0.42 / 0.43）は、構造ではなく **漏れ（holdout 分子を学習に含む）** が大部分と考えるのが自然
  （同じ構造の自前モデルを漏れなしで学習するとこの水準）。残る違いは FPNet の学習目的（±10ppm 候補の hard negative による
  対照ランキング損失）、10226 ビット、規模（d512 L8）、スペクトル併合ビュー。

（随時追記）

## 次
- run B: d512 L6 drop 0.2 6エポック（実行中、約10.5分/エポック）
- 対照ランキング損失（候補プール ±10ppm の hard negative、score = f·z）を BCE に足す — FPNet の学習目的
- phys_ana への組込み（TFM1 読み込み + v4k 語彙ビット）
