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

## 結果（融合 MRR@25、cqry / tqry）
| モデル | rev 単独 | 融合 wr=1 | 融合 wr=1.5 |
|---|---|---|---|
| rev なし | – | 0.5366 / 0.6228 | |
| v4k MLP (rev2_vh, 12 ep) | 0.3959 / 0.3435 | 0.5640 / 0.6345 | 0.5666 / 0.6398 |
| TFM d256 L4、**1エポックだけ**（smoke, clip=1） | 0.3663 / 0.3949 | 0.5629 / 0.6342 | 0.5699 / 0.6363 |

（随時追記）

## 次
- run A: d256 L4 12エポック（clip=200）→ 学習曲線とエポック毎の融合 MRR
- 良ければ d512 L6〜8 へ拡大
