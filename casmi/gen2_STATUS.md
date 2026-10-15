# 候補生成 C++ 版 (phys_gen) — 状態ファイル

2026-10-15 開始。前段の Python 試作は `gen_STATUS.md`。

## ファイル
- `cpp/rdkit-min/gen_ops.h` — 変換の実装（gen_recall.py の C++ 移植。±OH, ±Me, ±Ac, ±Hex, ±dHex, ±Pen, ±PenF, ±Api, ±GlcA, ±Mal, ±SO3, ±Prenyl, ±NO, ±H2, ±H2O、1段＋2段。RWMol 編集 → sanitize → 非立体カノニカルSMILES で重複除去）
- `cpp/rdkit-min/phys_gen.cpp` — phys_ana3 のコピー＋生成。引数は phys_ana3 と同じ＋ argv[24] にオプション文字列
  `gen=1,rm=0|1,cap=300,nan=80,two=1,prior=<file>,gw=a:b:c:d:e:f,det=<file>,thr=N,grid=0`
  - 生成: 上位80アナログ（variant 4 = modified cosine・6ピーク以上）の前駆体差が変換（または2段）に合うものを全部位で編集、プール窓（±ppm、スペクトルごとのアダクト）に入るものだけ、プール候補・互いと非立体キーで重複除去、(類似度降順, 段数, 順位) で cap
  - rm=1: 正解（とその非立体重複）をプール窓から除去（"正解がプールにない" シミュレーション）
  - 正解判定: プール側は InChIKey14 名、生成側は非立体カノニカルSMILES一致
  - 生成特徴（features ファイルの name の後ろに6列）: isGen, isGen·sim³, isGen·(2段), isGen·log1p(支持アナログ数), isGen·log1p(同じアナログ×変換の生成数), isGen·opPrior
  - 出力: `smiles` 指定で生成候補は SMILES、指定なしは正解なら正解の名前・他は G<k>（mrr.py でそのまま採点）
  - eval では物理スコアを列挙時に計算してフラグメント表を捨てる（メモリ）。fit モードは従来どおり
- `gen2_prior.py`（op 事前確率、cqry の rm 実行の det から）、`gen2_prep.py`（特徴ファイルに prior 列を埋める・cap/2段除外のシミュレーション）、`gen2_fit.py`（fuse_fit で 8+6 重み）、`gen2_hand.py`（v20 手の重み + 生成特徴の重みのグリッド＋座標探索）
- ビルド機: `C:\prog\casmi\fwdmodel\gen2\`（build.bat, run.bat, phys_gen.exe）、出力 `C:\prog\casmi\gen2o\`、prior `C:\prog\casmi\gen2o\gen2_prior_c.txt`

## 結果（途中, 2026-10-15 昼）

### 生成の再現率（正解が生成物に含まれる割合、cap 前 / cap1000 後）
| | C++ (観測前駆体 ±10ppm 窓) | Python Phase 1 (真の質量 ±max(3mDa,10ppm)) |
|---|---|---|
| cqry | 0.330 / 0.312 | 0.360 |
| tqry | 0.551 / 0.551 | 0.562 |
| sqry | 0.677 | – |
| oqry | 0.522 | – |
cap 別（rm、正解のある生成候補が生成順で何番目か）: cqry 125件中 cap25 96, cap100 108, cap200 111, cap300 115, cap1000 125（候補数/クエリ 15 / 50 / 87 / 118 / 253）。tqry 97件中 cap25 77, cap100 93, cap300 97。
正解の 80〜87% は1段変換（2段は生成物の 91〜94% を占めるのに正解は 13〜19%）。

### MRR@25（出力CSVを mrr.py で採点、v20 手の重み、生成特徴の重み 0 = 単純マージ）
| | base (生成なし) | +gen 単純マージ (cap1000) | +gen 正解除去 |
|---|---|---|---|
| tqry | 0.642 | 0.475 | 0.222 |
| cqry | 0.568 | 0.403 | 0.128 |
| sqry | 0.838 | 0.825 | – |
| oqry | 0.661 | 0.620 | – |

### 生成特徴の重み（cap300、選択 = cqry 正解あり:1 + cqry 正解除去:1 + sqry:2、評価 tqry/oqry）
- 手の重み v20 + gw = [isGen −2, sim³ 0, 2段 −1, log1p(支持) 4, nSib 0, prior 1.5]:
  cqry 0.547 / cqry除去 0.127 / sqry 0.836 / **tqry 0.624 / tqry除去 0.247 / oqry 0.659**
- fuse_fit 学習（inLib・COCONUT無効、cqry:cqry除去:sqry = 1:1:2）: tqry 0.601 / 除去 0.272、sqry 0.80（学習重みは sqry で手の重みに負ける、v21 の教訓どおり）
- 学習で cqry のみ: tqry 0.734 / 0.340 だが lib の重みが負（−5.2·lib）= holdout の作りの学習。使わない。

## 確定構成の検証（2026-10-15 12時、出力CSVを mrr.py で再採点）
構成: cap=300, prior=gen2_prior_c.txt, gw=-2:0:-1:4:0:1.5（isGen −2, 2段 −1, log1p(支持) 4, prior 1.5）, 基礎は v20 の手の重み（w 0.25, wr 1, wl 8, pow 32, wa 24, variant 4, pow 3, K 8000, rev2_vh.bin）

| holdout | base（生成なし） | +gen 確定構成 | 差 |
|---|---|---|---|
| tqry 正解あり | 0.642 | 0.624 | −0.018 |
| tqry 正解除去 | 0 | **0.247** | +0.247 |
| cqry 正解あり | 0.568 | 0.550（SMILES→InChIKey14 で採点 0.553） | −0.018 |
| cqry 正解除去 | 0 | **0.127** | +0.127 |
| sqry（ライブラリに正解） | 0.838 | 0.836 | −0.002 |
| oqry（ライブラリに正解） | 0.661 | 0.659 | −0.002 |
- 生成を入れて得になる「正解がプールにない割合」の分岐点: tqry 7%、cqry 12%。半々なら tqry 0.321→0.436、cqry 0.284→0.339。
- 生成候補 cap300: 104〜117/クエリ（中央値 34〜66）、生成後の正解保持 tqry 0.540 / cqry 0.287。
- SMILES 出力の経路確認: T4_gen_smiles.csv を RDKit で InChIKey14 に変換して採点 0.553、パース不能 0、上位25内の InChIKey 重複 21/8570。

### 実行時間（ビルド機 4スレッド, cqry 400 クエリ = 最も重い holdout）
- 列挙（アナログ検索＋候補スコア）壁時計: 生成なし 63 s → 生成あり 293 s（+230 s）。うち生成そのもの 387 thread-s（0.97 s/クエリ）。
- tqry（timsTOF）は生成 0.33 s/クエリ、候補数も少ない。Kaggle の CPU がビルド機の2倍遅くても +8分程度 → 予算 30 分内。

## 進捗
- [x] phys_gen.cpp / gen_ops.h 作成、ビルド
- [x] tqry/cqry（正解あり/除去）、sqry、oqry の評価
- [x] 生成特徴の重み（手＋学習）、確定構成で CSV 再採点
- [x] 4スレッドでの実行時間

## 次にやること
1. 正解除去時の順位改善（0.25/0.13 は正解ありの 0.62/0.55 に遠い）: 生成候補どうしの部位異性体の識別が主。MLP/物理/逆向きの部位感度を見る、op 事前確率を部位の化学（糖は OH の酸性度、メチル化はフェノール/カルボン酸優先など）で細かく。
2. 真偽判定を InChIKey 相当に（非立体SMILES一致は互変異性体を取りこぼす。SMILES→InChIKey 採点で 0.550→0.553）。
3. FPNet 併用時: 生成候補は FPNet スコアなし（クエリ平均 = z 0）で、未検証。FPNet の 10226 ビットは rdkit-min で計算できない（RDKitパス・AtomPair 等）。
4. LB で1回確認（v20/v22 構成 + gen）。正解がプール外の割合の推定にもなる。
