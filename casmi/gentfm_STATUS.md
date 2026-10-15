# 生成候補 × 自前 Transformer 逆向きモデル (phys_gentfm) — 状態ファイル

2026-10-15 13〜15時。目的: phys_gen の生成候補（部位異性体だらけ）を自前 TFM（tfm_A/B/C, xlib 学習・漏れなし）の
逆向きスコアで識別できるか。効くなら C++ ランカーに組込む。

## 結論（先に）
1. **仮説「TFM が生成異性体を識別して正解除去時の順位を上げる」はほぼ外れ**。正解除去（rm=1）の改善は cqry +0.002〜0.005、tqry +0.004〜0.006（同点平均、CSV でも 0.127→0.129 / 0.247→0.251）。
   - 生成候補の中だけの順位では TFM 単独は MLP より良い（tqry 0.213 → 0.24〜0.33、cqry 0.405 → 0.39〜0.44 でほぼ同程度）が、
     **融合後の生成候補内の順位はすでに 0.68（cqry）/ 0.74（tqry）**。正解を上回るのは主にプール候補（cqry 115件中69件、tqry 95件中64件で ≥1 件、中央値1件）で、
     差の主因は生成特徴の罰則（isGen −2, 2段 −1: 平均 3.6 / 3.0）とアナログ項（+2.0）。= 「部位異性体の識別」ではなく「生成物とプールの混ぜ方」が律速（`gentfm_diag.py`）。
2. **一方、正解がプールにある通常条件では TFM 項は全 holdout で効く**（CSV 再採点、下表）。シャッフル・逆符号の対照はどちらも大きく悪化 → 本物の情報。
3. 単一モデル A（d256 L4）で十分（ABC アンサンブルとの差はノイズ内）。CPU 推論は g++ -O3・4 スレッドで 22 ms/スペクトル → Kaggle 1200 スペクトルで約 30 秒（Kaggle CPU が2倍遅くても約1分）。
4. 全 train 学習版 `C:\prog\casmi\tfm_allA.bin`（alllib, hold=0, A と同じ設定 12 ep, 42.5 分）作成済み。phys_gentfm で読み込み・実走を確認（tqry は学習に含むので点は無意味: 0.679）。

## 結果: 確定構成 = v20 手の重み + gen2 確定構成 + `tfm=tfm_A.bin,tw=2,tv=0`（MLP REV2 wr=1 は残し、+2·z(TFM-A Σlogit) を追加）
出力 CSV を mrr.py で採点（基準は gen2_STATUS の確定構成の CSV 値）

| holdout | 基準（gen2 確定構成） | + TFM-A (tw=2) | 差 | 対照: シャッフル |
|---|---|---|---|---|
| cqry 正解あり | 0.550 | **0.561** | +0.011 | 0.487 |
| cqry 正解除去 | 0.127 | 0.129 | +0.002 | |
| tqry 正解あり | 0.624 | **0.639** | +0.015 | |
| tqry 正解除去 | 0.247 | 0.251 | +0.004 | |
| sqry | 0.836 | 0.841 | +0.005 | |
| oqry | 0.659 | **0.678** | +0.019 | |
注: sqry/oqry のクエリスペクトルは xlib に含まれる（TFM も MLP REV2 も同条件で学習時に見ている）ので、この2つの上げ幅は楽観的。

### 同点平均の MRR（`gentfm_eval.py`, 特徴ダンプ; 列 = A/B/C/ABC × Σlogit / tanimoto）
基準（MLP rev wr=1）: cqry 0.5485 / cqry除去 0.1268 / tqry 0.6238 / tqry除去 0.2472 / sqry 0.8361 / oqry 0.6594

| 構成 | cqry | cqry除去 | tqry | tqry除去 | sqry | oqry |
|---|---|---|---|---|---|---|
| A logit 追加 tw=2 | 0.5605 | 0.1291 | 0.6392 | 0.2509 | 0.8412 | 0.6777 |
| A logit 置換 wr=2 | 0.5645 | 0.1280 | 0.6322 | 0.2498 | 0.8373 | 0.6741 |
| ABC logit 追加 tw=1 | 0.5598 | 0.1293 | 0.6335 | 0.2534 | 0.8406 | 0.6786 |
| ABC logit 置換 wr=2 | 0.5678 | 0.1276 | 0.6392 | 0.2495 | 0.8379 | 0.6772 |
| ABC tanimoto 追加 tw=1.5 | 0.5614 | 0.1313 | 0.6476 | 0.2568 | 0.8427 | 0.6935 |
| 対照 ABC logit 追加 tw=1 シャッフル(5 seed) | 0.5215 | 0.1198 | 0.6045 | 0.2394 | 0.8252 | 0.6432 |
| 対照 ABC logit 追加 tw=1 逆符号 | 0.5107 | 0.1211 | 0.5939 | 0.2381 | 0.8253 | 0.6314 |
ブートストラップ（ABC logit 追加 tw=1 − 基準, 95% CI）: cqry +0.011 [+0.000, +0.022]、cqry除去 +0.0025 [+0.0001, +0.0058]、tqry +0.010 [−0.006, +0.028]、
tqry除去 +0.006 [+0.001, +0.014]、sqry +0.005 [+0.001, +0.009]、oqry +0.019 [+0.011, +0.029]。
グリッド（`gentfm_grid.py`, 選択 = (cqry + cqry除去 + 2 sqry)/4）は平坦で、上位はどれも ±0.003 内。

### 生成候補の中だけの順位（正解が生成物のクエリ、同点平均 MRR）
| スコア | cqry除去 (n=115) | tqry除去 (n=95) |
|---|---|---|
| MLP REV2 単独 | 0.405 | 0.213 |
| TFM A / B / C 単独 (logit) | 0.412 / 0.409 / 0.387 | 0.243 / 0.326 / 0.241 |
| TFM B tanimoto / ABC tanimoto 単独 | 0.440 / 0.427 | 0.307 / 0.301 |
| 融合（基準） | 0.684 | 0.736 |

## CPU 時間（ビルド機, 他ユーザーと共有中なので目安）
- `tfm_cputime`（g++ 14.2 -O3, Kaggle と同じフラグ, 4 スレッド）: A (d256 L4) cqry 21.9 ms/スペクトル（-march=x86-64-v3 で 16.7）、tqry（66.8 トークン）29.9 ms、B (d512 L6) 110 ms。
- phys_gentfm（MSVC /arch:AVX2, 8 スレッド）内: A 1 モデル 27〜45 ms/スペクトル、A+B+C で 53〜170 ms/スペクトル/モデル（混雑時）。
- → Kaggle は A 1 モデルなら +0.5〜1 分。ABC アンサンブルは +4〜6 分で、得はノイズ内なので不採用。

## 副次的な発見
- libSim（候補自身のライブラリスペクトルとのコサイン）が 1 を超える候補がある（最大 1.86、sqry 117 クエリ / oqry 48 クエリ）。
  各クエリピークを最良ライブラリピークに当てるだけで1対1でないため。8·lib^32 で 1.86^32 ≈ 4e8 になり強制1位。1 で頭打ちにすると oqry 0.6594→0.6632、sqry +0.0007、cqry/tqry 変化なし（`gentfm_libcap.py`）。v20 本番も同じ式。直すなら1対1照合か min(1,·)。

## ファイル
- `cpp/rdkit-min/phys_gentfm.cpp` — phys_gen のコピー + TFM の CPU 推論（`rev/tfm_model.hpp` を CPU float でビルド）。
  argv[24] に追加: `tfm=a.bin+b.bin+c.bin`（確率平均のアンサンブル）, `tw=`（+tw·z(列)）, `tv=`（融合に使う列）, `tshuf=1`（対照: 列をクエリ内で並べ替え）, `tsign=-1`（対照: 逆符号）。
  特徴ファイル: 6 生成列の後ろにモデルごと [Σlogit(p_b)（p を [1e-6, 1−1e-6] にクリップ）, tanimoto 型 Σp_b/(n+Σp−Σp_b)]、複数モデルならアンサンブルの2列、最後に候補 SMILES。
  分子の確率 = クエリのスペクトルの sigmoid 平均（tfm_train eval と同じ）。入力前処理も tfm_train の prepPeaks と同じ。
- `cpp/rev/tfm_model.hpp`（+ rdkit-min/rev/ にコピー）— CPU GEMM を行連続アクセスに（A 非転置のとき; 結果は丸め差のみ）
- `cpp/rev/parallel.hpp`（+ rdkit-min/rev のコピー）— `parallel_cap()`（スレッド上限, 0=全コア）と連続ブロック分割（要素ごとの書き込みで false sharing しない）
- `cpp/rev/tfm_cputime.cpp` — TFM の CPU 推論時間だけを測る（`g++ -O3 -std=c++17 -pthread`）
- `gentfm_eval.py`（同点平均 MRR: 単独/生成内/置換/追加/逆符号/シャッフル/ブートストラップ）、`gentfm_diag.py`、`gentfm_grid.py`、`gentfm_libcap.py`
- ビルド機: `C:\prog\casmi\fwdmodel\gentfm\`（build.bat, run.bat: `run.bat <qry> <lib> <tag> "<opts>" [wr]`。**opts は必ず "" で囲む**: cmd は , と = も引数区切りにする — 最初の実行は opts が "gen" だけになり生成なしで走った）、
  出力 `C:\prog\casmi\gtfo\`（f_*.txt ダンプ, *.csv, *.log。c_rm/t_rm/c_g/t_g/s_g/o_g = ABC 列入り, *_A = 確定構成, c_Ashuf = シャッフル対照）
- 全 train モデル: `C:\prog\casmi\tfm_allA.bin`（ログ gtfo\train_allA.log; `tfm_eval.exe train lib=alllib hold=0 out=tfm_allA d=256 L=4 epochs=12 clip=200`）

## 次にやること
1. Kaggle ノートブックへ: phys_gen → phys_gentfm に差し替え、データセットに tfm_allA.bin（51 MB）と rdkit-min/rev/tfm_model.hpp・parallel.hpp を追加、opts に `tfm=tfm_allA.bin,tw=2,tv=0`。
   g++ に `-march=x86-64-v3` を足すと TFM が約 25% 速い（任意）。提出前に出力 CSV を手元採点と照合（作業の前提どおり）。
2. 正解除去時の律速は「生成物 vs プール」の混ぜ方（isGen 罰則・アナログ項）。生成物の中の順位はすでに 0.7 前後。
   改善するなら: プール候補と生成候補を同じ基準で比べる特徴（例: 生成元アナログとの差スペクトルの説明度）、またはクエリごとに「正解がプールにない」確率を推定して isGen 罰則を可変に。
3. libSim > 1 の修正（1対1照合 or 上限 1）を phys_gen 系と本番に。
4. TFM の tanimoto 列（ABC tanimoto は tqry/oqry で最良）を単一モデルで再確認、ランキング損失版（C）の全 train 学習も候補。
