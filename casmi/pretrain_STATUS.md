# pretrain — own DreaMS-style masked-peak self-supervised pretraining (状態ファイル)

2026-10-21 開始。ユーザーの選択: DreaMS 型の自己教師あり事前学習を **全部自前のコード・自前の重み** で（第三者の重みなし、
RESUME.md 10-21 のルール: 賞の対象は学習コードを出せるものだけ）。

## 1. 設計と理由

**なぜ事前学習か.** 今いちばん強い部品は「記憶」= attention 検索エンジン（attn_STATUS.md、窓内 MRR hcqry 0.472 /
hxqry 0.612 単独）で、その類似度は手で作ったシフト対応エントロピー類似度（ment）。手の類似度は「同じ m/z のピークが
いくつ重なるか」しか見ないので、(i) 置換基 1 個ぶんずれた断片の系列、(ii) 装置・衝突エネルギーによる強度パターンの差、
(iii) ピーク同士の関係（中性損失のはしご、同位体、組成上ありえる組み合わせ）を「似ている」と言えない。
DreaMS（Bushuiev 2024）は MS/MS スペクトルだけで（構造ラベルなしに）ピーク集合の文脈から隠したピークを当てる
トランスフォーマーを学び、その埋め込みのコサインがスペクトル類似度・分子ネットワーク・下流の構造予測でよく効くと示した。
train の 2.5M スペクトル（構造ラベル付きだが、事前学習では使わない）は自己教師ありには十分な量で、
**ラベルを使わないので、後で評価する分子の構造がどこにも漏れない**（評価用には wlib = hcqry/hxqry/haqry の分子を全部除いた
ライブラリで事前学習する; 本番用は alllib で学び直す）。

**目的関数（masked peak）.** スペクトル = ピークのトークン列（既存の自前トランスフォーマー tfm_model.hpp と同じ:
正弦波 m/z 埋め込み + 中性損失 prec−m/z の正弦波 + sqrt 強度）+ プリカーサーのグローバルトークン（prec の正弦波、モード、
アダクト、装置の埋め込み）。各スペクトルでピークの 30% を（sqrt 強度に比例した確率で、重複なしに）選び、
**そのピークの特徴をすべて隠す**（m/z、中性損失、強度; トークン = btok + 学習する emask ベクトル）。
最終層の隠したトークンの位置から:
- **m/z の粗いビン（1 Da、nominal）を 1536 クラスの分類**（bin = round(0.9995·m/z); 0.9995 は CHNO の質量欠損の平均的な
  傾きで、ビン境界が質量欠損で割れないようにするため）
- **ビン内のずれ 10·(0.9995·m/z − bin) の回帰**（≈ 質量欠損、組成の情報）
- **強度 sqrt(I/Imax) の回帰**
損失 = CE(bin) + 1·MSE(offset) + 5·MSE(int)（隠したピークで平均）。
選んだ理由: 細かいビン（0.01 Da = 10 万クラス）の分類は出力層が巨大で、ほとんどのクラスが疎。粗いビンの分類 + ずれの回帰は
「どの nominal 質量か（断片の種類）」と「質量欠損（組成）」を分け、計算は 1536+2 出力で済む。
強度を一緒に当てるのは「どの断片がどのくらい出るか」= 結合の切れやすさ（物理モデルが手で書いているもの）を文脈から学ばせるため。
プリカーサーは常に見えている（グローバルトークン）ので、隠したピークは「この前駆体から、残りのピークと整合する断片」として
当てる = 中性損失の構造を学ぶ。
DreaMS との違い: DreaMS は m/z の細かいビン分類 + 保持時間順序の補助タスク、d=1024 L=7、2400 万スペクトル（GeMS）。
ここは 250 万スペクトル、小さめのモデル（GPU 共有）、強度も予測、自前のビン化。

**埋め込み.** 隠さずに順伝播して、最終層 LN 後の (g) グローバル（プリカーサー）トークン、(m) ピークトークンの平均、
(gm) 両者のコサインの平均、を L2 正規化。DreaMS は (g) に相当（プリカーサートークン）。

**使い方 (a) 検索の類似度として.** attn エンジンの ment 類似度を埋め込みコサインに置き換える（または z 和）:
クエリ分子ごとに、そのスペクトル × ライブラリ分子のスペクトルのコサインの最大（極性は attn と同じく一致させる）→ 上位 k 分子 →
softmax(sim/T) → fp3c16k ビット平均 → 窓の候補を連続 Tanimoto で採点（attn_eval と同じ式・同じ物差し）。
比較: ment attention 単独、埋め込み attention 単独、z(埋め込み attn) + w·z(ment attn)、+ fp3 モデル、+ pop、
対照 = クエリ q を q+7 の近傍で予測（SHUFFLED）。
**使い方 (b) 指紋予測の初期値として.** 事前学習のエンコーダ（パラメータの並びは tfm_model.hpp と lnfb まで同一）を
tfm::Model に写し、新しい指紋ヘッド（fp3c16k、16,384 ビット）を付けて tfm_train で微調整 vs ゼロから、同じ予算（エポック数・
ハイパーパラメータ）で比較、窓の物差し（fpown の win_eval、_mol.bin 経路）で。

**Kaggle の CPU 推論.** 4 コアで ~1,200 テストスペクトルの順伝播（d 512 L 6 でも 1 スペクトル数 ms）だけ。ライブラリの
埋め込みは事前に計算して fp16 で同梱（`pt_train embed`、2.5M × 512 × 2 B = 2.6 GB、(g) だけなら d 次元、
d 256 なら 1.3 GB; 必要なら int8 量子化で半分）。コサイン検索は 1,200 × 2.5M × d の内積 = 4 コアで数十秒〜数分
（ブロック化した GEMM）。

## 2. コード（cpp/pretrain/、既存ヘッダは変更なし）
- `mpm_model.hpp`: tfm_model.hpp のコピーを改造（namespace mpm）。エンコーダは同一、追加は emask・マスク入力・
  masked-peak ヘッド（Linear(d,HH) GELU Linear(HH,NC+2)）・損失（CE + 2 つの MSE、行ごとに warp で softmax）・
  ヘッドの逆伝播（隠した行だけ dYf に散らす）・emask の勾配。ファイル形式 MPM1。
- `pt_train.cpp`: `gradcheck`（double の CPU ビルドで有限差分）、`train`（wlib、hold 20 の分子を val、frac=部分集合、
  チェックポイント・resume・stopAfter）、`nbr`（埋め込み近傍 → attn_eval のキャッシュ形式 `ptnbr_<qry>_<tag>.bin`）、
  `embed`（fp16 埋め込みファイル、Kaggle 同梱用）、`export`（TFM1 へエンコーダを写す、(b) 用）、
  `mollogits`（TFM1 モデルの分子ロジット → fpown の win_eval）。
- `pt_eval.cpp`: attn_eval.cpp のコピー + `embnbr=`（近傍スロット 0–2 を埋め込み近傍に置き換え、スロット 3 = ment のまま）+
  2 つの attention の z 和（refcfg = ment k10 T0.1）とその対照。
- `build.bat`（ビルド機 `C:\prog\casmi\pretrain\`、ヘッダと tfm_train.cpp は cpp/rev からコピー）、`pilot.bat`、`night_pt.bat`（学習）+ `night_post.bat`（評価、単独でも実行可）、`smoke_post.bat`（評価経路の確認）。

## 3. 勾配チェック（必須、済）
`pt_check_cpu.exe gradcheck`（double、CPU、d 8 L 2 H 2、3 スペクトル、5 ピークを隠す、1 つはクラス上限、drop 0.1）:
**全 37 テンソル（emask・ヘッド・全ブロック・埋め込み）で worst rel(tensor) 2.3e-8 → PASS**。GPU float 版も 1.2e-3 で PASS。
ログ `C:\prog\casmi\pretrain\gradcheck_cpu.log`。

## 4. パイロット（2026-10-21 午後、`cpp/pretrain/pilot.bat`）
d 256 L 4 H 8（4.3M）、wlib の 10%（237k スペクトル）を 1 エポック、B 128、maxPeaks 64、lr 3e-4、学習 80 秒（GPU は他 2 ジョブと共有、92k tok/s）。
val = hold 20 の分子（学習に使わない分子）の 2 万スペクトル、マスクは固定シード:
| | CE(bin) | acc(bin) | MSE(10·offset) | MSE(sqrt int) | 合計 |
|---|---|---|---|---|---|
| ビンの周辺分布だけ（無条件の予測） | 6.057 | | | | |
| 初期値 | 5.899 | 0.010 | 0.771 | 0.100 | 7.171 |
| **1 エポック後** | **5.161** | 0.033 | 0.639 | 0.078 | **6.189** |
→ 損失は 3 項とも下がる（val は学習にない分子）。

窓の物差し（attn_eval と同じ窓・MRR@25・fp3c16k ビット・連続 Tanimoto; ログ `C:\prog\casmi\pretrain\pilot_eval_{hcqry,hxqry}.log`、
中心化あり = `pilotc_eval_*`）:
| MRR_in | hcqry | hxqry |
|---|---|---|
| 知名度だけ | 0.145 | 0.010 |
| fp3 モデル rev3_fp3c16k_big | 0.327 | 0.326 |
| ment attention（attn_STATUS の最良、基準） | 0.472 | 0.612 |
| **埋め込み attention（パイロット）**: embM（ピーク平均）k10 | **0.200**（T 0.03） | **0.108**（T 0.01） |
| 　embG（グローバル）/ embGM 最良 | 0.133 / 0.167 | 0.070 / 0.076 |
| 　対照 SHUFFLED（q ← q+7 の近傍） | 0.027 | 0.004 |
| 　中心化（ライブラリ平均を引く）embM | 0.200 | 0.107 |
| z(埋め込み attn) + w·z(ment attn)、w = 0.5 / 1 / 2 | 0.400 / 0.438 / 0.459 | 0.463 / 0.566 / 0.595 |
| 　対照 SHUF(埋め込み) + w·ment | 0.269 / 0.393 / 0.445 | 0.351 / 0.526 / 0.581 |
読み方: **パイロットの埋め込みだけで窓内 MRR 0.20 / 0.11、対照 0.03 / 0.004 → 自明でない検索ができている**（80 秒の学習、
4.3M パラメータ、10% のデータ）。ただし手の ment 類似度（0.47 / 0.61）には遠く、z 和は ment 単独より下（w ≤ 2 では埋め込みが
重すぎ; 夜は w 4, 8 も測る）。グローバルトークンよりピークトークン平均が強い（損失はピーク位置にしか掛からないので、
グローバルトークンには間接の勾配しか来ない）。最上位のコサインが 0.95〜0.99 と高い（埋め込みの異方性）→ 中心化を入れたが
パイロットでは効果なし。

## 5. 夜間ジョブ（2026-10-21 14:36 起動、launch.ps1 で切り離し）
| 名前 | PID | 中身 | 出力 |
|---|---|---|---|
| pt_night | 18032（cmd; 学習は pt_train_run.exe） | `C:\prog\casmi\pretrain\night_pt.bat`（= cpp/pretrain/night_pt.bat）: wlib 全体（2.38M スペクトル）で **d 512 L 6 H 8 HH 1024（21.8M）、10 エポック**、B 128、lr 2e-4 cosine、warm 3000、maxPeaks 64、stopAfter 660 分 → `call night_post.bat`: (a) 埋め込み近傍（中心化あり tag w: hcqry/hxqry/haqry、なし tag wnc: hc/hx）→ pt_eval（embG/embM/embGM × k 1/10/50 × T 0.001–0.1、ment 基準、z 和 w 0.5–8、対照）→ SUMMARY (a)。(b) export → tfm_train で fp3c16k を 1 エポック微調整（ft_pre）vs 同じ構成をゼロから 1 エポック（ft_scr）→ mollogits → fpown の win_eval（hcqry/hxqry）→ SUMMARY (b) | 進行 `C:\prog\casmi\pretrain\night_progress.txt`、学習ログ `night_train.log`、**`C:\prog\casmi\pretrain\SUMMARY.txt`**、モデル `C:\prog\casmi\pt_w.bin`（エポックごと `pt_w_ep*.bin`、`pt_w_ck.bin`）、`ft_pre.bin / ft_scr.bin` |
速度（起動直後、GPU 共有）: 26.7k tok/s = 1,000 step / 146 s → 1 エポック 18,580 step ≈ 45 分、10 エポック ≈ 7.5 時間 → (a) は ~22:30、(b) は各 ~1 時間で ~01:00〜02:00 に終わる見込み（他ジョブの具合で前後）。
確認: `powershell -c "Get-Process -Id 18032"`、`type C:\prog\casmi\pretrain\night_progress.txt`。止めるとき `taskkill /pid 18032 /t /f`（自分の PID だけ）。
途中で落ちたら同じ launch をもう一度（pt_w_ck.bin があれば resume=1 で続き）。評価だけやり直すなら `night_post.bat` を単独で（pt_w.bin が必要）。

勾配チェック・パイプラインの確認（14:40、`smoke_post.bat`、パイロットのモデルで）: export → tfm_train resume=1（"resumed ... at step 0"、BCE 1318 → 1139 に下がる）→
mollogits → win_eval が通る（2 分だけの微調整で hcqry tanimoto 0.078、対照 0.027）。中心化つき nbr も通る。

## 6. Kaggle の CPU 推論（見積もり）
d 512 L 6 の CPU（pt_train_cpu.exe、素朴なループ、4 スレッド、ビルド機は他ジョブで混雑）で hcqry 972 スペクトルの埋め込みが 151 秒
→ テスト ~1,200 スペクトルで ~3 分。ライブラリ側は事前計算して fp16 で同梱（`pt_train embed model=pt_all.bin set=alllib out=...`、
d 512 で 2.5M × 1 KB = 2.6 GB; embM だけでよい、int8 なら 1.3 GB）。コサイン検索は 1,200 × 2.5M × 512 = 1.5 TFLOP → ブロック化した
GEMM で数十秒、極性・プリカーサーで絞ればもっと少ない。**推論は Kaggle の 4 コアで十分に可能**。

## 7. 再開メモ（次にやること）
1. `type C:\prog\casmi\pretrain\SUMMARY.txt`（と night_progress.txt）。見る行:
   (a) `best attn fp3c16k: emb? k= T=` の MRR_in（パイロット 0.200 / 0.108 → 夜の d512 10 エポックでどこまで ment 0.472 / 0.612 に近づくか）、
   SHUFFLED が ~0.03 以下、`attn1 + w attn2`（埋め込み + ment の z 和）が ment 単独（config ment の `attn` 行）を超えるか（w 4, 8 = 埋め込みを薄く足す）、
   その SHUF 対照、`models + ...`、中心化の有無（wnc）。haqry も同じ。
   (b) `== hcqry / hxqry` の下の MRR_in: 2 モデル（ft_pre = 事前学習から、ft_scr = ゼロから、同じ 1 エポック）の logit / tanimoto。
2. 成功の判定: (a) 埋め込み + ment の z 和が ment 単独を hcqry・hxqry とも上回る、または (b) ft_pre > ft_scr が両方で。
   どちらも駄目なら: マスク率・ビン（細かい m/z の分類に変える = DreaMS 型）、グローバルトークンに直接の損失（DreaMS の保持時間の代わりに
   プリカーサー質量や「同じ分子の別スペクトルか」の対照損失）、エポック数を増やす。
3. 効いた場合の本番: alllib で事前学習し直し（`night_pt.bat` の lib=alllib、out=pt_all、評価なし）、`pt_train embed` でライブラリ埋め込みを作り
   データセットに同梱、phys_v34pc の attention に埋め込み類似度を足す（attn_STATUS の Integration と同じ場所、別スロット）。
4. ソースの場所: `cpp/pretrain/`（ビルド機 `C:\prog\casmi\pretrain\`、`build.bat` [gpu|cpu]）。tfm_model.hpp など既存ヘッダは変更なし（コピーして使う）。

## 8. 2026-10-22: 長い微調整（ft8）、MLP + CNN との比較・アンサンブル、プールの順位付け項、本番ジョブ、CPU コスト

**夜間の結果（SUMMARY.txt、再掲）.** pt_w.bin（wlib、d512 L6、10 エポック、val total 7.26 → 5.03）。
(a) 埋め込み attention は ment attention に届かず（最良 embM k10: hcqry 0.368 / hxqry 0.412 / haqry 0.320 vs ment 0.472 / 0.612）、
z 和（埋め込み + 8·ment）も SHUF 対照とほぼ同じ → **埋め込み検索は使わない**。
(b) 1 エポックの微調整: ft_pre（事前学習から）hcqry 0.318 / hxqry 0.414 vs ft_scr（ゼロから）0.241 / 0.266（窓 MRR_in、Tanimoto）。

### 8.1 ft8 = pt_w.bin から fp3c16k を 8 エポック微調整（wlib、holdout-safe）
`cpp/pretrain/ft8.bat`（launch 名 pt_ft8、09:15–10:39、GPU 単独で 1 エポック 610 s = 夜の 2.3 倍速）: export（エンコーダ + 新しいヘッド HH 2048）→
tfm_train resume=1 epochs=8 B128 lr 3e-4 cosine warm 2000 → `C:\prog\casmi\ft8_ep<N>.bin`（55.4M パラメータ、fp32 221 MB）。
評価 `cpp/pretrain/ft_eval.bat <stem>`: mollogits（hcqry/hxqry/haqry）→ `pretrain\we2\cnn_win_eval2.exe`（= cnn_win_eval + `_mol.bin` 入力 + `shufm=i`:
アンサンブルのメンバー i だけを q+7 の予測に置き換えた対照）で rev3_fp3c16k_big（MLP）・cnn_a_c（CNN）・FT を単独と z 和で。
表の作成 `py/fte_sum.py`、全表 `casmi/pool37ft/window_ft8.txt`。

窓 MRR_in（Tanimoto）、held-out（hold 20 の分子、2 万スペクトル）BCE / micro-F1:
| エポック | val BCE | F1 | hcqry FT | hxqry FT | haqry FT | hc CNN+FT (0:1:1) | hx CNN+FT | ha MLP+CNN+FT (1:1:1) |
|---|---|---|---|---|---|---|---|---|
| ft_pre（夜、1 ep 単独のスケジュール） | 795.7 | 0.613 | 0.318 | 0.414 | 0.277 | 0.376 | 0.596 | 0.330 |
| 1 | 815.1 | 0.606 | 0.306 | 0.442 | 0.261 | 0.357 | 0.596 | 0.330 |
| 2 | 798.0 | 0.633 | 0.296 | 0.535 | 0.283 | 0.358 | 0.639 | 0.339 |
| 3 | **791.3**（最小） | 0.646 | 0.312 | 0.596 | 0.292 | 0.360 | 0.676 | 0.342 |
| 4 | 808.7 | 0.657 | 0.326 | 0.634 | 0.317 | 0.382 | 0.678 | 0.346 |
| **5** | 840.4 | 0.663 | **0.339** | **0.659** | **0.319** | 0.380 | 0.697 | **0.355** |
| 6 | 864.0 | 0.668 | 0.339 | 0.662 | 0.312 | 0.374 | 0.693 | 0.355 |
| 7 | 894.7 | 0.669 | 0.340 | **0.682** | 0.304 | 0.378 | **0.717** | 0.357 |
| 8 | 909.9 | 0.670 | **0.347** | 0.680 | 0.300 | 0.377 | 0.716 | 0.357 |
| 基準: MLP rev3_fp3c16k_big | | | 0.327 | 0.326 | 0.280 | | | |
| 基準: CNN cnn_a_c | | | 0.294 | 0.546 | 0.247 | | | |
| 基準: MLP + CNN (1:1 / 0.5:1) | | | 0.354 / 0.352 | 0.559 / 0.576 | 0.308 / 0.291 | | | |
対照（shufm=2 = FT だけ q+7 の予測、ep8）: CNN+FT 0.226 / 0.362 / 0.189、MLP+CNN+FT 0.331 / 0.480 / 0.272 → 足した分はクエリ固有。

読み方:
- **単独で MLP と CNN の両方を超える**（ep5〜8）: hcqry 0.339–0.347 vs MLP 0.327、hxqry 0.66–0.68 vs CNN 0.546（MLP 0.326）、haqry 0.30–0.32 vs MLP 0.280。
  holdout 3 つとも最良の単一モデル。hxqry（timsTOF、テストに最も近い）は CNN より +0.11〜0.13。
- **MLP + CNN にも足される**: 最良の 2 モデル（MLP + CNN）に対して 3 モデル z 和は hcqry +0.03（0.354 → 0.38）、hxqry +0.14（0.576 → 0.71）、
  haqry +0.05（0.308 → 0.355）。CNN + FT の 2 モデルがほぼ最良（MLP はもう少ししか足さない: hx では MLP を足すと下がる）。
- held-out BCE は ep3 で最小、その後は上がる（確信度の過大 = 較正の劣化）が、F1・窓の Tanimoto は ep7 前後まで上がる。物差しは Tanimoto
  （生の確率の連続 Tanimoto、ランキングだけ見る）なので BCE の最小（ep3）では早すぎる。**本番の選択は ep5**（haqry が最大、hcqry/hxqry は ep7–8 と
  ±0.02 以内、BCE の悪化がまだ小さい）。ep3（BCE 最小）でも MLP+CNN には足される（hx 0.676、ha 0.342）。
- 夜の ft_pre（1 エポック単独スケジュール）→ ft8 ep1 で hcqry が下がるのは学習率がまだ高いため（cosine が 8 エポックに伸びた）、ep4 以降で追い越す。

### 8.2 ゼロから 8 エポック（ft8s、事前学習の価値）
`cpp/pretrain/ft8s.bat`（launch 名 pt_ft8s、ft8 の "all done" 待ち、10:39 開始）: 同じ構成・同じハイパーパラメータで pt_w.bin なし。
結果は 8.6 に追記（評価は ft_eval.bat ft8s_ep<N>、`C:\prog\casmi\pretrain\fte_ft8s_ep*_*.log`）。

### 8.3 プールの順位付け項（pool36 の CNN 項と同じ形）
`py/pool37ft.py`: pool36 の every-candidate ダンプ（`v36\out\p36{x,b,bs}_0_cf.txt`、v35 + attpool 2、rm=0）に列 ft = 候補の fp3c16k ビットと
FT の分子ごとの p（sigmoid(平均 logit)）の連続 Tanimoto を足し、score' = score + w z(cnn) + u z(ft) → pool36 の後段（top 60 → プールの知名度 → v35 の
Boltzmann merge）をそのまま再現、final リストの MRR@25、v35 と pool36 の選択（cnn 2）に対する対応ありブートストラップ。
対照: SHUF-query（クエリ q を q+7 の FT p で採点）、PERM（クエリ内で ft を並べ替え）。
**生成候補（G<k>、全行の 1/3）の SMILES がダンプにない**ので、`phys_v36pc` のコピーに `gsdump=` を足した `phys_v36gs`
（`cpp/pretrain/phys_v36gs.patch`、`build_gs.bat`、`run37gs.bat`、`gs3.bat`、ビルド機 `C:\prog\casmi\pretrain\p37\`）で同じ 3 つの実行をやり直し、
**cf ダンプが v36\out と fc /b で一致・MRR 一致**を確認したうえで G 候補の SMILES → fp3_prep apply（`pretrain\g{x,b,bs}_fp3c16k.bin`）。
プール候補のビット = `pretrain\p36pool_fp3c16k.bin`（119,046 分子、pool730 の SMILES）。
（注意: 最初は G 候補の値を補完していた。クエリ内の平均・最小で補完すると hbqry で +0.014〜0.017 と大きく出たが、rm=0 では正解が必ずプール側なので
「G 候補を一律に下げる」こと自体が点になる = 偽の利得（SHUF 対照も +0.005 に上がる）。CNN 列からの回帰で補完した値は実 SMILES の結果に近かった。
以下は全部、実 SMILES。）

final リスト MRR@25、v35 との差 / **pool36 の cnn 2 との差 [95% CI]**（表 `casmi/pool37ft/gridf_ep5.txt`、`gridf_ep8.txt`、`gridf_ep3*.txt`）:
| FT のエポック・項 | hxqry (v35 0.8731) | hbqry (0.7811) | hbsqry (0.8781) |
|---|---|---|---|
| cnn 2（pool36 の選択） | +0.0109 | +0.0212 | +0.0126 |
| ep5 ft 3 だけ（CNN なし） | +0.0146 | +0.0218 | +0.0219 |
| ep5 **cnn 2 + ft 2** | +0.0174 / **+0.0065 [+0.0004,+0.0134]** | +0.0289 / **+0.0077 [−0.0040,+0.0203]** | +0.0195 / **+0.0069 [−0.0003,+0.0150]** |
| ep5 cnn 2 + ft 3 | +0.0202 / +0.0093 [+0.0016,+0.0183] | +0.0274 / +0.0062 | +0.0230 / +0.0104 [+0.0011,+0.0207] |
| ep5 cnn 1 + ft 2 | +0.0168 / +0.0059 [+0.0012,+0.0120] | +0.0284 / +0.0072 | +0.0231 / +0.0105 [+0.0028,+0.0187] |
| ep8 cnn 2 + ft 2 | +0.0153 / +0.0044 | +0.0259 / +0.0047 | +0.0219 / +0.0093 [+0.0013,+0.0183] |
| ep8 cnn 1 + ft 2 | +0.0176 / +0.0066 | +0.0290 / +0.0078 | +0.0237 / +0.0111 [+0.0024,+0.0202] |
| ep3 cnn 2 + ft 2 | +0.0139 / +0.0030 | +0.0256 / +0.0044 | +0.0193 / +0.0068 |
| 対照 ep5 cnn 2 + ft 2 SHUF-query | +0.0097 / −0.0012 | +0.0080 / −0.0132 | +0.0086 / −0.0040 |
| 対照 ep5 cnn 2 + ft 2 PERM | +0.0017 / −0.0093 | −0.0165 / −0.0377 | +0.0015 / −0.0111 |
読み方: **FT 単独（ft 3）でもプールの順位付けで CNN 2 と同等以上**（hbsqry では +0.022 vs +0.013）。CNN に足すと timsTOF の 3 集合すべてで
+0.004〜+0.011 上乗せ（CI は hxqry・hbsqry で 0 の上、hbqry は 0 をまたぐ）、対照は 0 以下。窓での大きな差（hx +0.14）に比べてプールでの上乗せが
小さいのは、プールが 100 候補程度で v35 の最終リストが既に 0.87〜0.90 と飽和に近く、CNN と FT が同じ候補を持ち上げるため。
**推奨: + 2 z(CNN) + 2 z(FT ep5)**（重みは 3 集合で選んだ手置き、1:2 や 2:3 も同程度）。hcqry/hasqry（timsTOF なし）は未測定
（CNN 項はそこでは負だった → 同じく timsTOF のクエリだけに掛ける案は pool36 の Recommendation と同じ）。

### 8.4 本番ジョブ（切り離し済み）
`cpp/pretrain/prod_pt.bat`（launch 名 **pt_prod**、pid 18728、09:30 起動、ft8s の "all done" 待ち、最大 6 時間）:
(a) alllib で masked-peak 事前学習（wlib と同じ設定、10 エポック、`pt_train_prod.exe`）→ `C:\prog\casmi\pt_all.bin`（+ pt_all_ep*.bin、pt_all_ck.bin）、
(b) export + alllib で fp3c16k 微調整 8 エポック（ft8 と同じ）→ **`C:\prog\casmi\ft_all_ep<N>.bin`**（ep5 を使う、8.1）、`ft_all.bin` = ep8。
見込み → 実測: 12:05 に開始、事前学習は GPU 単独で 140k tok/s = 18,847 step/エポックが ~9 分 → 10 エポック ~1.5 時間（~13:35）、微調整 8 エポック ~1.5 時間 → **~15:00–15:30 完了**。
進行 `type C:\prog\casmi\pretrain\prod_progress.txt`、ログ `pretrain\prod_pt.log`、`prod_ft.log`、`night\pt_prod.log`。生存確認
`powershell -c "Get-Process -Id 18728"`、止めるとき `taskkill /pid 18728 /t /f`（自分の PID だけ）。再実行で事前学習は pt_all_ck.bin から再開。

### 8.5 Kaggle の CPU 推論コスト
`pt_train_cpu2.exe`（`build_cpu2.bat`、cl /O2 /arch:AVX2、tfm_model.hpp の素朴なループ）`mollogits model=ft_pre.bin qry=hxqry threads=4 B=32`:
**1,163 スペクトルで 246 s = 212 ms/スペクトル（4 スレッド）**、GPU の logit との差 最大 5.5e-5（|logit| 平均 10）。
→ テスト ~1,200 スペクトルで **~4 分**（Kaggle の CPU がビルド機より 1.5–2 倍遅くても 6–8 分）。CNN（30 ms/スペクトル）の約 7 倍だが十分収まる。
候補側は fp3c16k ビット（fp3_prep、プール 12 万分子で 12 s / 2 スレッド）と Tanimoto の内積だけ。モデルファイル 221 MB（fp32; fp16 なら 111 MB）。
高速化の余地: エンコーダの GEMM を Eigen/ブロック化に（素朴ループ）、ヘッド 2048 × 16384 はプール候補に現れるビットだけ計算すれば数分の 1。

### 8.6 ゼロから 8 エポック（ft8s）の結果 = 事前学習の価値（同じ 8 エポック、同じ構成・ハイパーパラメータ）
窓 MRR_in（Tanimoto; 表 `casmi/pool37ft/window_ft8s.txt`）、held-out BCE（最小 ep3 825.4 vs 事前学習あり 791.3）:
| | hcqry FT | hxqry FT | haqry FT | hc CNN+FT | hx CNN+FT | ha MLP+CNN+FT |
|---|---|---|---|---|---|---|
| scratch ep4 / 事前学習 ep4 | 0.303 / 0.326 | 0.580 / 0.634 | 0.308 / 0.317 | 0.371 / 0.382 | 0.648 / 0.678 | 0.343 / 0.346 |
| scratch ep5 / 事前学習 ep5 | 0.319 / 0.339 | 0.616 / 0.659 | 0.304 / 0.319 | 0.370 / 0.380 | 0.674 / 0.697 | 0.347 / 0.355 |
| scratch ep8 / 事前学習 ep8 | 0.337 / 0.347 | 0.644 / 0.680 | 0.310 / 0.300 | 0.372 / 0.377 | 0.695 / 0.716 | 0.349 / 0.357 |
プール（final、pool36 の cnn 2 との差、ゼロから ep8; `pool37ft/gridf_s8.txt`）: ft 3 単独 hx +0.0111 / hb +0.0161 / hbs +0.0152 vs v35
（事前学習 ep5: +0.0146 / +0.0218 / +0.0219）、cnn 2 + ft 2 は cnn 2 に対して +0.0043 / +0.0015 / +0.0087（事前学習 ep5: +0.0065 / +0.0077 / +0.0069）。
読み方: **利得の大部分はトランスフォーマー（ピーク集合の注意）という構造そのもの**で、ゼロから 8 エポックでも MLP・CNN を単独で超える（hx 0.644）。
事前学習は同じ微調整予算で **+0.01〜0.04（hx +0.036〜0.054、hc +0.01〜0.02）**、haqry は ep5 まで上・ep8 で逆転（ノイズ程度）、
プールでも事前学習ありのほうが一貫して少し強い。事前学習のコストは単独 GPU で ~1.5 時間（alllib）なので本番は事前学習ありで進める。

### 8.7 再開メモ
1. pt_prod の完了を確認（`prod_progress.txt` に "finetune rc=0" と "all done"）。`ft_all_ep5.bin` を本番モデルに。
2. 組み込み: phys_v36pc の cnnmodel/cnnpool と同じ形で `ftmodel=<TFM1>,ftpool=2`（tfm_model.hpp の CPU 順伝播 → 分子ごとの平均 logit →
   sigmoid → 候補の fp3c16k ビット（語彙は fp3c16k.bin、fp3.h でその場で計算）との連続 Tanimoto → + 2 z(ft)）。CNN 項はそのまま 2。
   C++ で 1 点（hbqry、cnnpool 2 + ftpool 2）を確認し、pool37ft.py のシミュレーション（ep5: final 0.8100）と一致を見る。
3. 窓のチャネル（PubChem 窓 = プール外の 2〜4%）でも REV2/fp3 の代わりに z(CNN) + z(FT) の z 和（窓 MRR hx 0.71、hc 0.38）を使える。
