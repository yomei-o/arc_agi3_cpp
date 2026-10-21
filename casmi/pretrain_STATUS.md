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
