# CASMI 2026 — 再開メモ

2026-10-14 夕方時点。次に続ける人（未来の自分）向け。**まず「現状と次にやること」と「作業の前提」を読むこと。** 理論・仮説・負の結果の詳細は `PHYSCHEM_IDEAS.md`。

## 現状と次にやること（2026-10-14 17時締め）
**明日の朝（提出枠は9時にリセット）**: kernel version 23 = **v22（v20の手の重み + 1.0·z(FPNet)）を push 済み**（2026-10-14 16:10）。ログで "fpnet stage done" と "output + 1 z(fpnet)" の行を確認してから version=23 を提出（失敗時は "FPNET DISABLED" と出て v20 相当の出力になる）。v21版ノートブックは scratchpad の casmi_match_v21.ipynb。
**2026-10-15**: v22（v20 + 1.0·z(公開FPNet)）= **0.299**（+0.008、LBノイズ内）→ 公開FPNetの手元での圧倒（tqry 0.733→0.931）は大半が holdout 分子の暗記。私たちの部品が本番ではFPNetの情報をほぼカバー。トップとの差の原因は候補範囲（プール外の正解）か本番の分子の種類（ライブラリにある割合）の可能性が高い → ライブラリ照合だけのプローブ（kernel casmi26-cpp-probe）で測定中。v23（v22 + REV2）提出済み・採点待ち。
**2026-10-15 午後**: **v24（v23 + 候補生成 phys_gen）= 0.322（ベスト）**。v23（v22 + REV2）は採点待ち（Kaggle混雑）→ 出たら REV2 と生成の内訳を確定し、本番の「正解がプール外」の割合を推定する。進行中: C++ LambdaMART ランカー（lmart_STATUS.md）、PubChem固定スロット（pcslot_STATUS.md）、Transformerによる生成候補の選別（gentfm_STATUS.md）。
**2026-10-15 夕方**: (a) **断片の大きさの効果を物理モデルに導入（phys_size.cpp）**: 切断で分かれる小さい側の重原子数ビン（1,2,3–4,5–8,9–15,16+）ごとの対数重み sizeW を fqry で合わせ込み → −0.5/−1/0/+1/+0.75/+1.25（パラメータ C:\prog\casmi\phys_size_fqry.txt）。物理モデル単独で cqry 0.433→0.445、tqry 0.372→0.390（CSV再採点）。融合での確認は未。(b) Transformer（tfm_allA.bin）を融合に追加（phys_gentfm.cpp）: cqry 0.550→0.561、tqry 0.624→0.639、oqry 0.659→0.678。正解除去時はほぼ不変 → 律速は生成候補とプール候補の混ぜ方（isGen罰則）。(c) 既存バグ: libSim が1を超える（最大1.86、ピーク照合が1対1でない）→ 上限1で oqry +0.004。**ユーザー判断: (a)〜(c) は小粒なので単独では提出しない**。LambdaMART・PubChemスロットと合わせて v25 で確かめる。
**2026-10-15 夜**: **v25（v24 + PubChem チャネル B、phys_pc、kernel version 26）= 0.321（v24 0.322 と同じ、PubChem は本番で効かず）**。候補: (1) 知名度の水増し（holdout の正解は公開ライブラリの分子）、(2) ゲート（libSim<0.9）と bias −7 が保守的すぎて本番でほぼ差し込まれていない、(3) 正解がプール外の割合が小さい（「LB 0.32 = 0.55(1−x)+0.13x → x≈0.55」の見積もりは外れ？）。明日、可視テスト以外での差し込み件数（ログの gate open）を見られないので、切り分けは holdout で gate/bias の感度と知名度の効き方を。攻め版 A（pcmin=1）のノートブックは `notebook/casmi_match_variantA_pcmin1.ipynb`（未提出、C++ で tqry 0.610 / 正解除去 0.497 を確認済み）。
**PubChem が効かなかった理由 = holdout の選び方の偏り（2026-10-15 夜、`py/pc_novel.py`）**: PubChem ±3ppm 窓で照合（各組最大300分子）。PubChem にある割合は train 無作為 91.7% / COCONUT のみ 91.6% / tqry 正解 98.3% / cqry 正解 91.6% → 「隠しテストの分子が PubChem にない」は否定。**知名度の平均は train 無作為 2.06・COCONUT のみ 2.08 に対し tqry 正解 7.65・cqry 正解 5.04、知名度だけの窓内 MRR は 0.05 / 0.05 に対し 0.34 / 0.32**。make_holdout.py が「timsTOF スペクトル2本以上の分子」を選ぶので、よく測られる有名な分子（標準品）に偏っていた。**手元と LB の差（0.55 vs 0.32）、v21・FPNet・PubChem が手元でだけ効いた件も、この偏りが一部原因の可能性**。→ **明日の最優先: スペクトル本数で選ばない（普通の分子の）holdout を作り直し、全部品を測り直す**。
**量子化学で障壁の法則を探す（ユーザーの長期チャレンジ、2026-10-15 夕方〜）**: 詳細と家での続け方は `py/qm/HOME_QM.md`。xTB の結合スキャン 600本で障壁の序列 アシル < グリコシド < C–ヘテロ < C–C < アリール–C（合わせ込みと同じ）。アリール–C は ipso プロトン化経路で 110 → 39 kcal/mol（248/254）。観測との対応は分子の最安定プロトン化を基準にすると AUC 0.65（独立450分子）、ipso の優位とオルト/パラ則は未確認。直接切断の障壁は「種類＋大きさ」を超えない（負）。自前の実空間格子 DFT（`cpp/rsdft/`、C++）: 水素原子 −0.49987、H₂⁺ 結合エネルギー 0.1028（正解 0.1026）、エッグボックス効果をセル平均で解消。次は2電子（H₂、ポアソン＋LDA）。
**2026-10-22 朝**: v33 は実行 570 分で提出拒否（FPNet の段階が 30,622 秒。単独でも起きる → 段階に時間制限が必要）。夜間の結果: 事前学習（pt_w.bin、10 エポック）から fp3 を1エポック学習 hcqry 0.318 / hxqry 0.414 vs ゼロから 0.241 / 0.266（事前学習が大きく効く）、埋め込み検索は 0.37 / 0.41 でエントロピー attention（0.47 / 0.61）に届かず。CNN 2本 + fp3 の窓 MRR 0.367 / 0.579。本番 CNN cnn_prod_c.bin（263 MB）完成。C++ での CNN 確認 hbqry 0.806 / 0.802（予想どおり）。今日: v35nofp（ビルド用カーネル → 提出）、v36nofp（自前 FPNet + 本番 CNN + 時間制限、v36_STATUS.md）、事前学習の延長と本番化（pretrain_STATUS.md）。
**明日（2026-10-22）の朝**: (1) v33（kernel v33 v1、FPNet あり）は 10-21 11時から 18時時点で7時間実行中 → 540 分超えなら提出不可。終わっていれば `bash <scratchpad>/pe/auto_sub3.sh casmi26-cpp-library-search-v33 1 kv33 ...` で確認・提出。(2) データセット casmi26-rdkit-min に v35 の版（phys_v35pc.cpp + pool730/rev3_all_fp3c16k_big.bin）をアップロード済み → ビルド用カーネル（C:\prog\casmi\kp_build、yomeiotani/casmi26-cpp-build）を push → ログで `failed mains: none` → v35nofp（C:\prog\casmi\kp_v35nofp、保守的な事前確率 0.1）を push → `PREBUILT ... used`、`alloc=boltz`、`attn=1` を確認して提出。手順は scratchpad pe/chain_v35.sh。**Kaggle は必ず1本ずつ**。(3) 夜間ジョブの結果: C:\prog\casmi\cnn\SUMMARY.txt・NIGHT_SUMMARY.txt（CNN）、C:\prog\casmi\pretrain\SUMMARY.txt（事前学習）、C:\prog\casmi\fpown\progress.txt（自前 FPNet、fpnet_own_STATUS.md）。(4) **pool36 の結果（casmi/pool36_STATUS.md）: CNN をプール候補の融合に +2 z(CNN Tanimoto) で足すと timsTOF の3組すべてで上がる（hbqry +0.021、hxqry +0.011、hbsqry +0.013、95% CI すべて 0 超え）、timsTOF なしの hasqry −0.005・hcqry −0.013。隠しテストは全部 timsTOF**。シャッフル対照は全部マイナス。phys_v36pc.cpp（cnnmodel=、cnnpool=w）は既定で v35 と同一。attpool は 2 のまま、fp3 をプールに足すのは逆効果。C++ での確認（hbqry、重み 2、期待値 最終 0.8023）を夜間に起動済み: C:\prog\casmi\v36\out\p36b_cnn2.log。
**次: 本番用の CNN を全 train（alllib）で学習 → v36nofp = v35nofp + cnnmodel + cnnpool=2（timsTOF のクエリだけに効かせる1行の条件も検討）。データセットに cnn_model.hpp・cnn_layers.hpp（rdkit-min/rev/）と CNN モデルを追加。**
本番用 CNN は夜間ジョブ cnn_prod（pid 9304、cnn_night の終了待ち → 学習約3時間）→ C:\prog\casmi\cnn_prod_c.bin、進み具合 C:\prog\casmi\cnn\prod_progress.txt。
**自前の FPNet 完成（fpnet_own_STATUS.md）**: 公開 FPNet と同じ構造・同じ 10226 ビット・同じファイル形式、train.parquet だけで自前の C++/CUDA で学習（公開の重みや出力は一切不使用）。本番ファイル C:\prog\casmi\fpown\ds_add\fpnet\fpnet_own_f16.bin（99.5 MB）。
窓 MRR hcqry 0.341 / hxqry 0.535（REV2 0.173 / 0.090）、**プール候補の並べ替えに +1.0 z で hcqry +0.013、hxqry +0.012、hasqry ±0**。
**v36nofp に入れる: データセットの fpnet/ に fpnet_own_f16.bin を入れ、FP_W = FPNET_SRC / "fpnet_own_f16.bin"、USE_FPNET = True（中身は自前）。最終データセットからは fpnet_full1_f16.bin を外す（ルール確認のため）。**
**プローブ = 0.023（2026-10-21）→ 隠しテストでプール外の正解は約 2%（合成系なら約 4%）**。「6〜7割がプール外」は誤り。本番の勝負は**プール内の約100候補の並べ替え**（手元 0.53〜0.86 vs 本番 0.34 = 隠しテストの分子が難しい）。PubChem 窓の改良（Tanimoto、fp3、attention、窓の物理、ボルツマン配分）は本番では 2〜4% × 伸び しか効かない → 新部品は**プール候補の並べ替え**への効果で判断する（attention の attpool 項、CNN・物理をプール融合の項に）。v35nofp はボルツマンの事前確率を保守的な 0.1 に変更（make_v35_nb.py の BOLTZ35）。
**2026-10-21 の自前部品（全部 FPNet なし）**: ボルツマン配分（v33）、attention エンジン（attn_STATUS.md、窓 +0.13〜0.31、プール融合 attpool=2 で hasqry +0.010）、窓の並べ替えに物理モデル（formula_STATUS.md、窓 +0.05〜0.07; 分子式そのものは物理と同じ情報）、1次元 CNN（cnn_STATUS.md、窓で fp3 に +0.03、hxqry 0.546）、v35 = v33 + v34（v35_STATUS.md、物差し +0.009 vs v33）、本番の通り道にバグなし（e2e_STATUS.md）、ビルド済み実行ファイルのカーネル（prebuilt_STATUS.md）、自前の事前学習（pretrain_STATUS.md、夜間）、FPNet と同じ構造の自前学習（fpnet_own_STATUS.md、学習中）、ICEBERG/GLACIER 評価（iceberg_STATUS.md、+0.01 前後、使わない）。
**v31 = 0.314（v28a 0.335 から −0.021）**: 関門付き LambdaMART v29（holdout2 の混合で学習、手元の物差し +0.005）＋ E2R2（+0.008 見込み）。**学習した並べ替えは3回続けて LB で失敗（v21 −0.041、v26 −0.035、v31 −0.021）**。新しい物差し＋関門でも判定できない → 学習した並べ替え・融合は最終版から外す。E2R2 単独の効果は分離できていない（同時投入）。効いているのは手の重みと理屈の部品だけ（物理、REV2、生成、プール内知名度、Tanimoto 採点）。
**ルール（2026-10-21、discussion 745841 の主催者回答）**: 第三者の重みは学習コード（と許可された学習データ）を提出できる場合のみ賞の対象。公開 FPNet（ahmedberatozer fpnet-full1、v22〜v30 で使用）は学習コード非公開 → **最終選択の2本は FPNet なしにする**（過去の提出は失格理由にならない）。ICEBERG/GLACIER・DreaMS・ChEBI/LIPID MAPS・train.parquet で学習したものは使用可。不適合はリーダーボードから外される可能性もある。FPNet の寄与は約 0.02（v27b→v27c）→ 穴埋め案: FPNet と同じ構造を自前の学習コードで train から学習。
**2026-10-21 本番経路の端から端までの検証（e2e_STATUS.md）**: hcqry+hxqry から偽の train/test.parquet（zlib と同一、本物の test と同じ列・型）を作り、v33nofp ノートブックの全セルをビルド機で実行 → submission.csv を公式の同一性で採点: **hcqry 0.5576 / hxqry 0.8613、同じ exe・設定の eval モード 0.5507 / 0.8613**（800 中 779 リストがバイト一致、差は eval だけが正解の重複コピーを落とすため = eval がやや悲観的）。**本番経路にスコアを下げるバグはない → LB の差は holdout と隠しテストの違い**。Kaggle 出力の点検も異常なし（解析不能 0・CCO 0・欠け 0）。修正版 casmi_match_v33nofp_fix.ipynb = rdkit-min の 3 回の単スレッドコンパイル（v30 ログで計 61 分）を並列・共有オブジェクト化（出力はバイト一致）。
**2026-10-21 自前 FPNet（fpnet_own_STATUS.md）**: FPNet と同じ構造・同じ 10226 出力ビットを自前 C++/CUDA（tfm_train fpnet=1）で train.parquet から学習、`fpnet_full1_f16.bin` と同形式（ggml 順伝播は学習器と 5.7e-6 一致）。公開重み・出力は学習に不使用。wlib 版（clean）: 窓内 MRR hcqry 0.341 / hxqry 0.535（rev2_w 0.173 / 0.090）、v30 holdout 融合 + 1.0 z(fpown) で hcqry +0.013 / hasqry +0.000 / hxqry +0.012。本番（alllib 6 ep）は夕方に `C:\prog\casmi\fpown\ds_add\fpnet\fpnet_own_f16.bin` へ → notebook は FP_W を差し替えるだけ。
**2026-10-21 夕方: 自前の DreaMS 型事前学習（masked peak）→ `pretrain_STATUS.md`**。勾配チェック PASS、パイロット（80 秒）で埋め込み検索の窓内 MRR hcqry 0.200 / hxqry 0.108（対照 0.03 / 0.004、ment 0.47 / 0.61 には未到達）。夜間 pt_night（d512 L6、wlib 10 エポック → (a) 埋め込み attention、(b) 微調整 vs ゼロから）→ 朝 `C:\prog\casmi\pretrain\SUMMARY.txt`。
**2026-10-21**: LB **v28a 0.335、v28b 0.334、v30（v28b + PubChem 窓を Tanimoto で採点、pcrev=tan）= 0.338（現ベスト）**。プローブ（kernel -probe v1）は 10-21 朝に提出、採点待ち。v31（v28a + 関門付き LambdaMART v29 + E2R2）は実行時間 665 分 > 上限 540 分で提出拒否（10-20 は3本同時に走らせ FPNet 段階が 5.4〜7.2 時間、ふだん 15 分）→ v31 v2 を単独で再実行中。v32（pc4_STATUS.md、自前 fp3 1.6万ビット大モデル 549MB で PubChem 窓を採点）は手元 +0.002 のみ: PubChem リスト内 MRR は +0.013・正解入り +10 だが、**最終リストへの差し込み規則（ゲート付き固定スロット＋厳しい昇格）が律速** → プローブで本番のプール外割合を見てから、差し込み規則を広げるか決める。自動提出スクリプトの不具合2件（待ち上限で誤って先へ進む、提出拒否を成功と表示）を修正済み（scratchpad pe/auto_sub3.sh）。
**明日（2026-10-20）の朝にやること**: (1) v28a（kernel v29、v27a + プール内の知名度）・v28b（kernel -b v2、+ PubChem v2）の LB を確認（2026-10-19 夕方は採点待ち）。(2) 夜間ジョブの確認: ビルド機 `C:\prog\casmi\night\*.log`（qm_night = 量子化学 400分子の障壁スキャン＋基準＋ipso＋分析、結果 C:\prog\casmi\ts_night_summary.txt / ts_*_n400*.csv）、fp3 エージェントの夜間学習（casmi/fp3_STATUS.md に名前・pid・出力）。起動の仕組みは casmi/night/launch.ps1（WMI 経由で ssh が切れても止まらない）。(3) 提出1: プローブ（casmi/probe_STATUS.md、kernel -probe、プール外の PubChem 候補だけのリスト）→ f_out = LB / m_out で本番のプール外割合を測る。(4) 提出2: v28 の良い方 + 関門付き LambdaMART v29（casmi/lmart2_STATUS.md、casmi_match_v29lm.ipynb、ds_add は C:\prog\casmi\lm29\ds_add）+ ライブラリ照合1位を2位に（casmi/rf_STATUS.md、casmi_match_v29rf.ipynb のセル7後処理、RF_MODE=e2r2, RF_GATE=1.0）。2つのノートブックの変更を合成する必要あり。(5) 本命は fp3（PubChem の窓 ~1.3万件の中で選べる自前の指紋モデル、物差しは hcqry 正解除去の窓内 MRR、今 ~0.05）。
**2026-10-19 夕方の方針転換（0.4 への道）**: LB 0.32 を S（ライブラリにスペクトルあり、手元 ~0.85、v6 から ~15%）・D（プール内でスペクトルなし ~0.53）・R（プール外 ~0.15）の混合で解くと **R ≈ 0.7（正解の 6〜7 割がプール外）**。holdout2 でも絶対値を合わせるには rm 56% が必要。PubChem が本番で効かなかったのは、窓（~1.3 万件）の中で正解を選んでいたのが知名度で、**自前の指紋予測（REV2）単独の窓内 MRR は 0.047 しかない**から。上位は FPNet（1 万ビット）で窓の中を選べる。→ (a) プール外の PubChem 候補だけのリストで1回プローブし本番のプール外割合を測る、(b) 本命: 1 万件の窓の中で選べる自前の指紋モデル（REV2 路線、ビット種類拡張、較正したベイズ採点）、物差しは hcqry 正解除去の PubChem 窓内 MRR、(c) 窓の上位を物理モデル（障壁の法則）で絞る。器（PubChem v2、ppop、dedup、E2R2、関門付き LambdaMART v29）は完成済み。
**2026-10-19 提出: v27a（v24 + 断片の大きさの重み phys_size_fqry.txt + libcap=1、phys_v27、kernel v28）= 0.323（+0.001、新ベスト）、v27b（v27a + 自前 Transformer tfm_allA tw=2、kernel -b v1）= 0.312（−0.011、手元 +0.010〜0.015 と逆）、v27c（v27b − 公開 FPNet = 全部自前、kernel -c v1）= 0.291（FPNet を外して −0.021）**。→ 公開 FPNet は本番で効いている。自前 Transformer は FPNet の代わりにならず害（重み過大 or train 分子の暗記）。賞を狙う全自前構成には、REV2 路線（本番 +0.014）を伸ばした自前の指紋予測が必要。後処理（postproc_STATUS.md）: 公式同一性での重複除去は +0.000〜0.001、組成式群の並べ替えは私たちには効かない。**手元の mrr.py（InChIKey14 名で判定）は公式（互変異性体を同一視）より 0.02〜0.04 低い**（cqry 0.552 vs 0.580）。
**LB 確定（2026-10-19 確認）: v23 = 0.313（REV2 で +0.014）、v26（v25 + LambdaMART、FPNet なし）= 0.286（−0.035）**。LambdaMART は v21（学習重み −0.041）と同じ型の失敗: 有名な分子に偏った holdout の構造（「ライブラリにあるのに合わない候補は答えでない」）を学習、学習時と本番で特徴を作るモデルが違う（xlib MLP / rev2_vh_xs vs 本番モデル）、FPNet なし。**holdout で学習・選択したもの（v21、v26、PubChem の条件）は 3 回続けて LB で効かないか悪化。手の重み＋理屈で作った部品（物理、REV2、生成）だけが効いている → holdout を作り直すまで学習による融合は使わない**。LambdaMART の記述（以下）は参考。LambdaMART（lmart_STATUS.md）: 6 holdout 5分割CV で手の融合比 cqry +0.019 / tqry +0.055 / oqry +0.013 / sqry ±0、ラベル置換対照は崩れる。本番モデル pool730/lmart_prod.txt、ソース phys_pclm.cpp。notebook/casmi_match.ipynb は v26。v25 版は scratchpad casmi_match_v25.ipynb。
**LB**: v15 0.188 → v16 0.235 → **v20 0.291（現ベスト）** → v21 0.250（v20＋学習した融合重み、kernel version 22）。トップ約0.49。
**v21の教訓**: 学習重み（cqry:sqry=1:0.3、「ライブラリにある構造を下げる」方向: inLib −2.36、lib の1次項が負）はLBで −0.041。**隠しテストにはライブラリに正解（またはそのスペクトル）がある分子が想定より多い**。次に重みを学習するときは sqry の比重を上げ（1:1〜1:2）、v20/v21両方のLBを説明できる比率を探す。それまで提出の基本は v20 構成。

**本番パイプライン（notebook/casmi_match.ipynb, kernel casmi26-cpp-library-search）**: match.cpp（ライブラリ検索）→ phys_ana（rdkit-min/phys_ana.cpp）が pool730（train+COCONUT）±10ppm の候補を融合スコアで並べる → プールのリストを先頭、残りをライブラリ結果で埋める（ライブラリ1位の強制は廃止）。データセット yomeiotani/casmi26-rdkit-min（ステージング ビルド機 C:\prog\casmi\ds_rdkit、pool730/ に pool・rev_all3072.bin・phys2_fqry.txt・fuse_w21.txt）。
融合: v20 = z(phys2) + 0.25 z(MLP v2) + 1.0 z(rev_all3072) + 8·libsim^32 + 24·Tanimoto·mcos^3（ずらし照合アナログ）。v21 = fuse_fit で学習した重み（fuse_w21.txt）。

**v22候補（手元で確認済み、tqry は学習に未使用）**: 学習重み + 差スペクトル幾何平均アナログ（phys_diff）+ 吸収マルコフ連鎖 深さ3（phys_dag、物理項を置換）→ tqry 0.704 → 0.733（cqry CVは横ばい）。phys_dag d3 はメモリ十数GB → Kaggle用にクエリ単位で DAG を捨てる改修が必要。特徴ファイル `C:\prog\casmi\comb_{cqry,sqry,tqry}.txt`、学習 `fuse_fit.exe comb_cqry.txt:1,comb_sqry.txt:0.3 comb_tqry.txt 0.001 8,9,10,11,12,13 - rz`。

**飛躍のための3本柱（エージェント作業中、状態ファイル参照）**:
1. アナログからの候補生成（正解がプールにない分子を拾う）→ `gen_STATUS.md`。**有望**: 上位80アナログに1〜2段の化学変換（±Me、±OH、±H2、±糖、Ac、N-oxide…）で正解を作り出せる率 cqry 36% / tqry 56%。正解をプールから抜いた条件で MRR 0 → 0.19（cqry）/ 0.33（tqry）、半々混合で 0.33→0.40 / 0.39→0.55（Python試作、gen_fuse3.log が正しい値）。単純マージは悪化、生成特徴（isGen、isGen·sim³、段数、支持アナログ数…）で学習し直しが必須。**試作の fit は inLib（−5.5）を使っているので、v21 の教訓どおり inLib なしで再確認すること**。C++（phys_gen.cpp）未着手、Kaggle追加 15〜25分の見積もり。
2. 逆向きモデル強化（容量・ビット・bit較正）→ `rev2_STATUS.md`。**効いたのはビット定義**: 学習ライブラリ頻度上位4096個の環境ハッシュ（ECFP/FCFP 半径0〜3）を衝突なしの語彙ビットに → 融合 cqry 0.550→0.564、tqry 0.612→0.635（`C:\prog\casmi
ev2_vh.bin`、phys_ana への組込みは改造コピーが必要、未作成）。今の rev_x3072 は tqry では融合を悪化させていた（rev なし 0.623）。Platt較正は単独は大きく上げるが融合でほぼ0、容量増・アンサンブルは効果なし。**FPNet の強さの多くはビット定義（10226ビット、多種類の部分構造）かもしれない → 明日まず MLP に FPNet と同じ10226ビットを予測させて、構造とビット定義を切り分ける**。その次に自前Transformer（FPNet構造: ピーク=トークン、m/zと中性損失の正弦波埋め込み、6層 d=512、6930ビット）を自前backend（rev/backend.hpp、CUDA/CPU同一ソース）で学習。
3. 公開FPNet（ahmedberatozer/casmi26-fpnet-full1）の推論を ggml で → `fpnet_STATUS.md`。**移植完了**: 実際の構造は 8層 d=512、10226ビット（ECFP4/6・FCFP4・RDKitパス・AtomPair・TopTorsion・MACCS の16551ビットから選択）、PyTorch との差 ≤1e-4、4スレッドで 40〜56 ms/スペクトル。pool730 の10226ビット指紋計算済み（933MB、公開プール指紋と300/300一致）。候補スコア `C:\prog\casmipnet_{cqry,tqry,sqry}.txt`。単独MRR cqry 0.750 / tqry 0.907 / sqry 0.756 だが**train全体で学習済みの重みなので大きく過大評価**（作者LBでも full1 差し替えで 0.380→0.378 と改善なし）。次: 自前Transformerを xlib で学習して漏れなし比較、公開重みの効果はLB 1回で確認。
ほか: PubChem固定スロット（公開上位で+0.03〜0.04、未着手）、知名度（COCONUTメタデータ、holdoutでは漏れの疑い大、LBで確認）。

**確かめること（次回の最初に）**:
1. 指紋予測モデルの公平な比較: 私たちのMLP（rev_x3072）／公開FPNet（漏れあり）／自前Transformer（xlibで学習、漏れなし）を同じholdout・同じ融合で。差がどこから来るか（m/z精度、ピーク間の関係、ビットの種類、規模）を1つずつ切り分ける。
2. 融合の各項のアブレーション（v20/v21構成から1項ずつ抜く）を cqry/tqry/sqry で。どの項が何点を担っているかの表を作る。
3. 隠しテストの正解がプール（train+COCONUT）の外にある割合の推定: LBプローブの設計（例: プール候補を全部抜いたリストで提出し、ライブラリ・PubChemだけの点を見る）。候補生成とPubChemスロットの価値の上限がこれで決まる。
4. ライブラリにある分子とない分子の比率: v20（手の重み）と v21（学習重み）のLB差から推定し、fuse_fit の cqry:sqry の重みを合わせ直す。**途中経過（2026-10-14）**: LB ∝ a·sqry + (1−a)·cqry で v20（s 0.840, c 0.550 → LB 0.291）と v21（s 0.722, c 0.610 → 0.250）を説明すると **a≈1**（ノイズ±0.03で不確か）→ 仮説「隠しテストの正解は、別装置のスペクトルがライブラリに載っている分子が大半」。正しければライブラリとの本人照合（装置・アダクトをまたぐ）を磨く方が候補生成より効く。**検証: ライブラリ一致（libsim）だけで並べたリストを1回提出**。なお学習重みは sqry の比重を上げても sqry 0.82 止まりで、手の重み（8·sim^32）に届かない（条件付きロジットでは強い本人一致の効き方が出しにくい）。
5. 手元とLBの比（v16: 0.67、v20: 0.48）がなぜ下がったか: 構成要素ごとに手元とLBの対応を見る。

## 作業の前提（2026-10-14 確立、必ず守る）
- **評価は天然物のholdoutで**: cqry（正解がライブラリにない、400）、tqry（timsTOF天然物、176、本番に最も近い）、sqry（正解の他スペクトルがライブラリにある、400、ライブラリslib）、fqry（1000、パラメータ合わせ専用）。モデルは xlib（train−cqry−tqry）で学習したもので評価。dqry（合成分子）は使わない。
- **提出前に出力CSVを手元評価に通し、グリッドの値と一致するか確認**（v16〜v19は出力経路のバグで別物が提出されていた）。
- **評価データの作りが特徴に漏れないか疑う**（pool の train フラグ、COCONUT掲載、知名度はすべて holdout では正解の目印になりうる）。
- **公開LB（約130分子）の標準誤差は約0.03**。0.01単位の調整は手元の大きいholdoutで決め、LBは大きな変更の方向確認に使う。
- **ツール内のMRRは同点を「すべて正解より上」と数える（悲観的）** → 候補ごとに少しでも違う値を足すだけで同点が崩れて点が上がる（2026-10-15、QMエージェントが逆符号の対照で発見。DAGの優位の約半分が同点崩し）。物理モデル単独の比較は `QM_TIES=avg`（同点平均）か、出力CSVを mrr.py で採点した値で行う。新しい項を足したら逆符号の対照も測る。
- **データセット更新直後のpushは古い版を使うことがある**（v21で発生）。ログで新機能の行を確認してから提出。

## 2026-10-14: 手元評価が本番より甘かった理由（測定済み）

- **dqryはCOCONUTと1分子も重ならない**（trainは9.5%=26,111分子がCOCONUT掲載）。dqryは「timsTOF・245〜461Da・2本以上」で選んでおり（見えているダミーtestに形を合わせた条件）、Envedaの合成・医薬品様ライブラリばかりを選んでいた。主催は天然物企業で、公開上位はtrain+COCONUTプールで0.40 → 隠しテストの正解は天然物寄りと推定（未確認）。
- 新holdout **cqry/clib**: `make_holdout.py --mode denovo --floor 0.002 --keys coconut_keys.txt --any-instr --prefix clib,cqry`（COCONUT掲載のtrain分子400個、1159スペクトル、装置不問）。順方向MLPはclibで学習し直し（`fwd_mlp_rank_clib.bin`）。
- `pool_candidate_rank` に `[pool_stem] [names]` 引数追加（プール差し替え・InChIKey14出力で評価）。評価は `C:\prog\casmi\mrr.py <truth_mols.txt> <csv...>`。

| holdout | 71万プール(train+COCONUT, `pool730_*`) | 437万プール(今の提出版) |
|---|---|---|
| dqry（合成分子） | **0.732** (393/400) | 0.639 (367/400) |
| cqry（天然物） | **0.392** (294/400) | 0.387 (289/400) |

- 見えているテストに合わせて足したPubChem約360万件は**有害か無益**（dqry −0.09、cqry −0.005）。提出版プールは71万件に戻すべき。
- 天然物では順方向MLPが0.73→0.39に半減。今後の評価はcqry（とtimsTOF天然物のtqry）を主にする。
- tqry（timsTOF天然物176分子、`--keys coconut_keys.txt`のみ）: 0.347 → 装置差は主因ではない、化学クラス差が主因。
- 物理・化学からの定式化を4観点のエージェントで検討 → `PHYSCHEM_IDEAS.md`。
- **E0（組成式oracle）**: cqryで真の組成式に絞っても 0.392→**0.437（+0.045）**。候補89件/分子のうち53件が真と同じ組成式。top1が誤った組成式なのは86/400。→ **ボトルネックは組成式ではなく同一組成式の異性体識別**。
- cqryで正解が候補に入らない39/400の内訳: 語彙外アダクト(tag負)、2M付加体（中性質量=(p−shift)/2 にしていない）、精度2桁のプリカーサー（±10ppm外）。ライブラリ由来の不備でほぼ化学と無関係。2Mは直すべき。
- **強度の切り分け（cqry / dqry）**: 全フラグメント同強度 0.351 / 0.540、順方向MLP 0.398 / 0.732 → 学習強度の上乗せは天然物で+0.047、合成で+0.19。天然物の点の大半は「どの質量の断片ができうるか（列挙）」から。pool_candidate_rankに `[intensity] uniform` と2M修正（中性質量を半分）を追加（2M修正でcqry 0.392→0.398）。
- **cosine→正規化尤度**（`[score] eps`、背景混合の対数尤度、スペクトル平均）: cqry 同強度 0.351→0.370、MLP 0.398→**0.407**（eps=0.6）。
- 物理モデル `cpp/rdkit-min/phys_rank.cpp`（結合クラス障壁＋プロトン親和力/酸性度による電荷保持＋H移動重み、約20パラメータ）。tqryで座標探索フィット→cqryで評価。
- **物理モデル phys_rank の結果**（パラメータは天然物1000分子 fqry/flib で座標探索。`make_holdout.py --keys coconut_fit_keys.txt --any-instr --n 1000 --seed 777 --prefix flib,fqry`、cqry/tqryと非重複）:
  | | fqry(調整) | cqry | tqry |
  |---|---|---|---|
  | 物理のみ | 0.384 | 0.379 | 0.352 |
  | MLPのみ（phys_rank内の同条件、同点は悲観的） | | 0.379 | 0.347 |
  | **物理 + 0.5×MLP（z和）** | | **0.403** | **0.377** |
  - 176分子(tqry)だけで合わせた版は過学習（tqry 0.394 → cqry 0.337）。1000分子で合わせ直して安定。
  - 結合クラスの効果大（外すとfqry 0.384→0.338）。電荷保持（PA表）は±0.003で効果なし。
  - フィット値（`C:\prog\casmi\phys_fqry.txt`）: E[intact]=−1, acyl −1.5, **glycosidic −1.95（最も切れやすい）**, C–het 0.75, aryl-C 1, C–C 1, ring 2, ring-aromatic 4.75, other 1.5; 正イオンはH移動0が主（切断点に電荷の偶数電子イオン）。
- **2本同時の開裂**（アシル/グリコシド/C–ヘテロの非環結合ペア、最大12本から）を phys_rank に追加、`phys2_fqry.txt`（E2=2）: 物理のみ cqry 0.379→0.396、tqry 0.352→0.352。**物理+0.5×MLP: cqry 0.411、tqry 0.388**。
- **v15**（pool730＋2M修正、cosine）push済み・実行中。**v16**（phys_rank＋0.5×MLP融合、pool730）はデータセット更新済み・ノートブック準備済み（`notebook/casmi_match.ipynb`、v15版は scratchpad に退避）、v15完了後にpush。
- **v15 = 0.188（pool730＋2M修正）。v16 = 0.235（物理＋0.5MLP、ベスト更新、0.2台達成）**。
- **逆向きモデル（tag修正、xlib学習 `rev_x3072.bin`）を第3項に**: cqry単独0.418（437万プール）。phys_rank融合 z(phys)+0.25z(MLP)+1.0z(rev): **cqry 0.490、tqry 0.491**（物理+MLPの0.415/0.380から大幅増）。wr 1〜1.5で安定、3以上は悪化。
- 公開上位の分析（エージェント、`scratchpad/pubnb/*.txt`、engine-2ソース `scratchpad/ext/casmi_engine.py`）: 最小構成（FPNet指紋予測 f·z＋ライブラリ特徴＋GBM）でLB 0.354。ライブラリはハードゲートではなく±10ppm（式由来質量）・付加イオンずらし照合・全スペクトルmaxの連続特徴。アナログ伝播（±200Da、上位80、Tanimoto×sim³）。外部スペクトルライブラリは誰も使っていない。隠しテストはtimsTOF、公開LBは約130分子。
- **ライブラリ類似度を連続スコアに**（phys_rank `[lib_stem] [wl] [lib_pow]`、候補自身の参照スペクトルとの最大cosine）。sibling天然物holdout **sqry/slib**（`--mode sibling --keys coconut_fit_keys.txt --any-instr --n 400 --seed 4242`）で評価: 線形はde novoを壊す（cqry 0.490→0.345@wl8）。**8·sim^32** で cqry 0.467 / sqry 0.816（ライブラリなし 0.490/0.617）。
- **v17準備済み**: phys2 + 0.25×MLP(v2) + 1.0×rev(全train `rev_all3072.bin`) + 8·sim^32(train)。逆向きモデル全train学習待ち。
- **アナログ伝播**（phys_rank、0.01Da転置インデックスで類似スペクトル検索、前駆体±200Da、上位80分子、`+ wa·max_a Tanimoto(c,a)·sim_a^3`）: wl=8,pow=32 固定で wa=8 → cqry 0.467→0.484、sqry 0.816→0.839、**tqry 0.460→0.530**。
- **v17 push済み（2026-10-14）**: phys2 + 0.25×MLP(v2) + 1.0×rev(`rev_all3072.bin`、全train floored `alllib`) + 8·sim^32 + 8·analog、ライブラリは notebook の `train`。可視テストでスモーク済み（400行、空なし）。データセットから rev_h3072.bin を削除（cascadeカーネルは再実行不可）。
- **v18（v17構成）= 0.235（v16と完全一致）→ 原因はバグ**: phys_rank の eval で `fused(..., &scores)` が冒頭で出力先 scores（=物理項の読み出し元）を空にしていた → 出力CSVは解放済みメモリ（物理スコアの残骸）で並べ替え。**v16〜v19の提出に、逆向き・ライブラリ類似度・アナログ伝播は一度も効いていない**（グリッドのMRRは out=nullptr 経路なので正しい）。加えてマージ（ライブラリ1位強制 θ=0.9）で可視テストは400/400がライブラリ1位。修正済み（phys_rank / phys_ana）、出力CSVでcqry 0.551＝グリッド値と一致を確認。
- **phys_ana（エージェント、ずらし照合のアナログ）**: 断片m/z＋中性損失の2本の転置インデックス→上位8000→厳密modified cosine（6ピーク以上一致）、`+24·Tanimoto·sim^3`: cqry 0.484→**0.550**、tqry 0.534→**0.612**、sqry 0.839→0.840。引数 `... train 8 32 24 "" 4 3 8000`。
- **v20 push済み**: phys_ana + 修正 + マージ廃止（プール融合リストのみ、残りをライブラリで埋める）。v19（マージ廃止、バグあり）は提出しない。
- 量子化学（xTB）と仮説検証の結果は `PHYSCHEM_IDEAS.md`。
- formula_infer.exe（部分組成式による組成式推定、`cpp/rdkit-min/formula_infer.cpp`）は初版が遅すぎて40分で打ち切り（F/Cl/Brまで列挙・大きい分子で部分式数爆発）。候補式をプール実在の式に限る形で作り直す予定。

## 現在のスコア

| 提出 | 変更 | publicScore |
|---|---|---|
| v4 | 質量窓ウィジェニング + Gaussian mass prior + instrument bonus | 0.108 |
| v6 | + ピーク強度フロア（ノイズ除去） | 0.121 |
| v7 | + COCONUT統合プール + analog propagation（構造類似度のみ、質量シフト無視） | 0.059（悪化） |
| v8 | v6ベース(analog propagation削除) + forward_rerank(ルールベース同一分子式再ランク) | 0.102(悪化) |
| v9 | v6ベース + forward_rerank_nn(学習済みMLPで再ランク) | 0.105(悪化) |
| **v10** | **v6 + pool_candidate_rank(順方向MLPでプール候補)、tight候補を先頭に固定してバックフィル** | **0.181（現ベスト）** |
| v11 | 逆向きモデル(ECFP予測) + 切替マージ(θ=0.9) | 0.160 |
| v12 | 逆向きモデル + v10と同じマージ（切り分け用） | 0.154 |
| v13 | v10の順方向MLP + 切替マージ(θ=0.9,k=1)（LBモデル予測≈0.22） | **0.180**（v10とほぼ同じ。マージ規則は隠しテストではほぼ無関係＝候補リスト上位に正解が来ること自体が稀） |
| v14 | v13 + 質量窓±10ppm + tagデコード修正(match.cpp/pool_candidate_rank) + アダクト表記ゆれ`[M-H2O+H]+` | カーネル実行完了。**提出スキップ**（2026-10-14、見込み0.18前後でv13と差が出ない判断。枠はランカー用に温存） |

### v13/cascadeの結論と次の本命
- v10≒v13 → マージ規則で点は動かない。cascade(0.109) → PubChem候補を先頭に置くと、ライブラリで届く分子の正解まで押し出して悪化。
- 公開0.438の得点の大半(~0.40)は**プール(train+COCONUT 71万)をLightGBMランカーで並べる**ことから来ている（PubChemは固定スロット補助、ICEBERG/GLACIERは同一分子式内微調整）。私たちはプールを順方向MLP 1本で並べていて0.18止まり。
- **次の本命: C++の学習型ランカー**（設計案）
  1. 学習データ: trainから多数の擬似クエリを作る（dlib/dqry方式＝分子の全スペクトルをライブラリから除外）。数千〜数万分子規模で、各クエリにプール候補（±10ppm）を付ける。
  2. 候補ごとの特徴量: 順方向MLPのcos / 逆向きlogit和 / 人気度（PubChem CID対応が必要、プールはInChIKey14名なのでCID-InChIKey対応表かSMILES照合）/ ライブラリ上位ヒット(match.cpp)との構造類似度（Tanimoto, ECFP）の最大値・平均 / 質量誤差ppm / train由来かCOCONUT由来か / 原子数・環数・元素組成などの記述子 / 同一分子式グループ内での順位。
  3. モデル: 勾配ブースティング（C++で自作、ヒストグラム法）か、3090でMLPランカー（listwise softmax損失）。dqryで評価、LBモデルで予測してから提出。
  4. PubChemは公開版と同じく「ランカー上位に差し込む少数の補助」として後で戻す。
| cascade v1 | PubChem全体(1.12億)+プール → 逆向き+人気度で1000件 → 順方向MLP再スコア融合 | **0.109（大幅悪化）** 別カーネル`casmi26-cpp-cascade` |

**cascade v1の悪化について（未解決）**: dqry（train構造除外）では0.345と良好だったのに実LBは0.109（v6の0.121未満）。仮説: 隠しテストの正解は一般PubChemよりプール（train構造＋COCONUT）側にある割合が高く、PubChem候補を1分子1.4万件足したことでプール内の正解が埋もれた。dqryで「プールのみMLP」をtrain除外条件で測っていなかったので、PubChem追加の価値を比較できていなかった（評価の穴）。v13（プールMLP＋同じ切替マージ）の結果で、マージ規則とPubChem追加のどちらが原因か切り分ける。PubChemは公開上位のように「固定スロットに少数だけ差し込む」補助扱いに変えるのが次の候補。

2026-10-13時点: **2353位/3178**（0.181）。100位0.441、500位0.438（公開ノートブック「ICEBERG where GLACIER is blind」0.438が大量コピーされた台地）。1位0.491。

## 2026-10-13: 大きな発見と方針転換（最重要）

### 発見1: 可視の`test.parquet`はダミー
上位公開ノートブックの解析で判明。可視テスト(400分子)は**trainから抜いたプレースホルダ**で、採点時の再実行で隠しテストに差し替わる。→ **可視テストの質量で絞ったPubChem拡張プール（pubchem_target_filter）は隠しテストではほぼ無意味だった。** 「全部timsTOF・245〜461Da」もダミー由来。可視テストに合わせたものは何も作ってはいけない。

### 発見2: 以前の順方向MLPのholdout数値は学習データ漏れ
MLPを`train`全体で学習→qry/dqryの分子のスペクトルも学習に含まれていた。**dlib（dqry 400分子の全スペクトルを抜いたtrain）で学習し直してdqryで測ると MRR@25 0.675**（漏れありで0.691）。今後の学習は必ずdlibで行い、dqryで評価すること。

### 発見3: dqryは実LBの傾向をちゃんと再現する／sibling holdout(qry)は楽観的すぎる
- dqry上: 順方向MLP 0.675 ＞ 逆向きモデル 0.417〜0.471 → 実LBでも v10(MLP)0.181 ＞ v12(逆向き)0.154 と一致。
- **LBモデル**: `LB ≈ f·qry + (1−f)·c·dqry` を v6/v10/v11/v12 の4点にフィット → **f=0.15（ライブラリ検索で届く割合）, c=0.18**, rmse 0.008。cが小さい＝**隠しテストの非ライブラリ分子の正解はプール(train+COCONUT)にほとんど無い** → PubChem全体が必要。
- 計算スクリプト: scratchpadの`calib.py`（qry/dqryのdirect・候補リスト・best_score・tightから合成MRRを計算）。

### 作ったもの（すべてC++、3090はビルドマシンにあった）
- `casmi/cpp/rev/`: **逆向きモデル**（スペクトル→ECFP 2048bit予測MLP、手書き逆伝播、yolov8_cppの`backend.hpp`流用で同一ソースがCPU/CUDA両対応）。`rev_train.cpp`(train/eval/rank、アンサンブル可)、`ecfp.h`(自作ECFP4)、`fp_prep.cpp`。3090で1エポック約50秒。dqry: 隠れ1024=0.417、2048=0.461、**3072=0.471**（`rev_h3072.bin`）。シードアンサンブルは効果なし。
- `casmi/cpp/pc_build.cpp`: **PubChem全体を質量ソート**（1.12億件、混合物除外、50〜1500Da）→ `pc_mass/off/smi/cid.bin`。
- `casmi/cpp/pop_build.cpp`: **人気度** `log1p(#SID)+log1p(#PMID)` → `pc_pop.bin`。dqryではほぼ効かない（+0.004）。
- `casmi/cpp/rev/cand_rank.cpp`: PubChem窓（±10ppm、実行時にrdkit-minでECFP）＋プールを逆向き＋人気度でスコア、指紋で重複除去。tagの符号化バグをデコード側で修正（後述）。
- `casmi/cpp/rdkit-min/fwd_rescore.cpp`: カスケード第2段。順方向MLPで再スコアし `z(fwd)+0.5·z(rev)+0.1·pop` で融合。
- match.cppに`best_score.csv`出力を追加（切替マージ用）。

### dqry（train構造を候補から除外＝PubChem/COCONUTからしか正解が来ない現実的条件）
| 方式 | MRR@25 |
|---|---|
| 逆向きのみ（PubChem全体、候補約1.7万/分子） | 0.132 |
| 順方向MLPのみ（1000件に絞った後） | 0.231 |
| **融合（1000件絞り込み、a=1,b=0.5,c=0.1）** | **0.345** |
PubChem全体では正解が候補に入る率94%、1000件絞り込み後でも73%。

### バグ: tag符号化
`tag=アダクト*16+装置`で装置未知(-1)だと`[M+Na]+`が`tag=31`→`[M-H]-`と誤解読（中性質量が約24Daずれる）。trainの約1割が語彙外の装置名（LC-ESI-QFT, qTof, ToF, 欠損…）。`tag%16==15`なら`アダクト=tag/16+1`と読み替えるデコードで回避（cand_rank, fwd_rescoreは対応済み、match.cppとpool_candidate_rankは未対応）。アダクトも`[M-H2O+H]+`（train表記）が語彙`[M+H-H2O]+`と綴り違いで未知扱い。

### Kaggle
- データセット: `casmi26-rdkit-min`（rdkit-min/autograd/rev ソース＋MLP重み＋rev_h3072）、`casmi26-pool-expanded`、`casmi26-rev-model`（pool_ecfp等）、`casmi26-pubchem-whole`（7.9GB）。
- カスケード版カーネル `casmi26-cpp-cascade`: RDKitを4並列で1回コンパイル(約4分)→cand_rank→fwd_rescore、全体23分。
- ビルドマシンにKaggle認証(`~/.kaggle/access_token`)を設置済み（ユーザー許可済み）。**大きいファイルはローカル(このノートPC)に持ってこないこと**（ユーザー指示）。Kaggle出力もビルドマシン側でダウンロードして検証する。

### 次にやること
1. cascade v1 と v13 のスコア確認。cascadeが効けばPubChem全体が本命と確定。
2. 1000件絞り込みでの正解残存73%を上げる（逆向きモデル強化：衝突エネルギー入力、極性別、4096bit、隠れ層拡大）。
3. 順方向MLPを強化（特徴量・グラフ情報、3090での大規模学習）。
4. match.cpp/pool_candidate_rankのtagデコード修正とアダクト表記ゆれ対応。
5. 公開上位がやっていてまだ無いもの: LightGBMランカー、ICEBERG/GLACIERでの同一分子式内再ランク、ChEBI/LIPID MAPS。

## 2026-10-07 朝: v8をKaggleに提出(本セッション)

Kaggleノートブック(`casmi26-cpp-library-search`)を更新してversion 8をpush:
- v7で使っていたanalog propagationのセル(ANALOG_SOURCE定義 + poolデータ
  セット読み込み + analog_match実行 + merge)を**全部削除**(実LBで悪化した
  手法なので土台から外した)。
- 代わりに`forward_rerank`(前日作ったGNN不使用の同一分子式再ランキング)を
  組み込み。RDKitのC++コア一式(`rdkit-min/`+`boost-min/`)を新規Kaggle
  Dataset `yomeiotani/casmi26-rdkit-min`としてアップロードし、カーネル内で
  `g++ -O3 -std=c++20 -pthread -I. -Iboost-min ...`で直接コンパイル
  (CMakeは使わず、Boostのfind_packageまわりの不確実性を避けた)。

**実地検証できた新事実**: `boost-min`はこれまで一度もビルドで実際に
使われたことがなかった(MSVC側は常にvcpkgのBoostを使っていたため)。
ビルドマシンのw64devkit(MinGW) g++で直接g++呼び出しを試したところ
`localtime_r`未定義エラー(RESUME.md既知のWIN32分岐問題、MinGW特有)で
失敗したが、**Kaggleの本物のLinux環境では一発でビルド成功**
(`forward_rerank compile rc: 0`、722秒)。出力も期待通り
(400/400分子、877グループ再ランク — 前日のholdout検証と完全一致)。

提出コマンド: `competition_submit_cli(file_name='submission.csv', kernel=
'yomeiotani/casmi26-cpp-library-search', version='8', ...)` → `PENDING`で
受理確認。

**結果: publicScore 0.102。v6(0.121)より悪化(-0.019)。**

### 重要な負の結果: forward_rerankはholdoutで良く見えたが実LBでは悪化

sibling holdout(400分子、729k train+COCONUTプールの同一質量デコイ)では
top1 35.2%・top5 70.0%・MRR 0.513、実テストの候補リストでも877グループ
(400/400分子)を整合性確認済みで再ランクできていた。それでも実LBは
0.121→0.102に悪化。ビルド・パイプライン自体の不具合ではない
(Kaggle上のLinuxビルド成功、出力フォーマット確認済み)。

**[[holdout-gain-does-not-transfer-1to1]]の別パターン**: 前回は「holdoutの
伸びが実LBでは縮む」だったが、今回は**符号が逆転**した。仮説(未検証):
1. sibling holdoutのデコイ(train由来の同一中性質量候補)は、実テスト
   セットの実際の候補群と構造分布が異なり、forward modelの暗黙の偏り
   (結合切断パターン・H転位モデルの粗さ)がholdoutでは「たまたま」正解
   寄りに効き、実データでは逆に働いた可能性。
2. match.cppの候補順序(cosine類似度ベース)が、同一分子式グループ内でも
   既に意味のある情報を持っていた可能性があり、forward_rerankがそれを
   purely構造ベースのスコアで上書きすることで、有用な既存シグナルを
   壊した可能性。

**現在のベストはv6(0.121)に戻る。forward_rerankは一旦保留
(使わない)。** 次に試すなら、同一分子式グループを無条件に並べ替える
のではなく、forward modelのスコア差が大きい(確信度が高い)場合だけ
採用する、あるいはmatch.cppの既存スコアとのブレンドにするなど、
「全面置き換え」ではなく「弱いシグナルとして混ぜる」方向を holdout で
検証してから再度試すべき。

## 今日(2026-10-06)作ったもの — forward model、完全にC++

5位Sho Sagaさんの解説（`casmi/reference/forward-model-sho-saga.html`、ICEBERGベース）
を読み、同じ発想を学習済みGNNなしでルールベース+C++で実装した。

**できたもの**:
1. `casmi/cpp/rdkit-min/` — RDKitのC++コアを自前CMakeLists.txtで最小抽出
   （Code/全部ではなく約95ファイルの.cppだけ、本家のCMakeLists.txtが列挙する
   正式なファイルリストを読んで抜き出した。詳細は `rdkit-min/README.md`）。
   **MSVCでもg++(MinGW/Linux想定)でもビルド確認済み。**
2. `casmi/cpp/rdkit-min/boost-min/` — 依存するboostヘッダーも同じやり方で
   最小化(146MB→21.5MB、2923ファイル)。g++の`-MM`依存解析で実際に使われる
   ヘッダーを機械的に特定。config/type_traits/mpl等の内部機能検出系モジュール
   だけは個別に追うと終わらないいたちごっこになったので丸ごとコピー(それでも
   小さい)。**Kaggleカーネル(Linux g++、ネット接続オフ)に持ち込む前提で
   ここまで絞った。**
3. `forward_model.h` — 候補構造→予測スペクトルの中核。結合を1本(非環)または
   同じ環内の2本切ってフラグメント列挙、各断片の質量+H転位(±2個まで、
   `exp(-decay*k^2)`で重み付け)を予測ピークとする。
4. `forward_holdout.cpp` — 729k候補プールから同じ中性質量の候補(同じ分子式
   相当)を集め、forward modelでスコアリングして正解が何位に来るか検証する
   ツール。**分子単位で集約(1分子の複数スペクトラのmaxスコア)、スレッド
   プールで並列化(match.cppと同じパターン)。**
5. `forward_rerank.cpp` — **実パイプラインへの統合。** match.cppが出した
   既存の候補リスト(submission.csv)を読み、各分子内で**厳密に同じ分子式**
   (原子番号ごとのカウント+暗黙水素数で判定、質量近似ではなく厳密一致)の
   候補グループだけをforward modelスコアで並べ替える。Sho Sagaさんの
   ページで実LBに+0.038〜+0.054効いた使い方と同じ。

## 測定結果(holdout、パラメータ確定済み)

パラメータスイープ(`forward_holdout.exe <dir> qry 20 10000 <decay> <maxHShift>`):
- `hShiftDecay`: 0→0.400, 1.0→0.472, **1.5〜2.0→0.484(プラトー、ここに決定)**, 5.0→0.477
- `maxHShift`: 0→0.471, 1→0.510, **2→0.513(ここで頭打ち)**, 3,4→変化なし

**最終設定: decay=2.0, maxHShift=2。**

分子単位集約(スペクトラ単位ではなく実際の採点単位)で測定:
**sibling holdout 400分子全件: top1 35.2%  top5 70.0%  MRR 0.513**
（スペクトラ単位では0.484だった→分子単位集約で向上）

実テストの候補リストに`forward_rerank`を適用: **400/400分子に同一分子式
グループがあり、合計877グループを並べ替えた。** 出力の整合性確認済み
(400行、各25件フル、重複なし)。ただし**実LBでの効果はまだ測っていない
（今日は提出していない）**。

## 踏んだバグ・教訓

- **候補配列のin-place並べ替え事故**: `cands[pos] = cands[other_pos]`を
  ループで行うと、同じループ内で既に上書き済みの値を読んでしまい重複
  候補が発生。元の文字列を全部コピーしてから書き戻すように修正
  (forward_rerank.cppで修正済み)。
- **PowerShellの`Copy-Item -Recurse`事故**: 既に存在するフォルダに同名
  フォルダをコピーすると中に入れ子になる(`boost/config/config/...`)。
  `Copy-Item "src\*" dst -Recurse`(ワイルドカードで中身だけ指定)が正しい。
- **RDKit単独ビルドの勘所** (`casmi/cpp/rdkit-min/README.md`に詳細):
  RDKit本家のCMakeLists.txtはほぼ全モジュールを無条件`add_subdirectory`
  する作りで、機能単位でつまみ食い的に無効化すると危険(どこかが暗黙に
  依存していて後でリンクエラー)。公式オプション(Python wrappers/テスト/
  boost serialization等)は使うが、本体は本家の`rdkit_library(...)`の
  ファイルリストをそのまま読んで必要なものだけ抜き出す方式に全面転換。
- **g++でのlocaltime_rエラーはWindows特有**: `#ifdef WIN32`の分岐で、
  MSVCビルドはWIN32が自動定義されるがg++手動呼び出しでは要`-DWIN32`。
  **Linux本番環境ではWIN32は未定義のままPOSIX分岐に入るので無関係。**

## PubChem候補プール拡充 — 進捗(2026-10-06夜、続き)

ダウンロード済み(`C:\prog\casmi\pubchem\`のCID-Mass.tsv 4.65GB、CID-SMILES.tsv)
を元に、一括質量レンジでのフィルタは効かない(61%も残る)ことが判明したため
方針転換:

1. `pubchem_target_filter.cpp` — テスト400分子**それぞれ**の中性質量
   (587種、アダクト補正済み)のppm窓だけで絞る。50ppmで1683万件、
   **5ppmで367万件**(COCONUTの約5倍相当、現実的なサイズと判断)。
2. `pubchem_join.cpp`で367万CIDにSMILESを結合 → `candidates.tsv`(231MB)。
3. `pubchem_parse.cpp`(新規، `rdkit-min/`に追加)でRDKitパース+重複排除。
   **実行中**(ビルドマシン上、バックグラウンド、20スレッド)。

### 重要な仕様変更: 重複排除キーがInChIKey14ではなくcanonical SMILES

`build_pool.py`はCOCONUT統合時、CSVに載っている`standard_inchi_key`列を
そのまま重複排除キーに使っていた(RDKitでInChIKeyを計算したのではない)。
今回のPubChem統合では列が無いのでRDKit自身に計算させる必要があるが、
**`rdkit-min`はINCHI-APIモジュール(外部IUPAC InChI Cライブラリへの
ラッパー)を一度も抽出していない** — これは今日の抽出作業で意図的に
除外したわけではなく、単に必要になるまで気づかなかった抜け。InChI
生成には別途外部Cライブラリの取り込みが要るため、今夜中には間に合わない
と判断し、**RDKit自身のcanonical SMILES(`MolToSmiles`)を重複排除キーに
使う**方式に切り替えた(`pubchem_parse.cpp`内にコメントで明記)。

canonical SMILESはInChIKeyと違い立体化学を区別する/表現非依存性が弱い
ので、同じ骨格の別スペリングを取りこぼす可能性がInChIKeyより高い
(= 新規プールに同じ構造の重複が残る可能性がInChIKey方式より高い)。
**ただしこれは重複が増える方向の誤差で、プールの再現率(recall)を
損なう誤差ではない** — 真の構造を取りこぼす心配はなく、無駄な候補が
少し増えるだけ。許容範囲と判断して進めた。InChIKey14での再重複排除は
後日、INCHI-APIを実際に抽出する余裕があれば一段の改善として残せる。
質量計算も同様の理由で`Descriptors`モジュールではなく、
`forward_model.h`と同じ原子質量の手計算(最頻同位体質量の総和)。

出力: `C:\prog\casmi\pubchem\pubchem_new.tsv`(`PUBCHEM:<cid>\tsmiles\tmass`、
既存プールと重複しない新規構造のみ)。**実行完了**: 367万候補中
3,637,867件が新規構造として確定(185件パース失敗、28,763件が既存プールと
重複)。

### 踏んだバグ: 未捕捉例外でバッチ全体が落ちる

`pubchem_parse.cpp`が729k中約46万件目で毎回同じ箇所(`exit code 9`)で
落ちた。原因はRDKit自体のバグではなく、**`RDKit::SmilesToMol`の
sanitize処理が投げる例外(`MolSanitizeException`)を一度もtry/catchして
いなかった**こと。実際にクラッシュしたのは
`O=[P](=O)(O)P(=O)(=O)(O)...`という、プレーンな`=O`だけでリン原子価6を
作ってしまっている不正なポリリン酸SMILES(プール内に実在)— RDKitは
これを正しく「Explicit valence for atom # 1 P, 6, is greater than
permitted」として例外送出していたが、呼び出し側が無防備だったため
プロセスごと終了していた。`safeParse()`(try/catchラッパー)を
`pubchem_parse.cpp`と`forward_rerank.cpp`の両方に追加して解決
(同じ無防備な`SmilesToMol`呼び出しがforward_rerank.cppにもあったため)。
**教訓: このrdkit-minを使う新しいツールは全て、SmilesToMol呼び出しを
必ずtry/catchすること。** RDKit自体を疑う前に、まず自分の呼び出し側の
例外処理を疑うべきだった。

### プール統合 — 完了

`casmi/cpp/pool_merge.cpp`(新規、RDKit不要・w64devkit g++で直接ビルド)で
`pubchem_new.tsv`を`pool_head.bin`/`pool_mols.txt`にマージ(has_spectrum=0、
COCONUT統合と同じ扱い)。**プール規模: 729,389件 → 4,367,256件(約6倍)。**
元のファイルは`pool_head.bin.bak`/`pool_mols.txt.bak`としてバックアップ済み。

## 拡大後プールでのholdout再測定(2026-10-06夜、続き)

`forward_holdout`を拡大後プール(436.7万件)に対して再実行:

| ppm窓 | プール規模 | top1 | top5 | MRR |
|---|---|---|---|---|
| 20 | 72.9万(旧) | 35.2% | 70.0% | **0.513** |
| 20 | 436.7万(新) | 29.2% | 60.2% | **0.435**(悪化) |
| 5  | 436.7万(新) | 40.1% | 72.2% | **0.538**(旧を上回る) |

**候補数の実態**: ppm=20だと1分子あたり候補が数千〜3万件超まで膨張
(例: candidates=32815)。これがMRR低下の直接原因 — 同じ中性質量の
デコイが単純に希釈分だけ増えた。ppm=5に絞ると候補は数十〜数百件に収まり、
むしろ旧プールより高いMRRが出た。**ppmウィンドウは「質量ゲート」ではなく
「候補密度のノブ」として効く**ことが実測で裏付けられた(match.cppの
コメントが指摘していた「実テストスペクトルは数ppmの機器誤差を持つ」
範囲には収まっている)。

**結論: プール拡充は有効、ただしppm=5(またはそれ以下)での運用が前提。**

### 踏んだ方法論ミス: 的外れな「100%カバー」測定

`pool_recall_check.cpp`で最初に測った「denovo-holdout(dqry)400分子の
正解構造がプールに何%存在するか」が**100.0%**と出たが、これは無意味な
数字だった。理由: `dqry`はtrainの構造そのもの(スペクトルだけ除去した
400分子)で、`dqry`の全キーがtrainの全キーの部分集合であることを確認
済み。`build_pool.py`はtrainの27.6万構造を丸ごとプールに含めている
(has_spectrum=1)ので、「プールに含まれるか」を単純に聞くと必ず
100%になる — 何も検証していない。

元の`pool_recall.py`はこの罠を避けるため、COCONUTのキー集合**だけ**を
held-outキーと突き合わせていた(train全体は見ていない)。今回はそれを
見落として「プール全体」に対して測ってしまった。`pool_head.bin`の
`has_spectrum`フラグを使い、**train由来ではない候補(COCONUT/PubChem)
だけでも正解に当たるか**を区別する方式に修正して再測定。

**結果: 400/400中、COCONUT/PubChem単独での一致は0件(0.0%)。**
ただしこれは「COCONUT/PubChemに本当に1件も無い」という意味ではなく、
判定方法(canonical SMILES完全一致)の限界が濃厚 — canonical SMILESは
InChIKeyと違い立体化学の差を畳み込まない。trainは立体を明示している
ことが多い一方、PubChem/COCONUTのバルクデータは立体未指定のことが多く、
**同じ骨格でも文字列としては一致しない**ケースが多発していると考えられる。
InChIKeyなしでの限界が、デデュープだけでなく再現率測定でも裏目に出た形。
**真の再現率を知るにはINCHI-API(外部IUPAC InChI Cライブラリ)の取り込みが
必要 — 今はやっていない。**

## 次回(明日)やること

1. ~~`pool_recall_check`修正版の結果確認~~ → 完了(上記、0.0%、ただし
   測定方法の限界ありと判断)。
2. **`forward_rerank`のppm窓もデフォルト値を見直す。** 現状
   `forward_rerank.cpp`は同一分子式グループを厳密一致で見つけており
   ppm窓は使っていない(質量ゲートはmatch.cpp側の仕事)ので、この
   知見は主に「将来、poolを新規候補ソースとして直接組み込む」フェーズ
   (denovo分子向け)に効いてくる。match.cpp自体はpool_head.binを
   まだ一切参照していない — pool拡充の効果が実際の提出に反映される
   のはそこを繋いでから。
3. 拡大後のプール(436.7万件)で`match.cpp`→`forward_rerank`を再実行し、
   実際の候補リストでの効果を確認。
4. **Kaggleノートブックへの移植**: 完了(下記「v8提出→実LB悪化」節)、
   ただし当初計画していたルールベースforward_rerankは実LBで悪化したため
   **使っていない**(現在は学習済みMLP版に差し替え、下記参照)。

## v8提出 → 実LB悪化(0.121→0.102)、ルールベースforward_rerankは保留に

holdoutでは878グループ・MRR 0.513相当だったforward_rerank(ルールベース)
をKaggle実環境(本物のLinux、rdkit-min+boost-minの初の実地ビルド成功)で
動かし提出。**実LBは0.121→0.102に悪化。** holdoutで良く見えた変更が
実LBでは符号ごと逆転した、[[holdout-gain-does-not-transfer-1to1]]より
強い負の結果。原因は未診断のまま保留 — 詳細はmemory
`casmi-forward-model-first-signal.md`。

## 学習ベースforward model(自作autograd、2026-10-07昼)

ユーザー指示「sagaさんに極力近づけよう」を受け、yomei-o/mini-yolov5-cpp
(scratch/の依存ゼロ自作autogradエンジン、勾配チェック済み)と
yomei-o/yolov8_cpp(pure/optim.hppのAdam/AdamW、cosine_lr)を調査・流用。
`casmi/cpp/autograd/`に設置(autograd.h/.cpp=mini-yolov5-cppからそのまま
コピー、optim.h/nn_ops.h/.cpp=新規、Tensor APIの違いを吸収する移植)。

**forward_model_nn.h**: フラグメントの量予想を固定式からMLPに置き換え。
入力特徴量(17次元): 質量・サイズ比・切断本数・元素組成比(C/H/N/O/その他
ヘテロ原子)・芳香族率・切断結合の種類(単/二重/芳香族)・H転位量(k,k²)・
アダクト/装置種別(スペクトルごとの文脈、`Head.tag`から)。
`train_forward_model_nn.cpp`(Phase1: 実測強度への回帰)、
`train_forward_model_rank.cpp`(Phase2: 正解がおとりより高得点になる
ようヒンジ損失でfine-tune、おとりはプールのppm窓同一質量候補)、
`forward_holdout_nn.cpp`(学習済みモデルでのholdout評価、既存の
forward_holdout.cppと同じ測定基準)。

### 踏んだバグ2つ(どちらも重要、今後の教訓)

1. **ゼロ過多回帰の退化**: 実スペクトルの大半のフラグメント位置には
   実ピークが無い(target=0)。素のMSEで学習すると「ほぼ全部0を予測」が
   最適解になり、見た目のlossは下がるが識別力が消える
   (MRR 0.151、未学習のルールベース0.538より悪化)。
   **修正: 加重MSE**(target>0の行は重み1、target=0の行は重み0.05)。
2. **出力層の死んだReLU**(より深刻、スケールすると再発): 最終層に
   `relu`を使うと、加重MSEでも多数派のゼロ行を減らそうとして出力前の
   値が負に寄り、出力が0に貼り付いて**勾配も0になり学習が止まる**。
   2000分子では偶然顕在化しなかった(MRR 0.614とたまたま良好)が、
   5000分子・20000分子では3回とも**完全に同一の数値**(MRR 0.149)に
   収束 — 重みが違うのに結果が同じという「あり得ない一致」が決め手に
   なった。Phase2のランキング学習でも loss が `margin` 値に**完全に
   固定**(850ステップ動かず)という形で同じ症状が出た(正解とおとりの
   スコアが両方0で一致、勾配ゼロ)。
   **修正: 出力層を`relu`→`sigmoid`に変更**(有界(0,1)で、ReLUのような
   完全な勾配消失が起きない)。修正後はデータを増やすほど素直に改善
   するようになった。
   **教訓: 出力に非負制約を掛けるとき、安易にreluを使うと「大多数が
   ゼロであるべき」という損失の構造と組み合わさって全体が死にうる。
   sigmoid/softplus系の、勾配が完全には消えない関数を選ぶこと。**

### 測定結果(holdout、ppm=5、拡大後プール436.7万件)

| 構成 | MRR |
|---|---|
| ルールベース(固定式) | 0.538 |
| 学習MLP Phase1のみ、2000分子(reluのまま、後に退化と判明) | 0.614(信頼できない) |
| 学習MLP Phase1のみ、5000/20000分子(relu、退化後) | 0.149(×3回、再現性あり) |
| 学習MLP Phase1のみ、5000分子(sigmoid修正後) | 0.650 |
| 学習MLP Phase1のみ、20000分子(sigmoid修正後) | 0.658 |
| **学習MLP Phase1+Phase2(ランキング学習)、20000分子** | **0.666** |

モデルファイル: `C:\prog\casmi\fwd_mlp_rank_v2.bin`(inDim=17, hidden=32)。

### v9提出: 学習済みMLPでのforward_rerank_nn → やはり悪化(0.121→0.105)

`forward_rerank_nn.cpp`(新規、forward_rerank.cppのMLP版)を実テスト候補
リストに適用、整合性確認(400/400分子、877グループ、v8と同一構造)。
Kaggle Dataset `casmi26-rdkit-min`を更新(autograd/を新規に同梱、
rdkit-min/にfwd_mlp_rank_v2.binを同梱)。本物のLinuxでビルド成功
(`forward_rerank_nn compile rc: 0`、717秒)、出力も期待通り。

**結果: publicScore 0.105。v8(0.102)より良いが、v6(0.121、再ランクなし)
より依然として悪い。**

**重要な結論: 2回連続(v8ルールベース、v9学習済み)で「同一分子式グループの
再ランク」そのものが実LBを悪化させた。** スコアリング関数の質(holdout
MRR 0.538→0.666まで本物に改善)を上げても符号が変わらなかったことから、
問題は forward model の精度ではなく、**match.cppが既に持っている候補
順序(実スペクトルとのcosine類似度)自体に、再ランクで壊してはいけない
有用な情報が入っている**という、より構造的な要因が濃厚。

**現在のベストは再びv6(0.121)。同一分子式グループの「全面置き換え」型
再ランクは筋が悪いと判断し、いったん保留。** 次に試すなら、
match.cppの順序を尊重しつつforward modelのスコアを「弱いシグナルとして
加点」するブレンド方式(Sho Sagaさんの「旧ベース: 最後の並びに点数を足す」
+0.054のパターン、今回の「同じ分子式の候補どうしだけ並べ替える」+0.038
より強かった)を試すべき。

### プール拡充の統合に着手: pool_candidate_rank.cpp

`pool_candidate_rank.cpp`(新規)—— match.cppの候補とは独立に、プール
(436.7万件)から直接、学習済みMLPで実スペクトルに対しスコアリングした
新規候補top25を生成するツール。denovo-holdout(dqry)で測定:
**top25内に正解あり93.0%、top1 57.2%、top5 84.5%、MRR@25 0.691。**
(注意: dqryの正解はtrainの構造そのものなのでプールに必ず存在する —
これは「何千件もの同質量候補からtop25へ正解を押し上げられるか」という
ランキング精度の検証であり、「COCONUT/PubChem由来の真に新しい構造まで
同じ精度で届くか」はInChIKeyなし判定の限界により未測定のまま。)

**次回: この`pool_candidate_rank`の出力をmatch.cppの直接候補とどう
マージするかの設計**(例: tight_countが低い/direct候補の確信度が低い
分子だけプール候補で補完する、等)が残作業。

### v10: pool候補バックフィルを実際にマージして提出(本セッション終盤)

`pool_candidate_rank`を実テストセット(test_head.bin)で実行(367万件
プール→各分子の質量窓→top25、ppm=5)。`merge_submissions.py`のロジック
(tight_countで「direct検索が自信を持って見つけた分だけ残し、残り枠を
pool候補で埋める」)をノートブックに組み込み、version 10としてpush。
**354/400分子が平均16.6件のpool候補で補完された。**

**セッション終了時点でKaggle実行がまだ完了していない(RUNNINGのまま、
2時間以上経過)。** おそらくKaggle環境のCPUコア数がビルドマシン(20コア)
よりずっと少なく(標準4コア程度)、同じ作業に5倍近く時間がかかっている
だけだと思われる(ビルドマシンでの同等処理は約25〜30分で完了)。
バグではなく、単にCPU律速で遅いだけの可能性が高い。

**次回最初にやること: Kaggleのkernel実行状況を確認し(`kaggle kernels
status yomeiotani/casmi26-cpp-library-search`)、COMPLETEなら
ログ取得→整合性確認→`competition_submit_cli`で提出→スコア確認。**
残り提出枠は本日2回分(日付が変わっていれば5回にリセットされている
はず)。

## GNPS参照スペクトル統合(本セッション終盤、2026-10-07)

ユーザー提案「84%の届かない分子向けに参照スペクトルそのものを増やす」を
受け、GNPS(Global Natural Products Social、公開MS/MSスペクトルDB)の
「cleaned」版(Zenodo record 11193898、CC-BY-4.0、542,778件、MGF+CSV)を
ダウンロードし、`casmi/cpp/merge_gnps.cpp`(新規)でtrainの
train_head.bin/train_peaks.bin/train_mols.txtに統合(train2_*として出力、
元ファイルは上書きしない)。

**踏んだバグ: ピークオフセット計算が2倍ズレていた。** `head.off`は
「mz配列・inten配列(それぞれ全スペクトルのピーク数の合計で確保)への
インデックス」であって、peaks.binファイル自体のバイトオフセットでは
ない(peaks.binは1スペクトルあたり[mz×n, inten×n]の2n個のfloatを書くが、
offに数えるべきはnだけ)。最初「ファイルサイズ/4」で計算してしまい、
常に正しい値のちょうど2倍になっていた — match.exeが配列範囲外アクセスで
無言クラッシュ(exit code 5、エラーメッセージなし)。**教訓:
「何も出力せず即終了」はアクセス違反の典型パターン。まず自分の
オフセット/インデックス計算を疑うこと。** 既存の全head recordの`n`を
実際に積算する方式に修正して解決。

**結果: 487,389件のGNPSスペクトルを統合できたが、新規構造(train未収録)
はわずか2,363件(27.6万構造の+0.86%)。** 実テストセット(400分子)に対し
`match.exe`をtrain2で実行し`tight_count`を比較したところ、**厳密一致
候補が増えた分子は8件のみ、ゼロ件→発見ありに変わった分子は0件。**

**結論: GNPSはtrainと構造的にかなり重複しており(CASMIの訓練データ自体が
公開ライブラリ由来と思われる)、このデータセットでの参照スペクトル拡充の
効果は現時点では小さそうだと判明。** ただし本当に効果があるかは実テストの
正解が分からないため未確定(候補が増えた8分子のどれかが正解を含む可能性は
ゼロではない)。GNPS2本体(290万件、ALL_GNPSの完全版)やMassBank/MoNA等、
他のスペクトルDBならもっと新規構造が多い可能性もあるが、今日はここまで。

ファイル: `C:\prog\casmi\gnps\ALL_GNPS_cleaned.{csv,mgf}`、
`C:\prog\casmi\train2_{head.bin,peaks.bin,mols.txt}`、
`C:\prog\claude4\arc\arc3\casmi\cpp\merge_gnps.cpp`。

## 明日やること(優先順)

1. Kaggle v10の結果確認→未提出なら提出→スコア確認。
2. v10が0.121を超えたら、pool候補バックフィルの効果は本物と判断できる。
   超えなければ、理由を考察(プール候補のランキング精度 vs
   マージ戦略(tight_count gating)自体の妥当性、どちらが悪いか切り分け)。
3. GNPS統合は効果が小さそうだった(8/400分子のみ)。やるなら
   (a) GNPS2本体(290万件)でもっと新規構造を狙う、(b) MassBank/MoNA等
   別ソースを試す、(c) 参照スペクトル拡充より候補プール拡充
   (pool_candidate_rank)の精度向上の方を優先する、のいずれかを検討。
4. 土台(v6=0.121)とSagaさん(0.374〜0.412)の差が大きい根本原因は
   「ライブラリ検索だけでは84%が届かない」こと(本セッションで判明)。
   これを埋めるのが最優先レバー——pool_candidate_rankの精度・マージ
   戦略の改善に時間を使うのが筋が良い。

## 保留事項

- `C:\prog\casmi\iceberg_msg_all\`(ICEBERGのGNN学習済み重み)と
  `C:\prog\casmi\ms-pred-main\`(公式実装ソース)は今回の方針では使わない
  が、消さずに残してある。フラグメント列挙のロジックやヒューリスティック
  のヒントとして読む価値はある。
- `casmi/third_party/`(RDKit本家フルビルド手順)は使っていない。
  `casmi/cpp/rdkit-min/`が本命。

### FPNet を融合に入れた参考値（2026-10-14、漏れあり）
- comb2_{q}.txt（comb + FPNetスコア列）で fuse_fit: cqry CV 0.602→0.768、tqry 0.733→0.931。重みは FPNet 4.5、rev 0.93→0.06、MLP 0.21→0.01 → holdout上は FPNet が逆向き・MLPをほぼ包含。ただし公開重みは holdout 分子を学習済みで、作者LBでも差し替え効果なし。
- 明日のLB確認案: (a) v20構成 + 1.0·z(FPNet)、(b) rev を FPNet に置換。差が公開FPNetの真の上乗せで、自前Transformerの目標値になる。

### 公開FPNetのKaggle統合（準備済み、未push）→ `fpnet_kaggle_STATUS.md`
- ビルド機で提出と同じ手順のE2Eテスト済み（visible test、400行、FPNet推論 約1.5分/4コア）。cpp/fpnet/{fpnet_ggml.cpp, fpnet_pool.cpp, to_f16.py}、rdkit-min/phys_fp.cpp（phys_ana + `[fpnet_scores.txt] [wf]`）、notebook/fpnet_cells.py（cell 5 後に挿入するセル＋cell 6 の1行変更、失敗時は従来構成で続行）。
- 重み f16 化（99.5MB）。**全テンソル f16 は不可**（SinEmb の周波数が丸められ位相が崩れる）→ 1次元テンソルは f32 のまま。
- プール指紋は別データセット `yomeiotani/casmi26-fpnet-pool`（非公開、2026-10-14 夕方アップロード開始。完了したか status を確認）。コード側は ds_rdkit\fpnet\（未アップロード）。
- **注意: phys_fp は v21（学習重み fuse_w21.txt）の上に wf·z(FPNet) を足す作り。明日は v20 構成（重みファイルなし）＋FPNet で出すこと**。案: (a) v20 + 1.0·z(FPNet)、(b) rev を FPNet に置換。

### 切り分け: MLP に FPNet と同じ10226ビットを予測させる（2026-10-14、xlibで学習＝漏れなし）→ `fp10226_STATUS.md`
- 融合 cqry/tqry: 2048ビット 0.550/0.612、**v4k（衝突なし4096）0.564/0.635（最良）**、f10k（FPNetと同じ10226）0.553/0.619。rev単独は f10k が +0.01 だが融合で下がる。予測が当たる（F1≥0.5）のは10226中2605ビットだけ。
- **結論: FPNet の強さはビット定義では説明できない** → 源は (1) Transformer 構造（正確な m/z の正弦波埋め込み、ピーク間注意）、(2) 学習のさせ方、(3) holdout 分子の暗記（漏れ）のどれか。(1)(2) は自前Transformerを xlib で学習して確認、(3) は公開重みを LB で確認。
- ファイル: ビルド機 C:\prog\casmi\rev2_f10k.bin、xlib_f10k.bin、fcand_f10k.bin。

### 新しい逆向きモデル（v4k）の組込み完了（2026-10-14 16:20）→ `rev2_integ_STATUS.md`
- `cpp/rdkit-min/phys_ana2.cpp`（phys_ana + REV2形式の逆向きモデル: 先頭が "REV2" なら rev2::Model と埋め込み語彙のビット、p を [0.01,0.99] にクリップ）。ヘッダ `rdkit-min/rev/{fp2.h, rev2_model.hpp}`。g++ -std=c++20 でコンパイル確認済み。
- 実ツールで出力CSVを採点: cqry 0.551→**0.568**（wr=1）/ 0.573（wr=1.5）、tqry 0.612→**0.642** / 0.643（xlib学習の rev2_vh.bin、漏れなし）。
- 本番用 `rev2_all_v4k.bin`（alllib全体、hold=0、12エポック、248MB）。データセット casmi26-rdkit-min に phys_ana2.cpp・ヘッダ・rev2_all_v4k.bin（pool730/）をアップロード済み（2026-10-14 16:30）。
- **明日 v23**: v22（FPNet）の結果を見てから、phys_fp の FPNet 項を phys_ana2 に移植した版（REV2 + FPNet）を作り、引数は `rev2_all_v4k.bin 1.0`（wr 1.0〜1.5）で push。phys_ana2 単体なら v20 の行から 実行ファイル・revファイル・wr の3点を変えるだけ。
- **phys_ana3 = phys_ana2（REV2）+ phys_fp の FPNet 項**（`cpp/rdkit-min/phys_ana3.cpp`、データセットにアップロード済み 16:30）。確認済み: FPNetなしで cqry 0.568（phys_ana2と一致）、FPNetありで visible test 400行・全候補にスコア。
- **v23 ノートブック準備済み: `notebook/casmi_match_v23_ready.ipynb`**（cell A で phys_ana3 をビルド、rev を rev2_all_v4k.bin に）。手順: (1) データセットはアップロード済み、(2) casmi_match.ipynb を v23_ready で置き換えて push、(3) ログで "fpnet stage done" と FPNet/REV2 の行を確認して提出。
