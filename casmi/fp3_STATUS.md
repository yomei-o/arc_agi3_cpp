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
（下に追記）
