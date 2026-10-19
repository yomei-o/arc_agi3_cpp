# pc2 — PubChem チャネル v2（公開式のゲート・固定スロット・昇格）とプール内の知名度（状態ファイル）

2026-10-19。動機は gap_STATUS.md の #1（プール内の知名度、公開 LB +0.010）と #4（PubChem の昇格）。手の重みだけ、学習した融合はなし。

## 結論（先に）
- **プール内の知名度（ppop）は効く。知名度を合わせた holdout（hcqry）でも +0.015**（0.518→0.533）。ただし有名な分子に偏った holdout ほど大きく見える（tqry +0.087〜0.127、cqry +0.033〜0.039）ので、重みは公開の 0.15 のまま（cqry/tqry では選んでいない）。
  ライブラリに同じ分子のスペクトルがある sibling 型（hasqry）では知名度が本人一致を上書きして −0.006 → **ライブラリ類似度 ≥ 0.9 の分子では並べ替えない（ppgate=0.9）** と −0.001、hcqry の利得は +0.016→+0.015 でほぼ保たれる。
- **PubChem チャネル v2（公開式: z_window(REV2)+0.15·pop、固定スロット 4,8,12,16,20、rel/S による 1 位昇格）は、旧チャネル B（pcslot、得点で差し込み）を超えない**。正解あり −0.004、正解除去 +0.05〜0.07、分岐点 7〜8%（B は同じ基盤・同じ holdout で −0.004 / +0.06〜0.10、分岐点 4.5〜6%）。
  公開側の PubChem が効く理由は窓の中で正解を当てる指紋モデル（FPNet の f·z）と、昇格を拒否する第 2 エンジンにある。私たちの REV2 だけの並びは弱く（rm=1 で窓の最良 50 件に正解が入るのは 25〜40%）、知名度を軽くすると B の重い知名度（pcpp 20）より当たらない。
- 昇格（relz>3 または S>6 かつ pop≥5）は小さくプラス（rm=1 で +0.003〜0.012、正解ありでの損失は増えない）。S>4・pop≥5 にすると正解除去で +0.03〜0.05 だが、これは有名な分子だけを当てる知名度依存なので holdout の偏りで過大（hcqry では +0.004）。既定は公開と同じ S>6。

## ファイル
- `cpp/rdkit-min/phys_v28pc.cpp` — phys_v27 + phys_pc のチャネル（pcv=1、旧 B）+ 新チャネル pcv=2 + ppop/ppgate。オプションはヘッダ参照。**全部 off なら phys_v27 と同じ出力**（pc=0、ppop=0）。MSVC でビルド・実行、w64devkit gcc 14 で `-fsyntax-only` 通過
- `pc2_poolpop_ik.py` — pool730 → PubChem 知名度。**pool730 の名前は InChIKey14 なので、PubChem の CID-InChI-Key（7.4GB、ビルド機 C:\prog\casmi\pubchem）と結合するだけ**（構造照合なし、メトリクスと同じ同一性）。pop = log1p(#SID)+log1p(#PMID)（pc_pop.bin から、表にない CID 45,621 件は CID-SID/PMID から数える）、同じ IK14 の CID の最大。最初に作った C++ 構造照合版（質量窓→組成式→非立体カノニカル SMILES）は PubChem 1.1 億件のパースで数時間かかるため中止・削除
- `pc2_grid.py` — dump（`_pp.txt` = 知名度並べ替え前の上位 60、`_det.txt` = PubChem 候補と V2 統計）から ppop・スロット・昇格を再現（`--pop`、`--asis` = pcrank=fused の順をそのまま、`--pair` = 正解あり/除去の組と分岐点）。**C++ の出力 CSV と一致を確認**（hcV0: シミュ 0.528 = C++ 0.528、ppop のみ 0.533 = 0.533、hcV1: 0.201 = 0.201）
- `bat_pc2/build.bat`, `run.bat`, `all1..4.bat`（ビルド機 C:\prog\casmi\fwdmodel\pc2、出力・ログ・dump は C:\prog\casmi\pc2）
- ビルド機の成果物: **`C:\prog\casmi\pc2\pool730_pop.bin`**（float32 × 729,389 = 2,917,556 バイト、pool730_mols.txt の行順）、`pool730_cid.bin`（int32、最大 pop の CID、なしは −1）

## pool730 → PubChem 知名度（pc2_poolpop_ik.py）
| | 分子 | PubChem にある（IK14） | 知名度の平均（なし=0） | 見つかった分子だけ |
|---|---|---|---|---|
| 全体 | 729,389 | **689,702（94.56%）** | 2.129 | 2.252 |
| train | 275,810 | 93.87% | 2.041 | 2.175 |
| COCONUT のみ | 453,579 | 94.98% | 2.183 | 2.298 |
pop ≥ 5 は 6.1%。CID-InChI-Key 1.247 億行、一致 CID 1,986,478（pc 表にあるもの 1,940,857）。

## 評価の条件
- 基盤 = v27a（phys_size_fqry.txt、libcap=1、生成 gen=1,cap=300、REV2、手の融合; FPNet は holdout で漏れるので外す）。holdout2（hcqry = 知名度を合わせた COCONUT、haqry = COCONUT 1 本以上、hasqry = sibling）は ylib・fwd_mlp_rank_ylib・rev2_y、cqry/tqry は xlib・fwd_mlp_rank_xlib・rev2_vh。
- rm=1 = 正解（と非立体の重複）をプールから除く。CSV を mrr.py（1/順位）で採点、表の値は pc2_grid.py（C++ と一致）。
- pcv=2 の PubChem リストは 2 通り: rev（z_window(REV2)+0.15·pop だけ）と fused（その上位 300 → 重複除去後の上位 50 に本体の物理・MLP・アナログを計算し、z(手の融合)+0.15·pop で並べる）。fused の方が rm=1 で +0.005〜0.01 良い → 既定の推奨は fused。

## 結果: MRR@25（正解あり / 正解除去）
| 構成 | hcqry | haqry | tqry | cqry | hasqry | 平均 Δ あり | 平均 Δ 除去 | 分岐点 f* |
|---|---|---|---|---|---|---|---|---|
| v27a（基盤） | 0.518 / 0.151 | 0.551 / 0.143 | 0.622 / 0.247 | 0.552 / 0.127 | 0.840 | 0 | 0 | – |
| ppop 0.10 | 0.533 / – | 0.581 | 0.718 | 0.584 | 0.837 | | | |
| **ppop 0.15** | **0.534** | 0.590 | 0.749 | 0.591 | 0.834 | +0.055（4 組） | ±0 | – |
| ppop 0.25 | 0.535 | 0.600 | 0.785 | 0.608 | 0.826 | | | |
| ppop 0.50 | 0.503 | 0.586 | 0.813 | 0.606 | 0.796 | | | |
| **ppop 0.15 + ppgate 0.9** | **0.533** | 0.577 | 0.709 | 0.585 | **0.839** | +0.040 | ±0 | – |
| PubChem v2 fused, slots 4..20 だけ | 0.514 / 0.196 | 0.545 / 0.197 | 0.618 / 0.293 | 0.548 / 0.191 | 0.841 | −0.0043 | +0.052 | 7.6% |
| + relhi 6（rel が大きいと 2..10） | 0.514 / 0.198 | 0.545 / 0.198 | 0.618 / 0.293 | 0.548 / 0.191 | | −0.0043 | +0.053 | 7.6% |
| + 昇格 relz>3 または S>6 かつ pop≥5 | 0.514 / 0.201 | 0.545 / 0.209 | 0.618 / 0.297 | 0.548 / 0.191 | | −0.0044 | +0.058 | 7.1% |
| （参考）昇格 relz>2 | 0.509 / 0.203 | 0.540 / 0.205 | 0.594 / 0.291 | 0.531 / 0.201 | | −0.017 | +0.058 | 23% |
| （参考）昇格 S>4 かつ pop≥5 | 0.513 / 0.200 | 0.542 / 0.232 | 0.618 / 0.344 | 0.548 / 0.229 | | −0.006 | +0.084 | 6.3% |
| **推奨: v2 fused + relhi 6 + 昇格 + ppop 0.15 + ppgate 0.9** | **0.528 / 0.201** | 0.573 / 0.209 | 0.707 / 0.297 | 0.581 / 0.191 | | +0.037 | +0.058 | 0 |
| （比較）旧 B（pcv=1 既定） | 0.515 / 0.209 | 0.547 / 0.244 | – | – | | −0.004 | +0.080 | 4.5% |
| PubChem v2 rev のみ（fused なし）slots | 0.514 / 0.190 | 0.545 / 0.191 | 0.618 / 0.274 | 0.548 / 0.185 | | −0.004 | +0.043 | 8.5% |

- 昇格の当たり（rm=1、推奨構成）: hcqry 19 件中正解 4、haqry 17 中 7、tqry 1 中 1、cqry 20 中 5。正解ありの条件で昇格するのは主に「窓にプール候補がない（relz=1e9）」分子で、もともと点がないので損失は増えない。
- relhi（密なスロット）は ±0.001。

## 本番（LB）への影響の見積もり
f = 隠しテストのうち正解がプール外の割合。ΔLB ≈ (1−f)·Δあり + f·Δ除去。Δあり は知名度を合わせた hcqry（+0.015、ppop+ppgate）と sibling 型（hasqry −0.001）の間、Δ除去 は 4 組の平均。
| f | ppop 0.15+gate だけ（hcqry 値） | PubChem v2 だけ | 推奨（両方, hcqry 値） |
|---|---|---|---|
| 0.05 | +0.014 | −0.001 | +0.012 |
| 0.10 | +0.013 | +0.002 | +0.015 |
| 0.20 | +0.012 | +0.008 | +0.020 |
| 0.30 | +0.010 | +0.014 | +0.024 |
| 0.50 | +0.007 | +0.027 | +0.034 |
注意: (1) 公開 LB の 1 分子 = ±0.004〜0.008、標準誤差 ~0.03 → 1 回の提出で +0.01 は見えない。(2) v25（旧 B）は holdout で分岐点 2.7% だったのに LB ±0 → holdout の正解除去の利得は LB に移りにくい（知名度の偏り、REV2 は本番では rev2_all_v4k で較正が違う）。(3) 本番は FPNet 項が本体に入るので、本体の正解ありでの強さが上がり、PubChem 差し込みの損失（Δあり）はやや大きくなりうる。

## Kaggle への組み込み
1. データセット yomeiotani/casmi26-rdkit-min に `phys_v28pc.cpp`（rdkit-min/、phys_v27.cpp と同じ場所）と **`pool730/pool730_pop.bin`（2.9MB、ビルド機 C:\prog\casmi\pc2\pool730_pop.bin）** を追加。PubChem 表は既存の yomeiotani/casmi26-pubchem-whole（pc_mass/pc_off/pc_pop/pc_smi、pc_cid なしで可: CID の代わりに表の位置を使う。SMILES 出力では不要）。
2. ノートブック: `srcs_fp = ["phys_v28pc.cpp"] + srcs[1:]`。argv[24] の末尾に
   - 知名度だけ（安全側、まずこれ）: `,ppop=0.15,ppgate=0.9,ppopf=<pool730_pop.bin の絶対パス>`
   - 推奨（PubChem v2 も）: `,ppop=0.15,ppgate=0.9,ppopf=<...>,pc=1,pcv=2,pcrank=fused,relhi=6,relp=3,sp=6,pcdir=<pc_mass.bin のあるディレクトリ>`
   （pcout を付けなければ out.csv = 差し込み済みのリスト。オプション文字列は Python のリストの 1 要素なので cmd の分割の問題はない）
3. ログで確認する行:
   - `pool popularity: <path>, 729389 values, weight 0.15 over the top 60 (pool candidates only), only when best libSim < 0.9`（`FILE MISSING, POOL POP DISABLED` が出たら失敗）
   - `pool popularity: weight 0.15, top 60 re-ordered, N list places changed over M queries`（N > 0）
   - `pubchem v2: 112357994 compounds; list = z_window(rev2) + 0.15 pop (top by it, then ordered by z(hand fusion) + pop), slots lo:4:8:12:16:20 hi:2:4:6:8:10 (relz > 6); promote if relz > 3 or (S > 6 and pop >= 5); gate libSim < 0.9`
   - `pubchem: ... thread-s (... ms/query); window .../query`、`pubchem slots: gate open for N / M queries`、`pubchem v2: dense slots N, promoted to rank 1 N / M queries`
4. 実行時間・メモリ（Kaggle 4 コア、~30GB）: ppop はほぼ 0 秒・+3MB。PubChem v2 は 1.4〜2.5 thread-s/クエリ（ビルド機、他ユーザーと共用で 6 スレッド）→ 400 クエリで 4 コア 3〜4 分（Kaggle の CPU が 2 倍遅ければ 8 分）+ 表の読み込み 1〜2 分。メモリ +約 1.8GB（mass 450MB + off 900MB + pop 450MB、SMILES 6GB は窓の範囲だけ読む）。ビルド機での全体のピーク 6.2GB（ylib、PubChem 込み）。
