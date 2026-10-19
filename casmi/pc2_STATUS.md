# pc2 — PubChem チャネル v2（公開式のゲート・固定スロット・昇格）とプール内の知名度（状態ファイル）

2026-10-19 開始。動機は gap_STATUS.md の #1（プール内の知名度 +0.010）と #4（PubChem の昇格）。手の重みだけ、学習した融合はなし。

## ファイル
- `cpp/rdkit-min/pool_pop.cpp` — pool730 の各分子 → PubChem 知名度（float32、pool730_mols.txt の行順）。質量 ±0.003 Da の窓 → 組成式シグネチャ一致 → 非立体カノニカル SMILES 一致、複数一致は pop の最大
- `cpp/rdkit-min/phys_v28pc.cpp` — phys_v27 + phys_pc のチャネル（pcv=1）+ 新チャネル pcv=2 + ppop。オプションはヘッダ参照。全部 off なら phys_v27 と同じ出力
- `bat_pc2/build.bat`, `bat_pc2/run.bat`（ビルド機 C:\prog\casmi\fwdmodel\pc2、出力 C:\prog\casmi\pc2）
- `pc2_grid.py` — dump（_pp.txt / _det.txt）から ppop・スロット・昇格のグリッドを再現

## 進捗
- [x] phys_v28pc.cpp（MSVC ビルド済み）
- [ ] pool730_pop.bin（実行中）
- [ ] holdout 実行（cqry/tqry/haqry/hcqry × rm 0/1、hasqry）
- [ ] グリッド・確定構成・CSV 再採点
