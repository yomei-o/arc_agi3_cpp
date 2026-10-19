# pc3 — PubChem 窓内の REV2 採点を連続 Tanimoto に（phys_v30pc、状態ファイル）

2026-10-19 夕方。動機: fp3_STATUS.md の物差しで、窓内の並べ方を logit 和 → 連続 Tanimoto にすると hcqry 窓内 MRR 0.185 → 0.266（+0.15 pop 込み 0.297 → 0.367）。これを本番チャネル（phys_v28pc の pcv=2）に入れる。

## ファイル
- `cpp/rdkit-min/phys_v30pc.cpp` = phys_v28pc + argv[24] オプション
  - `pcrev=logit|tan|cal`（既定 logit = phys_v28pc と同じ）: PubChem 窓の候補の REV2 得点。前段の並び（z_window(rev)+0.15 pop）、pcMean/pcSd、relz、S、fused リストの rev 項（z はプール候補の同じ形の得点で取る）すべてに効く。
    - tan = Σ_{候補のビット} p_b / (n_bits + Σ_all p − Σ_{候補のビット} p_b)、p = クエリのスペクトル平均（クリップなし）
    - cal = Σ Platt 較正 logit（calA·logit(p, clip 1e-6) + calC、±13.8 でクリップ; モデルに較正がなければ logit）
    - どちらも cpp/rev3/win_eval.cpp と同じ式。候補のビットは REV2 語彙（fp2::Vocab::bits、重複なし）。
  - `poolrev=logit|tan|cal`（既定 logit）: プール・生成候補の本体融合の rev 項。未測定（既定は変えない）。
  - ログ行: `reverse model score form: PubChem window tanimoto, pool logit (Platt calibration in the model: N)`
- `bat_pc3/run.bat` — bat_pc2/run.bat に v28b のオプションを固定、exe とスレッド数を引数に（ビルド機 `C:\prog\casmi\fwdmodel\pc2\run3.bat`、出力 `C:\prog\casmi\pc3\`）。ビルドは `bat_pc2/build.bat phys_v30pc`（fwdmodel\pc2 にソースを置く）。
- `notebook/casmi_match_v30.ipynb` = v28b ノートブック + `srcs_fp = ["phys_v30pc.cpp"] + ...` + PubChem 有効時に `,pcrev=tan`。`notebook/kernel-metadata-v30.json`（id yomeiotani/casmi26-cpp-library-search-v30、dataset_sources は -b と同じ）。
- Kaggle 用のステージ: ビルド機 `C:\prog\casmi\pc3\ds_add\phys_v30pc.cpp`（データセット casmi26-rdkit-min の rdkit-min/ に phys_v28pc.cpp と並べて追加する。追加のヘッダは不要: ds_rdkit の rev/rev2_model.hpp に calA/calC がある）。**データセットの更新・push・提出はしていない。**

## 検証
RESULTS_PLACEHOLDER
