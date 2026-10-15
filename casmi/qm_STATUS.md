# qm_STATUS — xTB 由来の障壁・プロトン親和力を物理モデルへ（2026-10-15）

## 状態: 完了（13:00）
- [x] 1. QMデータ 1357分子（fqry 857 + COCONUT train 500）: PA 13301部位、開裂 28116方向 / 14058結合
- [x] 2. 安い式: PA 表（R² 0.44）、方向つき開裂エネルギー（R² 0.83）、結合単位（R² 0.378、CV 0.374）
- [x] 3. phys_qm.cpp（phys_rank 改造）/ phys_dag_qm.cpp（phys_dag 改造）に組込み、fqry で grid、cqry/tqry 評価
- [x] 4. PHYSCHEM_IDEAS.md 追記（「2026-10-15: xTB（GFN2）から作った安い式を物理モデルへ」）

## 結論（詳細と表は PHYSCHEM_IDEAS.md）
1. **方法上の発見**: phys_rank/phys_dag の MRR は同点を悲観的に数えるので、候補ごとに違う摂動は何でも点を上げる（QM と逆符号でも上がる）。`QM_TIES=avg`（同点はランダム順の期待値）で比べると、DAG の phys2 に対する優位の約半分は同点崩しだった（cqry +0.028 → +0.012、tqry +0.022 → +0.011）。午前のスモーク「+0.014 / +0.020」もこの人工物。
2. **電荷の分け前（QM の方向エネルギー差、gDir）は小さいが本物**: 両モデル・両 holdout で符号どおり（DAG: 0.4282/0.3772 → 0.4305/0.3805、逆符号 0.4174/0.3671）。モデルを通さない検証でも見える側を 65%（PA 表 61%、大きい方 60%）。
3. **障壁補正（κ）は phys_rank 型でだけ効き、DAG では効かない**。QM の結合エネルギーで「どの結合の断片が見えるか」を当てる力は分子内 AUC 0.55〜0.57、式では 0.48（偶然と同じ）。
4. **融合（MLP・逆向き・ライブラリ込み）では差なし（±0.003）→ 提出構成には入れない**。QM の PA 表も差なし。

## 数値（同点平均、cqry / tqry）
| | 物理単独 | 融合 w=.25 wr=1 |
|---|---|---|
| phys_rank 型 基準（p_refitT） | 0.4147 / 0.3621 | 0.4894 / 0.5418 |
| + κ 0.03, gDir 0.1 | 0.4233 / 0.3729 | 0.4886 / 0.5435 |
| DAG d2 基準（pd_refitT） | 0.4282 / 0.3772 | 0.4960 / 0.5555 |
| DAG + gDir 0.1 | 0.4305 / 0.3805 | 0.4984 / 0.5555 |

悲観計数（従来の数え方）: phys_rank 型 QM 0.4125 / 0.3604（phys2 0.3947 / 0.3522）、DAG+gDir 0.4241 / 0.3803（d2 0.4228 / 0.3743）。

## プロトコル（qm2.py）
GFN2-xTB `--opt crude`、ETKDG+MMFF 初期構造、8並列。
- PA_site = E(M) − E(MH+ at site)（N/O/S 全原子と芳香族C–H、対称等価は1回）。最適化後に付加Hの最寄り重原子を記録し、移動したものは PA の回帰から除外。
- 非環の単結合（重原子間、C–X も C–C も）を両方向で: MH+ → A+ + B–H（Bの切断原子にH）。基準は最安 MH+（REF=local で脱離原子に陽子を置いた MH+）。
- 所要: fqry 1000分子 97分、追加500 29分、芳香族C補完 10分。phys_qm の評価 1分/holdout、grid 数分、全パラメータ再フィット 約30分。

## ファイル
- C++: `cpp/rdkit-min/phys_qm.cpp`, `cpp/rdkit-min/phys_dag_qm.cpp`（ビルド機 `C:\prog\casmi\fwdmodel\qm\`、`build_qm.bat <name>` は cl 直呼びで共有ビルドに触れない）
- 環境変数: `QM_COEF=<coef.txt> QM_DIR=1 QM_PA=1 QM_FIT=i,j QM_GRID="k,..;g,.." QM_TIES=avg`。パラメータ番号 phys_qm: κ 25、gDir 26、betaPos 9 / DAG: κ 29、gDir 30。
- 係数 `py/qm/qm_coef_final.txt`（dir/bond 25項、med、pa 24型 + ln + donors + median）、回帰ログ `py/qm/qm_fit_final.log`、検証ログ `py/qm/qm_obs_final.log`。パラメータ（ビルド機 `C:\prog\casmi\qm\`）: p_refitT / p_gF（phys_qm）、pd_refitT / pdc_g / pd_q（DAG）。評価ログ `C:\prog\casmi\qm\ev_*.log, evd_*.log`。
- Python: `py/qm/qm2.py`（xTB 駆動、再開可）、`qm_fit.py`（回帰と係数出力）、`qm_obs.py`（モデルなし検証）、`qm_check.py`（C++ と一致確認）、`mk_extra.py`。データ `py/qm/data/*.csv`（2.2 MB）。
- 注意: 最初の qm2.py は `AddHs` 後に `GetTotalNumHs()` を見ていたため芳香族 C–H 部位を取りこぼし、`QM_ONLY_ARC=1` で補完した（*_arc_*.csv）。cmd のバッチ引数ではカンマも区切りなので `QM_FIT` は "25,26" と引用符で渡すこと。

## 次にやるなら
- 遷移状態: 反応エネルギーでは C–C が C–X より「切れやすい」と出る（H移動を要する経路の速度論が見えない）。xTB の経路探索で代表的な数十結合の障壁を見る。
- 電荷の分け前は式より生の QM が良い（65.3 vs 63.9%）→ 方向の式（陽イオン安定性）の改良。
- 同点平均の MRR を phys_* の既定にするか検討（過去の比較は悲観計数で、同点崩しを改善と見誤る危険）。
