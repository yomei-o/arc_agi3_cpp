# ICEBERG / GLACIER（ms-pred 公開 MassSpecGym 重み）は私たちのパイプラインに情報を足すか — 状態ファイル

2026-10-21。主催者回答（RESUME 2026-10-21）で ICEBERG/GLACIER（MassSpecGym 学習の ms-pred チェックポイント）は名指しで使用可。
公開ノートブックは同一組成式群の中で位置を保って再ランク（LB +0.007 GLACIER 追加、+0.006 [M+H]+ のみ、+0.007 最後を GLACIER のみ）。
私たちの順方向モデル（phys/MLP/REV2/TFM）は融合に入っているので群内再ランクは効かなかった（postproc_STATUS.md）。ICEBERG/GLACIER は入っていない。
**ノートブックへの統合・アップロード・提出はしていない。**

## 結論（先に）
1. **GLACIER は情報を足す。ただし効き方は holdout の種類で大きく違う。** 一番テストに近い hxqry（timsTOF・[M+H]+ 主体・CE 全件あり・MassSpecGym 分子 2%＝漏れなし）で
   **融合スコアに +w·z(GLACIER)（w=0.25〜1）が +0.011〜+0.017 MRR（95% CI が 0 を含まない）**、ランダム対照は 0〜−0.02。
   hcqry +0.004〜+0.015（CI は 0 を含む、得の大半は MassSpecGym に載っている分子＝漏れ）、hcqry rm=1 は群内再ランクで +0.022（CI 外）、
   **hasqry（sibling）は w≤0.5 で ±0.002、w=1 で −0.017**（正解の別スペクトルがライブラリにあり、私たちの上位がすでに強い）。
2. 物差しの混合 0.5·S + 0.42·hcqry + 0.08·hcqry rm: **S=hasqry では +0.002〜+0.004（有意でない）**、S=hxqry では +0.007〜+0.015（有意）。
   推奨候補は **add_w0.25〜0.5、[M+H]+ のみ**（sibling で損なし: has +0.0006 [−0.005, +0.006] / −0.002 [−0.008, +0.003]、hx +0.013 / +0.015）。
3. **公開ノートブック流の「組成式群の中だけ・位置保持」は私たちのリストでは弱い**（fgr_l1: hx +0.005、hc +0.006、has −0.015、hc rm +0.023）。
   理由は postproc_STATUS と同じ: 私たちの 40 件リストはほぼ全部が同じ組成式（±10 ppm の窓）なので「群内」＝「全体」。GLACIER だけで並べる（fgr_pure）は −0.10〜−0.21。
   PubChem の固定スロット（rm=1 で正解を運ぶ）を動かせるのは群内方式だけなので、rm 成分では fgr が効き add は 0。両方（comb）は has で −0.026 と悪い。
4. **漏れ**: ICEBERG/GLACIER は MassSpecGym（MoNA/MassBank/GNPS 由来）で学習。hcqry の正解 48%・hasqry 60%（train/val fold）が MassSpecGym 分子＝重みが正解のスペクトルを見ている。
   train.parquet の enveda-180（timsTOF 18.3 万分子）は 0.0%、enveda-np-examples 90%、mona 69%、gnps 39%。隠しテストが Enveda の新しい timsTOF データなら漏れは小さく、hxqry が一番近い物差し。
   漏れのない部分（MassSpecGym 以外の正解）だけで見ても hxqry add_w0.5 +0.015 [+0.006, +0.024]、混合（S=hx）+0.010 [+0.001, +0.018]。
5. **ICEBERG は Windows で GPU が使えない**（DGL の Windows wheel は 2.2.1 CPU 版が最後、torch ≤ 2.3）。CPU で 2.2 予測/秒（2 スレッド）→ hxqry 先頭 120 問だけ評価（下記）。
   公開側も最後の並べ替えは GLACIER だけ（ICE+GL 0.402 < GL のみ 0.417）。
6. Kaggle: GLACIER は CPU 4 コアで ~9〜12 予測/秒 → 400 問 × 40 候補 × ~2 条件 ≈ 3 万予測で **約 45〜60 分**（GPU T4 なら 10〜20 分）。持ち込みは ms-pred の部分集合・チェックポイント 1 本（60 MB）・小さな DGL/torch_scatter スタブ・pure-python wheel 数本で足りる（下記）。

## 1. 入手したもの（ビルド機のみ、ラップトップには置いていない）
| もの | 場所（ビルド機） | 出所 / 検証 |
|---|---|---|
| ms-pred ソース | `C:\prog\casmi\iceberg\src\ms-pred-708148c2a8eb0c32120644436fefd2fe90cd2214\` | github coleygroup/ms-pred（samgoldman97/ms-pred の移転先）commit **708148c2a8eb0c32120644436fefd2fe90cd2214** の zip（codeload）。MIT（LICENSE, Copyright (c) 2023 Samuel Goldman）。公開データセットの MANIFEST と同じ commit |
| GLACIER | `...\iceberg\dl\dropbox\glacier_msg\best.ckpt` | ms-pred README の「GLACIER trained on the MassSpecGym dataset」Dropbox。**sha256 5a47cecca707d3abd5a49c7dbac99d100aa2a586d5d4f848f1e8e35140d7db11** = 公開 Kaggle データセット ahmedberatozer/casmi26-glacier の MANIFEST `ckpt_source_sha256` と一致（そのデータセットの ckpt/glacier.pt はこれを state_dict だけに保存し直したもの） |
| ICEBERG 2.1 msg_all | `...\dl\dropbox\ice_msg_all\gen\best.ckpt`（sha256 77d78adf7b60e2fdf68b45db39346c7bde2a1a2dbf57c5e7fc2b05223d43f16a）、`...\inten_contr\best.ckpt`（d2785ff38beb52a1a22200832ed07b9817709470aa03d45bf6af75905bdaa7fa） | README「Pretrained ICEBERG 2.1 Model Weights on MassSpecGym」の msg_all。公開データセット ahmedberatozer/casmi26-iceberg の README/MANIFEST は「msg_all、{hyper_parameters, state_dict} だけに保存し直し、テンソルはビット一致」（gen.pt sha256 deef0451…, inten.pt c49e330e… は保存し直し後の値なので直接は比べられない）。msg_simulation 版（gen 1eda5f3d…, inten e074c039…）も落としたが未使用 |
| MassSpecGym.tsv / 1.5 | `...\dl\MassSpecGym*.tsv` | HF roman-bushuiev/MassSpecGym。漏れの判定用（231,104 スペクトル / 28,929 分子、fold train 22,746 / val 3,185 / test 2,998） |
公開ノートブック（pubnb2）は datasets `ahmedberatozer/casmi26-iceberg` と `ahmedberatozer/casmi26-glacier` を添付（kernel-metadata の dataset_sources）。その MANIFEST/README/LICENSE-NOTICE（テキスト 3 本ずつ、計 ~20 KB）だけを kaggle CLI で読んだ（ラップトップの scratchpad、コード・重みは落としていない）。
訓練データ: README は「MassSpecGym-trained」。ICEBERG msg_all は「CE がある部分で学習 → CE を補完 → 全体で再学習」（README）。hparams.yaml にデータ名はない。

## 2. 環境（ビルド機、システム Python は触っていない）
- `C:\prog\casmi\iceberg\venv`（CPU 参照）: torch 2.3.0+cpu、**dgl 2.2.1**（Windows の最後の wheel、graphbolt の dll が torch 2.3.0 まで → 2.3.1 は不可）、torchdata 0.8.0、pydantic、rdkit **2025.3.6**（MANIFEST が 2026.03 は Kekulize が違うとして固定）、pytorch-lightning 2.4.0、torch-scatter（pyg torch-2.3.0+cpu）、LinSATNet、pygmtools、numpy 1.26（numpy 2 は torch 2.3 と不整合）。`venv_freeze.txt`。
- `C:\prog\casmi\iceberg\venv_gpu`（GLACIER 用）: torch 2.6.0+cu124、torch-scatter 2.1.2+pt26cu124、**dgl なし**。GLACIER の予測経路は密（graphormer + LinSATNet）で、DGL は (1) import と (2) 位置符号用の `dgl.graph(...)` 1 か所（glacier/dataset.py get_pe_for_tensor）だけ → `ice_score.py --dgl-stub` が import 用の空モジュールと最小のグラフ（edges / num_nodes / ndata / edata）を登録。
  **検証**: 参照環境（本物の DGL、CPU）と比べて 1,680 組（15 分子）のエントロピー類似度の差 平均 0.0006・最大 0.021、1 位候補は 14 分子中 13 で一致（GPU の浮動小数とバッチ構成の差）。smoke 6 分子では 3 桁一致。
- ICEBERG は DGL のグラフ演算（node_subgraph / reorder_graph / batch / unbatch / update_all など）を本当に使うので、GPU（DGL の CUDA 版）は Windows では無理。CPU 参照環境で動かす（torch_scatter が int32 の添字を拒むので long に変換するラッパを ice_score.py に入れた）。ckpt は PosixPath を pickle しているので Windows では WindowsPath に差し替えて読む。

## 3. 評価の作り方
- **リスト**: v30 構成（phys_v30pc、v28b オプション + pcrev=tan、holdout 安全なモデル ylib/zlib, fwd_mlp_rank_ylib, rev2_y/rev2_z、FPNet なし）を特徴ダンプ付きで再実行（`casmi/iceberg/v30dump.bat`, `laneA/B.bat`、出力 `C:\prog\casmi\iceberg\v30\`）。最終リスト（`_ins.csv`）は既存の結果を再現: hcqry 0.5255（pc3 hcT0 と同じ）、hcqry rm=1 0.2152（hcT1）、hasqry 0.8374（pc4 has30_0）、hxqry **0.8596**（新規）。
- **候補**: 最終 25 件（PubChem スロット込み）＋融合リスト（pp ダンプ、上位 60）の続きで 40 件まで（33〜39 件/問）。名前→SMILES は特徴ダンプ（プール・生成）、det ダンプ（PubChem）、rm=1 で正解を運ぶ PubChem スロット（IK14 名で書かれる）は mols.txt から。
- **クエリ**: holdout の bin に CE がないので train.parquet から照合して回収（`q_extract.py`、全スペクトル一致）。CE がある割合: hcqry 222/972、hasqry 286/1094、**hxqry 1163/1163**。CE がないスペクトルは 30 eV で予測（フラグ付き）。CE は平均を 5 eV に丸め、装置は timsTOF/QTOF 系→QTOF、Orbitrap/ESI-QFT→Orbitrap。正イオンの [M+H]+ / [M+Na]+ だけ採点（公開と同じ）。
- **スコア** = 候補の予測スペクトル（m/z 4 桁で合算（ICE）/最大（GL）、上位 100 本）とクエリの重み付きエントロピー類似度（0.01 Da）のスペクトル平均。コサインも出したが has で一貫して悪い（add_w1 −0.035 vs −0.017）→ エントロピーを主に。
- **変種**（`ice_eval.py`）: base（= v30 の最終 25）、fgr_pure（同一組成式群内を GLACIER だけで、位置保持）、fgr_l1（群内を z(1/(3+r)) + 1·z(GL)、公開流）、add_wX（PubChem 以外の全項目を z(融合+0.15·pop) + w·z(GL) で並べ直し、PubChem スロットは固定。add_w0 = 融合だけで並べ直した参照、base との差 −0.002〜+0.002）、gate_*（ライブラリ類似度 < 0.9 の問だけ）、comb（add の後に fgr_l1）、それぞれ `_mh`（[M+H]+ のスペクトルだけ）と `_rand`（GL を一様乱数に、同じ被覆）。
- MRR は 25 件で切った CSV（`out_<tag>_<variant>.csv`、mrr.py と同じ規則、base は mrr.py の値と一致）。95% CI は問ごとの差のブートストラップ 2000 回。MSG / nonMSG = 正解が MassSpecGym（train+val）分子か。

## 4. 結果（GLACIER、エントロピー、K=40、差は base から）
被覆: 採点できた問 hx 354/400、hc 291、has 302（負イオン・[M+NH4]+ だけの問は採点なし）。予測数 hx 30,406 / hc 13,183 / has 14,110。

| 変種 | hxqry (S′) | hasqry (S) | hcqry (D) | hcqry rm (R) | 混合 S=has [95% CI] | 混合 S=hx [95% CI] | 混合 S=hx・非MSG |
|---|---|---|---|---|---|---|---|
| base MRR | 0.8596 | 0.8374 | 0.5255 | 0.2152 | | | |
| fgr_pure | −0.119 | −0.210 | −0.099 | +0.011 | −0.145 | −0.100 | −0.106 |
| fgr_l1 | +0.005 [+0.001,+0.010] | −0.015 [−0.025,−0.007] | +0.006 | **+0.023** [+0.013,+0.034] | −0.003 [−0.010,+0.003] | +0.007 [+0.002,+0.012] | +0.003 |
| fgr_l1_mh | +0.006 | −0.014 | +0.007 | +0.024 | −0.002 | +0.008 [+0.003,+0.014] | +0.003 |
| add_w0（参照） | +0.002 | −0.002 | +0.002 | 0 | +0.000 | +0.002 | +0.002 |
| add_w0.25 | +0.011 [+0.005,+0.018] | +0.001 [−0.005,+0.006] | +0.004 | 0 | +0.002 [−0.003,+0.007] | +0.007 [+0.002,+0.013] | +0.006 |
| **add_w0.25_mh** | **+0.013** [+0.006,+0.021] | **+0.001** [−0.005,+0.006] | +0.005 | 0 | +0.002 [−0.003,+0.007] | **+0.008** [+0.003,+0.014] | +0.006 [−0.001,+0.013] |
| **add_w0.5_mh** | **+0.015** [+0.006,+0.024] | −0.002 [−0.008,+0.003] | +0.011 [−0.002,+0.024] | 0 | +0.004 [−0.003,+0.010] | **+0.012** [+0.005,+0.019] | +0.009 [+0.001,+0.018] |
| add_w1_mh | +0.017 [+0.006,+0.029] | **−0.017** [−0.028,−0.007] | +0.015 | 0 | −0.002 | +0.015 [+0.005,+0.024] | +0.008 |
| gate_add_w0.5_mh | +0.007 | −0.001 | +0.006 | 0 | +0.002 | +0.006 | +0.004 |
| comb_w0.5_mh | +0.011 | −0.022 | +0.009 | +0.020 | −0.006 | +0.011 | +0.005 |
| 対照 fgr_l1_rand | −0.061 | −0.087 | −0.041 | −0.009 | −0.062 | −0.048 | |
| 対照 add_w0.5_rand | +0.000 | −0.012 | −0.025 | 0 | −0.017 | −0.011 | |
（hc は CI がすべて 0 を含む。hc の得は MSG 分子に偏る: add_w0.5_mh で MSG +0.018 / 非MSG +0.004、add_w1_mh +0.031 / −0.001。has も MSG +0.000 / 非MSG ±0.004。）
読み:
- GLACIER は乱数より明確に良い（同じ変種の rand 対照は −0.01〜−0.09）→ 情報はある。
- それが**私たちの融合に足して得になる**のは hxqry（timsTOF、xinstr）と hcqry rm（PubChem 候補の並べ替え）。sibling（has）では w を上げると損 → 物差しの関門（sibling/xinstr で損なし）を通るのは **w ≤ 0.5**。
- [M+H]+ だけ（_mh）はわずかに良い（公開の +0.006 と同じ向き）。
- ライブラリ類似度のゲート（< 0.9）は has の損を消すが hx の得も半分にする → 使わない。
- 公開流の群内再ランクは、私たちには PubChem スロット（rm 成分）でだけ効く。プール候補は add の方が良い。

ICEBERG: RESULTS_ICE

## 5. 実行時間（ビルド機、他エージェントで CPU 91% 使用中の値）
| モデル | 装置 | 予測/秒 | 1 予測 = 候補 1 つ × 条件 1 つ |
|---|---|---|---|
| GLACIER | RTX 3090（共有、他のジョブが GPU 91%）、batch 32 | 18〜28 | hxqry 400 問 30,406 予測 1,124 s |
| GLACIER | CPU 2 スレッド（参照環境） | 8.7 | hxqry 15 問 1,280 予測 147 s |
| ICEBERG | CPU 2 スレッド | 2.2 | hxqry 120 問 |
| ICEBERG | CPU 4 スレッド（smoke、小バッチ） | 2.0 | |
可視テスト（プレースホルダ、400 分子 1,213 スペクトル）は全部 timsTOF・CE あり、[M+H]+ 959 本 / 正イオン 987 本 → hxqry と同じ形。
Kaggle CPU 4 コアの見積り: GLACIER [M+H]+ のみ、40 候補、~2 条件/問 → ~3 万予測 ÷ 9〜12/s ≈ **45〜60 分**（候補を同一組成式の群だけ・25 件にすれば ~30 分）。ICEBERG は CPU で ~2/s → 4 時間超、CPU では非現実的。
競技の制限: 私たちの CPU ノートブックは 540 分で拒否された実績（v31、665 分）。公開ノートブックは `enable_gpu: true`（T4、`machine_shape NvidiaTeslaT4`）で、ICEBERG 5400 s・GLACIER 4000 s の予算を付けて 9 時間以内に収めている（n1a の説明）。GPU ノートブックも同じ 9 時間枠の扱い（公開ノートブックの記述より。競技ページの規則本文は未確認）。

## 6. Kaggle にオフラインで持ち込むもの
公開側のやり方（ahmedberatozer/casmi26-iceberg・-glacier の README/MANIFEST より、コードは写していない）:
- データセットに ms-pred の推論に要るモジュールだけ（`src/ms_pred/`、commit 708148c、LICENSE 同梱、plot/tune 系は空の置き換え）、チェックポイントを `{hyper_parameters, state_dict}` に保存し直したもの（ICE 14.6 + 13.8 MB、GL 60.5 MB）、`shim/`（DGL と torch_scatter の推論で使う関数の純 torch 版、自作）、`wheels/*.whl.ice`（rdkit 2025.3.6 cp312、pytorch_lightning 2.6.6、torchmetrics 1.9.0、lightning_utilities 0.15.3、einops 0.8.2。ノートブック本体の `rdkit-*.whl` の glob に拾われないよう拡張子を変えてある）、`ice_runner.py` / `gl_runner.py` / `fuse.py`。
- 実行: 本体プロセスとは別の subprocess で `pip install --no-index --no-deps --target <site>` → `shim/`・`src/`・site を sys.path の先頭に → 予測（予算時間で打ち切り、失敗は全部 null = 並べ替えなし）。rdkit は 2025.3.6 に固定（2026.03 は Kekulize が違い予測が変わる）。Kaggle の画像が cp313 になったとき 2025.3.6 の cp313 wheel を別データセット（eveliaveldrine/rdkit-wheels）から入れる修正が入った。torch は Kaggle 画像のもの。
私たちがやるなら（GLACIER だけ）:
- 自前データセット: ms-pred の `src/ms_pred` から glacier・common・nn_utils・magma（commit 708148c、MIT、LICENSE 同梱）、`glacier_msg/best.ckpt`（182 MB、または state_dict のみに保存し直して ~60 MB）、ice_score.py 相当の採点コード（`--dgl-stub` で DGL 不要）、torch_scatter の代わり（GLACIER の予測経路は `ts.scatter_max` だけ → `torch.Tensor.scatter_reduce(amax)` で書ける。未実装）、wheels: rdkit 2025.3.6（Kaggle の Python 版に合う cp タグ）、LinSATNet、pygmtools（pure python）、pytorch_lightning（画像にあれば不要）。
- 入力は v30 の最終 40 件（ランカーが nout=40 で書く、v28 から）。GLACIER は [M+H]+ のスペクトルだけ、`add_w0.25〜0.5`: プール候補の融合スコアに w·z(GL) を足して並べ直し、PubChem スロットは固定。C++ 側の変更なし（Python の後処理で済む）。
- 注意: 漏れのない検証は hxqry だけ。sibling（has）では w=1 で損。LB で確かめるなら w=0.5・[M+H]+ のみ 1 本。

## 7. ファイル
`casmi/iceberg/`（グルーだけ。ms-pred 本体はビルド機の上記パスに commit 指定で置き、リポジトリには入れていない）
- `v30dump.bat`, `laneA.bat`, `laneB.bat` — v30 を特徴ダンプ付きで再実行
- `q_extract.py` — holdout クエリの CE・イオン化モードを train.parquet から回収 → `q_<qry>.json`
- `msg_overlap.py` — holdout 正解・train の各ライブラリと MassSpecGym の重なり → `msg_ik14_<fold>.txt`
- `ice_score.py` — ICEBERG / GLACIER で候補のスペクトルを予測して採点（再開可、meta.json に時間）
- `ice_eval.py` — `build`（40 件リスト・候補 json・ライブラリ類似度）と `eval`（変種 CSV・問ごとの RR・ブートストラップ）
- `ice_mix.py` — 物差しの混合とその CI
- `eval_all.bat` — 4 組を作り直して評価（`eval_all.bat gl` / `eval_all.bat ice`）
- `setup_ref.bat`, `setup_gpu.bat` — 2 つの venv、`gl_run.bat`, `ice_run.bat`, `job_*.bat` — 実行（launch.ps1 で切り離し）
- `cmp_scores.py`（GPU スタブ vs 本物の DGL）、`peek.py`、`rgrep.py`（ビルド機での読み取り補助）
ビルド機の出力 `C:\prog\casmi\iceberg\`: `gl_<tag>.json`（予測スコア）、`list_/cands_/lib_<tag>.json`、`out_<tag>_<variant>.csv`、`rr_<tag>.tsv`、`eval_gl_<tag>.txt`、`mix_gl_has.txt` / `mix_gl_hx.txt`。

## 続き（再開手順）
- ICEBERG hxqry 先頭 120 問: `C:\prog\casmi\night\ice_ice_hx0.log`、出力 `ice_hx0.json` → `ice_eval.py eval ... hx0 ... ice_hx0.json`（120 問以外は base のまま = 被覆の差に注意）。
- 統合するなら: v30/v32 ノートブックの後処理に GLACIER（[M+H]+、add_w0.5）。Kaggle 用の torch_scatter 代替と rdkit 2025.3.6 wheel の用意が先。
