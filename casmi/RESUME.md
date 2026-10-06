# CASMI 2026 — 再開メモ

2026-10-06 時点。次に続ける人（未来の自分）向け。

## 現在のスコア

| 提出 | 変更 | publicScore |
|---|---|---|
| v4 | 質量窓ウィジェニング + Gaussian mass prior + instrument bonus | 0.108 |
| **v6** | **+ ピーク強度フロア（ノイズ除去）** | **0.121（現在のベスト）** |
| v7 | + COCONUT統合プール + analog propagation（構造類似度のみ、質量シフト無視） | 0.059（悪化、原因はmodified cosine未実装と推定） |

順位: 約2186位/2487チーム中（v6時点）。6位(Sho Saga)が0.436、1位が0.471。
今日の提出5回は使い切り済み（2026-10-05分）。日付が変われば5回に回復。

詳細は [[casmi-analog-propagation-regression]] [[casmi-reading-notes]] [[holdout-gain-does-not-transfer-1to1]] （`~/.claude/projects/.../memory/` のメモ）。

## 今日（2026-10-06）やったこと・わかったこと

5位のSho Sagaさんが公開した解説ページ（`casmi/reference/forward-model-sho-saga.html`）を読んだ。
中身は ICEBERG（Goldman et al. 2023, NeurIPS; arXiv:2304.13136）をベースにした「順方向モデル」:

- 候補分子ごとに「結合を1〜2本切ってできるかけらを列挙→GNNで強度予測→予想スペクトルを実測と照合」
- 質量だけでは分子式が同じ異性体（中央値46個/問）を区別できないが、壊れ方の違いは区別できる
- 実LBで +0.038〜+0.054（同じ分子式の候補どうしだけ並べ替える使い方）

公式実装 `github.com/samgoldman97/ms-pred`（MITライセンス）に、
MassSpecGym学習済みのオープンな重み（NISTライセンス不要）がDropboxで公開されている:
`gen/best.ckpt` + `inten_contr/best.ckpt`。ビルドマシンに既にダウンロード済み
（`C:\prog\casmi\iceberg_msg_all\`）。

**方針決定**: GNN推論自体をPyTorch/DGLのままPythonで動かすのは技術的に可能
（GPU律速でPythonオーバーヘッドは無視できる）だが、ユーザーの明確な意向
「Pythonでやるものには興味ない」により不採用。代わりに:

**同じ「順方向モデル」のアイデアを、学習済みGNNではなくルールベースのヒューリスティックで、
完全にC++で実装する。** 化学的に正しいフラグメント列挙（結合切断・環判定・価数）は自前実装が
危険すぎるので、RDKitの**C++コア本体**（Pythonバインディングではなく）をソースからMSVCで
ビルドしてリンクする。

## 進行中の作業（中断時点）

ビルドマシン（192.168.6.14）で:

1. PyTorch 2.6+cu124, pytorch_lightning等をインストール済み（**今回の方針転換で不要になった。
   消さなくても害はないが、次回air使わない**）
2. ICEBERGの学習済み重みをダウンロード済み: `C:\prog\casmi\iceberg_msg_all\{gen,inten_contr}\best.ckpt`
   （**これも今回の方針では使わない**。参考実装として ms-pred のソース
   (`C:\prog\casmi\ms-pred-main\`) は残しておく価値あり — フラグメント列挙のロジックや
   ヒューリスティックのヒントとして読む分には有用）
3. RDKitソースをダウンロード・展開済み: `C:\prog\rdkit-Release_2026_03_1\`
4. vcpkgをブートストラップ済み: `C:\prog\vcpkg\`
5. **`vcpkg install boost:x64-windows` を実行中だったが、中断時点で未完了。
   SSH接続が切れると、ビルド中のプロセスごと終了している可能性が高い
   （CLAUDE.mdに記載の既知の挙動）。次回は `tasklist | findstr cl.exe` で
   生きているか確認し、死んでいたら `cd C:\prog\vcpkg && vcpkg.exe install boost:x64-windows --disable-metrics`
   を再実行すること。**

## 次回やること

1. Boostのインストールを確認・再実行
2. RDKitをCMake configureする。要点:
   ```
   cmake -S C:\prog\rdkit-Release_2026_03_1 -B C:\prog\rdkit-build -G Ninja ^
     -DCMAKE_TOOLCHAIN_FILE=C:/prog/vcpkg/scripts/buildsystems/vcpkg.cmake ^
     -DVCPKG_TARGET_TRIPLET=x64-windows ^
     -DCMAKE_BUILD_TYPE=Release ^
     -DCMAKE_INSTALL_PREFIX=C:/prog/rdkit-install ^
     -DRDK_BUILD_PYTHON_WRAPPERS=OFF ^
     -DRDK_BUILD_CPP_TESTS=OFF ^
     -DRDK_USE_BOOST_SERIALIZATION=OFF ^
     -DRDK_USE_BOOST_IOSTREAMS=OFF ^
     -DRDK_BUILD_DESCRIPTORS3D=OFF ^
     -DRDK_BUILD_COORDGEN_SUPPORT=OFF
   ```
   （MSVCの `vcvars64.bat` を通した環境で実行すること。cmake/ninjaはVS Build Tools同梱、
   `vswhere` で探す — CLAUDE.mdの手順参照）
3. ビルド・インストール（`ninja install`、20コアなので数十分の見込み、未検証）
4. `casmi/cpp/` に新しいツール（例: `forward_model.cpp`）を作り、RDKitのC++ APIで:
   - 候補SMILESをパース
   - 結合を1〜2本切ってフラグメントを列挙（環は2本切り or 1本+鎖1本が必要なことに注意）
   - 各フラグメントの精密質量を計算
   - ヒューリスティックな強度スコア（結合の種類・切断本数などから、学習済みではなくパラメータ化）
   - 予想スペクトルを構築し、既存の `cosine()`/`entropy_sim()` で実測と照合
5. 既存の候補リスト（match.cppの出力）のうち、**同じ分子式の候補どうしだけ**をこのスコアで
   並べ替える（Sho Sagaさんのページで実LBに効いたのはこの使い方）
6. 効果はholdoutで測定してから提出（提出は貴重、5回/日）

## 保留事項

- ヒューリスティックのパラメータ（結合切断のペナルティ、ヘテロ原子隣接のボーナス等）は
  当て推量ではなく、既存のholdout測定の枠組みで sweep して決めること（これまでのプロジェクトの流儀）
- RDKitのC++ APIは本体のドキュメントが薄い。`Code/GraphMol/`配下のヘッダを直接読むのが早い
  （特に `RWMol`, `MolOps::getMolFrags`, `Bond` クラス）
