# CASMI 2026 — 再開メモ

2026-10-06 夜 時点。次に続ける人（未来の自分）向け。

## 現在のスコア

| 提出 | 変更 | publicScore |
|---|---|---|
| v4 | 質量窓ウィジェニング + Gaussian mass prior + instrument bonus | 0.108 |
| **v6** | **+ ピーク強度フロア（ノイズ除去）** | **0.121（現在のベスト、未提出の改善あり↓）** |
| v7 | + COCONUT統合プール + analog propagation（構造類似度のみ、質量シフト無視） | 0.059（悪化） |

順位は約2186位/2487チーム中（v6時点）。6位(Sho Saga)が0.436、1位が0.471。
提出は2026-10-05分の5回を使い切り、2026-10-06は**まだ1回も提出していない**
（今日作った新手法はholdoutで検証済みだが、実際のLB提出はまだ）。

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

## 次回(明日)やること

1. **PubChemデータのダウンロード状況を確認。** `C:\prog\casmi\pubchem\`に
   `CID-Mass.gz`(1.3GB)と`CID-SMILES.gz`(1.4GB)をダウンロード開始したが、
   SSH接続が切れるとプロセスごと終了している可能性が高い(既知の挙動)。
   `dir C:\prog\casmi\pubchem`で完了しているか確認し、途中なら
   `C:\prog\dl_pubchem.ps1`を再実行。
2. **PubChemスケールの候補プール拡充**(ユーザー発案、Phase Aの延長):
   - `CID-Mass.gz`でまず質量フィルタ(245〜461Da程度)——RDKit不要、
     テキスト処理だけなので1.1億件でも現実的
   - 絞ったCIDだけ`CID-SMILES.gz`と突き合わせてSMILES取得
   - その後だけRDKitでパース・質量確認・InChIKey14で重複排除
   - `build_pool.py`と同じ要領でpool_head.bin/pool_mols.txtに統合
3. **Kaggleノートブックへの移植**:
   - `rdkit-min/`+`boost-min/`をKaggle Datasetとしてアップロード
     (ビルドマシン上で既にg++ビルド確認済み、Linux本番でも動くはず)
   - ノートブック内でg++により`forward_rerank`をコンパイル
   - match.cppの出力→forward_rerankで同一分子式グループ並べ替え→提出
4. **期待値**: forward_rerank単体では実LBで+0.03〜0.05程度の上乗せが
   現実的な見込み(0.121→0.15〜0.17くらい)。Sho Sagaさんの+0.038〜0.054は
   **もっと強いベース(0.374〜0.412)の上に乗せた数字**なので、同じ絶対
   幅が乗っても到達点は変わる。0.4に近づくには候補プール拡充
   (PubChemスケール)と組み合わせる必要がある。

## 保留事項

- `C:\prog\casmi\iceberg_msg_all\`(ICEBERGのGNN学習済み重み)と
  `C:\prog\casmi\ms-pred-main\`(公式実装ソース)は今回の方針では使わない
  が、消さずに残してある。フラグメント列挙のロジックやヒューリスティック
  のヒントとして読む価値はある。
- `casmi/third_party/`(RDKit本家フルビルド手順)は使っていない。
  `casmi/cpp/rdkit-min/`が本命。
