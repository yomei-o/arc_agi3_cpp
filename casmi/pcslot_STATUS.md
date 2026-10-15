# PubChem 固定スロット (phys_pc) — 状態ファイル

2026-10-15 開始、同日 15時に一区切り。公開上位ノートブックの「ライブラリの証拠が弱い分子では、PubChem 全体（プール外）の候補を最終 top-25 の固定スロットに差し込む」を自前の C++ で再現し、手元の holdout で効くかを測る。

## ファイル
- `cpp/rdkit-min/phys_pc.cpp` — phys_gen.cpp のコピー＋PubChem チャネル（オプション文字列 argv[24] に `pc=1,...`、ヘッダ参照）
  - 窓: 本番と同じ ±10ppm・スペクトルごとのアダクト（2M は 1/2）。PubChem 表（pc_build.cpp: 1.12 億化合物 50〜1500 Da、質量順、SMILES ≤250）から窓ごとに連続読み出し
  - 前段: 窓内の全化合物を rdkit-min で読み、REV2 語彙ビットの logit 和 + pcpp·知名度（pc_pop.bin）で並べ上位 pcpre 件
  - 非立体カノニカル SMILES でプール窓・生成候補・互いと重複除去 → 上位 pcn 件に本体と同じ特徴（物理・MLP・アナログ）
  - PubChem リストの得点: 本体の手の融合（z はメイン候補の平均・標準偏差で）+ pcpop·知名度
  - 差し込み（既定 pcmode=score）: ゲート（メイン候補の libSim 最大 < gate）が開いたとき、PubChem リスト上位 pck 件を、得点（融合 + pcsw·pop + pcbias）が本体の次の候補（FPNet 項を足す前の得点）を超える位置に、pcmin 位以降で入れる。pcmode=slot で公開式の固定スロット
  - 出力: pcout なし → out.csv が差し込み済み（本番）。pcout あり → out.csv = 差し込みなし、pcout = 差し込みあり。pcdet = ゲート統計・本体 top25 の得点・PubChem 候補の特徴（Python グリッド用）
- `pcslot_grid.py`（slot/score のグリッド）、`pcslot_grid2.py`（score の細かいグリッド、cqry で選び tqry で評価）、`pcslot_pre.py`（窓内の前段の並べ方）、`pcslot_write.py`（dump から 1 構成の CSV を書く → mrr.py）、`pcslot_ikcheck.py`（RDKit InChIKey14 で真偽判定を照合）、`pcslot_build.bat` / `pcslot_run.bat`（ビルド機用）
- ビルド機: `C:\prog\casmi\fwdmodel\pcslot\`（build.bat, run.bat, phys_pc.exe）、出力・ログ・dump `C:\prog\casmi\pcslot\`（確定構成は t3*/c3*/s3*/o3*、A は *_A.csv）

## 結論（2026-10-15 15時）
- **holdout の正解の 84〜98% は PubChem 全体の ±10ppm 窓にある**（tqry 173/176、cqry 338/400、sqry 335/400）。窓は 1クエリ 7千〜1.3万件。
- **窓の中で正解を当てる主役は知名度（pc_pop = log1p #SID + log1p #PMID）**。REV2 の logit 和だけでは窓内 MRR 0.05、知名度を主・REV2 を同点崩しにすると 0.59（tqry 正解除去）。**漏れの疑いあり**（下記）。
- 公開ノートブック式の固定スロット（2,4,6,8,10 / k=2〜5）は、正解がプールにある分子で −0.03 失う（分岐点 17〜21%）。
- **スコア比較で差し込む方式（pcmode=score）が固定スロットより全面的に良い**: PubChem 候補の得点を本体と同じ尺度（本体の z 統計で標準化した手の融合）に置き、`+1·pop −7` が本体の次の候補の得点を超えたときだけ入れる。
  - 確定構成 B（既定値, 2位以降に最大2件）: 正解ありでの損失がほぼ 0、正解除去で +0.11〜0.14、分岐点 **2.7%**。
  - 攻め構成 A（`pcmin=1`, 1位にも入れる）: 正解除去で +0.20〜0.25、正解ありで −0.014〜−0.021、分岐点 7%。プール外の割合が 12% を超えるなら A が B より上。

### MRR@25（出力 CSV を mrr.py で再採点。B は C++ の出力、A と slot は同じ dump から書いた CSV / C++ 出力）
| 構成 | cqry | cqry 正解除去 | tqry | tqry 正解除去 | sqry | oqry |
|---|---|---|---|---|---|---|
| 差し込みなし（v24 = phys_gen と同一） | 0.550 | 0.127 | 0.624 | 0.247 | 0.836 | 0.659 |
| 公開式 slots 2,4,6,8,10 k5, gate 0.9, pcpop 10 | 0.518 | 0.274 | 0.592 | 0.347 | 0.831* | 0.652 |
| 公開式 slots 2,4 (k2), gate 0.9, pcpop 2 | 0.521 | 0.284 | 0.595 | 0.380 | 0.832 | 0.653 |
| **B: score, k2, pop 1, bias −7, 2位以降, gate 0.9（既定）** | **0.544** | **0.237** | **0.623** | **0.386** | **0.836** | **0.660** |
| A: B + `pcmin=1` | 0.529 | 0.330 | 0.610 | 0.497 | 0.833 | 0.655 |
（* sqry の k5 行は pcpp=1 の実行）。グリッドは `pcslot_grid.py`（slot 864 通り + score 1728 通り）と `pcslot_grid2.py`（細かい score グリッド + cqry で選んで tqry で評価、逆も）。cqry で選んだ構成は tqry でもほぼ同じ利得（f=0.2 で +0.046）→ 2 パラメータ（pop, bias）の選び方は holdout 間で移る。

### 分岐点（「正解がプール外の分子の割合」f がこれを超えれば得、天然物 cqry/tqry の平均で）
| 構成 | 正解ありの損失 | 正解除去の利得 | 分岐点 f* |
|---|---|---|---|
| 公開式 slots 2,4,6,8,10 k5 | −0.032 | +0.124 | 21% |
| 公開式 slots 2,4 k2 | −0.029 | +0.145 | 17% |
| **B（既定）** | −0.0035 | +0.125 | **2.7%** |
| A（pcmin=1） | −0.0175 | +0.227 | 7.2% |
ライブラリ系（sqry/oqry）は B で ±0.001、A で −0.003〜−0.004。

### 正解が PubChem の ±10ppm 窓にある割合（rm=1・pcrecall=1、非立体カノニカルSMILES一致、組成式で前絞り）
| | 正解が窓にある | 窓の大きさ（/クエリ, 中央値） |
|---|---|---|
| tqry | 173/176 (0.983) | 12756 (11070) |
| cqry | 338/400 (0.845) | 7450 (5533) |
| sqry | 335/400 (0.838) | 7806 (6791) |
cqry/sqry の残り 15% は未分解（SMILES>250・混合物として除外、語彙外アダクト/精度2桁プリカーサーでの窓外、互変異性体・電荷形の表記違いなど）。
本番経路で拾える割合（前段 pcpp=20 の上位300 → 重複除去後の上位50 に入る）: tqry 正解除去 67/176（生成候補が正解の 95 件は PubChem 側から除くので、残り 81 件中 67）、cqry 正解除去 183/400。
InChIKey14 での照合（RDKit）: 選ばれた PubChem 候補で C++ の真偽判定と食い違うのは「InChIKey14 では正解なのに非立体SMILESでは別」の 2/352（tqry）、6/792（cqry）だけ（互変異性体・電荷形）→ 表の値はわずかに悲観的。

### 窓内での正解の順位（同点は悲観的; tqry 正解除去 176 / cqry 正解除去 400）
| 並べ方 | tqry top1 / top25 / 窓内MRR | cqry top1 / top25 / 窓内MRR |
|---|---|---|
| REV2 logit和のみ | 4 / 26 / 0.047 | 40 / 114 / 0.133 |
| rev + 1·pop | 10 / 41 / 0.086 | 53 / 143 / 0.177 |
| rev + 8·pop | 57 / 108 / 0.403 | 103 / 219 / 0.327 |
| rev + 20·pop（前段の既定） | 81 / 145 / 0.575 | 119 / 264 / 0.392 |
| pop のみ | 78 / 136 / 0.531 | 78 / 228 / 0.282 |
重複除去後の上位50 件の中では、本体の手の融合（物理・MLP・REV2・アナログ 24·Tanimoto·mcos³）+ 1〜2·pop が最良（tqry 正解除去 0.353、cqry 0.395、pop だけ 0.25/0.33）。

### 漏れの疑い（必ず LB で確認）
holdout の正解は train（= 公開スペクトルライブラリの分子）なので、MassBank/GNPS/MoNA などの登録で PubChem の SID が増えている可能性が高い（正解の pop 平均 2.2〜2.3 vs 窓内の他 1.16〜1.18）。隠しテストの分子が同じ性質か（公開ライブラリに載っている化合物か、Enveda 社内の測定か）は手元では判定できない。公開ノートブックは知名度込みで LB +0.03〜0.04 を報告しているので少なくとも一部は効くはず。切り分け案: PubChem の SID-Map（SID → 登録元）で「スペクトルライブラリ由来の SID を除いた知名度」を作り、holdout での効きが残るかを見る（未着手）。v21 の教訓（inLib）と同じく、**知名度を本体（プール候補）の融合に入れるのは holdout の作りを学ぶ危険が大きい**ので今回はやっていない。

## 実行時間
PubChem チャネルの thread 時間（ビルド機, 8スレッド同時・他ユーザーと共用）: tqry 2.7〜4.6 s/クエリ（窓 1.3万件の SMILES パース + REV2 語彙ビット、上位50 の物理・MLP・アナログ）、cqry 1.4〜2.7 s、sqry 1.3〜2.0 s、oqry 1.6〜2.5 s。表の読み込み（mass 450MB + off 900MB + pop 450MB をメモリ、SMILES 6GB はクエリごとに窓の範囲だけ読む）。
→ Kaggle 4コアで隠しテスト 400 クエリ（timsTOF 天然物 ≈ tqry）: 400 × 3.5 s / 4 ≈ 6 分（Kaggle の CPU が 2倍遅ければ 12 分）＋表の読み込み 1〜2 分。今のパイプライン 45 分 + 約 10〜15 分で予算 9 時間に十分収まる。メモリ +約 2GB（読み込み時の一時バッファで瞬間 +1GB）。pcrecall=1 は評価専用（遅くなる）。

## Kaggle への組み込み手順
1. データセット `yomeiotani/casmi26-pubchem-whole`（7.9GB; kaggle CLI で確認: pc_mass.bin 449,431,976 / pc_off.bin 898,863,960 / pc_pop.bin 449,431,976 / pc_smi.bin 6,060,456,641 バイト = ビルド機 C:\prog\casmi の表と同じ大きさ）をノートブックに追加。**pc_cid.bin は入っていない** → phys_pc はなければ表の位置を ID に使う（SMILES 出力では不要）。
2. `cpp/rdkit-min/phys_pc.cpp` を casmi26-rdkit-min データセット（phys_gen.cpp と同じ場所, gen_ops.h と rev/ を使う）に追加。ノートブックの `srcs_fp = ["phys_gen.cpp"] + srcs[1:]` を `["phys_pc.cpp"] + srcs[1:]` に。w64devkit gcc 14 で `-fsyntax-only` 通過を確認済み（MSVC でビルド・実行済み）。
3. pc_mass.bin のあるディレクトリを 4階層までの glob で探し（rglob は遅い）、オプション文字列（argv[24]）の末尾に `,pc=1,pcdir=<そのディレクトリ>` を追加。既定値 = 構成 B。攻め構成 A なら `,pcmin=1` も。
4. pcout を付けなければ out.csv（pool_candidates.csv）が差し込み済みのリストになる（SMILES は PubChem の表記のまま）。後段の `p_list + d_list` のマージはそのままでよい。
5. ログで確認する行: `pubchem: 112357994 compounds; prefilter 300 (pop 20), full 50, insert 2 by score (pop 1, bias -7, from position 2), gate libSim < 0.9 ...`、`pubchem: ... thread-s ... window .../query`、`pubchem slots: gate open for N / 400 queries`。
6. 注意: FPNet の z 項は PubChem 候補にないので、比較には FPNet を足す前の本体スコアを使う（実装済み）。本番の逆向きモデルは rev2_all_v4k.bin（holdout は rev2_vh.bin で較正）なので bias −7 の最適値は少しずれうる。LB は B → A の順に 1 回ずつ見るのが安全（B と A の差から「プール外の割合」も推定できる: A−B の差 ≈ f·0.10 − (1−f)·0.014）。

## 次にやること
1. LB で B（安全）を 1 回、次に A。両方とも v24（phys_gen）と比べる。
2. 知名度の漏れの切り分け（SID-Map で登録元を見る）。
3. 窓内の PubChem で正解が前段 300 件に入らない分（tqry 9/173, cqry 19/338）と、重複除去後の上位50 に入らない分の改善（前段にアナログ Tanimoto を安く入れる、など）。
4. 正解が窓にない cqry/sqry の 15% の内訳。

## 運用上の注意（ビルド機）
- cmd の `run.bat` 引数はカンマと = で分割される → オプション文字列は `"..."` で渡し、bat 側は `%~4`。
- **TaskStop でローカルの ssh を止めても、リモートの `cmd /c "a && b && ..."` の連鎖は生き残って次の run.bat を実行し続けた**（実行中の phys_pc は手で taskkill、しかし連鎖の残りが後から同じログに書き込み、新しい連鎖の実行と衝突）。止めるときはリモートの cmd.exe 親プロセスを Get-CimInstance で探して taskkill /T。

## 進捗
- [x] phys_pc.cpp 作成・ビルド（ビルド機 MSVC、gcc 14 で構文確認）
- [x] holdout の正解が PubChem 窓にある割合
- [x] 6 条件（cqry/tqry 正解あり・除去、sqry、oqry）の実行
- [x] 差し込み方のグリッド（固定スロット / スコア比較）
- [x] 確定構成の CSV 再採点、実行時間、Kaggle 統合手順
