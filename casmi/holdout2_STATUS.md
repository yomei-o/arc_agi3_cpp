# holdout2 — LB を再現する holdout の作り直し（状態ファイル）

2026-10-19 開始。動機: cqry/tqry（「スペクトル2本以上」の COCONUT 分子）は有名な分子に偏り（PubChem 知名度 5〜7.7 vs 普通の分子 ~2）、
その上で選んだ変更（v21 学習重み、v26 LambdaMART、PubChem）が LB で効かないか悪化した。LB の差分を再現する holdout を探す。

## 分子の母集団（train.parquet、`py/holdout2_stats.py` → `C:\prog\casmi\h2_molstats.tsv`）
- train 275,810 分子 / 2,539,608 スペクトル。**65% (182,941 分子) は ingest_lib = enveda-180**（Enveda の合成・医薬品様ライブラリ、timsTOF、平均 6.5 本、COCONUT 掲載は 51 分子だけ）。
- timsTOF・前駆体 245–461 のスペクトルを持つ分子 179,518 → ほぼ全部 enveda-180。うち COCONUT 202 分子。
- **enveda-np-examples（250 分子）**: Enveda の天然物の例。249 が COCONUT、全部が他のライブラリにも載る（平均 227 本!）超有名分子。tqry はほぼこれ。
- COCONUT 掲載 26,111 分子: スペクトル 1 本 4,225 / 2 本以上 21,886。

### PubChem 知名度 log1p(#SID)+log1p(#PMID)（`py/pop_sample.py stats 300 10`、各 300 分子、旧 holdout 分子は除外）
| 群 | PubChem にある | 知名度 平均 / 中央値 | 知名度だけの窓内 MRR |
|---|---|---|---|
| train 全体（無作為） | 88.3% | 1.87 / 1.39 | 0.043 |
| COCONUT・1本だけ | 93.3% | 3.32 / 2.48 | 0.168 |
| COCONUT・2本以上 | 91.0% | 5.23 / 3.95 | 0.258 |
| enveda-180 | 92.0% | 1.33 / 1.10 | 0.002 |
| enveda-np-examples | 95.1% | 9.30 / 9.33 | 0.489 |
| timsTOF 245–461 | 93.7% | 1.33 / 1.39 | 0.003 |
| COCONUT 以外・enveda 以外 | 80.7% | 2.80 / 2.30 | 0.094 |
| (参考) COCONUT にあって train にない pool730 分子 | 91.9% | 1.99 / 1.79 | 0.055 |

→ 「2本以上」を外しても COCONUT の train 分子はまだ有名（3.3）。知名度を合わせるには明示的な抽出が要る。

## 作った holdout（すべて 400 分子、互いに非重複、cqry/tqry/sqry/oqry/fqry/dqry とも非重複、--floor 0.002、ジッタなし）
make_holdout.py に追加したオプション（既定値は従来どおり＝旧 holdout は再現可能）: `--min-spec N`（既定 2）、`--exclude f1,f2`（その mols ファイルの分子を除外）、`--no-lib`（クエリだけ書く）。

| 名前 | 中身 | モード | コマンド（cwd C:\prog\casmi） | スペクトル |
|---|---|---|---|---|
| haqry | COCONUT・1本以上・装置不問 | denovo | `make_holdout.py C:\prog\casmi --mode denovo --floor 0.002 --keys coconut_keys.txt --any-instr --min-spec 1 --exclude cqry_mols.txt,tqry_mols.txt,sqry_mols.txt,oqry_mols.txt,fqry_mols.txt,dqry_mols.txt --seed 2601 --no-lib --prefix halib,haqry` | 1050 |
| hbqry | timsTOF 245–461・種類不問（= ほぼ enveda-180）・1本以上 | denovo | `... --mode denovo --floor 0.002 --min-spec 1 --exclude <上記>,haqry_mols.txt --seed 2602 --no-lib --prefix hblib,hbqry` | 1156 |
| hcqry | COCONUT・知名度を「COCONUT にあって train にない分子」に合わせて抽出 | denovo | `pop_match.py hc_keys.txt 4000 800 400 haqry_mols.txt,hbqry_mols.txt 10 2603` → `make_holdout.py ... --mode denovo --floor 0.002 --keys hc_keys.txt --any-instr --min-spec 1 --seed 2603 --no-lib --prefix hclib,hcqry` | 972 |
| hasqry | COCONUT・2本以上・装置不問 | sibling（他のスペクトルはライブラリに残る） | `... --mode sibling --floor 0.002 --keys coconut_keys.txt --any-instr --exclude <上記>,haqry,hbqry,hcqry --seed 2604 --no-lib --prefix haslib,hasqry` | 1094 |
| hbsqry | timsTOF 245–461・種類不問・2本以上 | sibling | `... --mode sibling --floor 0.002 --exclude <上記>,hasqry_mols.txt --seed 2605 --no-lib --prefix hbslib,hbsqry` | 1057 |

hcqry の知名度: 選んだ 400 分子 平均 1.99→**2.09**（目標 1.99、元の COCONUT train 4.85）、知名度だけの MRR 0.077（目標 0.055、元 0.237）。

**共有ライブラリ ylib**（`build_ylib.py C:\prog\casmi alllib ylib haqry,hbqry,hcqry hasqry,hbsqry`）: alllib（train 全体）から denovo 3 組の分子のスペクトルを全部（25,089 本）、sibling 2 組はクエリのスペクトルだけ（2,151/2,151 本、完全一致で照合）を抜いたもの。2,512,368 スペクトル / 274,610 分子。5 組すべてこの 1 つのライブラリで評価する。

**holdout で安全なモデル（ylib で学習 = クエリのスペクトルも denovo 分子も見ていない）**:
- 順方向 MLP `fwd_mlp_rank_ylib.bin`（train_np.bat と同じ 2 段: `train_forward_model_nn ... ylib 20000 32 30000 256 0.002` → `train_forward_model_rank ... 3000 20 5 0.1 5000 0.0002`）
- REV2 `rev2_y.bin`（`fp2_prep apply ylib v4k.bin` → `rev2_train train lib=ylib fp=v4k out=rev2_y.bin epochs=12 hold=20`、rev2_vh と同じ設定）
- rev v1 `rev_y3072.bin`（`fp_prep ylib` → `rev_train_ens train ... ylib 15 512 0.0006 rev_y3072.bin 0.4 1.0 cqry 10 3072 31`、rev_x.bat と同じ）
- 固定のまま使うもの: phys2_fqry.txt（fqry で合わせ込み、非重複）、fuse_w21.txt（cqry+sqry で学習、非重複）、gen2_prior_c.txt（cqry）、lmart_prod.txt（cqry/tqry/sqry/oqry、非重複）、PubChem 表
- **FPNet は全版で外す**（公開重みは train 全体で学習 = どの holdout でも漏れる）。したがって手元の v22 = v20、v23 は「v20 → REV2」の差、v25 → v26 は「FPNet の有無を除いた LambdaMART の差」と比べる。

## 版の再現（`bat_h2/h2_v.bat <qry> <ver> <rm> <lib> <mlp> <rev1> <rev2> <tag>`、出力 C:\prog\casmi\h2o\）
- v20 = phys_ana.exe、手の重み、rev v1。v21 = 同 + fuse_w21.txt。
- v23 = phys_pclm.exe（生成なし）+ REV2。v24 = + 生成（gen=1,cap=300,prior,gw=-2:0:-1:4:0:1.5）。v25 = + pc=1（out.csv = 挿入後の本番リスト）。v26 = + lmart_prod.txt。
- 同じ bat で tqry・xlib モデルを流して既知値を再現: v23 0.642、v20 0.612（rev2_STATUS / more.bat と一致）。
- rm=1（正解をプールから除く、生成と PubChem でしか当たらない）を denovo 3 組で v24–v26 に。v20/v21/v23 は rm 時 0。

## 結果（出力CSVを mrr.py で採点、ylib モデル、FPNet なし）

### 版ごとの MRR@25（各 400 分子、rm = 正解をプールから除去）
| holdout | v20 | v21 | v23 | v24 | v25 | v26 |
|---|---|---|---|---|---|---|
| haqry（COCONUT≥1, denovo） | 0.551 | 0.612 | 0.560 | 0.542 | 0.538 | 0.543 |
| hcqry（知名度合わせ, denovo） | 0.531 | 0.590 | 0.537 | 0.517 | 0.512 | 0.516 |
| hbqry（timsTOF 種類不問, denovo） | 0.787 | 0.895 | 0.803 | 0.782 | 0.781 | 0.804 |
| hasqry（COCONUT≥2, sibling） | 0.833 | 0.697 | 0.834 | 0.833 | 0.832 | 0.821 |
| hbsqry（timsTOF, sibling） | 0.873 | 0.741 | 0.875 | 0.859 | 0.859 | 0.883 |
| haqry rm | 0 | 0 | 0 | 0.143 | 0.245 | 0.270 |
| hcqry rm | 0 | 0 | 0 | 0.151 | 0.207 | 0.234 |
| hbqry rm | 0 | 0 | 0 | 0.058 | 0.109 | 0.123 |
| **LB**（v23–v25 は FPNet 込み） | 0.291 | 0.250 | 0.313 | 0.322 | 0.321 | 0.286 |

### LB の差分の再現（`py/h2_deltas.py`、括弧は対応のある SE、n=400）
| 変更 | LB | haqry | hcqry | hbqry | hasqry | hbsqry | haqry rm | hcqry rm | hbqry rm |
|---|---|---|---|---|---|---|---|---|---|
| v20→v21 学習重み | **−0.041** | +0.061 (.010) | +0.059 (.011) | +0.108 (.011) | **−0.136** (.015) | **−0.131** (.015) | 0 | 0 | 0 |
| v20→v23 REV2（LB は v22→v23） | **+0.014** | +0.009 (.008) | +0.005 (.007) | +0.015 (.005) | +0.001 (.004) | +0.002 (.003) | 0 | 0 | 0 |
| v23→v24 生成 | **+0.009** | −0.018 (.005) | −0.020 (.006) | −0.021 (.005) | −0.001 (.001) | −0.016 (.005) | **+0.143** | **+0.151** | **+0.058** |
| v24→v25 PubChem | **−0.001** | −0.004 | −0.005 | −0.000 | −0.000 | −0.000 | +0.101 (.010) | +0.057 (.007) | +0.051 (.007) |
| v25→v26 LambdaMART（LB は FPNet 外しも込み） | **−0.035** | +0.005 (.007) | +0.004 (.008) | +0.023 (.007) | −0.012 (.005) | +0.024 (.008) | +0.026 (.005) | +0.027 (.005) | +0.014 (.004) |

公開 LB（約130分子）でこの差分が持つ標準誤差の目安（同じ分子ごとのばらつきを n=130 に換算）: 学習重み 0.02〜0.026、REV2 0.01〜0.015、生成 0.01（rm 側 0.027）、LambdaMART 0.01〜0.014。

**どの単独の holdout も LB の符号の並びを再現しない**:
- denovo（正解はプールにあるがスペクトルなし）は全部 v21 を +0.06〜+0.11 と見る（LB −0.041）。学習重みの inLib −2.36 が「ライブラリにない候補 = 正解」を当てるため。知名度を合わせても（hcqry）変わらない → v21 の失敗は知名度ではなく**モード（正解がライブラリにあるか）**の問題。
- sibling（正解の別スペクトルがライブラリにある）だけが v21 の符号を再現（−0.13）。
- 生成は「正解がプールにある」側では必ず −0.02、rm 側でだけ +0.06〜+0.15 → LB +0.009 には正解がプール外の分子が要る。
- PubChem はプールにある側で 0、rm 側で +0.05〜+0.10 → LB ≈0 は「正解がプール外」の割合が小さいことを示す。
- **LambdaMART はどの成分でも LB の −0.035 を再現しない**（最悪 hasqry −0.012）。
- 絶対値: 成分はどれも 0.51〜0.88（rm 0.06〜0.27）で、LB 0.29〜0.32 と合う単独 holdout はない。

### 混合（`py/h2_mixfit.py`、LB = Σ π·成分、v23–v25 の LB から FPNet 分 0.008 を引く）
- 差分だけで合わせる（絶対値は自由）: **S=hasqry 0.50 / D=haqry or hcqry 0.42 / R=rm 0.08**（drmse 0.010）。予測差分: v21 −0.042、REV2 +0.004、生成 +0.004〜+0.006、PubChem +0.003、**LambdaMART −0.002（LB −0.035 は再現せず）**。絶対値 0.62〜0.65 で LB の約 2 倍（係数 0.45〜0.49）。
- 絶対値も合わせると R（rm）が 0.56 必要になり、その代わり PubChem +0.027・LambdaMART +0.008 と差分が崩れる（rmse 0.028）。→ 隠しテストは「どの成分で見ても手元より約 2 倍難しい」。成分の割合では絶対値と差分を同時に説明できない。

### LambdaMART の LB 悪化の切り分け（どれも再現せず）
1. 本番モデル（fwd_mlp_rank_v2 / rev2_all_v4k、train 全体で学習 = 漏れあり、手の融合とランカーに同じ漏れ）で v25→v26: hasqry +0.002（0.833→0.835）、hcqry **+0.030**（0.550→0.580）、haqry rm **+0.039**（0.261→0.300）→ 「学習時と本番で特徴を作るモデルが違う」ズレでは悪化しない。
2. 前駆体ジッタ 1.5 ppm（timsTOF の実測誤差 precursor_error_ppm は enveda-180 で中央値 0.98・p90 2.76 ppm、train の precursor_mz は計算値でジッタ 0 だと正解の merr ≈ 0 が漏れになる; merr は lmart_prod で 214 分岐に使用）: hajqry +0.017（0.535→0.552）、hcjqry −0.001（0.511→0.510）、hasjqry −0.013（0.835→0.822）→ ジッタでも悪化は再現せず。
3. **xinstr holdout hxqry**（timsTOF・245–461 のクエリ、正解の他装置スペクトルだけがライブラリに残る = 「隠しテストの分子が train に別装置で載っている」場合。`make_holdout.py C:\prog\casmi --mode xinstr --floor 0.002 --exclude <旧6組+ha,hb,hc,has,hbs> --seed 2606 --no-lib --prefix hxlib,hxqry`、400 分子 1163 スペクトル、知名度 1.30 = 普通の分子。ライブラリ zlib = `build_ylib.py ... alllib zlib haqry,hbqry,hcqry hasqry,hbsqry hxqry`（hx 分子の timsTOF 2630 本を除く）、rev 系は zlib で学び直し rev2_z.bin / rev_z3072.bin、MLP は ylib のもの（17次元・2万分子、漏れは無視できる）:
   v20 0.870 / v21 0.779 / v23 0.874 / v24 0.864 / v25 0.864 / v26 **0.886** → 学習重み −0.091、REV2 +0.004、生成 −0.010、PubChem 0、**LambdaMART +0.022**。sibling と同じ型で、LambdaMART の悪化は出ない。

ジッタ版の他の版（v20/v21/v23/v24）: hajqry 0.548/0.609/0.557/0.539、hcjqry 0.528/0.586/0.533/0.513、rm v25/v26 hajqry 0.242/0.268・hcjqry 0.204/0.235 → ジッタ 1.5 ppm は全体を −0.003 ずらすだけで差分は変わらない（途中で止めた残り: hasjqry v21–v24, rm v24）。

### PubChem 知名度（holdout の正解 400 分子全部、`pop_sample.py keys`）
haqry 4.94（≥1 にしても有名）/ hasqry 5.31 / **hcqry 2.09** / hbqry 1.30 / **hxqry 1.30**。

## 結論
- **LB の差分の並びを単独で再現する holdout はない**。各成分で見た符号:
  denovo（ha/hc/hb）= v21 が大きく＋（LB −）、生成が −（LB ＋）。sibling/xinstr（has/hbs/hx）= v21 は正しく −、生成は −。rm（正解がプール外）= 生成・PubChem・LambdaMART がすべて大きく＋。
- 差分（v21・REV2・生成・PubChem の 4 つ）は混合 **0.5·hasqry + 0.42·hcqry + 0.08·hcqry rm** でほぼ再現（v21 −0.042 / REV2 +0.004 / 生成 +0.006 / PubChem +0.003、LB は −0.041 / +0.014 / +0.009 / −0.001）。ただし LambdaMART は −0.002（LB −0.035、FPNet 外しの分を引いても −0.027）で再現しない。
- 絶対値は混合でも 0.62 前後で LB の約 2 倍。割合を変えて絶対値に合わせると差分が崩れる → 隠しテストは成分の割合ではなく「1 問ごとに手元より難しい」（未知の要因: スペクトルの取り方・分子の種類・候補窓の混み具合など）。
- **LambdaMART の悪化は、知名度・モード（denovo/sibling/xinstr/プール外）・本番モデルとのズレ・前駆体ジッタのどれでも再現しない**（手元は 9 成分中 7 で＋）。LB の SE（この差分で 0.01〜0.015）と FPNet 外しを考えると、偶然の可能性も否定できない。

## 次の物差し（推奨）
- 新しい変更は **hasqry（or hxqry）・hcqry・hcqry rm の 3 成分を別々に** 測り、0.5 / 0.42 / 0.08 の混合を主指標に、成分ごとの符号も見る。学習を伴うもの（融合の重み、ランカー）は「S 成分で悪化しないか」を必須条件にする（v21 の失敗は S で −0.13 として見えていた）。
- 絶対値の比較はしない（LB ≈ 0.47 × 混合）。
- LambdaMART は手元では判定できない。LB で切り分けるなら v25 から FPNet だけ外した版（LambdaMART なし）を 1 回出すと「FPNet 外し」と「ランカー」を分離できる。
