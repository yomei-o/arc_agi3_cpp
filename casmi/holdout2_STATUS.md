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

## 結果
（測定中）
