# models38 — stronger own models where it matters: POOL ranking on the timsTOF holdouts (状態ファイル)

2026-10-22 午後。目的: 自前モデル（own FPNet、CNN、FT = 事前学習 Transformer の微調整）を、LB を決める **プールの順位付け項** として強くする。
物差しは pool36/pool37ft と同じ: phys_v36gs の every-candidate ダンプ（v35 + attpool 2、rm=0、生成候補の SMILES 付き = gsdump）を
オフラインで再採点し、final リスト MRR@25 を hxqry / hbqry / hbsqry（timsTOF）で、対応ありブートストラップ 2000 の 95% CI、
対照 = SHUF-query（クエリ q を q+7 の予測で採点）と PERM（クエリ内で列を並べ替え）。窓 MRR は二次的。
何もアップロード・push・提出していない。

## 道具
- `py/pool38.py` = pool37ft.py の一般化。列: `cf:`（別の phys_v36gs ダンプの cnn 列 = 別の CNN ファイル）、
  `tan:<mol.bin>#<bits>`（連続 Tanimoto、FT/CNN 形）、`dot:<mol.bin>#<bits>`（ロジット内積 = fpnet_pool 形）、`avg:`。
  config 文字列 `cnn:2,ft:2`、`@perm=<col>`、`<col>~` = SHUF-query。基準 2 つ（v35 と ref=）との差。
  **検証**: pool37ft の ep5 の行を完全に再現（hxqry cnn 2 + ft 2 final 0.8904 / +0.0065 [+0.0004,+0.0134]、hbqry 0.8100、SHUF/PERM 行も一致）。
- ビルド機 `C:\prog\casmi\models38\`: `D\`（基準ダンプ p36{x,b,bs}_0 + mols、stage.bat）、`py\`、`out\`（新しいダンプ）、`fpn\`（FPNet の分子ロジット）。
  ソース `cpp/models38/*.bat`。
- `cpp/cnn/cnn_win_eval.cpp` に `molout=<prefix>`（CNN の分子ごとの平均 p → logit を `_mol.bin` に書く、窓不要）を追加。
  CPU 版 `C:\prog\casmi\models38\cnnb\cnn_win_eval_cpu.exe`（build_cnnb.bat、/arch:AVX2）。

## 3. FT（微調整 Transformer）: 長く（ep8 vs ep5）
表 `casmi/models38/ft_grid.txt`（列 f<N> = ft8_ep<N>、wlib、事前学習 pt_w.bin から）。final、v35 との差 / **cnn 2 + ft(ep5) 2 との差 [95% CI]**:
| config | hxqry | hbqry | hbsqry |
|---|---|---|---|
| cnn 2（pool36） | +0.0109 / −0.0065 | +0.0212 / −0.0077 | +0.0126 / −0.0069 |
| **cnn 2 + ep5 2（現行の選択）** | +0.0174 | +0.0289 | +0.0195 |
| cnn 2 + ep3 2 | +0.0139 / −0.0034 | +0.0256 / −0.0033 | +0.0193 / −0.0001 |
| cnn 2 + ep7 2 | +0.0153 / −0.0021 | +0.0264 / −0.0025 | +0.0219 / +0.0024 |
| cnn 2 + ep8 2 | +0.0153 / −0.0020 [−0.0061,+0.0013] | +0.0259 / −0.0030 [−0.0085,+0.0016] | +0.0219 / +0.0024 [−0.0001,+0.0062] |
| cnn 2 + ep8 3 | +0.0173 / −0.0001 | +0.0328 / +0.0038 [−0.0031,+0.0109] | +0.0225 / +0.0030 |
| cnn 1 + ep8 2 | +0.0176 / +0.0002 | +0.0290 / +0.0001 | +0.0237 / +0.0042 [−0.0009,+0.0094] |
| cnn 2 + ep5 1 + ep8 1（エポックのアンサンブル） | +0.0165 / −0.0008 | +0.0299 / +0.0010 | +0.0207 / +0.0012 |
| cnn 2 + ep5 1.5 + ep8 1.5 | +0.0188 / +0.0014 | +0.0323 / +0.0034 | +0.0226 / +0.0032 |
| 対照 cnn 2 + ep8~ 2（SHUF） | +0.0085 / −0.0089 | +0.0027 / −0.0262 | +0.0078 / −0.0116 |
| 対照 cnn 2 + ep8 2 PERM | +0.0009 / −0.0165 | −0.0180 / −0.0469 | +0.0001 / −0.0194 |
**読み方: ep8 は ep5 と区別できない**（同じ重みで −0.002 / −0.003 / +0.002、CI はすべて 0 をまたぐ）。窓 MRR の ep5→ep8 の伸び（hx +0.02）は
プールでは見えない。ep5 + ep8 の和もほぼ同じ（エポックを跨いだ平均では多様性が足りない）。→ **本番は ep5 のまま**（ft_all_ep5.bin）。
重み 3 や cnn 1 の点が少し上に見えるのは 3 集合で選んだ手置きの揺れの範囲（どれも ±0.004、CI は 0 をまたぐ）。
第 2 シード（別の初期化・データ順）は夜間ジョブ ftb で学習中（下）。

CPU 推論コスト（Kaggle 4 コア）: FT 1 モデル 212 ms/スペクトル（pretrain_STATUS 8.5）→ テスト ~1,200 スペクトルで ~4 分。2 シードなら ~8 分。

## 2. CNN: 2 シードの平均（cnn_a + cnn_a33）と cnn_f16（fp3c16k ビット）vs cnn_a 単独
a33 の列 = phys_v36gs を `cnnmodel=cnn_a33.bin` で同じ 3 実行（`cpp/models38/gs38.bat`、a33{x,b,bs}.bat、ビルド機 `models38\out\a33*_cf.txt`、
cnnpool 0 の MRR は基準と同一 0.8731 / …）。f16 の列 = `cnn_win_eval_cpu molout`（`f16mol.bat`）の分子ロジット + fp3c16k ビット（プール・生成候補とも実 SMILES）。
表 `casmi/models38/cnn_a33_grid.txt`、`cnn_f16_grid.txt`。final、v35 との差 / **cnn_a 2（pool36 の選択）との差 [95% CI]**:
| config | hxqry | hbqry | hbsqry |
|---|---|---|---|
| cnn_a 2 | +0.0109 | +0.0212 | +0.0126 |
| a33 2（第 2 シード単独） | +0.0095 / −0.0014 | +0.0162 / −0.0050 | +0.0123 / −0.0003 |
| **2 シード平均** a 1 + a33 1（= avg 2） | +0.0107 / −0.0002 [−0.0037,+0.0028] | +0.0193 / −0.0019 [−0.0074,+0.0038] | +0.0143 / +0.0017 [−0.0004,+0.0051] |
| 2 シード a 1.5 + a33 1.5 | +0.0134 / +0.0024 | +0.0229 / +0.0016 | +0.0142 / +0.0017 |
| （同じ総重みの単独）cnn_a 3 | +0.0126 / +0.0017 | +0.0223 / +0.0011 | +0.0122 / −0.0003 |
| **f16 2**（fp3c16k の CNN 単独） | +0.0023 / **−0.0086 [−0.0173,−0.0009]** | +0.0090 / **−0.0123 [−0.0236,−0.0011]** | +0.0072 / −0.0053 |
| cnn_a 2 + f16 1 | +0.0109 / −0.0000 | +0.0179 / −0.0034 | +0.0165 / +0.0040 [+0.0000,+0.0090] |
| cnn_a 1.5 + f16 1.5 | +0.0121 / +0.0012 | +0.0178 / −0.0034 | +0.0170 / +0.0044 |
| FT ep5 2 の上で: cnn_a 2 + ft 2 | +0.0174 / +0.0065 | +0.0289 / +0.0077 | +0.0195 / +0.0069 |
| 　avg(a,a33) 2 + ft 2 | +0.0182（cnn_a 2 + ft 2 との差 +0.0009） | +0.0291（+0.0001） | +0.0208（+0.0014） |
| 　cnn_a 2 + f16 1 + ft 2 | +0.0185（+0.0011） | +0.0260（−0.0029） | +0.0231（+0.0036） |
| 対照 a33 2 PERM / f16~ 2（SHUF）/ f16 2 PERM | −0.0056 / −0.0078 / −0.0046 | −0.0432 / −0.0202 / −0.0314 | −0.0159 / −0.0069 / −0.0158 |
**読み方**:
- (a) **2 シード平均は単独の cnn_a と区別できない**（同じ総重み 2 で −0.000 / −0.002 / +0.002、総重み 3 でも単独 cnn_a 3 と同程度）。
  第 2 シード a33 単独は cnn_a より少し弱い（hb −0.005）= シード間の揺れの大きさ。FT を足した後でも +0.001 以下。
  窓（NIGHT_SUMMARY、hxqry Tanimoto MRR_in）では a 0.546、a33 0.505、a + a33 0.564 と少し効いたが、プールの順位では見えない（プール ~100 候補、v35 の final 0.87〜0.90 で飽和に近い）。
- (b) **cnn_f16 は cnn_a より弱い**（プール項として hx −0.009、hb −0.012 で CI が 0 の下）。窓でも f16 単独は hx 0.480 と cnn_a 0.546 より下
  （f16 + cnn_a の z 和は窓で 0.596 と上がった）。cnn_a に足しても hbs だけ +0.004（CI 下限 0.000）、hb −0.003 → ノイズ。
- 対照はすべて負（f16 の SHUF −0.02、PERM −0.03 on hb）→ 列の計算は正しく、クエリ固有の情報は乗っているが cnn_a の情報と重なる。
- **結論: CNN の本番の追加学習（第 2 シード / f16 の alllib 版）は行わない**（GPU は FPNet と FT 第 2 シードに回した）。本番は cnn_prod_c.bin 1 本のまま。

CPU 推論コスト（ビルド機、4 スレッド、他ジョブで負荷あり、hxqry 1,163 スペクトル）: cnn_a 48.8 s = **42 ms/スペクトル**、cnn_f16 59.6 s = **51 ms/スペクトル**
（16k ビットの出力層の分だけ重い）。テスト ~1,200 スペクトルで 1 モデル ~1 分、2 シードで ~2 分。どちらも時間的には余裕があるが、利得がないので足さない。

## 1. own FPNet: 長く学習（wlib 12 エポック、プール項の指標で選ぶ）
列 = fpnet_ggml（notebook と同じソース）の分子ロジット、候補の 10226 ビット（`cpp/fpown/fp10226.py` を p36pool_mols.txt 119,046 分子と
生成候補 g{x,b,bs}_mols.txt に適用 → ビルド機 `models38\{p36pool,gx,gb,gbs}_fp10226.bin`、壊れた SMILES 0）。
本番の形 = ロジット内積（fpnet_pool）の + 1.0 z、ここでは cnn_a 2 の上に足す（v36nofp と同じ組）。注意: 本番の phys は生成候補に
FPNet の値を持たせず「クエリ平均」で埋める（rm=0 ではそれ自体が点になる、pretrain_STATUS 8.3 の注意）→ ここでは生成候補にも実ビットで値を付けた
（モデルの強さの比較として公平）。
### 現行モデル（fpown_w = 5 エポック、本番 fpnet_own_f16.bin と同じレシピ）の基準（表 `casmi/models38/fpn_base_grid.txt`、`fpn_tan_grid.txt`）
final、v35 との差 / cnn 2 との差 [95% CI]:
| config | hxqry | hbqry | hbsqry |
|---|---|---|---|
| cnn 2 + fpn 0.5 | +0.0113 / +0.0004 | +0.0245 / +0.0033 [+0.0000,+0.0076] | +0.0155 / +0.0029 [+0.0000,+0.0067] |
| **cnn 2 + fpn 1（本番の形）** | +0.0112 / **+0.0003 [−0.0005,+0.0011]** | +0.0281 / **+0.0069 [+0.0017,+0.0136]** | +0.0147 / **+0.0021 [−0.0021,+0.0067]** |
| cnn 2 + fpn 1.5 | +0.0112 / +0.0003 | +0.0290 / +0.0078 | +0.0188 / +0.0063 [+0.0004,+0.0129] |
| cnn 2 + fpn 2 | +0.0126 / +0.0017 | +0.0269 / +0.0056 | +0.0188 / +0.0063 |
| fpn 1 だけ（CNN なし） | +0.0032 | +0.0113 | +0.0083 |
| cnn 2 + fpn(Tanimoto 形) 1 / 2 | +0.0008 / +0.0032 [+0.0002,+0.0070] | +0.0075 / +0.0080 | +0.0063 [+0.0004,+0.0129] / **+0.0094 [+0.0010,+0.0187]** |
| 対照 Tanimoto 形 2 の SHUF / PERM（cnn 2 + ft 2 との差） | −0.0052 / −0.0192 | −0.0240 / −0.0486 | −0.0082 / −0.0189 |
| 5 エポック版の途中 ep2 / ep3 / ep4 / ep5、cnn 2 + fpn 1 の cnn 2 との差 | −0.0001 / +0.0000 / +0.0003 / +0.0003 | +0.0039 / +0.0034 / **+0.0083** / +0.0069 | +0.0033 / +0.0029 / +0.0046 / +0.0021 |
| 対照 cnn 2 + fpn~ 1（SHUF） / PERM | +0.0001 / −0.0033 | −0.0083 / −0.0131 | −0.0018 / +0.0010 |
| FT の上で: cnn 2 + ft 2 + fpn 1 / 0.5 | +0.0069 / +0.0077（cnn 2 + ft 2 は +0.0065） | +0.0061 / +0.0056（同 +0.0077） | +0.0111 / +0.0094（同 +0.0069） |
FT の上（v37 の組 cnn 2 + ft 2 との差、表 `fpn_tan2_grid.txt`）: + fpn 内積 1 = +0.0005 / −0.0015 / +0.0042（3 集合の和 +0.003）、
**+ fpn Tanimoto 1 = +0.0039 [+0.0001,+0.0088] / −0.0004 / +0.0050 [−0.0001,+0.0113]（和 +0.0085）**、Tanimoto 2 = +0.0026 / +0.0024 / +0.0032（和 +0.0082）。
読み方: CNN の上での FPNet 項は **hbqry で +0.007（CI 0 の上）、hxqry ~0、hbsqry +0.002**。hxqry で効かないのは、ロジット内積が候補のビット数に強く
引っ張られる形（Tanimoto 形にすると w 2 で +0.003 と少し出る）と、CNN が同じ情報を既に持つため。FT（ep5）を入れた後の上乗せは −0.002〜+0.004 で
ノイズ程度。**エポックによる差はプールでは小さい**: 5 エポック版の途中 ep2〜ep5 で hb +0.003〜+0.008、hbs +0.002〜+0.005、hx ~0 と順序が付かない
（窓 MRR は同じ区間で hx 0.38 → 0.54 と大きく伸びたのに）。→ 12 エポックも、プール項としては大きな差は期待しにくい（夜間の表で確認）。
**むしろ項の形**: 同じ fpown_w を Tanimoto 形（p = sigmoid(logit) の連続 Tanimoto、CNN/FT と同じ形）にすると内積形より 3 集合の和で +0.005〜0.01 良い
（FT の上でも +0.0085 vs +0.003）。CI は個別には 0 をまたぐが、対照（SHUF/PERM）はすべて負。本番の fpnet_pool は内積を出しているので、
Tanimoto に変えるのは fpnet_pool の出力（p と候補ビット、pool730_fp10226.bin は既にある）の数行の変更 + 生成候補の扱い（本番は平均で補完）。tune38（重み探索）側の候補。
### 12 エポック（夜間、結果は fpn_grid.txt）
`cpp/models38/fpn12.bat`（m38_fpn12）: wlib 12 エポック（cosine を 12 エポックに、seed 31、他は fpown_train.bat と同じ、threads 2）→ fpown12_w_ep<N>.bin、
続けて alllib 12 エポック（hold 0）→ fpown12_all_ep<N>.bin、ep5〜12 を FPN2 f16 に変換して `models38\fpn_all\fpown12_all_ep<N>_f16.bin`。
`fpn_watch.bat`（m38_fpn_watch）: 次のエポックのファイルが出たら前のエポックを fpn_eval.bat（tfm2fpnet → fpnet_ggml 2 スレッド、3 集合 ~6 分）で評価、
12 エポック後に `fpn_grid.bat` → **`C:\prog\casmi\models38\fpn_grid.txt`**（e1..e12 と f5、o1..o4 = 5 エポック版の途中、t = Tanimoto 形、SHUF/PERM 対照、
基準 = cnn 2 + f5 1）。速度: GPU を FT 第 2 シードと共有して 1 エポック ~39 分 → wlib ~22:50 終了、alllib ~07:00〜08:00 終了の見込み。
**選び方**: hx/hb/hbs の「cnn 2 + e<N> 1 vs cnn 2 + f5 1」の差が 3 集合とも ≥ 0 で、合計が最大のエポック N。良ければ本番は
`fpown12_all_ep<N>_f16.bin`（同じ 12 エポックのスケジュールの N エポック目 = wlib で測ったものと同じ位置）を fpnet_own_f16.bin の代わりに。
差がノイズ内（±0.003）なら今の fpnet_own_f16.bin のまま。Kaggle の CPU コストは同じ構造なので変わらない（FPNet 段 75 分の上限内、v36_STATUS）。

### 3b. FT 第 2 シード（夜間）
`cpp/models38/ftb.bat`（m38_ftb）: pt_w.bin を head seed 32 で export → wlib 8 エポック（ft8 と同じ、seed 32、threads 2）→ `C:\prog\casmi\ft8b_ep<N>.bin`、
ep3/5/8 の分子ロジット（hxqry/hbqry/hbsqry）→ `pretrain\ft8b_ep<N>_<q>_mol.bin`。続けて（CNN の本番追加は無し = cnnprod.bat を置いていない）
alllib で同じ seed 32 の微調整 8 エポック（pt_all.bin から）→ `C:\prog\casmi\ft_allb_ep<N>.bin`（本番の第 2 シード候補）。
速度: GPU 共有で 1 エポック ~28 分 → wlib ~18:45、分子ロジット ~19:15、alllib ~23:30 の見込み。
評価（翌朝、CPU 1 コア数分）: 
```
python casmi/py/pool38.py <D> <out> p36x_0:hxqry:x,p36b_0:hbqry:b,p36bs_0:hbsqry:bs "cnn:2,f5:2;cnn:2,s5:2;cnn:2,f5:1,s5:1;cnn:2,f5:1.5,s5:1.5;cnn:2,f5:2,s5:2;cnn:2,s5~:2;cnn:2,s5:2,@perm=s5;cnn:2,f5:1,s8:1" ^
  col=f5=tan:pretrain\ft8_ep5_{q}_mol.bin#<FT bits> col=s5=tan:pretrain\ft8b_ep5_{q}_mol.bin#<FT bits> col=s8=tan:pretrain\ft8b_ep8_{q}_mol.bin#<FT bits> ref=cnn:2,f5:2
<FT bits> = pretrain\p36pool_fp3c16k.bin::pretrain\p36pool_mols.txt;pretrain\g{s}_fp3c16k.bin::pretrain\g{s}_mols.txt
```
（ビルド機なら `cd /d C:\prog\casmi` で `"C:\Program Files\Python312\python.exe" models38\py\pool38.py models38\D models38\grid_out ...`）。
2 シードの和（f5 1 + s5 1 など）が cnn 2 + f5 2 を 3 集合で上回れば、本番は ft_all_ep5 + ft_allb_ep5 の 2 モデル（Kaggle CPU 2 × ~4 分）。

## 夜間ジョブ（2026-10-22 15:02〜、すべて launch.ps1 で切り離し、自分の PID だけ）
| 名前 | PID | 中身 | 進行 / 出力 |
|---|---|---|---|
| m38_fpn12 | 18040 | FPNet wlib 12 ep → alllib 12 ep（GPU、2 スレッド） | `models38\fpn12_progress.txt`、ログ `models38\fpn12_w.log` / `fpn12_all.log`、`C:\prog\casmi\fpown12_{w,all}_ep<N>.bin`、`models38\fpn_all\*_f16.bin` |
| m38_fpn_watch | 9208 | エポックごとの評価 → fpn_grid.txt（CPU 2 スレッド） | `models38\fpn12_progress.txt`（"eval ep<N> done"）、`models38\fpn\*_mol.bin`、**`models38\fpn_grid.txt`** |
| m38_ftb | 10860 | FT seed 32 wlib 8 ep → 分子ロジット → alllib 8 ep（GPU、2 スレッド） | `models38\ftb_progress.txt`、`ftb_train.log` / `ftb_train_all.log`、`ft8b_ep*.bin`、`ft_allb_ep*.bin` |
| m38_fpn_old | 20944 | 5 エポック版の ep1〜4 の評価（CPU 2、~16:00 に終わる） | `models38\fpn\fpown_w_ep*_mol.bin` |
確認: `type C:\prog\casmi\models38\fpn12_progress.txt C:\prog\casmi\models38\ftb_progress.txt`、
生存 `powershell -c "Get-Process -Id 18040,9208,10860"`、止めるとき `taskkill /pid <pid> /t /f`（これらだけ）。
落ちた場合: 同じ名前で `C:\prog\casmi\models38\L.bat "fpn12"`（または ftb / fpn_watch）を再実行 → _ck.bin から再開・既にある評価は飛ばす。
（注意: ssh 経由の cmd で `timeout` は "入力のリダイレクトはサポートされていません" で即終了するが、launch.ps1（WMI）で起動したジョブでは正常に待つ。）
