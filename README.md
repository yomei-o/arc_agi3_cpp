# arc_agi3_cpp — ARC-AGI-3 を C++ で解く

[ARC Prize 2026 — ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)
への挑戦。エージェントの「考える部分」を依存ゼロの C++ で書き、Python は環境との
通信だけを担当する。

## なぜ C++ なのか（採点式から来る必然）

このコンペの1レベルあたりの得点は

```
level_score = min( (人間の行動数 / エージェントの行動数)^2 , 1.15 )   ※未クリアなら 0
game_score  = Σ(level_score_i × i) / Σ(i)        ← i は1始まりのレベル番号（後半ほど重い）
final_score = 全ゲームの game_score の平均
```

つまり **実際の1手は高価で、考えることは無料**。2倍の手数をかけたら得点は 1/4、
5倍なら 1/25 に落ちる。だから「大量に探索して、ほとんど動かない」設計に寄せる価値が
あり、その探索を担うのが C++ コア。

## 構成

```
cpp/arc3_core.cpp     エージェントの本体（物体抽出・アバター同定・通行可能性の学習・
                      経路計画・前レベルの勝ち手の再現）。依存ゼロ C++17、CPU のみ
py/agent_template.py  ctypes で C++ を呼ぶ薄い Python 層。ARC の Agent 契約を満たす
py/harness.py         公開25ゲームを本番と同じ採点式で評価する
py/debug_play.py      1ゲームを再生し、C++ コアが何を信じているかを出力する
py/perception_test.py 各アクションがフレームをどう変えるかを観察する
scripts/bundle_agent.py  C++ ソースを agent_template.py に埋め込んで単一ファイル化
```

Kaggle のノートブックは1ファイルしか運べないので、C++ ソースは文字列として
エージェントに同梱し、**ノートブック起動時に g++ でコンパイル**して ctypes で読む。
コンパイルできなかった場合に備えて純 Python の代替も積んである（提出を落とさないため）。

## 環境について分かっていること（実測）

* 観測は **64×64 の int8 グリッド1枚**。色は 0〜15
* 行動は RESET / ACTION1-4（上下左右）/ ACTION5（文脈依存の決定）/ ACTION6（x,y 座標クリック）
  / ACTION7（アンドゥ）。ゲームごとに使える集合が違い、`available_actions` で分かる
* 画面は**レンダリング結果**であって論理盤面ではない。論理1マスは数ピクセル
  （LS20 では 5×5）。加えて HUD（残り手数バー、インベントリ）が毎手勝手に変化するので、
  「フレームが変わらない＝壁」は使えない。予測位置と実際の位置を比べて壁を学習する
* ローカルはゲームが Python クラスとして降ってくるので `copy.deepcopy(env)` で
  完全なシミュレータが作れる。**Kaggle 本番は HTTP gateway 越しのリモートなので複製は不可**
* `metadata.json` に **人間ベースライン行動数**（レベルごと）が入っている

## 使い方

```sh
# 公開25ゲームを本番と同じ採点式で評価
python py/harness.py --agent agent/my_agent.py --jobs 10

# 1ゲームだけ、中身を見ながら
python py/debug_play.py --game ls20 --steps 300

# C++ を編集したら
python scripts/bundle_agent.py
```

必要な環境: Python 3.12（`arc-agi` の要件）、g++、`pip install arc-agi kaggle pandas pyarrow`。

## 現状

| 版 | 内容 | 25ゲームでのスコア | クリアしたレベル数 |
|---|---|---|---|
| random | 付属のランダム方策 | 0.00 | 0 |
| v1 | C++ コア（アバター同定＋通行可能性学習＋フロンティア探索） | 0.00 | 3 |

v1 はレベルをクリアできるようになったが、予算を使い切ってからのクリアなので
得点にならない（LS20 のレベル1は人間22手に対して数千手）。**次の課題は効率**。
