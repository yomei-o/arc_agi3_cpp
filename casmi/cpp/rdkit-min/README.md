# 最小RDKit（SMILES解析＋分子グラフ＋環判定だけ）

`casmi/third_party/` のRDKit単独フルビルド手順は、実際には使わない機能
（描画、3D配座、反応処理、Pythonバインディング……）のビルドシステムと
丸一日格闘する羽目になった。うちが実際に要るのは「SMILESを読んで、原子・
結合のグラフを作り、環を判定する」だけ。

この下のソースは、RDKit本家（`Release_2026_03_1`タグ）の該当モジュールの
CMakeLists.txtが列挙する**正式なファイルリスト**をそのまま使い、本家の
ビルドシステム自体は一切使わず、自前の`CMakeLists.txt`一本でビルドする。

## 中身

| モジュール | 役割 | .cppファイル数 |
|---|---|---|
| RDGeneral | 基本型・例外・ログ | 7 |
| DataStructs | ビットベクタ等（内部で使用） | 11 |
| Geometry | 座標型（Atom/Conformerが参照） | 6 |
| GraphMol | 分子グラフ本体（Atom/Bond/ROMol/環判定/芳香族性等） | 36 |
| GraphMol/CIPLabeler | R/S立体配置判定（sanitize時に呼ばれる、省略不可） | 15 |
| GraphMol/SmilesParse | SMILES/SMARTSパーサー本体 | 11 |
| Numerics, Query | ヘッダオンリー（Geometry/GraphMolの依存先） | 0 |

実行ファイルで約1.1MB、ビルド1分弱。

## どうやって必要なファイルを決めたか

本家のCMakeLists.txtの`rdkit_library(モジュール名 ファイル1.cpp ...)`を
そのまま読む（推測しない）→ コンパイル → 足りないヘッダ/シンボルを
エラーメッセージで確認 → 該当ファイルだけ追加、の繰り返し。

躓いた点:
- `Numerics/Vector.h`, `Query/QueryObjects.h` — ヘッダオンリーの依存
  モジュール。ディレクトリごとコピーするだけで解決。
- `RDGeneral/hash/*.hpp` — 同上。
- `GraphMol/FileParsers/MolFileStereochem.h`,
  `MolSGroupParsing.h`, `GraphMol/MolEnumerator/LinkNode.h` —
  **FileParsers・MolEnumerator丸ごとは不要**で、実体はこの3ヘッダ
  だけ(全部 inline / 宣言のみ)。これらが属する本物のモジュールは
  重い依存チェーン(ChemReactions, ChemTransforms, SubstructMatch等)
  を持つが、使われているのは型宣言だけなのでヘッダ単体コピーで足りた。
- `GraphMol/CIPLabeler/` — 最初は「立体配置のラベル付けは質量計算に
  関係ないはず」と外したが、`MolOps.cpp`や`Chirality.cpp`など中核
  ファイルから`Atropisomers`関連シンボルがリンクできずエラーに
  なり、**sanitize処理の一部として不可欠**と判明。モジュールごと
  取り込んだ(49ファイル)。
- `lex.yysmiles.cpp`/`lex.yysmarts.cpp`が`unistd.h`(POSIX専用)を
  無条件includeしてMSVCでビルド不可 → 本家のSmilesParse/CMakeLists.txt
  に`if(MSVC) add_definitions("/D YY_NO_UNISTD_H") endif()`という
  対処が既にあったので、同じdefineを自分のCMakeLists.txtに移植。

## ビルド

```
cmake -S . -B build -G Ninja ^
  -DCMAKE_TOOLCHAIN_FILE=C:/prog/vcpkg/scripts/buildsystems/vcpkg.cmake ^
  -DVCPKG_TARGET_TRIPLET=x64-windows ^
  -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Boost（vcpkg経由、ヘッダオンリー利用のみ）とThreadsだけが外部依存。
Freetype・Eigen・Python・Catch2・flex/bisonは一切不要。

## 拡張するとき

次に要る機能（分子式・精密質量の計算など）が出てきたら、同じやり方で:
1. 本家の該当モジュールのCMakeLists.txtから`rdkit_library(...)`のファイル
   リストを読む
2. ディレクトリを作ってそのファイルだけコピー
3. `CMakeLists.txt`に4行足す（`add_library` / `file(GLOB...)` /
   `target_include_directories` / `target_link_libraries`）
4. ビルドしてエラーが出たら、エラーが要求するヘッダ/シンボルの実体を
   同じ手順で足す
