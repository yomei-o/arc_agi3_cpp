# RDKit を C++ コアとして単独ビルドする

Pythonバインディングではなく、RDKitの**C++コア本体**をソースからMSVCでビルドし、
`casmi/cpp/` のツールから直接リンクして使うための手順。RDKitのソース自体はここに
コミットしない（llama.cppのソースを一度コミットして `.git` が2.3GBに膨れた教訓）。
その代わり、同じ手順をいつでも再現できるようスクリプトと手順をここに残す。

## 結論だけ先に

RDKitの `CMakeLists.txt` はほぼ全モジュールを `add_subdirectory` で**無条件に**追加する
作りで、機能単位でつまみ食い的に無効化するのは危険（どこかのモジュールが暗黙に
依存していて、あとで気づかずリンクエラーになる）。なので:

- **公式にオプションとして用意されている無効化**（`RDK_BUILD_PYTHON_WRAPPERS`,
  `RDK_BUILD_CPP_TESTS`, `RDK_USE_BOOST_SERIALIZATION`, `RDK_USE_BOOST_IOSTREAMS`,
  `RDK_BUILD_DESCRIPTORS3D`, `RDK_BUILD_COORDGEN_SUPPORT` 等）は使う
- **単に依存が足りないだけのもの**（Freetype, Eigen3, zlib）は無効化せず、素直に入れて
  フルビルドを通す

という方針。

## 前提

- VS 2022 Build Tools（`vswhere` で見つける。cmake・ninja は同梱されている）
- Python 3.12（`find_package(Python3)` のため。ビルド自体には使わない）
- git は不要（Catch2・better-enums を zip でダウンロードして
  `FETCHCONTENT_SOURCE_DIR_*` で差し替えているため）

## 手順

1. **vcpkg を用意する**（CMakeの `REQUIRED CONFIG` モードでBoostを解決するため。
   公式のプレビルドBoostバイナリにはCMake config packageが付属しないので必須）。
   ```powershell
   Invoke-WebRequest -Uri https://github.com/microsoft/vcpkg/archive/refs/heads/master.zip -OutFile C:\prog\vcpkg.zip
   Expand-Archive C:\prog\vcpkg.zip C:\prog\ ; Rename-Item C:\prog\vcpkg-master C:\prog\vcpkg
   C:\prog\vcpkg\bootstrap-vcpkg.bat
   ```

2. **依存ライブラリをインストール**（Boostは丸ごと入れてしまったが、本当に必要なのは
   ほぼヘッダオンリーの一部＋Freetype＋Eigen3だけ。詳細は後述の「実際に使っている
   boostヘッダ」参照）。
   ```powershell
   cd C:\prog\vcpkg
   .\vcpkg.exe install boost:x64-windows freetype:x64-windows eigen3:x64-windows --disable-metrics
   ```

3. **RDKitのソースを取得**（git不要、タグのzipで十分）。
   ```powershell
   Invoke-WebRequest -Uri https://github.com/rdkit/rdkit/archive/refs/tags/Release_2026_03_1.zip -OutFile C:\prog\rdkit-src.zip
   Expand-Archive C:\prog\rdkit-src.zip C:\prog\
   ```

4. **CMakeが内部でgit cloneしようとする2つの依存（Catch2・better-enums）を
   先にzipで取得**し、`FETCHCONTENT_SOURCE_DIR_*` で差し替える。
   ```powershell
   Invoke-WebRequest -Uri https://github.com/catchorg/Catch2/archive/refs/tags/v3.4.0.zip -OutFile C:\prog\catch2.zip
   Expand-Archive C:\prog\catch2.zip C:\prog\
   Invoke-WebRequest -Uri https://github.com/aantron/better-enums/archive/c35576bed0295689540b39873126129adfa0b4c8.zip -OutFile C:\prog\better-enums.zip
   Expand-Archive C:\prog\better-enums.zip C:\prog\
   ```

5. **`configure_rdkit.bat` → `build_rdkit.bat` の順に実行。**
   両方ともこのディレクトリに置いてある。パスはすべて `C:\prog\...` 前提
   （ビルドマシン固有。別の場所に置くならバッチ内のパスを書き換える）。
   `build_rdkit.bat` は `ninja install` まで実行し、
   `C:\prog\rdkit-install\` にヘッダと `.lib`/`.dll` 一式が入る。

## 実際に使っている boost ヘッダ（次に最小構成でやり直すときのメモ）

`Code/` 以下を `#include <boost/` で全文検索した結果、ほとんどがヘッダオンリー
（`algorithm/*`, `any`, `dynamic_bitset`, `flyweight`, `foreach`, `format`,
`functional`, `graph/*`（Boost Graph Library、ヘッダオンリー）, `multi_array`,
`optional`, `property_map`, `property_tree` 等）で、コンパイルが要るのは:

| ヘッダ | 要るboostライブラリ | うちでの扱い |
|---|---|---|
| `archive/*` | serialization | `RDK_USE_BOOST_SERIALIZATION=OFF` で無効化 |
| `iostreams/*` | iostreams | `RDK_USE_BOOST_IOSTREAMS=OFF` で無効化 |
| `program_options/*` | program_options | コマンドラインツール用、ライブラリ本体には不要 |
| `python/*` | python | `RDK_BUILD_PYTHON_WRAPPERS=OFF` で不要 |
| `log/*`, `mpi.hpp` | log, mpi | 使っていない機能のはず、未検証 |

理論上は `boost-algorithm`, `boost-dynamic-bitset`, `boost-flyweight`,
`boost-foreach`, `boost-format`, `boost-functional`, `boost-graph`,
`boost-multi-array`, `boost-optional`, `boost-property-tree` あたりの
ヘッダオンリーport individually を vcpkg で入れるだけで足りたはず
（未検証 — 今回は時間節約のため `vcpkg install boost`（166モジュール全部）を
先に実行してしまってから気づいた）。
