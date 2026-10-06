@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set CMAKE=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
set NINJA=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe
"%CMAKE%" -S C:\prog\rdkit-Release_2026_03_1 -B C:\prog\rdkit-build -G Ninja ^
  -DCMAKE_MAKE_PROGRAM="%NINJA%" ^
  -DCMAKE_TOOLCHAIN_FILE=C:/prog/vcpkg/scripts/buildsystems/vcpkg.cmake ^
  -DVCPKG_TARGET_TRIPLET=x64-windows ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_INSTALL_PREFIX=C:/prog/rdkit-install ^
  -DFETCHCONTENT_SOURCE_DIR_CATCH2=C:/prog/Catch2-3.4.0 ^
  -DFETCHCONTENT_SOURCE_DIR_BETTER_ENUMS=C:/prog/better-enums-c35576bed0295689540b39873126129adfa0b4c8 ^
  -DRDK_BUILD_PYTHON_WRAPPERS=OFF ^
  -DRDK_BUILD_CPP_TESTS=OFF ^
  -DRDK_USE_BOOST_SERIALIZATION=OFF ^
  -DRDK_USE_BOOST_IOSTREAMS=OFF ^
  -DRDK_BUILD_DESCRIPTORS3D=OFF ^
  -DRDK_BUILD_COORDGEN_SUPPORT=OFF ^
  -DRDK_BUILD_CHEMDRAW_SUPPORT=OFF
