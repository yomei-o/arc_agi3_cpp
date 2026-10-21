@echo off
rem build.bat : formula tools on the build machine (C:\prog\casmi\formula), CPU only (MSVC).
rem   win_dump     : headers rev2_model.hpp backend.hpp parallel.hpp copied from ..\attn
rem   formula_eval : rdkit-min (fwdmodel\rdkit-min headers incl. forward_model.h, libs fwdmodel\ana\build\rdkitmin)
setlocal
for /f "usebackq tokens=*" %%i in (`"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.Component.MSBuild -property installationPath`) do set VSPATH=%%i
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d %~dp0
copy /y ..\attn\rev2_model.hpp . >nul
copy /y ..\attn\backend.hpp . >nul
copy /y ..\attn\parallel.hpp . >nul
cl /nologo /O2 /EHsc /std:c++17 /MD /wd4819 /arch:AVX2 win_dump.cpp /Fe:win_dump.exe > cl_win_dump.log 2>&1 || (type cl_win_dump.log & echo CLFAIL win_dump)
set L=C:\prog\casmi\fwdmodel\ana\build\rdkitmin
cl /nologo /DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG /std:c++20 /MD /wd4819 /arch:AVX2 /DYY_NO_UNISTD_H /IC:\prog\casmi\fwdmodel\rdkit-min /IC:\prog\casmi\fwdmodel /external:IC:\prog\vcpkg\installed\x64-windows\include /external:W0 formula_eval.cpp /Fe:formula_eval.exe /link %L%\SmilesParse.lib %L%\GraphMol.lib %L%\CIPLabeler.lib %L%\GraphMol.lib %L%\Geometry.lib %L%\DataStructs.lib %L%\RDGeneral.lib > cl_formula_eval.log 2>&1 || (type cl_formula_eval.log & echo CLFAIL formula_eval)
dir /b *.exe
