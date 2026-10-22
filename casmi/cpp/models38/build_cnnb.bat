@echo off
rem build_cnnb.bat : CPU (AVX2) build of cnn_win_eval with molout= (models38) in C:\prog\casmi\models38\cnnb (own dir, the
rem   GPU tools in fwdmodel\cnn are left alone). Headers copied from fwdmodel\cnn.
setlocal
for /f "usebackq tokens=*" %%i in (`"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.Component.MSBuild -property installationPath`) do set VSPATH=%%i
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d C:\prog\casmi\models38\cnnb
for %%h in (backend.hpp parallel.hpp rev2_model.hpp cnn_layers.hpp cnn_model.hpp) do copy /y C:\prog\casmi\fwdmodel\cnn\%%h . >nul
cl /nologo /O2 /EHsc /std:c++20 /wd4819 /arch:AVX2 cnn_win_eval.cpp /Fe:cnn_win_eval_cpu.exe > cl_we.log 2>&1 || (type cl_we.log & echo CLFAIL)
dir /b *.exe
