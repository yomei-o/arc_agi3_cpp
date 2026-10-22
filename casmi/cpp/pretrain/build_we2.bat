@echo off
rem build_we2.bat : cnn_win_eval with _mol.bin models and shufm= (C:\prog\casmi\pretrain\we2\cnn_win_eval2.exe)
rem sources next to this file: cnn_win_eval.cpp cnn_model.hpp cnn_layers.hpp backend.hpp parallel.hpp rev2_model.hpp
setlocal
for /f "usebackq tokens=*" %%i in (`"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.Component.MSBuild -property installationPath`) do set VSPATH=%%i
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d %~dp0
nvcc -x cu -std=c++17 --extended-lambda -DUSE_CUDA -DUSE_CUBLAS -O2 -arch=sm_86 -Xcompiler "/wd4819 /EHsc" cnn_win_eval.cpp -o cnn_win_eval2.exe -lcublas > nv_we2.log 2>&1 || (type nv_we2.log & echo NVFAIL)
dir /b *.exe
