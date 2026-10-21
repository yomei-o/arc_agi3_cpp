@echo off
rem build.bat : CNN reverse-model tools on the build machine (C:\prog\casmi\fwdmodel\cnn).
rem Headers backend.hpp parallel.hpp rev2_model.hpp are copied unchanged from cpp\rev next to this file.
rem   cnn_train.exe / cnn_win_eval.exe / cnn_gradcheck.exe : CUDA (cuBLAS), float
rem   cnn_gradcheck_cpu.exe : CPU, double (the strict finite-difference check)
rem   cnn_train_cpu.exe     : CPU, float (cputime = Kaggle-like inference timing)
setlocal
for /f "usebackq tokens=*" %%i in (`"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.Component.MSBuild -property installationPath`) do set VSPATH=%%i
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d %~dp0
for %%s in (cnn_train cnn_win_eval cnn_gradcheck) do (
  nvcc -x cu -std=c++17 --extended-lambda -DUSE_CUDA -DUSE_CUBLAS -O2 -arch=sm_86 -Xcompiler "/wd4819 /EHsc" %%s.cpp -o %%s.exe -lcublas > nv_%%s.log 2>&1 || (type nv_%%s.log & echo NVFAIL %%s)
)
cl /nologo /O2 /EHsc /std:c++20 /wd4819 /DCNN_REAL=double cnn_gradcheck.cpp /Fe:cnn_gradcheck_cpu.exe > cl_gc.log 2>&1 || (type cl_gc.log & echo CLFAIL gradcheck)
cl /nologo /O2 /EHsc /std:c++20 /wd4819 cnn_gradcheck.cpp /Fe:cnn_gradcheck_cpuf.exe > cl_gcf.log 2>&1 || (type cl_gcf.log & echo CLFAIL gradcheck_f)
cl /nologo /O2 /EHsc /std:c++20 /wd4819 /arch:AVX2 cnn_train.cpp /Fe:cnn_train_cpu.exe > cl_tr.log 2>&1 || (type cl_tr.log & echo CLFAIL train_cpu)
dir /b *.exe
