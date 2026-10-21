@echo off
rem build.bat : masked-peak pretraining tools on the build machine (C:\prog\casmi\pretrain).
rem Headers backend.hpp parallel.hpp tfm_model.hpp rev2_model.hpp and tfm_train.cpp are copied unchanged from cpp\rev
rem next to this file (tfm_train.exe here = the (b) fine-tuning / from-scratch runs; own copy, other agents' exes untouched).
rem   pt_train.exe / pt_eval.exe : CUDA (cuBLAS), float
rem   pt_check_cpu.exe           : CPU, double (strict finite-difference gradient check: pt_check_cpu gradcheck)
rem   pt_train_cpu.exe           : CPU, float (Kaggle-like CPU embedding timing)
setlocal
for /f "usebackq tokens=*" %%i in (`"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.Component.MSBuild -property installationPath`) do set VSPATH=%%i
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d %~dp0
if "%1"=="cpu" goto cpu
for %%s in (pt_train pt_eval tfm_train) do (
  nvcc -x cu -std=c++17 --extended-lambda -DUSE_CUDA -DUSE_CUBLAS -O2 -arch=sm_86 -Xcompiler "/wd4819 /EHsc" %%s.cpp -o %%s.exe -lcublas > nv_%%s.log 2>&1 || (type nv_%%s.log & echo NVFAIL %%s)
)
if "%1"=="gpu" goto end
:cpu
cl /nologo /O2 /EHsc /std:c++20 /wd4819 /DTFM_DOUBLE pt_train.cpp /Fe:pt_check_cpu.exe > cl_gc.log 2>&1 || (type cl_gc.log & echo CLFAIL check)
cl /nologo /O2 /EHsc /std:c++20 /wd4819 /arch:AVX2 pt_train.cpp /Fe:pt_train_cpu.exe > cl_tr.log 2>&1 || (type cl_tr.log & echo CLFAIL train_cpu)
:end
dir /b *.exe
