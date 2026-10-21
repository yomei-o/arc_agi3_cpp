@echo off
rem build the own FPNet-format trainer (tfm_train, CUDA) and exporter (tfm2fpnet, CPU float) in C:\prog\casmi\fpown\build
rem sources: cpp\rev\{tfm_train.cpp,tfm_model.hpp,backend.hpp,parallel.hpp,rev2_model.hpp,fp2.h} + cpp\fpown\tfm2fpnet.cpp copied next to this file
setlocal
for /f "usebackq tokens=*" %%i in (`"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.Component.MSBuild -property installationPath`) do set VSPATH=%%i
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d %~dp0
nvcc -x cu -std=c++17 --extended-lambda -DUSE_CUDA -DUSE_CUBLAS -O2 -arch=sm_86 -Xcompiler "/wd4819 /EHsc" tfm_train.cpp -o tfm_train.exe -lcublas > nv.log 2>&1 || (type nv.log & echo NVFAIL)
cl /nologo /O2 /EHsc /std:c++20 /wd4819 tfm2fpnet.cpp /Fe:tfm2fpnet.exe > cl.log 2>&1 || (type cl.log & echo CLFAIL)
nvcc -x cu -std=c++17 --extended-lambda -DUSE_CUDA -DUSE_CUBLAS -O2 -arch=sm_86 -Xcompiler "/wd4819" win_eval.cpp -o win_eval.exe -lcublas > nv_we.log 2>&1 || (type nv_we.log & echo NVFAIL win_eval)
dir /b *.exe
