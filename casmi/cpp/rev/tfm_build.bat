@echo off
setlocal
for /f "usebackq tokens=*" %%i in (`"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.Component.MSBuild -property installationPath`) do set VSPATH=%%i
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d C:\prog\casmi\fwdmodel\tfm
if "%1"=="cpu" goto cpu
nvcc -x cu -std=c++17 --extended-lambda -DUSE_CUDA -DUSE_CUBLAS -O2 -arch=sm_86 -Xcompiler "/wd4819 /EHsc" tfm_train.cpp -o tfm_train.exe -lcublas > nv.log 2>&1 || (type nv.log & echo NVFAIL)
if "%1"=="gpu" goto end
:cpu
cl /nologo /O2 /EHsc /std:c++20 /wd4819 /DTFM_DOUBLE tfm_train.cpp /Fe:tfm_check_cpu.exe > cl.log 2>&1 || (type cl.log & echo CLFAIL)
:end
dir /b *.exe
