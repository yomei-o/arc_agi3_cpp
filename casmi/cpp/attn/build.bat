@echo off
rem build.bat : attn tools on the build machine (C:\prog\casmi\attn; headers rev2_model.hpp backend.hpp parallel.hpp copied from ..\fwdmodel\rev3)
setlocal
for /f "usebackq tokens=*" %%i in (`"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.Component.MSBuild -property installationPath`) do set VSPATH=%%i
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d %~dp0
for %%s in (attn_eval) do (
  nvcc -x cu -std=c++17 --extended-lambda -DUSE_CUDA -DUSE_CUBLAS -O2 -arch=sm_86 -Xcompiler "/wd4819" %%s.cpp -o %%s.exe -lcublas > nv_%%s.log 2>&1 || (type nv_%%s.log & echo NVFAIL %%s)
)
dir /b *.exe
