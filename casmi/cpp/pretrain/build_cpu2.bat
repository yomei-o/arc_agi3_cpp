@echo off
rem build_cpu2.bat : CPU float pt_train (mollogits / embed timing, Kaggle-like) as pt_train_cpu2.exe
setlocal
for /f "usebackq tokens=*" %%i in (`"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.Component.MSBuild -property installationPath`) do set VSPATH=%%i
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d %~dp0
cl /nologo /O2 /EHsc /std:c++20 /wd4819 /arch:AVX2 pt_train.cpp /Fo:pt_train_cpu2.obj /Fe:pt_train_cpu2.exe > cl_cpu2.log 2>&1 || (type cl_cpu2.log & echo CLFAIL)
dir /b pt_train_cpu2.exe
