@echo off
rem build.bat : rev3 tools on the build machine (C:\prog\casmi\fwdmodel\rev3; headers fp2.h rev2_model.hpp backend.hpp parallel.hpp copied from ..\rev2)
setlocal
for /f "usebackq tokens=*" %%i in (`"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.Component.MSBuild -property installationPath`) do set VSPATH=%%i
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d C:\prog\casmi\fwdmodel\rev3
set L=C:\prog\casmi\fwdmodel\ana\build\rdkitmin
set LIBS=%L%\SmilesParse.lib %L%\CIPLabeler.lib %L%\GraphMol.lib %L%\Geometry.lib %L%\DataStructs.lib %L%\RDGeneral.lib
for %%s in (win_build fp3_prep) do (
  cl /nologo /O2 /EHsc /std:c++20 /MD /wd4819 /DYY_NO_UNISTD_H /IC:\prog\casmi\fwdmodel\rdkit-min /IC:\prog\vcpkg\installed\x64-windows\include %%s.cpp /Fe:%%s.exe /link %LIBS% > cl_%%s.log 2>&1 || (type cl_%%s.log & echo CLFAIL %%s)
)
for %%s in (win_eval rev3_train) do (
  nvcc -x cu -std=c++17 --extended-lambda -DUSE_CUDA -DUSE_CUBLAS -O2 -arch=sm_86 -Xcompiler "/wd4819" %%s.cpp -o %%s.exe -lcublas > nv_%%s.log 2>&1 || (type nv_%%s.log & echo NVFAIL %%s)
)
dir /b *.exe
