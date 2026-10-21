@echo off
rem build the DEPLOYED fpnet_ggml / fpnet_pool source (C:\prog\casmi\ds_rdkit\fpnet = the notebook's copy) in C:\prog\casmi\fpown\gg
for /f "usebackq delims=" %%i in (`"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath`) do set VS=%%i
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul
if not exist C:\prog\casmi\fpown\gg xcopy /e /i /q C:\prog\casmi\ds_rdkit\fpnet C:\prog\casmi\fpown\gg >nul
cd /d C:\prog\casmi\fpown\gg
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release > cm.log 2>&1 || (type cm.log & exit /b 1)
cmake --build build -j 6 > bd.log 2>&1 || (type bd.log & exit /b 1)
dir /b build\*.exe
