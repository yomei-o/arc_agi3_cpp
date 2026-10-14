@echo off
setlocal
for /f "usebackq tokens=*" %%i in (`"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.Component.MSBuild -property installationPath`) do set VSPATH=%%i
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
cmake -S C:\prog\casmi\fwdmodel\dag -B C:\prog\casmi\fwdmodel\rdkit-min\build_dag -G Ninja -DCMAKE_BUILD_TYPE=Release -DBoost_DIR=C:/prog/vcpkg/installed/x64-windows/share/boost >nul
ninja -C C:\prog\casmi\fwdmodel\rdkit-min\build_dag -j 12 phys_dag
