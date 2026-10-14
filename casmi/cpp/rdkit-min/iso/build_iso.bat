@echo off
setlocal
for /f "usebackq tokens=*" %%i in (`"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.Component.MSBuild -property installationPath`) do set VSPATH=%%i
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
if not exist C:\prog\casmi\fwdmodel\iso\build mkdir C:\prog\casmi\fwdmodel\iso\build
cd /d C:\prog\casmi\fwdmodel\iso\build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DBoost_DIR=C:/prog/vcpkg/installed/x64-windows/share/boost -Dboost_headers_DIR=C:/prog/vcpkg/installed/x64-windows/share/boost_headers >nul
ninja -j8 phys_iso
