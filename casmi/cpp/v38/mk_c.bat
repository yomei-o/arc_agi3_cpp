@echo off
rem build the final phys_v38pc.cpp (C:\prog\casmi\v38\phys_v38pc.cpp) into C:\prog\casmi\v38\c (MSVC, flags as build_v38.bat)
setlocal
for /f "usebackq tokens=*" %%i in (`"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.Component.MSBuild -property installationPath`) do set VSPATH=%%i
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
copy /y C:\prog\casmi\v38\phys_v38pc.cpp C:\prog\casmi\v38\c\phys_v38pc.cpp
cd /d C:\prog\casmi\v38\c
set L=C:\prog\casmi\fwdmodel\ana\build\rdkitmin
cl /nologo /DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG /std:c++20 /MD /wd4819 /arch:AVX2 /IC:\prog\casmi\fwdmodel\v27 /IC:\prog\casmi\fwdmodel\rdkit-min /IC:\prog\casmi\fwdmodel /external:IC:\prog\vcpkg\installed\x64-windows\include /external:W0 phys_v38pc.cpp /Fe:phys_v38pc.exe /link %L%\SmilesParse.lib %L%\GraphMol.lib %L%\autograd.lib %L%\CIPLabeler.lib %L%\GraphMol.lib %L%\Geometry.lib %L%\DataStructs.lib %L%\RDGeneral.lib > cl_phys_v38pc.log 2>&1 || (type cl_phys_v38pc.log & echo CLFAIL)
certutil -hashfile phys_v38pc.cpp MD5
dir phys_v38pc.exe
