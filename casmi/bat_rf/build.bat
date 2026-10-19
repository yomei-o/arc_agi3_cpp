@echo off
rem build.bat <source stem> : MSVC build of <stem>.cpp in C:\prog\casmi\fwdmodel\rf (same flags/libs as bat_pc2\build.bat)
setlocal
for /f "usebackq tokens=*" %%i in (`"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.Component.MSBuild -property installationPath`) do set VSPATH=%%i
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d C:\prog\casmi\fwdmodel\rf
set L=C:\prog\casmi\fwdmodel\ana\build\rdkitmin
cl /nologo /DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG /std:c++20 /MD /wd4819 /arch:AVX2 /IC:\prog\casmi\fwdmodel\v27 /IC:\prog\casmi\fwdmodel\rdkit-min /IC:\prog\casmi\fwdmodel /external:IC:\prog\vcpkg\installed\x64-windows\include /external:W0 %~1.cpp /Fe:%~1.exe /link %L%\SmilesParse.lib %L%\GraphMol.lib %L%\autograd.lib %L%\CIPLabeler.lib %L%\GraphMol.lib %L%\Geometry.lib %L%\DataStructs.lib %L%\RDGeneral.lib > cl_%~1.log 2>&1 || (type cl_%~1.log & echo CLFAIL)
dir %~1.exe
