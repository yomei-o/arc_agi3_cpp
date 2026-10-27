@echo off
rem build.bat : own C++ GLACIER inference on the build machine (MSVC, links the rdkit-min libraries of
rem C:\prog\casmi\fwdmodel\rdkit-min\build; GasteigerCharges compiled in from rdkit-min\GraphMol\PartialCharges).
rem Sources in C:\prog\casmi\glacier (= casmi/cpp/glacier), rdkit-min headers C:\prog\casmi\fwdmodel\rdkit-min.
setlocal
for /f "usebackq tokens=*" %%i in (`"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath`) do set VSPATH=%%i
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
set RM=C:\prog\casmi\fwdmodel\rdkit-min
set RB=%RM%\build
set G=C:\prog\casmi\glacier
cd /d %G%
if "%SRC%"=="" set SRC=gl_check.cpp
if "%EXE%"=="" set EXE=gl_check.exe
set INC=/I%RM% /I%RM%\GraphMol /IC:\prog\vcpkg\installed\x64-windows\include /I%G%
set LIBS=%RB%\SmilesParse.lib %RB%\CIPLabeler.lib %RB%\GraphMol.lib %RB%\Geometry.lib %RB%\DataStructs.lib %RB%\RDGeneral.lib
cl /nologo /std:c++20 /O2 /EHsc /MD /DNDEBUG /DYY_NO_UNISTD_H /bigobj /openmp /wd4819 %INC% %* %SRC% %G%\pc\GasteigerCharges.cpp %G%\pc\GasteigerParams.cpp %LIBS% /Fe:%EXE% /link /LIBPATH:C:\prog\vcpkg\installed\x64-windows\lib
echo build rc %errorlevel%
