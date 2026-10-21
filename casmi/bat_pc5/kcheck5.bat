@echo off
rem kcheck5.bat : stage the v33 files for the Kaggle dataset (C:\prog\casmi\pc5\ds_add, ds_rdkit untouched) and
rem g++ -std=c++20 -fsyntax-only with the Kaggle include layout. phys_v33pc.cpp includes rev\fp3_v32.h (from v32,
rem not yet in the dataset either), so both are staged. Check tree pc5\kcheck\rdkit-min = new files + ds_rdkit rev\.
setlocal
set S=C:\prog\casmi\fwdmodel\pc2
set DS=C:\prog\casmi\ds_rdkit
set A=C:\prog\casmi\pc5\ds_add
set K=C:\prog\casmi\pc5\kcheck
powershell -c "New-Item -ItemType Directory -Force %A%\rdkit-min\rev, %K%\rdkit-min\rev" >nul
copy /y %S%\phys_v33pc.cpp %A%\rdkit-min\ >nul
copy /y C:\prog\casmi\fwdmodel\v27\rev\fp3_v32.h %A%\rdkit-min\rev\ >nul
copy /y %DS%\rdkit-min\rev\*.* %K%\rdkit-min\rev\ >nul
copy /y %A%\rdkit-min\rev\fp3_v32.h %K%\rdkit-min\rev\ >nul
copy /y %A%\rdkit-min\phys_v33pc.cpp %K%\rdkit-min\ >nul
cd /d %K%\rdkit-min
C:\prog\w64devkit\bin\g++.exe -std=c++20 -fsyntax-only -w -I %K%\rdkit-min -I %DS%\rdkit-min -I %DS%\rdkit-min\boost-min -I %DS% phys_v33pc.cpp > C:\prog\casmi\pc5\kcheck.log 2>&1
echo g++ rc %errorlevel% >> C:\prog\casmi\pc5\kcheck.log
type C:\prog\casmi\pc5\kcheck.log
