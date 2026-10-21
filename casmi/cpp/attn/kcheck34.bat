@echo off
rem kcheck34.bat : stage the v34 files for the Kaggle dataset (C:\prog\casmi\attn\ds_add; ds_rdkit untouched) and
rem g++ -std=c++20 -fsyntax-only with the Kaggle include layout (as casmi/bat_pc4/kcheck.bat for v32: the check tree
rem attn\kcheck\rdkit-min = the new file + a copy of ds_rdkit\rdkit-min\rev + the v32 rev\fp3_v32.h).
setlocal
set DS=C:\prog\casmi\ds_rdkit
set A=C:\prog\casmi\attn\ds_add
set K=C:\prog\casmi\attn\kcheck
powershell -c "New-Item -ItemType Directory -Force %A%\rdkit-min\rev, %K%\rdkit-min\rev" >nul
copy /y C:\prog\casmi\attn\v34\phys_v34pc.cpp %A%\rdkit-min\ >nul
copy /y C:\prog\casmi\fwdmodel\v27\rev\fp3_v32.h %A%\rdkit-min\rev\ >nul
copy /y %DS%\rdkit-min\rev\*.* %K%\rdkit-min\rev\ >nul
copy /y %A%\rdkit-min\rev\fp3_v32.h %K%\rdkit-min\rev\ >nul
copy /y %A%\rdkit-min\phys_v34pc.cpp %K%\rdkit-min\ >nul
cd /d %K%\rdkit-min
C:\prog\w64devkit\bin\g++.exe -std=c++20 -fsyntax-only -w -I %K%\rdkit-min -I %DS%\rdkit-min -I %DS%\rdkit-min\boost-min -I %DS% phys_v34pc.cpp > C:\prog\casmi\attn\kcheck.log 2>&1
echo g++ rc %errorlevel% >> C:\prog\casmi\attn\kcheck.log
type C:\prog\casmi\attn\kcheck.log
