@echo off
rem kcheck35.bat : stage the v35 files for the Kaggle datasets (C:\prog\casmi\v35\ds_add; ds_rdkit untouched) and
rem g++ -std=c++20 -fsyntax-only with the Kaggle include layout (as cpp/attn/kcheck34.bat).
rem   ds_add\rdkit-min\phys_v35pc.cpp, ds_add\rdkit-min\rev\fp3_v32.h   -> dataset casmi26-rdkit-min
rem   ds_add\pool730\rev3_all_fp3c16k_big.bin (production fp3 model)    -> dataset casmi26-pool-expanded, pool730/
setlocal
set DS=C:\prog\casmi\ds_rdkit
set A=C:\prog\casmi\v35\ds_add
set K=C:\prog\casmi\v35\kcheck
powershell -c "New-Item -ItemType Directory -Force %A%\rdkit-min\rev, %A%\pool730, %K%\rdkit-min\rev" >nul
copy /y C:\prog\casmi\v35\phys_v35pc.cpp %A%\rdkit-min\ >nul
copy /y C:\prog\casmi\fwdmodel\v27\rev\fp3_v32.h %A%\rdkit-min\rev\ >nul
if not exist %A%\pool730\rev3_all_fp3c16k_big.bin copy /y C:\prog\casmi\rev3_all_fp3c16k_big.bin %A%\pool730\ >nul
copy /y %DS%\rdkit-min\rev\*.* %K%\rdkit-min\rev\ >nul
copy /y %A%\rdkit-min\rev\fp3_v32.h %K%\rdkit-min\rev\ >nul
copy /y %A%\rdkit-min\phys_v35pc.cpp %K%\rdkit-min\ >nul
cd /d %K%\rdkit-min
C:\prog\w64devkit\bin\g++.exe -std=c++20 -fsyntax-only -w -I %K%\rdkit-min -I %DS%\rdkit-min -I %DS%\rdkit-min\boost-min -I %DS% phys_v35pc.cpp > C:\prog\casmi\v35\kcheck.log 2>&1
echo g++ rc %errorlevel% >> C:\prog\casmi\v35\kcheck.log
type C:\prog\casmi\v35\kcheck.log
dir %A% /s /b
