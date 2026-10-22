@echo off
rem kcheck38.bat : stage phys_v38pc.cpp for the Kaggle dataset in C:\prog\casmi\speed\ds_add (ds_rdkit untouched)
rem and g++ -std=c++20 -fsyntax-only with the Kaggle include layout (ds_rdkit + v36 ds_add headers + v38), as kcheck37.bat.
rem   ds_add\rdkit-min\phys_v38pc.cpp -> casmi26-rdkit-min rdkit-min/   (no new headers, no new data file)
setlocal
set DS=C:\prog\casmi\ds_rdkit
set A6=C:\prog\casmi\v36\ds_add
set A=C:\prog\casmi\speed\ds_add
set K=C:\prog\casmi\speed\kcheck
powershell -c "New-Item -ItemType Directory -Force %A%\rdkit-min, %K%\rdkit-min\rev" >nul
copy /y C:\prog\casmi\v38\phys_v38pc.cpp %A%\rdkit-min\ >nul
copy /y %DS%\rdkit-min\rev\*.* %K%\rdkit-min\rev\ >nul
copy /y %A6%\rdkit-min\rev\*.hpp %K%\rdkit-min\rev\ >nul
copy /y %A%\rdkit-min\phys_v38pc.cpp %K%\rdkit-min\ >nul
cd /d %K%\rdkit-min
C:\prog\w64devkit\bin\g++.exe -std=c++20 -fsyntax-only -w -I %K%\rdkit-min -I %DS%\rdkit-min -I %DS%\rdkit-min\boost-min -I %DS% phys_v38pc.cpp > C:\prog\casmi\speed\kcheck.log 2>&1
echo g++ rc %errorlevel% >> C:\prog\casmi\speed\kcheck.log
type C:\prog\casmi\speed\kcheck.log
certutil -hashfile %A%\rdkit-min\phys_v38pc.cpp MD5
dir %A% /s
