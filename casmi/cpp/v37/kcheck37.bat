@echo off
rem kcheck37.bat : stage the v37nofp files for the Kaggle datasets in C:\prog\casmi\v37\ds_add (ds_rdkit untouched)
rem and g++ -std=c++20 -fsyntax-only with the Kaggle include layout (ds_rdkit + v36 ds_add + v37 ds_add), as kcheck36.bat.
rem   ds_add\rdkit-min\phys_v37pc.cpp                                 -> casmi26-rdkit-min rdkit-min/
rem   ds_add\pool730\ft_all_ep5.bin (production FT, pt_all + ft_all)  -> pool730/ (next to pool730_head.bin, cnn_prod_c.bin)
rem   no new rev/ headers: phys_v37pc uses rev/tfm_model.hpp + rev/fp3_v32.h (already in the dataset) and the v36
rem   rev/cnn_model.hpp + cnn_layers.hpp (v36\ds_add, upload that first if it is not uploaded yet)
setlocal
set DS=C:\prog\casmi\ds_rdkit
set A6=C:\prog\casmi\v36\ds_add
set A=C:\prog\casmi\v37\ds_add
set K=C:\prog\casmi\v37\kcheck
powershell -c "New-Item -ItemType Directory -Force %A%\rdkit-min, %A%\pool730, %K%\rdkit-min\rev" >nul
copy /y C:\prog\casmi\v37\phys_v37pc.cpp %A%\rdkit-min\ >nul
if exist C:\prog\casmi\ft_all_ep5.bin if not exist %A%\pool730\ft_all_ep5.bin copy /y C:\prog\casmi\ft_all_ep5.bin %A%\pool730\ >nul
copy /y %DS%\rdkit-min\rev\*.* %K%\rdkit-min\rev\ >nul
copy /y %A6%\rdkit-min\rev\*.hpp %K%\rdkit-min\rev\ >nul
copy /y %A%\rdkit-min\phys_v37pc.cpp %K%\rdkit-min\ >nul
cd /d %K%\rdkit-min
C:\prog\w64devkit\bin\g++.exe -std=c++20 -fsyntax-only -w -I %K%\rdkit-min -I %DS%\rdkit-min -I %DS%\rdkit-min\boost-min -I %DS% phys_v37pc.cpp > C:\prog\casmi\v37\kcheck.log 2>&1
echo g++ rc %errorlevel% >> C:\prog\casmi\v37\kcheck.log
type C:\prog\casmi\v37\kcheck.log
certutil -hashfile %DS%\rdkit-min\rev\tfm_model.hpp MD5
certutil -hashfile C:\prog\casmi\fwdmodel\v27\rev\tfm_model.hpp MD5
if exist %A%\pool730\ft_all_ep5.bin certutil -hashfile %A%\pool730\ft_all_ep5.bin MD5
if exist C:\prog\casmi\ft_all_ep5.bin certutil -hashfile C:\prog\casmi\ft_all_ep5.bin MD5
dir %A% /s
