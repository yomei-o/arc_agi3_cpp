@echo off
rem kcheck36.bat : stage the v36nofp files for the Kaggle datasets in C:\prog\casmi\v36\ds_add (ds_rdkit untouched)
rem and g++ -std=c++20 -fsyntax-only with the Kaggle include layout (ds_rdkit + staged files), as v35\kcheck35.bat.
rem   ds_add\rdkit-min\phys_v36pc.cpp, ds_add\rdkit-min\rev\cnn_model.hpp, cnn_layers.hpp -> casmi26-rdkit-min
rem   ds_add\fpnet\fpnet_own_f16.bin (own FPNet, production alllib)                      -> casmi26-rdkit-min fpnet/
rem   ds_add\pool730\cnn_prod_c.bin (production CNN, all of train)                       -> pool730/ (next to pool730_head.bin)
rem   DELETE from casmi26-rdkit-min: fpnet/fpnet_full1_f16.bin (public weights, compliance)
setlocal
set DS=C:\prog\casmi\ds_rdkit
set A=C:\prog\casmi\v36\ds_add
set K=C:\prog\casmi\v36\kcheck
powershell -c "New-Item -ItemType Directory -Force %A%\rdkit-min\rev, %A%\pool730, %A%\fpnet, %K%\rdkit-min\rev" >nul
copy /y C:\prog\casmi\v36\phys_v36pc.cpp %A%\rdkit-min\ >nul
copy /y C:\prog\casmi\fwdmodel\v27\rev\cnn_model.hpp %A%\rdkit-min\rev\ >nul
copy /y C:\prog\casmi\fwdmodel\v27\rev\cnn_layers.hpp %A%\rdkit-min\rev\ >nul
if not exist %A%\pool730\cnn_prod_c.bin copy /y C:\prog\casmi\cnn_prod_c.bin %A%\pool730\ >nul
if not exist %A%\fpnet\fpnet_own_f16.bin copy /y C:\prog\casmi\fpown\ds_add\fpnet\fpnet_own_f16.bin %A%\fpnet\ >nul
echo fpnet/fpnet_full1_f16.bin> %A%\DELETE_FROM_casmi26-rdkit-min.txt
copy /y %DS%\rdkit-min\rev\*.* %K%\rdkit-min\rev\ >nul
copy /y %A%\rdkit-min\rev\*.hpp %K%\rdkit-min\rev\ >nul
copy /y %A%\rdkit-min\phys_v36pc.cpp %K%\rdkit-min\ >nul
cd /d %K%\rdkit-min
C:\prog\w64devkit\bin\g++.exe -std=c++20 -fsyntax-only -w -I %K%\rdkit-min -I %DS%\rdkit-min -I %DS%\rdkit-min\boost-min -I %DS% phys_v36pc.cpp > C:\prog\casmi\v36\kcheck.log 2>&1
echo g++ rc %errorlevel% >> C:\prog\casmi\v36\kcheck.log
type C:\prog\casmi\v36\kcheck.log
certutil -hashfile %A%\pool730\cnn_prod_c.bin MD5
certutil -hashfile C:\prog\casmi\cnn_prod_c.bin MD5
certutil -hashfile %A%\fpnet\fpnet_own_f16.bin MD5
certutil -hashfile C:\prog\casmi\fpown\ds_add\fpnet\fpnet_own_f16.bin MD5
dir %A% /s
