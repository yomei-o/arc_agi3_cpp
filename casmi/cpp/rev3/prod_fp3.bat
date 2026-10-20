@echo off
rem prod_fp3.bat : production fp3 model (pc4_STATUS.md). Same recipe as rev3_fp3c16k_big
rem (fp3c16k vocabulary, 4096x2, 20 epochs, hold=20 -> Platt calibration) but on alllib (all of train).
rem Out: C:\prog\casmi\rev3_all_fp3c16k_big.bin ; logs C:\prog\casmi\pc4\prod_*.log, progress C:\prog\casmi\pc4\prod_progress.txt
setlocal
set D=C:\prog\casmi
set X=C:\prog\casmi\fwdmodel\rev3b
set O=C:\prog\casmi\pc4
set T=8
cd /d %D%
powershell -c "New-Item -ItemType Directory -Force C:\prog\casmi\pc4" >nul
echo start %date% %time% > %O%\prod_progress.txt
if not exist %D%\alllib_fp3c16k.bin %X%\fp3_prep.exe apply %D% alllib_mols.txt fp3c16k.bin alllib_fp3c16k.bin %T% > %O%\prod_apply.log 2>&1
echo bits done %time% >> %O%\prod_progress.txt
if not exist %D%\rev3_all_fp3c16k_big.bin %X%\rev3_train.exe train dir=%D% lib=alllib fp=fp3c16k out=rev3_all_fp3c16k_big.bin epochs=20 H=4096,4096 hold=20 eval=- threads=%T% > %O%\prod_train.log 2>&1
echo all done %date% %time% >> %O%\prod_progress.txt
