@echo off
rem prod_cnn.bat : production cnn_a-recipe model on alllib (all of train), v4k, fixed calibration (cnn_calib.exe).
rem Waits for night_cnn.bat ("all done" in night_progress.txt, max 20 h). Output C:\prog\casmi\cnn_prod_c.bin
setlocal
set D=C:\prog\casmi
set X=C:\prog\casmi\fwdmodel\cnn
set O=C:\prog\casmi\cnn
cd /d %D%
echo start %date% %time% > %O%\prod_progress.txt
set /a W=0
:wait
findstr /c:"all done" %O%\night_progress.txt >nul 2>&1 && goto go
set /a W+=1
if %W% GEQ 240 goto go
powershell -c "Start-Sleep -Seconds 300"
goto wait
:go
echo night done or timeout (%W% x 5 min) %time% >> %O%\prod_progress.txt
if not exist %D%\alllib_v4k.bin %X%\fp3_prep.exe apply %D% alllib_mols.txt v4k.bin alllib_v4k.bin 5 > %O%\prod_apply_v4k.log 2>&1
%X%\cnn_calib.exe train dir=%D% lib=alllib fp=v4k out=cnn_prod_c.bin epochs=12 hold=20 threads=5 > %O%\prod_train.log 2>&1
echo all done %date% %time% >> %O%\prod_progress.txt
