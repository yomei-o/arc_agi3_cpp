@echo off
rem prod_cnn.bat : production CNN (cnn_a recipe, REV2 4096 vocabulary) on alllib = all of train, for the v36 pool term
rem (pool36_STATUS.md: +2 z(CNN Tanimoto) in the pool fusion helps every timsTOF holdout).
rem Waits for night_cnn.bat ("all done" in night_progress.txt; at most 10 h), then trains. 5 CPU threads + GPU.
rem Output: C:\prog\casmi\cnn_prod_c.bin, log C:\prog\casmi\cnn\prod_train.log, marker C:\prog\casmi\cnn\prod_progress.txt
setlocal
set D=C:\prog\casmi
set X=C:\prog\casmi\fwdmodel\cnn
set O=C:\prog\casmi\cnn
cd /d %D%
echo queued %date% %time% > %O%\prod_progress.txt
set /a W=0
:wait
findstr /c:"all done" %O%\night_progress.txt >nul 2>&1 && goto go
set /a W+=1
if %W% GEQ 120 goto go
powershell -c "Start-Sleep -Seconds 300"
goto wait
:go
echo start %date% %time% (waited %W% x 5 min) >> %O%\prod_progress.txt
%X%\cnn_calib.exe train dir=%D% lib=alllib fp=v4k out=cnn_prod_c.bin epochs=12 hold=20 threads=5 > %O%\prod_train.log 2>&1
echo all done rc=%errorlevel% %date% %time% >> %O%\prod_progress.txt
