@echo off
rem ft8s.bat : control for ft8.bat - the same fp3c16k transformer (d 512 L 6 H 8 HH 2048, maxPeaks 64) from SCRATCH,
rem   same 8 epochs / hyper-parameters on wlib. Waits for ft8.bat's "all done" (max 4 h) so the two do not share the GPU.
setlocal
cd /d C:\prog\casmi
set P=C:\prog\casmi\pretrain
set PR=%P%\ft8s_progress.txt
echo queued %date% %time% > %PR%
set /a N=0
:wait
findstr /c:"all done" %P%\ft8_progress.txt > nul 2>&1 && goto go
set /a N+=1
if %N% gtr 48 goto go
timeout /t 300 /nobreak > nul
goto wait
:go
echo start %date% %time% (waited %N% x 5 min) >> %PR%
copy /y %P%\tfm_train.exe %P%\tfm_train_ft8s.exe > nul
%P%\tfm_train_ft8s.exe train lib=wlib fp=fp3c16k out=ft8s d=512 L=6 H=8 ff=4 HH=2048 maxPeaks=64 epochs=8 B=128 lr=3e-4 warm=2000 eval=none hold=20 ckptEvery=0 threads=3 > %P%\ft8s_train.log 2>&1
echo train rc=%errorlevel% %date% %time% >> %PR%
echo all done >> %PR%
