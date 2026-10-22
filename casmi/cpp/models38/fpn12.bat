@echo off
rem fpn12.bat (launch name m38_fpn12) : own FPNet trained longer (models38_STATUS.md)
rem   (1) wlib, 12 epochs (cosine over 12), hold 20, same recipe as cpp/fpown/fpown_train.bat otherwise -> fpown12_w_ep<N>.bin
rem       (holdout-safe, evaluated per epoch by fpn_watch.bat as a pool term)
rem   (2) alllib, 12 epochs, hold 0 -> fpown12_all_ep<N>.bin; every epoch from 5 on exported to FPN2 f16
rem       C:\prog\casmi\models38\fpn_all\fpown12_all_ep<N>_f16.bin (pick the epoch the wlib evaluation picks).
rem Waits for pt_prod ("all done" in pretrain\prod_progress.txt, max 3 h). GPU + 2 CPU threads.
rem Re-running resumes from the _ck.bin of the unfinished run.
setlocal
cd /d C:\prog\casmi
set M=C:\prog\casmi\models38
set PR=%M%\fpn12_progress.txt
set B=C:\prog\casmi\fpown\build
set COMMON=dir=C:\prog\casmi fp=fp10226 fpnet=1 eval=none B=128 lr=3e-4 wd=0.05 drop=0.1 clip=1 warm=2000 threads=2 logEvery=1000 ckptEvery=2000 seed=31
echo queued %date% %time% >> %PR%
set /a N=0
:wait
findstr /c:"all done" C:\prog\casmi\pretrain\prod_progress.txt > nul 2>&1 && goto go
set /a N+=1
if %N% gtr 36 goto go
timeout /t 300 /nobreak > nul
goto wait
:go
echo start %date% %time% (waited %N% x 5 min) >> %PR%
copy /y %B%\tfm_train.exe %M%\tfm_train_f12.exe > nul
if exist C:\prog\casmi\fpown12_w.bin goto wdone
set RES=0
if exist C:\prog\casmi\fpown12_w_ck.bin set RES=1
%M%\tfm_train_f12.exe train lib=wlib out=fpown12_w epochs=12 hold=20 resume=%RES% %COMMON% >> %M%\fpn12_w.log 2>&1
echo wlib rc=%errorlevel% %date% %time% >> %PR%
:wdone
echo wlib done %date% %time% >> %PR%
if exist C:\prog\casmi\fpown12_all.bin goto adone
set RES=0
if exist C:\prog\casmi\fpown12_all_ck.bin set RES=1
%M%\tfm_train_f12.exe train lib=alllib out=fpown12_all epochs=12 hold=0 resume=%RES% %COMMON% >> %M%\fpn12_all.log 2>&1
echo alllib rc=%errorlevel% %date% %time% >> %PR%
:adone
powershell -c "New-Item -ItemType Directory -Force C:\prog\casmi\models38\fpn_all | Out-Null"
for %%e in (5 6 7 8 9 10 11 12) do (
  if exist C:\prog\casmi\fpown12_all_ep%%e.bin if not exist %M%\fpn_all\fpown12_all_ep%%e_f16.bin (
    %B%\tfm2fpnet.exe C:\prog\casmi\fpown12_all_ep%%e.bin %M%\fpn_all\tmp_f32.bin %M%\fpn_all\fpown12_all_ep%%e_f16.bin >> %M%\fpn_all\export.log 2>&1
    del %M%\fpn_all\tmp_f32.bin
  )
)
echo all done %date% %time% >> %PR%
