@echo off
rem night_hn_gpu.bat : GPU chain for the hnqry holdout-safe models on nlib: CNN (cnn_calib train, as cnn_a_c) -> own FPNet (tfm_train, as fpown_w).
rem Waits for night_hn_cpu.bat's bits.done (nlib_v4k / nlib_fp10226). Launch detached with night\launch.ps1 hn_gpu.
setlocal
set D=C:\prog\casmi
set O=C:\prog\casmi\hn
set X=C:\prog\casmi\fwdmodel\cnn
set B=C:\prog\casmi\fpown\build
cd /d %D%
echo [%date% %time%] start gpu chain > %O%\gpu_progress.txt
set /a W=0
:wait
if exist %O%\bits.done goto go
set /a W+=1
if %W% gtr 720 (echo bits.done never appeared >> %O%\gpu_progress.txt & exit /b 1)
ping -n 31 127.0.0.1 > nul
goto wait
:go
echo [%date% %time%] bits ready >> %O%\gpu_progress.txt

rem ---- 1. CNN on nlib (as prod_cnn.bat / cnn_a_c: cnn_calib train = train + calibrate, v4k, 12 epochs, hold 20)
if not exist %D%\cnn_n_c.bin %X%\cnn_calib.exe train dir=%D% lib=nlib fp=v4k out=cnn_n_c.bin epochs=12 hold=20 threads=5 > %O%\train_cnn_n_c.log 2>&1
echo [%date% %time%] cnn_n_c done >> %O%\gpu_progress.txt

rem ---- 2. own FPNet on nlib (as fpown_train.bat wlib: 5 epochs, hold 20), exported to the f16 inference file
cd /d C:\prog\casmi\fpown
set COMMON=dir=C:\prog\casmi fp=fp10226 fpnet=1 eval=none B=128 lr=3e-4 wd=0.05 drop=0.1 clip=1 warm=2000 threads=6 logEvery=1000 ckptEvery=2000 seed=31
if exist %D%\fpown_n.bin goto ndone
if exist %D%\fpown_n_ck.bin (
  %B%\tfm_train.exe train lib=nlib out=fpown_n epochs=5 hold=20 resume=1 %COMMON% > %O%\train_fpown_n.log 2>&1
) else (
  %B%\tfm_train.exe train lib=nlib out=fpown_n epochs=5 hold=20 %COMMON% > %O%\train_fpown_n.log 2>&1
)
:ndone
echo [%date% %time%] fpown_n done >> %O%\gpu_progress.txt
%B%\tfm2fpnet.exe %D%\fpown_n.bin %D%\fpown\fpown_n_f32.bin %D%\fpown\fpown_n_f16.bin > %O%\export_fpown_n.log 2>&1
echo [%date% %time%] gpu all done >> %O%\gpu_progress.txt
