@echo off
rem Own FPNet-format training (C:\prog\casmi\fpown\fpown_train.bat, launched detached with night\launch.ps1 fpown_train).
rem 1) holdout-safe evaluation model on wlib (5 epochs, per-epoch models fpown_w_ep*.bin), exported to f16
rem 2) production model on all of train (alllib, hold=0, 6 epochs), exported to f16 and staged in C:\prog\casmi\fpown\ds_add\fpnet
rem Only train.parquet-derived data (wlib/alllib spectra, our RDKit 10226-bit labels) and our own code; no third-party weights.
setlocal
cd /d C:\prog\casmi\fpown
set B=C:\prog\casmi\fpown\build
set COMMON=dir=C:\prog\casmi fp=fp10226 fpnet=1 eval=none B=128 lr=3e-4 wd=0.05 drop=0.1 clip=1 warm=2000 threads=6 logEvery=1000 ckptEvery=2000 seed=31
echo [%date% %time%] start wlib > progress.txt
if exist C:\prog\casmi\fpown_w.bin goto wdone
if exist C:\prog\casmi\fpown_w_ck.bin (
  %B%\tfm_train.exe train lib=wlib out=fpown_w epochs=5 hold=20 resume=1 %COMMON%
) else (
  %B%\tfm_train.exe train lib=wlib out=fpown_w epochs=5 hold=20 %COMMON%
)
:wdone
echo [%date% %time%] wlib done >> progress.txt
%B%\tfm2fpnet.exe C:\prog\casmi\fpown_w.bin C:\prog\casmi\fpown\fpown_w_f32.bin C:\prog\casmi\fpown\fpown_w_f16.bin
echo [%date% %time%] wlib exported >> progress.txt
if exist C:\prog\casmi\fpown_all.bin goto adone
if exist C:\prog\casmi\fpown_all_ck.bin (
  %B%\tfm_train.exe train lib=alllib out=fpown_all epochs=6 hold=0 resume=1 %COMMON%
) else (
  %B%\tfm_train.exe train lib=alllib out=fpown_all epochs=6 hold=0 %COMMON%
)
:adone
echo [%date% %time%] alllib done >> progress.txt
powershell -c "New-Item -ItemType Directory -Force C:\prog\casmi\fpown\ds_add\fpnet | Out-Null"
%B%\tfm2fpnet.exe C:\prog\casmi\fpown_all.bin C:\prog\casmi\fpown\fpown_all_f32.bin C:\prog\casmi\fpown\ds_add\fpnet\fpnet_own_f16.bin
echo [%date% %time%] all done >> progress.txt
