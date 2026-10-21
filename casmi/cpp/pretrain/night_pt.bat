@echo off
rem night_pt.bat : overnight masked-peak pretraining on wlib (pretrain_STATUS.md), then night_post.bat:
rem   (a) embedding-neighbour attention on the window yardstick (hcqry, hxqry, haqry) -> SUMMARY.txt
rem   (b) fingerprint fine-tuning (fp3c16k) from the pretrained encoder vs from scratch, 1 epoch each, window MRR -> SUMMARY.txt
rem Training runs a copy (pt_train_run.exe) and the evaluation lives in night_post.bat, so pt_train.exe and
rem night_post.bat can still be fixed while the pretraining runs (a running .bat must not be edited).
rem Re-running the same bat resumes the pretraining from pt_w_ck.bin (resume=1) and redoes the evaluations.
rem Launch detached: powershell -ExecutionPolicy Bypass -File C:\prog\casmi\night\launch.ps1 pt_night C:\prog\casmi\pretrain\night_pt.bat
setlocal
cd /d C:\prog\casmi
set P=C:\prog\casmi\pretrain
set PR=%P%\night_progress.txt
set EP=10
set STOP=660
echo start %date% %time% >> %PR%

rem ---------------- 1. pretraining (d 512, L 6; stopAfter %STOP% min caps it so (a) runs before morning)
set RES=0
if exist C:\prog\casmi\pt_w_ck.bin set RES=1
echo pretrain resume=%RES% %time% >> %PR%
copy /y %P%\pt_train.exe %P%\pt_train_run.exe > nul
%P%\pt_train_run.exe train lib=wlib out=pt_w epochs=%EP% B=128 d=512 L=6 H=8 ff=4 HH=1024 maxPeaks=64 lr=2e-4 warm=3000 logEvery=1000 ckptEvery=3000 stopAfter=%STOP% threads=3 resume=%RES% >> %P%\night_train.log 2>&1
echo pretrain done rc=%errorlevel% %time% >> %PR%
call %P%\night_post.bat
