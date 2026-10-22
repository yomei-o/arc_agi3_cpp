@echo off
rem prod_pt.bat : PRODUCTION path of the pretrained-transformer fingerprint model (pretrain_STATUS.md section 8)
rem   (a) masked-peak pretraining on alllib (all of train), the same settings as the wlib run (night_pt.bat: d 512 L 6 H 8
rem       HH 1024, 10 epochs, B 128, lr 2e-4 cosine, warm 3000, maxPeaks 64) -> C:\prog\casmi\pt_all.bin (+ pt_all_ep*.bin)
rem   (b) export the encoder + fresh fp3c16k head (HH 2048) and fine-tune on alllib for 8 epochs (= ft8.bat recipe)
rem       -> C:\prog\casmi\ft_all_ep<N>.bin each epoch (+ held-out BCE in prod_ft.log), ft_all.bin = epoch 8.
rem       Pick the epoch by the wlib run's yardstick / held-out loss (pretrain_STATUS.md section 8).
rem Waits for ft8s.bat's "all done" (max 6 h) so it does not share the GPU with the wlib control.
rem Re-running resumes the pretraining from pt_all_ck.bin; the fine-tuning restarts only if ft_all.bin is missing.
rem Launch: powershell -ExecutionPolicy Bypass -File C:\prog\casmi\night\launch.ps1 pt_prod C:\prog\casmi\pretrain\prod_pt.bat
setlocal
cd /d C:\prog\casmi
set P=C:\prog\casmi\pretrain
set PR=%P%\prod_progress.txt
echo queued %date% %time% >> %PR%
set /a N=0
:wait
findstr /c:"all done" %P%\ft8s_progress.txt > nul 2>&1 && goto go
set /a N+=1
if %N% gtr 72 goto go
timeout /t 300 /nobreak > nul
goto wait
:go
echo start %date% %time% (waited %N% x 5 min) >> %PR%
copy /y %P%\pt_train.exe %P%\pt_train_prod.exe > nul
if exist C:\prog\casmi\pt_all.bin goto ft
set RES=0
if exist C:\prog\casmi\pt_all_ck.bin set RES=1
%P%\pt_train_prod.exe train lib=alllib out=pt_all epochs=10 B=128 d=512 L=6 H=8 ff=4 HH=1024 maxPeaks=64 lr=2e-4 warm=3000 logEvery=1000 ckptEvery=3000 threads=3 resume=%RES% >> %P%\prod_pt.log 2>&1
echo pretrain rc=%errorlevel% %date% %time% >> %PR%
if not exist C:\prog\casmi\pt_all.bin (echo pretrain FAILED: no pt_all.bin >> %PR% & goto end)
:ft
if exist C:\prog\casmi\ft_all.bin goto end
%P%\pt_train_prod.exe export model=pt_all.bin lib=alllib fp=fp3c16k HH=2048 out=ft_all_ck.bin > %P%\prod_export.log 2>&1
echo export rc=%errorlevel% %time% >> %PR%
copy /y %P%\tfm_train.exe %P%\tfm_train_prod.exe > nul
%P%\tfm_train_prod.exe train lib=alllib fp=fp3c16k out=ft_all resume=1 epochs=8 B=128 lr=3e-4 warm=2000 eval=none hold=20 ckptEvery=0 threads=3 > %P%\prod_ft.log 2>&1
echo finetune rc=%errorlevel% %date% %time% >> %PR%
:end
echo all done %date% %time% >> %PR%
