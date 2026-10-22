@echo off
rem ft8.bat : longer fine-tuning of a fp3c16k fingerprint head from the masked-peak pretrained encoder pt_w.bin (wlib, holdout-safe)
rem   export pt_w.bin -> ft8_ck.bin (encoder + fresh head HH 2048), tfm_train resume=1 for 8 epochs (cosine over 8 epochs),
rem   writes ft8_ep<N>.bin each epoch + held-out BCE/F1 per epoch in ft8_train.log; final ft8.bin.
rem   per-epoch window yardstick is run separately (ft8_eval.bat <N>).
setlocal
cd /d C:\prog\casmi
set P=C:\prog\casmi\pretrain
set PR=%P%\ft8_progress.txt
echo start %date% %time% > %PR%
copy /y %P%\tfm_train.exe %P%\tfm_train_ft8.exe > nul
%P%\pt_train.exe export model=pt_w.bin lib=wlib fp=fp3c16k HH=2048 out=ft8_ck.bin > %P%\ft8_export.log 2>&1
echo export rc=%errorlevel% %time% >> %PR%
%P%\tfm_train_ft8.exe train lib=wlib fp=fp3c16k out=ft8 resume=1 epochs=8 B=128 lr=3e-4 warm=2000 eval=none hold=20 ckptEvery=0 threads=3 > %P%\ft8_train.log 2>&1
echo train rc=%errorlevel% %date% %time% >> %PR%
echo all done >> %PR%
