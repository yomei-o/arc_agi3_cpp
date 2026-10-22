@echo off
rem ftb.bat (launch name m38_ftb) : second fine-tune seed of the pretrained transformer (pretrain_STATUS.md 8, models38_STATUS.md)
rem   (1) wlib: export pt_w.bin with head seed 32 -> ft8b_ck.bin, 8 epochs (= ft8.bat recipe, seed 32) -> ft8b_ep<N>.bin;
rem       molecule logits of ep3/5/8 on hxqry/hbqry/hbsqry -> pretrain\ft8b_ep<N>_<q>_mol.bin (pool38 tan: columns)
rem   (2) CNN production counterpart (models38\cnnprod.bat, chosen from the pool38 CNN grid) if that file exists
rem   (3) alllib: the same seed-32 fine-tune from pt_all.bin -> ft_allb_ep<N>.bin (production counterpart of seed 2)
rem Waits for pt_prod ("all done", max 3 h). GPU + 2-3 CPU threads.
setlocal
cd /d C:\prog\casmi
set P=C:\prog\casmi\pretrain
set M=C:\prog\casmi\models38
set PR=%M%\ftb_progress.txt
echo queued %date% %time% >> %PR%
set /a N=0
:wait
findstr /c:"all done" %P%\prod_progress.txt > nul 2>&1 && goto go
set /a N+=1
if %N% gtr 36 goto go
timeout /t 300 /nobreak > nul
goto wait
:go
echo start %date% %time% (waited %N% x 5 min) >> %PR%
copy /y %P%\tfm_train.exe %M%\tfm_train_ftb.exe > nul
copy /y %P%\pt_train.exe %M%\pt_train_ftb.exe > nul
if exist C:\prog\casmi\ft8b.bin goto wdone
if not exist C:\prog\casmi\ft8b_ck.bin %M%\pt_train_ftb.exe export model=pt_w.bin lib=wlib fp=fp3c16k HH=2048 seed=32 out=ft8b_ck.bin > %M%\ftb_export.log 2>&1
%M%\tfm_train_ftb.exe train lib=wlib fp=fp3c16k out=ft8b resume=1 epochs=8 B=128 lr=3e-4 warm=2000 eval=none hold=20 ckptEvery=0 threads=2 seed=32 >> %M%\ftb_train.log 2>&1
echo wlib train rc=%errorlevel% %date% %time% >> %PR%
:wdone
for %%e in (3 5 8) do for %%q in (hxqry hbqry hbsqry) do (
  if not exist %P%\ft8b_ep%%e_%%q_mol.bin %M%\pt_train_ftb.exe mollogits model=ft8b_ep%%e.bin qry=%%q out=pretrain\ft8b_ep%%e_%%q_mol.bin > %M%\ftb_mol_%%e_%%q.log 2>&1
)
echo wlib mollogits done %date% %time% >> %PR%
if exist %M%\cnnprod.bat (
  call %M%\cnnprod.bat
  echo cnnprod rc=%errorlevel% %date% %time% >> %PR%
)
if exist C:\prog\casmi\ft_allb.bin goto adone
if not exist C:\prog\casmi\ft_allb_ck.bin %M%\pt_train_ftb.exe export model=pt_all.bin lib=alllib fp=fp3c16k HH=2048 seed=32 out=ft_allb_ck.bin > %M%\ftb_export_all.log 2>&1
%M%\tfm_train_ftb.exe train lib=alllib fp=fp3c16k out=ft_allb resume=1 epochs=8 B=128 lr=3e-4 warm=2000 eval=none hold=20 ckptEvery=0 threads=2 seed=32 >> %M%\ftb_train_all.log 2>&1
echo alllib train rc=%errorlevel% %date% %time% >> %PR%
:adone
echo all done %date% %time% >> %PR%
