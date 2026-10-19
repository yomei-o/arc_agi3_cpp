@echo off
rem night_ctr.bat : within-window contrastive training (fp3_STATUS.md). Waits for night_fp3.bat ("all done"
rem in C:\prog\casmi\fp3\progress.txt), then decoys + fine-tunes + evals. 8 CPU threads + GPU.
rem Tools: C:\prog\casmi\fwdmodel\rev3b (rev3_train with ctr=/init=). Results C:\prog\casmi\fp3\ctr_*.log, CTR_SUMMARY.txt
setlocal
set D=C:\prog\casmi
set X=C:\prog\casmi\fwdmodel\rev3b
set O=C:\prog\casmi\fp3
set PY="C:\Program Files\Python312\python.exe"
set T=8
cd /d %D%
set /a W=0
:wait
findstr /c:"all done" %O%\progress.txt >nul 2>&1 && goto go
set /a W+=1
if %W% GTR 300 goto go
ping -n 121 127.0.0.1 >nul
goto wait
:go
echo ctr start %date% %time% > %O%\ctr_progress.txt

rem ---- decoys: 30000 random wlib molecules, ~64 random PubChem compounds of each one's +-10 ppm window
if not exist %D%\dq_head.bin %PY% %X%\fp3_dq.py %D% wlib dq 30000 7 > %O%\ctr_dq.log 2>&1
if not exist %D%\d_wlib_win.bin %X%\win_build.exe %D% dq %D% d_wlib 10 %T% 64 11 > %O%\ctr_decoys.log 2>&1
for %%v in (v4k fp3b8k) do if not exist %D%\d_wlib_%%v.bin %X%\fp3_prep.exe apply %D% d_wlib_wmols.txt %%v.bin d_wlib_%%v.bin %T% > %O%\ctr_apply_%%v.log 2>&1
echo decoys done %time% >> %O%\ctr_progress.txt

rem ---- fine-tune from the BCE models (3 epochs, lr 2e-4) with lam 0.5 / 2, scale 0.2
%X%\rev3_train.exe train dir=%D% lib=wlib fp=v4k init=rev2_w.bin out=rev3_ctr_v4k_l05.bin epochs=3 lr=2e-4 warm=100 hold=20 eval=- ctr=d_wlib ctrq=dq lam=0.5 ctrS=0.2 threads=%T% > %O%\ctr_train_v4k_l05.log 2>&1
%X%\rev3_train.exe train dir=%D% lib=wlib fp=v4k init=rev2_w.bin out=rev3_ctr_v4k_l2.bin epochs=3 lr=2e-4 warm=100 hold=20 eval=- ctr=d_wlib ctrq=dq lam=2 ctrS=0.2 threads=%T% > %O%\ctr_train_v4k_l2.log 2>&1
rem control: same fine-tune without the contrastive term
%X%\rev3_train.exe train dir=%D% lib=wlib fp=v4k init=rev2_w.bin out=rev3_ft_v4k.bin epochs=3 lr=2e-4 warm=100 hold=20 eval=- threads=%T% > %O%\ctr_train_v4k_ft.log 2>&1
%X%\rev3_train.exe train dir=%D% lib=wlib fp=fp3b8k init=rev3_fp3b8k.bin out=rev3_ctr_fp3b8k_l05.bin epochs=3 lr=2e-4 warm=100 hold=20 eval=- ctr=d_wlib ctrq=dq lam=0.5 ctrS=0.2 threads=%T% > %O%\ctr_train_fp3b8k_l05.log 2>&1
echo fine-tunes done %time% >> %O%\ctr_progress.txt
rem from scratch with the contrastive term (12 epochs)
%X%\rev3_train.exe train dir=%D% lib=wlib fp=v4k out=rev3_ctr_v4k_s.bin epochs=12 hold=20 eval=- ctr=d_wlib ctrq=dq lam=0.5 ctrS=0.2 threads=%T% > %O%\ctr_train_v4k_s.log 2>&1
echo scratch done %time% >> %O%\ctr_progress.txt

for %%q in (hcqry hxqry) do (
  %X%\win_eval.exe dir=%D% qry=%%q win=w_%%q models=rev2_w.bin:v4k,rev3_ft_v4k.bin:v4k,rev3_ctr_v4k_l05.bin:v4k,rev3_ctr_v4k_l2.bin:v4k,rev3_ctr_v4k_s.bin:v4k threads=%T% > %O%\ctr_eval_v4k_%%q.log 2>&1
  %X%\win_eval.exe dir=%D% qry=%%q win=w_%%q models=rev3_fp3b8k.bin:fp3b8k,rev3_ctr_fp3b8k_l05.bin:fp3b8k threads=%T% > %O%\ctr_eval_fp3b8k_%%q.log 2>&1
)
(
echo ==== ctr summary %date% %time%
type %O%\ctr_decoys.log | findstr /c:"truth in window"
for %%f in (%O%\ctr_train_*.log) do (
  echo ---- %%f
  type %%f | findstr /c:"contrastive:" /c:"ctr CE" /c:"calibration:" /c:"wrote"
)
for %%f in (%O%\ctr_eval_*.log) do (
  echo ---- %%f
  type %%f
)
) > %O%\CTR_SUMMARY.txt 2>&1
echo ctr all done %date% %time% >> %O%\ctr_progress.txt
