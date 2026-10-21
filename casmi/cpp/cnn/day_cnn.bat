@echo off
rem day_cnn.bat : first CNN reverse-model experiments (casmi/cnn_STATUS.md). Sequential, 6 CPU threads + GPU.
rem Results: C:\prog\casmi\cnn\*.log, progress.txt, SUMMARY.txt ; models C:\prog\casmi\cnn_*.bin, rev2_w_s32.bin
setlocal
set D=C:\prog\casmi
set X=C:\prog\casmi\fwdmodel\cnn
set R=C:\prog\casmi\fwdmodel\rev3b
set O=C:\prog\casmi\cnn
set T=6
cd /d %D%
powershell -c "New-Item -ItemType Directory -Force C:\prog\casmi\cnn" >nul
echo start %date% %time% > %O%\progress.txt

rem ---- 1. control for the ensemble question: REV2 MLP, same data/settings as rev2_w, another seed
if not exist %D%\rev2_w_s32.bin %R%\rev3_train.exe train dir=%D% lib=wlib fp=v4k out=rev2_w_s32.bin epochs=12 hold=20 eval=- seed=32 threads=%T% > %O%\train_rev2_w_s32.log 2>&1
echo rev2_w_s32 done %time% >> %O%\progress.txt

rem ---- 2. CNN A: flat + global-pool readout, 12 epochs (= rev2_w epochs)
if not exist %D%\cnn_a.done (
  %X%\cnn_train.exe train dir=%D% lib=wlib fp=v4k out=cnn_a.bin epochs=12 hold=20 threads=%T% > %O%\train_cnn_a.log 2>&1 && echo ok > %D%\cnn_a.done
)
echo cnn_a done %time% >> %O%\progress.txt
for %%q in (hcqry hxqry) do %X%\cnn_win_eval.exe dir=%D% qry=%%q win=w_%%q models=rev2_w.bin:v4k,cnn_a.bin:v4k ens=1 threads=%T% > %O%\eval_a_%%q.log 2>&1
%X%\cnn_train_cpu.exe cputime dir=%D% model=cnn_a.bin qry=hcqry threads=4 max=1000 B=64 > %O%\cputime_a.log 2>&1
echo eval a done %time% >> %O%\progress.txt

rem ---- 3. CNN G: global pool only (pure translation-invariant readout)
if not exist %D%\cnn_g.done (
  %X%\cnn_train.exe train dir=%D% lib=wlib fp=v4k out=cnn_g.bin epochs=12 hold=20 flat=0 threads=%T% > %O%\train_cnn_g.log 2>&1 && echo ok > %D%\cnn_g.done
)
for %%q in (hcqry hxqry) do %X%\cnn_win_eval.exe dir=%D% qry=%%q win=w_%%q models=rev2_w.bin:v4k,cnn_g.bin:v4k ens=1 threads=%T% > %O%\eval_g_%%q.log 2>&1
echo cnn_g done %time% >> %O%\progress.txt

rem ---- 4. control ensemble: REV2 + REV2 (other seed) = what any second model adds
for %%q in (hcqry hxqry) do %X%\cnn_win_eval.exe dir=%D% qry=%%q win=w_%%q models=rev2_w.bin:v4k,rev2_w_s32.bin:v4k ens=1 threads=%T% > %O%\eval_ctl_%%q.log 2>&1
echo control eval done %time% >> %O%\progress.txt

rem ---- 5. CNN NP: no pair-difference channel (pairK=0), flat readout
if not exist %D%\cnn_np.done (
  %X%\cnn_train.exe train dir=%D% lib=wlib fp=v4k out=cnn_np.bin epochs=12 hold=20 pairK=0 threads=%T% > %O%\train_cnn_np.log 2>&1 && echo ok > %D%\cnn_np.done
)
for %%q in (hcqry hxqry) do %X%\cnn_win_eval.exe dir=%D% qry=%%q win=w_%%q models=cnn_np.bin:v4k,cnn_a.bin:v4k threads=%T% > %O%\eval_np_%%q.log 2>&1
echo cnn_np done %time% >> %O%\progress.txt

rem ---- 6. CNN T2: equal wall-clock to rev2_w (2 epochs ~ 9 min)
if not exist %D%\cnn_t2.done (
  %X%\cnn_train.exe train dir=%D% lib=wlib fp=v4k out=cnn_t2.bin epochs=2 hold=20 threads=%T% > %O%\train_cnn_t2.log 2>&1 && echo ok > %D%\cnn_t2.done
)
for %%q in (hcqry hxqry) do %X%\cnn_win_eval.exe dir=%D% qry=%%q win=w_%%q models=cnn_t2.bin:v4k threads=%T% > %O%\eval_t2_%%q.log 2>&1
echo cnn_t2 done %time% >> %O%\progress.txt

rem ---- 7. summary
(
echo ==== CNN day summary %date% %time%
for %%f in (%O%\train_*.log) do (
  echo ---- %%f
  type %%f | findstr /c:"cnn cfg" /c:"training spectra" /c:"epoch" /c:"calibration:" /c:"wrote"
)
type %O%\cputime_a.log | findstr /c:"cputime"
for %%f in (%O%\eval_a_hcqry.log %O%\eval_a_hxqry.log %O%\eval_g_hcqry.log %O%\eval_g_hxqry.log %O%\eval_ctl_hcqry.log %O%\eval_ctl_hxqry.log %O%\eval_np_hcqry.log %O%\eval_np_hxqry.log %O%\eval_t2_hcqry.log %O%\eval_t2_hxqry.log) do (
  echo ---- %%f
  type %%f | findstr /v /c:"SHUFFLED logit_c01" /c:"SHUFFLED cal_wq" /c:"z(logit_c01)" /c:"z(cal_wq)"
)
) > %O%\SUMMARY.txt 2>&1
echo all done %date% %time% >> %O%\progress.txt
