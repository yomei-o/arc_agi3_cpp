@echo off
rem night_cnn.bat : CNN follow-ups (casmi/cnn_STATUS.md). Waits for day_cnn.bat ("all done"), then sequential, 5 CPU threads + GPU.
rem Results: C:\prog\casmi\cnn\n_*.log, night_progress.txt, NIGHT_SUMMARY.txt
setlocal
set D=C:\prog\casmi
set X=C:\prog\casmi\fwdmodel\cnn
set O=C:\prog\casmi\cnn
set T=5
cd /d %D%
echo start %date% %time% > %O%\night_progress.txt
set /a W=0
:wait
findstr /c:"all done" %O%\progress.txt >nul 2>&1 && goto go
set /a W+=1
if %W% GEQ 120 goto go
powershell -c "Start-Sleep -Seconds 300"
goto wait
:go
echo day job done or timeout (%W% x 5 min) %time% >> %O%\night_progress.txt

rem ---- 1. CNN A on the fp3 16k vocabulary (the MLP gained 0.254 -> 0.303 / 0.299 -> 0.270 from fp3c16k)
if not exist %D%\cnn_f16.done (
  %X%\cnn_train.exe train dir=%D% lib=wlib fp=fp3c16k out=cnn_f16.bin epochs=12 hold=20 threads=%T% > %O%\n_train_cnn_f16.log 2>&1 && echo ok > %D%\cnn_f16.done
)
echo cnn_f16 done %time% >> %O%\night_progress.txt
for %%q in (hcqry hxqry) do (
  %X%\cnn_win_eval.exe dir=%D% qry=%%q win=w_%%q models=rev3_fp3c16k_big.bin:fp3c16k,cnn_f16.bin:fp3c16k ens=1 threads=%T% > %O%\n_eval_f16_%%q.log 2>&1
  %X%\cnn_win_eval.exe dir=%D% qry=%%q win=w_%%q models=rev2_w.bin:v4k,rev3_fp3c16k_big.bin:fp3c16k,cnn_a.bin:v4k,cnn_f16.bin:fp3c16k ens=1 threads=%T% > %O%\n_eval_all4_%%q.log 2>&1
  %X%\cnn_win_eval.exe dir=%D% qry=%%q win=w_%%q models=rev2_w.bin:v4k,rev3_fp3c16k_big.bin:fp3c16k ens=1 threads=%T% > %O%\n_eval_mlp2_%%q.log 2>&1
  %X%\cnn_win_eval.exe dir=%D% qry=%%q win=w_%%q models=rev3_fp3c16k_big.bin:fp3c16k,cnn_a.bin:v4k ens=1 threads=%T% > %O%\n_eval_fbig_a_%%q.log 2>&1
)
echo eval f16 done %time% >> %O%\night_progress.txt

rem ---- 2. regularized CNN A (held-out loss rose after epoch 4 in cnn_a): decoupled weight decay
if not exist %D%\cnn_awd.done (
  %X%\cnn_train.exe train dir=%D% lib=wlib fp=v4k out=cnn_awd.bin epochs=12 hold=20 wd=0.05 drop=0.5 threads=%T% > %O%\n_train_cnn_awd.log 2>&1 && echo ok > %D%\cnn_awd.done
)
for %%q in (hcqry hxqry) do %X%\cnn_win_eval.exe dir=%D% qry=%%q win=w_%%q models=cnn_a.bin:v4k,cnn_awd.bin:v4k,rev2_w.bin:v4k threads=%T% > %O%\n_eval_awd_%%q.log 2>&1
echo cnn_awd done %time% >> %O%\night_progress.txt

rem ---- 3. second CNN seed: CNN+CNN ensemble vs CNN+MLP (is the ensemble gain diversity of architecture?)
if not exist %D%\cnn_a33.done (
  %X%\cnn_train.exe train dir=%D% lib=wlib fp=v4k out=cnn_a33.bin epochs=12 hold=20 seed=33 threads=%T% > %O%\n_train_cnn_a33.log 2>&1 && echo ok > %D%\cnn_a33.done
)
for %%q in (hcqry hxqry) do %X%\cnn_win_eval.exe dir=%D% qry=%%q win=w_%%q models=cnn_a.bin:v4k,cnn_a33.bin:v4k ens=1 threads=%T% > %O%\n_eval_a33_%%q.log 2>&1
echo cnn_a33 done %time% >> %O%\night_progress.txt

rem ---- 4. summary
(
echo ==== CNN night summary %date% %time%
for %%f in (%O%\n_train_*.log) do (
  echo ---- %%f
  type %%f | findstr /c:"cnn cfg" /c:"training spectra" /c:"epoch" /c:"calibration:" /c:"wrote"
)
for %%f in (%O%\n_eval_*.log) do (
  echo ---- %%f
  type %%f | findstr /v /c:"SHUFFLED logit_c01" /c:"SHUFFLED cal_wq" /c:"z(logit_c01)" /c:"z(cal_wq)"
)
) > %O%\NIGHT_SUMMARY.txt 2>&1
echo all done %date% %time% >> %O%\night_progress.txt
