@echo off
rem night_fp3.bat : fp3 overnight pipeline (fp3_STATUS.md). Self-contained, sequential, 8 CPU threads + GPU.
rem Results: C:\prog\casmi\fp3\*.log and C:\prog\casmi\fp3\SUMMARY.txt
setlocal
set D=C:\prog\casmi
set X=C:\prog\casmi\fwdmodel\rev3
set O=C:\prog\casmi\fp3
set PY="C:\Program Files\Python312\python.exe"
set T=8
cd /d %D%
powershell -c "New-Item -ItemType Directory -Force C:\prog\casmi\fp3" >nul
echo start %date% %time% > %O%\progress.txt

rem ---- 1. window caches (hcqry made interactively; skip if present)
if not exist %D%\w_hxqry_win.bin %X%\win_build.exe %D% hxqry %D% w_hxqry 10 %T% > %O%\win_hxqry.log 2>&1
if not exist %D%\w_haqry_win.bin %X%\win_build.exe %D% haqry %D% w_haqry 10 %T% > %O%\win_haqry.log 2>&1
if not exist %D%\w_hcqry_win.bin %X%\win_build.exe %D% hcqry %D% w_hcqry 10 %T% > %O%\win_hcqry.log 2>&1
echo windows done %time% >> %O%\progress.txt

rem ---- 2. wlib = train minus every spectrum of haqry,hbqry,hcqry,hxqry molecules, minus sibling query spectra
if not exist %D%\wlib_head.bin %PY% %D%\build_ylib.py %D% alllib wlib haqry,hbqry,hcqry,hxqry hasqry,hbsqry > %O%\wlib.log 2>&1
echo wlib done %time% >> %O%\progress.txt

rem ---- 3. REV2 vocabulary bits (v4k) on wlib and on the windows
if not exist %D%\wlib_v4k.bin %X%\fp3_prep.exe apply %D% wlib_mols.txt v4k.bin wlib_v4k.bin %T% > %O%\apply_wlib_v4k.log 2>&1
for %%w in (hcqry hxqry) do if not exist %D%\w_%%w_v4k.bin %X%\fp3_prep.exe apply %D% w_%%w_wmols.txt v4k.bin w_%%w_v4k.bin %T% > %O%\apply_w%%w_v4k.log 2>&1
echo v4k bits done %time% >> %O%\progress.txt

rem ---- 4. baseline yardstick: rev2_y (ylib), rev2_z (zlib; hxqry molecules' other-instrument spectra seen)
%X%\win_eval.exe dir=%D% qry=hcqry win=w_hcqry models=rev2_y.bin:v4k threads=%T% > %O%\eval_base_hcqry.log 2>&1
%X%\win_eval.exe dir=%D% qry=hxqry win=w_hxqry models=rev2_y.bin:v4k,rev2_z.bin:v4k threads=%T% > %O%\eval_base_hxqry.log 2>&1
echo baseline eval done %time% >> %O%\progress.txt

rem ---- 5. REV2 retrained on wlib (clean for both hcqry and hxqry), same settings as rev2_vh/rev2_y
if not exist %D%\rev2_w.bin %X%\rev3_train.exe train dir=%D% lib=wlib fp=v4k out=rev2_w.bin epochs=12 hold=20 eval=- threads=%T% > %O%\train_rev2_w.log 2>&1
echo rev2_w done %time% >> %O%\progress.txt

rem ---- 6. fp3 vocabularies: document frequency on wlib (>=100), in-window entropy on haqry windows (not evaluated)
if not exist %D%\fp3_cnt.bin %X%\fp3_prep.exe count %D% wlib fp3_cnt.bin 100 %T% > %O%\fp3_count.log 2>&1
if not exist %D%\fp3a8k.bin %X%\fp3_prep.exe select %D% fp3_cnt.bin w_haqry fp3a8k.bin 8192 400 3000 %T% - > %O%\fp3_sel_a8k.log 2>&1
if not exist %D%\fp3b8k.bin %X%\fp3_prep.exe select %D% fp3_cnt.bin w_haqry fp3b8k.bin 8192 400 3000 %T% v4k.bin > %O%\fp3_sel_b8k.log 2>&1
if not exist %D%\fp3c16k.bin %X%\fp3_prep.exe select %D% fp3_cnt.bin w_haqry fp3c16k.bin 16384 400 3000 %T% v4k.bin > %O%\fp3_sel_c16k.log 2>&1
for %%v in (fp3a8k fp3b8k fp3c16k) do (
  if not exist %D%\wlib_%%v.bin %X%\fp3_prep.exe apply %D% wlib_mols.txt %%v.bin wlib_%%v.bin %T% > %O%\apply_wlib_%%v.log 2>&1
  for %%w in (hcqry hxqry) do if not exist %D%\w_%%w_%%v.bin %X%\fp3_prep.exe apply %D% w_%%w_wmols.txt %%v.bin w_%%w_%%v.bin %T% > %O%\apply_w%%w_%%v.log 2>&1
)
echo fp3 vocab+bits done %time% >> %O%\progress.txt

rem ---- 7. train on fp3 vocabularies (same architecture/settings)
for %%v in (fp3b8k fp3a8k fp3c16k) do (
  if not exist %D%\rev3_%%v.bin %X%\rev3_train.exe train dir=%D% lib=wlib fp=%%v out=rev3_%%v.bin epochs=12 hold=20 eval=- threads=%T% > %O%\train_rev3_%%v.log 2>&1
  echo trained %%v %time% >> %O%\progress.txt
)

rem ---- 8. window MRR of everything
for %%q in (hcqry hxqry) do (
  %X%\win_eval.exe dir=%D% qry=%%q win=w_%%q models=rev2_w.bin:v4k,rev3_fp3b8k.bin:fp3b8k,rev3_fp3a8k.bin:fp3a8k,rev3_fp3c16k.bin:fp3c16k ens=1 threads=%T% > %O%\eval_all_%%q.log 2>&1
)
echo eval done %time% >> %O%\progress.txt

rem ---- 8b. longer/wider on the largest vocabulary (20 epochs, 4096x2)
if not exist %D%\rev3_fp3c16k_big.bin %X%\rev3_train.exe train dir=%D% lib=wlib fp=fp3c16k out=rev3_fp3c16k_big.bin epochs=20 H=4096,4096 hold=20 eval=- threads=%T% > %O%\train_rev3_fp3c16k_big.log 2>&1
for %%q in (hcqry hxqry) do (
  %X%\win_eval.exe dir=%D% qry=%%q win=w_%%q models=rev3_fp3c16k_big.bin:fp3c16k,rev3_fp3c16k.bin:fp3c16k ens=1 threads=%T% > %O%\eval_big_%%q.log 2>&1
)
echo big done %time% >> %O%\progress.txt

rem ---- 9. summary
(
echo ==== fp3 overnight summary %date% %time%
for %%f in (%O%\win_hcqry.log %O%\win_hxqry.log %O%\win_haqry.log) do type %%f 2>nul | findstr /c:"truth in window"
type %O%\fp3_count.log | findstr /c:"count wlib"
for %%f in (%O%\fp3_sel_*.log) do type %%f | findstr /c:"vocab"
for %%f in (%O%\train_*.log) do type %%f | findstr /c:"calibration:" /c:"wrote"
for %%f in (%O%\eval_base_hcqry.log %O%\eval_base_hxqry.log %O%\eval_all_hcqry.log %O%\eval_all_hxqry.log %O%\eval_big_hcqry.log %O%\eval_big_hxqry.log) do (
  echo ---- %%f
  type %%f
)
) > %O%\SUMMARY.txt 2>&1
echo all done %date% %time% >> %O%\progress.txt
