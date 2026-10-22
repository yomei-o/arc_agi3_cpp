@echo off
rem fpn_watch.bat (launch name m38_fpnw) : evaluates each wlib epoch of fpn12.bat as soon as the next one exists (or wlib is done),
rem   with fpn_eval.bat (fpnet_ggml molecule logits on hxqry/hbqry/hbsqry), then the pool38 grid -> models38\fpn_grid.txt.
rem   Reference = the current wlib model fpown_w.bin (5-epoch run = the recipe of production fpnet_own_f16.bin).
setlocal enabledelayedexpansion
cd /d C:\prog\casmi
set M=C:\prog\casmi\models38
set PR=%M%\fpn12_progress.txt
call %M%\fpn_eval.bat fpown_w >> %M%\fpn_watch_detail.log 2>&1
for /l %%e in (1,1,12) do call :one %%e
call %M%\fpn_grid.bat > %M%\fpn_grid.txt 2>&1
echo watch all done %date% %time% >> %PR%
goto :eof

:one
set /a NX=%1+1
set /a W=0
:w1
if exist C:\prog\casmi\fpown12_w_ep%NX%.bin goto ev
findstr /c:"wlib done" %PR% > nul 2>&1 && goto ev
set /a W+=1
if %W% gtr 200 (echo epoch %1 timeout >> %PR% & goto :eof)
timeout /t 300 /nobreak > nul
goto w1
:ev
if not exist C:\prog\casmi\fpown12_w_ep%1.bin (echo epoch %1 missing >> %PR% & goto :eof)
call %M%\fpn_eval.bat fpown12_w_ep%1 >> %M%\fpn_watch_detail.log 2>&1
echo eval ep%1 done %date% %time% >> %PR%
goto :eof
