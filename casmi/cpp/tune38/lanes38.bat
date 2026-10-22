@echo off
rem lanes38.bat <lane> : v37 production configuration (holdout-safe stand-ins) with tdump, one lane per call (2 threads)
rem   a = hxqry (zlib), b = hbqry (ylib), c = hbsqry (ylib), d = hasqry + hcqry (ylib), sequential
rem   REV2 = rev2_w.bin (wlib: clean on every h* set), MLP = fwd_mlp_rank_ylib.bin, fp3 = rev3_fp3c16k_big (wlib),
rem   CNN = cnn_a_c.bin (wlib), FT = ft8_ep5.bin (wlib), own FPNet = fpown_w_f16.bin scores (wlib)
setlocal
set R=C:\prog\casmi\tune38\run38.bat
set O=,pcrev=tan,pcfp=rev3_fp3c16k_big.bin,attn=1,pcrank=rev,attpool=2,alloc=boltz,bT1=3,bT2=0.5,bpi=0.1,ba=0.5,cnnmodel=cnn_a_c.bin,cnnpool=2,cnninst=0,ftmodel=ft8_ep5.bin,ftpool=2
if "%~1"=="a" call %R% hxqry zlib fwd_mlp_rank_ylib.bin rev2_w.bin t38x 2 h2o\own_ep5_hxqry_fp.txt "%O%"
if "%~1"=="b" call :waitfp
if "%~1"=="b" call %R% hbqry ylib fwd_mlp_rank_ylib.bin rev2_w.bin t38b 2 tune38\fp_hbqry.txt "%O%"
if "%~1"=="c" call :waitfp
if "%~1"=="c" call %R% hbsqry ylib fwd_mlp_rank_ylib.bin rev2_w.bin t38bs 2 tune38\fp_hbsqry.txt "%O%"
if "%~1"=="d" call %R% hasqry ylib fwd_mlp_rank_ylib.bin rev2_w.bin t38s 2 h2o\own_ep5_hasqry_fp.txt "%O%"
if "%~1"=="d" call %R% hcqry ylib fwd_mlp_rank_ylib.bin rev2_w.bin t38c 2 h2o\own_ep5_hcqry_fp.txt "%O%"
echo ALLDONE %~1
goto :eof
:waitfp
findstr /c:"ALLDONE" C:\prog\casmi\night\t38_fp.log >nul 2>&1 && goto :eof
timeout /t 30 /nobreak >nul
goto waitfp
