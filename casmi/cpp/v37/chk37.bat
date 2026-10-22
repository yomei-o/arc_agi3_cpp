@echo off
rem chk37.bat : (a) default = phys_v36pc (same options as cpp/v36/chkb.bat, compare v37\out\c37b_cnn2* with v36\out\p36b_cnn2* by fc /b)
rem   (b) the pool37ft.py point: hbqry, cnnpool 2 + ftpool 2 with ft8_ep5.bin (holdout-safe), simulated final 0.8100
set O=,pcrev=tan,pcfp=rev3_fp3c16k_big.bin,attn=1,pcrank=rev,attpool=2,alloc=boltz,bT1=2,bT2=0.5,bpi=0.2,ba=0.5,cnnmodel=cnn_a_c.bin,cnnpool=2
if "%~1"=="a" call C:\prog\casmi\v37\run37.bat hbqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin c37b_cnn2 3 "%O%"
if "%~1"=="b" call C:\prog\casmi\v37\run37.bat hbqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin c37b_ft2 3 "%O%,ftmodel=ft8_ep5.bin,ftpool=2,ftdump=v37\out\c37b_ft2_mol.bin,cfdump=v37\out\c37b_ft2_cf.txt"
echo ALLDONE %~1
