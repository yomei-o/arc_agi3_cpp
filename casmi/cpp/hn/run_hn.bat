@echo off
rem run_hn.bat <tag> <lib32 w> <analog w> <fpnet scores (rel. to C:\prog\casmi)> <fpnet w> <extra opts (leading comma)>
rem   = tune38\run38.bat on hnqry with the nlib holdout-safe models (hn_STATUS.md), v36 production configuration (no FT),
rem   6 threads. Outputs C:\prog\casmi\hn\out\<tag>.csv (pool list), <tag>_ins.csv (final), <tag>.log (+ mrr.py scores).
setlocal
cd /d C:\prog\casmi
if not exist hn\out mkdir hn\out
set O=,pcrev=tan,pcfp=rev3_n_fp3c16k_big.bin,attn=1,pcrank=rev,attpool=2,alloc=boltz,bT1=3,bT2=0.5,bpi=0.1,ba=0.5,cnnmodel=cnn_n_c.bin,cnnpool=2,cnninst=0
C:\prog\casmi\tune38\phys_v38td.exe eval C:\prog\casmi hnqry pool730 10 phys_size_fqry.txt hn\out\%~1.csv fwd_mlp_rank_nlib.bin 0.25 "" rev2_n.bin 1.0 nlib %~2 32 %~3 "" 4 3 8000 "" %~4 %~5 "gen=1,cap=300,prior=gen2o\gen2_prior_c.txt,gw=-2:0:-1:4:0:1.5,thr=6,grid=0,libcap=1,ppdet=hn\out\%~1_pp.txt,pc=1,pcv=2,pcout=hn\out\%~1_ins.csv,pcdet=hn\out\%~1_det.txt,pcrank=fused,relhi=6,relp=3,sp=6,ppgate=0.9,tdump=hn\out\%~1_td.txt%O%%~6" > hn\out\%~1.log 2>&1
"C:\Program Files\Python312\python.exe" C:\prog\casmi\mrr.py C:\prog\casmi\hnqry_mols.txt C:\prog\casmi\hn\out\%~1.csv C:\prog\casmi\hn\out\%~1_ins.csv >> hn\out\%~1.log 2>&1
echo done %~1
