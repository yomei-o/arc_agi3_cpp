@echo off
rem gs38.bat <qry> <lib> <mlp> <rev2> <tag> <cnnmodel> : one phys_v36gs run (= pretrain\gs3.bat options, 2 threads) with another
rem   CNN file in cnnmodel= -> every-candidate dump C:\prog\casmi\models38\out\<tag>_cf.txt whose cnn column is that model
rem   (models38_STATUS.md). Everything else identical to the p37 dumps (pool36 lanes + gsdump).
setlocal
cd /d C:\prog\casmi
set O=,pcrev=tan,pcfp=rev3_fp3c16k_big.bin,attn=1,pcrank=rev,attpool=2,alloc=boltz,bT1=2,bT2=0.5,bpi=0.2,ba=0.5,cnnmodel=%~6
C:\prog\casmi\pretrain\p37\phys_v36gs.exe eval C:\prog\casmi %~1 pool730 10 phys_size_fqry.txt models38\out\%~5.csv %~3 0.25 "" %~4 1.0 %~2 8 32 24 "" 4 3 8000 "" "" 1 "gen=1,cap=300,prior=gen2o\gen2_prior_c.txt,gw=-2:0:-1:4:0:1.5,thr=2,grid=0,libcap=1,ppopf=pc2\pool730_pop.bin,ppdet=models38\out\%~5_pp.txt,pc=1,pcv=2,pcout=models38\out\%~5_ins.csv,pcdet=models38\out\%~5_det.txt,pcrank=fused,relhi=6,relp=3,sp=6,ppop=0.15,ppgate=0.9%O%,cfdump=models38\out\%~5_cf.txt" > models38\out\%~5.log 2>&1
echo done %~5
