@echo off
rem run38.bat <qry> <lib> <mlp> <rev2> <tag> <threads> <fpnet scores (rel. to C:\prog\casmi)> <extra opts>
rem   = v37/run37.bat with phys_v38td.exe (phys_v37pc + tdump), the own-FPNet scores as argv[22] (+ 1.0 z, as the notebook),
rem   outputs C:\prog\casmi\tune38\out\<tag>.csv (pool list), <tag>_ins.csv (final), _det/_pp/_td dumps, <tag>.log
setlocal
cd /d C:\prog\casmi
C:\prog\casmi\tune38\phys_v38td.exe eval C:\prog\casmi %~1 pool730 10 phys_size_fqry.txt tune38\out\%~5.csv %~3 0.25 "" %~4 1.0 %~2 8 32 24 "" 4 3 8000 "" %~7 1 "gen=1,cap=300,prior=gen2o\gen2_prior_c.txt,gw=-2:0:-1:4:0:1.5,thr=%~6,grid=0,libcap=1,ppopf=pc2\pool730_pop.bin,ppdet=tune38\out\%~5_pp.txt,pc=1,pcv=2,pcout=tune38\out\%~5_ins.csv,pcdet=tune38\out\%~5_det.txt,pcrank=fused,relhi=6,relp=3,sp=6,ppop=0.15,ppgate=0.9,tdump=tune38\out\%~5_td.txt%~8" > tune38\out\%~5.log 2>&1
"C:\Program Files\Python312\python.exe" C:\prog\casmi\mrr.py C:\prog\casmi\%~1_mols.txt C:\prog\casmi\tune38\out\%~5.csv C:\prog\casmi\tune38\out\%~5_ins.csv >> tune38\out\%~5.log 2>&1
echo done %~5
