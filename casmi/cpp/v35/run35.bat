@echo off
rem run35.bat <qry> <lib> <mlp> <rev2> <tag> <threads> <extra opts>
rem   = cpp/attn/run34.bat (same fixed v28b options, pcrank=fused unless overridden) with phys_v35pc.exe and
rem   outputs in C:\prog\casmi\v35\out\<tag>.csv (plain), <tag>_ins.csv (inserted / merged), _det/_pp dumps, <tag>.log + mrr.py
setlocal
cd /d C:\prog\casmi
C:\prog\casmi\v35\phys_v35pc.exe eval C:\prog\casmi %~1 pool730 10 phys_size_fqry.txt v35\out\%~5.csv %~3 0.25 "" %~4 1.0 %~2 8 32 24 "" 4 3 8000 "" "" 1 "gen=1,cap=300,prior=gen2o\gen2_prior_c.txt,gw=-2:0:-1:4:0:1.5,thr=%~6,grid=0,libcap=1,ppopf=pc2\pool730_pop.bin,ppdet=v35\out\%~5_pp.txt,pc=1,pcv=2,pcout=v35\out\%~5_ins.csv,pcdet=v35\out\%~5_det.txt,pcrank=fused,relhi=6,relp=3,sp=6,ppop=0.15,ppgate=0.9%~7" > v35\out\%~5.log 2>&1
"C:\Program Files\Python312\python.exe" C:\prog\casmi\mrr.py C:\prog\casmi\%~1_mols.txt C:\prog\casmi\v35\out\%~5.csv C:\prog\casmi\v35\out\%~5_ins.csv >> v35\out\%~5.log 2>&1
echo done %~5
