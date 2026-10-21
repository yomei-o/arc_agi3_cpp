@echo off
rem run34.bat <qry> <lib> <mlp> <rev2> <tag> <threads> <extra opts>
rem   = casmi/bat_pc4/run4.bat (same fixed v28b options) with phys_v34pc.exe (C:\prog\casmi\attn\v34) and
rem   outputs in C:\prog\casmi\attn\pc\<tag>.csv (plain), <tag>_ins.csv (PubChem inserted), _det/_pp dumps, <tag>.log + mrr.py
setlocal
cd /d C:\prog\casmi
C:\prog\casmi\attn\v34\phys_v34pc.exe eval C:\prog\casmi %~1 pool730 10 phys_size_fqry.txt attn\pc\%~5.csv %~3 0.25 "" %~4 1.0 %~2 8 32 24 "" 4 3 8000 "" "" 1 "gen=1,cap=300,prior=gen2o\gen2_prior_c.txt,gw=-2:0:-1:4:0:1.5,thr=%~6,grid=0,libcap=1,ppopf=pc2\pool730_pop.bin,ppdet=attn\pc\%~5_pp.txt,pc=1,pcv=2,pcout=attn\pc\%~5_ins.csv,pcdet=attn\pc\%~5_det.txt,pcrank=fused,relhi=6,relp=3,sp=6,ppop=0.15,ppgate=0.9%~7" > attn\pc\%~5.log 2>&1
"C:\Program Files\Python312\python.exe" C:\prog\casmi\mrr.py C:\prog\casmi\%~1_mols.txt C:\prog\casmi\attn\pc\%~5.csv C:\prog\casmi\attn\pc\%~5_ins.csv >> attn\pc\%~5.log 2>&1
echo done %~5
