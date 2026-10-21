@echo off
rem run5.bat <qry> <lib> <mlp> <rev2> <tag> <exe stem> <threads> <extra opts>
rem   = bat_pc4\run4.bat with outputs in C:\prog\casmi\pc5 (v28b/v30 options fixed: pcrank=fused,relhi=6,relp=3,sp=6,ppop=0.15,ppgate=0.9).
rem   exe in C:\prog\casmi\fwdmodel\pc2 (phys_v30pc / phys_v32pc / phys_v33pc).
rem   out: C:\prog\casmi\pc5\<tag>.csv (plain), <tag>_ins.csv (PubChem inserted), _det/_pp dumps, <tag>.log + mrr.py
setlocal
cd /d C:\prog\casmi
C:\prog\casmi\fwdmodel\pc2\%~6.exe eval C:\prog\casmi %~1 pool730 10 phys_size_fqry.txt pc5\%~5.csv %~3 0.25 "" %~4 1.0 %~2 8 32 24 "" 4 3 8000 "" "" 1 "gen=1,cap=300,prior=gen2o\gen2_prior_c.txt,gw=-2:0:-1:4:0:1.5,thr=%~7,grid=0,libcap=1,ppopf=pc2\pool730_pop.bin,ppdet=pc5\%~5_pp.txt,pc=1,pcv=2,pcout=pc5\%~5_ins.csv,pcdet=pc5\%~5_det.txt,pcrank=fused,relhi=6,relp=3,sp=6,ppop=0.15,ppgate=0.9%~8" > pc5\%~5.log 2>&1
"C:\Program Files\Python312\python.exe" C:\prog\casmi\mrr.py C:\prog\casmi\%~1_mols.txt C:\prog\casmi\pc5\%~5.csv C:\prog\casmi\pc5\%~5_ins.csv >> pc5\%~5.log 2>&1
echo done %~5
