@echo off
rem run.bat <qry> <lib> <mlp> <rev2> <tag> <extra opts>
rem   phys_v28pc eval in the v27a configuration (fragment-size physics params, libcap, generation, REV2, hand fusion;
rem   FPNet off - its public weights leak on train-derived holdouts) + PubChem channel v2 + pool popularity dumps.
rem   out: C:\prog\casmi\pc2\<tag>.csv (plain = no PubChem insertion, ppop as in opts), <tag>_ins.csv (inserted),
rem   <tag>_det.txt (pcdet), <tag>_pp.txt (ppdet), <tag>.log with mrr.py (tie-averaged) of both CSVs. 6 threads.
setlocal
cd /d C:\prog\casmi
C:\prog\casmi\fwdmodel\pc2\phys_v28pc.exe eval C:\prog\casmi %~1 pool730 10 phys_size_fqry.txt pc2\%~5.csv %~3 0.25 "" %~4 1.0 %~2 8 32 24 "" 4 3 8000 "" "" 1 "gen=1,cap=300,prior=gen2o\gen2_prior_c.txt,gw=-2:0:-1:4:0:1.5,thr=6,grid=0,libcap=1,ppopf=pc2\pool730_pop.bin,ppdet=pc2\%~5_pp.txt,pc=1,pcv=2,pcout=pc2\%~5_ins.csv,pcdet=pc2\%~5_det.txt%~6" > pc2\%~5.log 2>&1
"C:\Program Files\Python312\python.exe" C:\prog\casmi\mrr.py C:\prog\casmi\%~1_mols.txt C:\prog\casmi\pc2\%~5.csv C:\prog\casmi\pc2\%~5_ins.csv >> pc2\%~5.log 2>&1
echo done %~5
