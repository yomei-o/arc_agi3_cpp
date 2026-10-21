@echo off
rem v30dump.bat <qry> <lib> <mlp> <rev2> <tag> <threads> <extra opts>
rem   = bat_pc4\run4.bat (v30 = phys_v30pc, v28b options + pcrev=tan) with the per-candidate feature dump
rem   (argv[17], all pool + generated candidates with SMILES) so ICEBERG/GLACIER can score the final lists.
rem   out: C:\prog\casmi\iceberg\v30\<tag>.csv (plain), <tag>_ins.csv (final), _det/_pp dumps, f_<tag>.txt, <tag>.log
setlocal
cd /d C:\prog\casmi
C:\prog\casmi\fwdmodel\pc2\phys_v30pc.exe eval C:\prog\casmi %~1 pool730 10 phys_size_fqry.txt iceberg\v30\%~5.csv %~3 0.25 "" %~4 1.0 %~2 8 32 24 iceberg\v30\f_%~5.txt 4 3 8000 "" "" 1 "gen=1,cap=300,prior=gen2o\gen2_prior_c.txt,gw=-2:0:-1:4:0:1.5,thr=%~6,grid=0,libcap=1,ppopf=pc2\pool730_pop.bin,ppdet=iceberg\v30\%~5_pp.txt,pc=1,pcv=2,pcout=iceberg\v30\%~5_ins.csv,pcdet=iceberg\v30\%~5_det.txt,pcrank=fused,relhi=6,relp=3,sp=6,ppop=0.15,ppgate=0.9,pcrev=tan%~7" > iceberg\v30\%~5.log 2>&1
"C:\Program Files\Python312\python.exe" C:\prog\casmi\mrr.py C:\prog\casmi\%~1_mols.txt C:\prog\casmi\iceberg\v30\%~5.csv C:\prog\casmi\iceberg\v30\%~5_ins.csv >> iceberg\v30\%~5.log 2>&1
echo done %~5
