@echo off
rem run.bat <qry> <lib> <mlp> <rev2> <tag> <extra opts>
rem   phys_v29rf eval in the v28a configuration (v27a + pool popularity 0.15 gated at library sim 0.9;
rem   FPNet off - its public weights leak on train-derived holdouts), names not SMILES, 40 deep,
rem   + the single-engine lists rf\<tag>_{rev,phys,mlp,ana}.csv. 6 threads.
rem   out: C:\prog\casmi\rf\<tag>.csv + .log (mrr.py of every list appended)
setlocal
cd /d C:\prog\casmi
C:\prog\casmi\fwdmodel\rf\phys_v29rf.exe eval C:\prog\casmi %~1 pool730 10 phys_size_fqry.txt rf\%~5.csv %~3 0.25 "" %~4 1.0 %~2 8 32 24 "" 4 3 8000 "" "" 1 "gen=1,cap=300,prior=gen2o\gen2_prior_c.txt,gw=-2:0:-1:4:0:1.5,thr=6,grid=0,libcap=1,ppop=0.15,ppgate=0.9,ppopf=pc2\pool730_pop.bin,outn=40,alt=rf\%~5%~6" > rf\%~5.log 2>&1
"C:\Program Files\Python312\python.exe" C:\prog\casmi\mrr.py C:\prog\casmi\%~1_mols.txt C:\prog\casmi\rf\%~5.csv C:\prog\casmi\rf\%~5_rev.csv C:\prog\casmi\rf\%~5_phys.csv C:\prog\casmi\rf\%~5_mlp.csv C:\prog\casmi\rf\%~5_ana.csv >> rf\%~5.log 2>&1
echo done %~5
