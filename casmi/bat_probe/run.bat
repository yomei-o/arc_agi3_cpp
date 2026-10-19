@echo off
rem run.bat <qry> <lib> <mlp> <rev2> <tag> <extra opts>
rem   phys_probe eval, v27a config as bat_pc2\run.bat (FPNet off) + PubChem v2 fused + probe=1:
rem   out C:\prog\casmi\probe\<tag>.csv = non-pool PubChem list (names; truth = stereo-blind canonical key),
rem   <tag>_smi.csv = same list as SMILES (for InChIKey14 scoring), <tag>_plain.csv = normal list. 6 threads.
setlocal
cd /d C:\prog\casmi
C:\prog\casmi\fwdmodel\probe\phys_probe.exe eval C:\prog\casmi %~1 pool730 10 phys_size_fqry.txt probe\%~5.csv %~3 0.25 "" %~4 1.0 %~2 8 32 24 "" 4 3 8000 "" "" 1 "gen=1,cap=300,prior=gen2o\gen2_prior_c.txt,gw=-2:0:-1:4:0:1.5,thr=6,grid=0,libcap=1,ppop=0.15,ppgate=0.9,ppopf=pc2\pool730_pop.bin,pc=1,pcv=2,pcrank=fused,probe=1,probesmi=probe\%~5_smi.csv,pcout=probe\%~5_plain.csv%~6" > probe\%~5.log 2>&1
"C:\Program Files\Python312\python.exe" C:\prog\casmi\mrr.py C:\prog\casmi\%~1_mols.txt C:\prog\casmi\probe\%~5.csv C:\prog\casmi\probe\%~5_plain.csv >> probe\%~5.log 2>&1
"C:\Program Files\Python312\python.exe" C:\prog\casmi\probe\probe_ik_mrr.py C:\prog\casmi\%~1_mols.txt C:\prog\casmi\probe\%~5_smi.csv >> probe\%~5.log 2>&1
echo done %~5
