@echo off
rem run.bat <qry> <lib> <mlp> <rev2> <tag> <extra opts> [model] [thr]
rem   phys_v29lm eval in the v28a configuration (fragment-size physics, libcap=1, generation, REV2, hand fusion,
rem   pool popularity ppop=0.15 ppgate=0.9; FPNet off - its public weights leak on train-derived holdouts;
rem   no PubChem) + the v29 ranker dump (lm29\<tag>_f.txt). With [model] (argv[25]) the ranker orders the list.
rem   out: C:\prog\casmi\lm29\<tag>.csv, <tag>.log (+ mrr.py tie-averaged on the CSV). Affinity: cores 0-9 at most.
setlocal
cd /d C:\prog\casmi
set THR=%~8
if "%THR%"=="" set THR=5
set AFF=1F
if "%THR%"=="10" set AFF=3FF
if "%~9"=="hi" set AFF=3E0
start "" /b /wait /affinity %AFF% C:\prog\casmi\fwdmodel\lm29\phys_v29lm.exe eval C:\prog\casmi %~1 pool730 10 phys_size_fqry.txt lm29\%~5.csv %~3 0.25 "" %~4 1.0 %~2 8 32 24 lm29\%~5_f.txt 4 3 8000 "" "" 1 "gen=1,cap=300,prior=gen2o\gen2_prior_c.txt,gw=-2:0:-1:4:0:1.5,thr=%THR%,grid=0,libcap=1,ppop=0.15,ppgate=0.9,ppopf=pc2\pool730_pop.bin%~6" %~7 > lm29\%~5.log 2>&1
"C:\Program Files\Python312\python.exe" C:\prog\casmi\mrr.py C:\prog\casmi\%~1_mols.txt C:\prog\casmi\lm29\%~5.csv >> lm29\%~5.log 2>&1
echo done %~5
