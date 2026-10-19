@echo off
rem pp_run.bat <exe> <qry> <lib> <mlp> <rev2> <tag> "<extra opts>"
rem v27a configuration (phys_size params, libcap=1, generation; no FPNet - it leaks on train-derived holdouts)
rem with the per-candidate feature dump pp\f_<tag>.txt; outputs C:\prog\casmi\pp\<tag>.csv/.log; 8 cores (affinity FF000)
setlocal
cd /d C:\prog\casmi
set E=%~1
set Q=%~2
set L=%~3
set M=%~4
set R=%~5
set T=%~6
set X=%~7
start "" /b /wait /affinity FF000 %E% eval C:\prog\casmi %Q% pool730 10 phys_size_fqry.txt pp\%T%.csv %M% 0.25 "" %R% 1.0 %L% 8 32 24 pp\f_%T%.txt 4 3 8000 "" "" 1 "gen=1,cap=300,prior=gen2o\gen2_prior_c.txt,gw=-2:0:-1:4:0:1.5,thr=8,grid=0,libcap=1%X%" > pp\%T%.log 2>&1
"C:\Program Files\Python312\python.exe" C:\prog\casmi\mrr.py C:\prog\casmi\%Q%_mols.txt C:\prog\casmi\pp\%T%.csv >> pp\%T%.log 2>&1
echo done %T%
