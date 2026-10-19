@echo off
rem holdout2: reproduce one LB version on one holdout (FPNet off everywhere; it leaks on train-derived holdouts)
rem h2_v.bat <qry> <ver v20|v21|v23|v24|v25|v26> <rm 0|1> <lib> <mlp> <rev1> <rev2> <tag>
rem   v20 phys_ana hand fusion (rev v1);  v21 = v20 + fuse_w21.txt learned weights
rem   v23 phys_pclm gen off, REV2;  v24 + generation;  v25 + PubChem channel;  v26 + LambdaMART lmart_prod.txt
rem outputs C:\prog\casmi\h2o\<tag>.csv/.log, mrr.py (tie-averaged) appended to the log; 12 cores (affinity FFF)
setlocal
cd /d C:\prog\casmi
set Q=%~1
set V=%~2
set RM=%~3
set L=%~4
set M=%~5
set R1=%~6
set R2=%~7
set T=%~8
set A=C:\prog\casmi\fwdmodel\ana\build\phys_ana.exe
set P=C:\prog\casmi\fwdmodel\lmart\phys_pclm.exe
set G=gen=1,cap=300,prior=gen2o/gen2_prior_c.txt,gw=-2:0:-1:4:0:1.5,thr=12,grid=0,rm=%RM%
if "%V%"=="v20" start "" /b /wait /affinity FFF %A% eval C:\prog\casmi %Q% pool730 10 phys2_fqry.txt h2o\%T%.csv %M% 0.25 "" %R1% 1.0 %L% 8 32 24 "" 4 3 8000 > h2o\%T%.log 2>&1
if "%V%"=="v21" start "" /b /wait /affinity FFF %A% eval C:\prog\casmi %Q% pool730 10 phys2_fqry.txt h2o\%T%.csv %M% 0.25 "" %R1% 1.0 %L% 8 32 24 "" 4 3 8000 fuse_w21.txt > h2o\%T%.log 2>&1
if "%V%"=="v23" start "" /b /wait /affinity FFF %P% eval C:\prog\casmi %Q% pool730 10 phys2_fqry.txt h2o\%T%.csv %M% 0.25 "" %R2% 1.0 %L% 8 32 24 "" 4 3 8000 "" "" 0 "thr=12,grid=0,rm=%RM%" > h2o\%T%.log 2>&1
if "%V%"=="v24" start "" /b /wait /affinity FFF %P% eval C:\prog\casmi %Q% pool730 10 phys2_fqry.txt h2o\%T%.csv %M% 0.25 "" %R2% 1.0 %L% 8 32 24 "" 4 3 8000 "" "" 0 "%G%" > h2o\%T%.log 2>&1
if "%V%"=="v25" start "" /b /wait /affinity FFF %P% eval C:\prog\casmi %Q% pool730 10 phys2_fqry.txt h2o\%T%.csv %M% 0.25 "" %R2% 1.0 %L% 8 32 24 "" 4 3 8000 "" "" 0 "%G%,pc=1" > h2o\%T%.log 2>&1
if "%V%"=="v26" start "" /b /wait /affinity FFF %P% eval C:\prog\casmi %Q% pool730 10 phys2_fqry.txt h2o\%T%.csv %M% 0.25 "" %R2% 1.0 %L% 8 32 24 "" 4 3 8000 "" "" 0 "%G%,pc=1" lmart_prod.txt > h2o\%T%.log 2>&1
"C:\Program Files\Python312\python.exe" C:\prog\casmi\mrr.py C:\prog\casmi\%Q%_mols.txt C:\prog\casmi\h2o\%T%.csv >> h2o\%T%.log 2>&1
echo done %T%
