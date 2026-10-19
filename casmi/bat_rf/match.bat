@echo off
rem match.bat <qry> <lib> : the notebook's match.cpp (library search, same args as cell 4 but 40 deep)
rem   in its own directory C:\prog\casmi\rf\m_<qry> (hard links to the bins; match writes submission.csv there)
rem   out: C:\prog\casmi\rf\lib_<qry>.csv (SMILES), lib_<qry>_best.csv, lib_<qry>_tight.csv
setlocal
set D=C:\prog\casmi\rf\m_%~1
if not exist %D% mkdir %D%
for %%s in (head.bin peaks.bin mols.txt) do (
  if not exist %D%\%~2_%%s mklink /H %D%\%~2_%%s C:\prog\casmi\%~2_%%s >nul
  if not exist %D%\%~1_%%s mklink /H %D%\%~1_%%s C:\prog\casmi\%~1_%%s >nul
)
start "" /b /wait /affinity 3 C:\prog\casmi\rf\match.exe %D% 0.05 0.02 40 %~2 %~1 3.0 0.1 1 1 0.25 > C:\prog\casmi\rf\lib_%~1.log 2>&1
copy /y %D%\submission.csv C:\prog\casmi\rf\lib_%~1.csv >nul
copy /y %D%\best_score.csv C:\prog\casmi\rf\lib_%~1_best.csv >nul
copy /y %D%\tight_count.csv C:\prog\casmi\rf\lib_%~1_tight.csv >nul
echo done match %~1
