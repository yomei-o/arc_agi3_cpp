@echo off
rem run_all.bat <N> <evalN list, quoted> : win_dump (if missing) + formula_eval for haqry, hcqry, hxqry (4 threads, CPU)
rem   logs C:\prog\casmi\formula\fe_<qry>_N<N>.log ; progress in formula\run_all.txt
setlocal
cd /d C:\prog\casmi
for %%q in (haqry hcqry hxqry) do (
  if not exist formula\%%q_top.bin formula\win_dump.exe qry=%%q N=1000 threads=4 > formula\dump_%%q.log 2>&1
  echo start %%q %TIME% >> formula\run_all.txt
  formula\formula_eval.exe qry=%%q N=%~1 threads=4 "evalN=%~2" > formula\fe_%%q_N%~1.log 2>&1
  echo done %%q %TIME% >> formula\run_all.txt
)
echo all done >> formula\run_all.txt
