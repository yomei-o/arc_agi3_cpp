@echo off
rem go_pop.bat : popularity grid from the cached features, logs formula\fpop_<qry>.log (d is vs base + 0.15 pop)
cd /d C:\prog\casmi
for %%q in (haqry hcqry hxqry) do formula\formula_eval.exe qry=%%q N=1000 threads=4 "evalN=200,1000" popgrid=1 > formula\fpop_%%q.log 2>&1
echo all done
