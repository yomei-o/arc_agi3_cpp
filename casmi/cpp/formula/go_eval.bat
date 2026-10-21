@echo off
rem go_eval.bat : re-evaluate from the cached features (formula\<qry>_feat_N1000.bin), logs formula\fe2_<qry>.log
cd /d C:\prog\casmi
for %%q in (haqry hcqry hxqry) do formula\formula_eval.exe qry=%%q N=1000 threads=4 "evalN=200,1000" > formula\fe2_%%q.log 2>&1
echo all done > formula\go_eval.txt
