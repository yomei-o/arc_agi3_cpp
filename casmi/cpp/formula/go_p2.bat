@echo off
rem go_p2.bat : same features with the older phys2_fqry.txt parameters (the main-pool physics term), logs formula\fp2_<qry>.log
cd /d C:\prog\casmi
for %%q in (haqry hcqry hxqry) do formula\formula_eval.exe qry=%%q N=1000 threads=4 "evalN=200,1000" params=phys2_fqry.txt feat=formula\%%q_feat_N1000_p2.bin > formula\fp2_%%q.log 2>&1
echo all done > formula\go_p2.txt
