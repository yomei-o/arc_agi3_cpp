@echo off
rem pc_t.bat <qry> <lib> <tag> <model or ""> : phys_pclm with the PubChem channel (score mode defaults), plain CSV + inserted CSV
cd /d C:\prog\casmi\fwdmodel\lmart
phys_pclm.exe eval C:\prog\casmi %1 pool730 10 phys2_fqry.txt lmo\%3.csv fwd_mlp_rank_xlib.bin 0.25 "" rev2_vh_xs.bin 1.0 %2 8 32 24 "" 4 3 8000 "" "" 1 "gen=1,cap=300,prior=gen2o/gen2_prior_c.txt,gw=-2:0:-1:4:0:1.5,thr=6,grid=0,pc=1,pcdir=C:/prog/casmi,pcout=lmo/%3_ins.csv" %4 > C:\prog\casmi\lmo\%3.log 2>&1
"C:\Program Files\Python312\python.exe" C:\prog\casmi\mrr.py C:\prog\casmi\%1_mols.txt C:\prog\casmi\lmo\%3.csv C:\prog\casmi\lmo\%3_ins.csv >> C:\prog\casmi\lmo\%3.log
echo done %3
