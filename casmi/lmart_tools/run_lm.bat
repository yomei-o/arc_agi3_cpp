@echo off
rem run_lm.bat <qry> <lib> <tag> <rm 0|1> <rev.bin> [lmart model] : phys_lm eval, hand config v20+gw (CSV = hand baseline unless a model is given), features dump
cd /d C:\prog\casmi\fwdmodel\lmart
phys_lm.exe eval C:\prog\casmi %1 pool730 10 phys2_fqry.txt lmo\%3.csv fwd_mlp_rank_xlib.bin 0.25 "" %5 1.0 %2 8 32 24 lmo\f_%3.txt 4 3 8000 "" "" 1 gen=1,cap=300,prior=gen2o/gen2_prior_c.txt,gw=-2:0:-1:4:0:1.5,thr=6,grid=0,rm=%4 %6 > C:\prog\casmi\lmo\%3.log 2>&1
"C:\Program Files\Python312\python.exe" C:\prog\casmi\mrr.py C:\prog\casmi\%1_mols.txt C:\prog\casmi\lmo\%3.csv >> C:\prog\casmi\lmo\%3.log
echo done %3
