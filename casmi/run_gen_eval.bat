@echo off
rem run_gen_eval.bat <q> <gcfile> <tag>   (q = tqry/cqry)
cd /d C:\prog\casmi
"C:\Program Files\Python312\python.exe" fwdmodel\gen\build_genpool.py C:\prog\casmi pool730 %1_mols.txt fwdmodel\gen\%2 pg_%3A 0
"C:\Program Files\Python312\python.exe" fwdmodel\gen\build_genpool.py C:\prog\casmi pool730 %1_mols.txt fwdmodel\gen\%2 pg_%3B 1
C:\prog\casmi\fwdmodel\ana\build\phys_ana.exe eval C:\prog\casmi %1 pg_%3A 10 phys2_fqry.txt "" fwd_mlp_rank_xlib.bin 0 "" rev_x3072.bin 0 xlib 0 32 0 gen_f_%3A.txt 4 3 8000 > fwdmodel\gen\pa_%3A.log 2>&1
C:\prog\casmi\fwdmodel\ana\build\phys_ana.exe eval C:\prog\casmi %1 pg_%3B 10 phys2_fqry.txt "" fwd_mlp_rank_xlib.bin 0 "" rev_x3072.bin 0 xlib 0 32 0 gen_f_%3B.txt 4 3 8000 > fwdmodel\gen\pa_%3B.log 2>&1
echo finished %3
