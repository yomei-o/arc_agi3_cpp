@echo off
rem lane A: v32 (pcfp = rev3_fp3c16k_big, wlib = clean for hcqry) hcqry answer-removed; then v32 WITHOUT pcfp on the same
rem (must equal pc3\hcT1.csv / hcT1_ins.csv = phys_v30pc pcrev=tan rm=1)
call C:\prog\casmi\fwdmodel\pc4\run4.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hc32_1 phys_v32pc 2 ",rm=1,pcrev=tan,pcfp=rev3_fp3c16k_big.bin"
call C:\prog\casmi\fwdmodel\pc4\run4.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hc32nofp_1 phys_v32pc 2 ",rm=1,pcrev=tan"
fc /b C:\prog\casmi\pc4\hc32nofp_1.csv C:\prog\casmi\pc3\hcT1.csv > C:\prog\casmi\pc4\eq_csv.txt 2>&1
fc /b C:\prog\casmi\pc4\hc32nofp_1_ins.csv C:\prog\casmi\pc3\hcT1_ins.csv > C:\prog\casmi\pc4\eq_ins.txt 2>&1
echo ALLDONE
