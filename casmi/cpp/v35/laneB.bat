@echo off
rem v35 lane B (2 threads each; see v35_STATUS.md)
call C:\prog\casmi\v35\run35.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin vbo_1 2 ",rm=1,pcrev=tan,alloc=boltz,bT1=2,bT2=1,bpi=0.2,ba=0.25"
fc /b C:\prog\casmi\v35\out\vbo_1.csv C:\prog\casmi\pc5\hc33r_1.csv > C:\prog\casmi\v35\out\eq_bo_csv.txt 2>&1
fc /b C:\prog\casmi\v35\out\vbo_1_ins.csv C:\prog\casmi\pc5\hc33r_1_ins.csv > C:\prog\casmi\v35\out\eq_bo_ins.txt 2>&1
call C:\prog\casmi\v35\run35.bat haqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin haap2_1 2 ",rm=1,pcrev=tan,pcfp=rev3_fp3c16k_big.bin,attn=1,pcrank=rev,attpool=2"
call C:\prog\casmi\v35\run35.bat hbsqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hbsap2_0 2 ",pcrev=tan,pcfp=rev3_fp3c16k_big.bin,attn=1,pcrank=rev,attpool=2"
call C:\prog\casmi\v35\run35.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hcap15_1 2 ",rm=1,pcrev=tan,pcfp=rev3_fp3c16k_big.bin,attn=1,pcrank=rev,attpool=1.5"
echo ALLDONE
