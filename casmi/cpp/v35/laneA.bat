@echo off
rem v35 lane A (2 threads each; see v35_STATUS.md)
call C:\prog\casmi\v35\run35.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin vdef_1 2 ",rm=1,pcrev=tan,pcfp=rev3_fp3c16k_big.bin"
fc /b C:\prog\casmi\v35\out\vdef_1.csv C:\prog\casmi\pc4\hc32_1.csv > C:\prog\casmi\v35\out\eq_def_csv.txt 2>&1
fc /b C:\prog\casmi\v35\out\vdef_1_ins.csv C:\prog\casmi\pc4\hc32_1_ins.csv > C:\prog\casmi\v35\out\eq_def_ins.txt 2>&1
call C:\prog\casmi\v35\run35.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin vatt_1 2 ",rm=1,pcrev=tan,pcfp=rev3_fp3c16k_big.bin,attn=1,pcrank=rev,attpool=2"
fc /b C:\prog\casmi\v35\out\vatt_1.csv C:\prog\casmi\attn\pc\hc34rp2_1.csv > C:\prog\casmi\v35\out\eq_att_csv.txt 2>&1
fc /b C:\prog\casmi\v35\out\vatt_1_ins.csv C:\prog\casmi\attn\pc\hc34rp2_1_ins.csv > C:\prog\casmi\v35\out\eq_att_ins.txt 2>&1
call C:\prog\casmi\v35\run35.bat hxqry zlib fwd_mlp_rank_ylib.bin rev2_z.bin hxap2_0 2 ",pcrev=tan,pcfp=rev3_fp3c16k_big.bin,attn=1,pcrank=rev,attpool=2"
call C:\prog\casmi\v35\run35.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hcap15_0 2 ",pcrev=tan,pcfp=rev3_fp3c16k_big.bin,attn=1,pcrank=rev,attpool=1.5"
echo ALLDONE
