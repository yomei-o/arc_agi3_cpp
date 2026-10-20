@echo off
rem lane B: v32 hcqry with the answer in the pool (v30 baseline = pc3\hcT0)
call C:\prog\casmi\fwdmodel\pc4\run4.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hc32_0 phys_v32pc 2 ",pcrev=tan,pcfp=rev3_fp3c16k_big.bin"
echo ALLDONE
