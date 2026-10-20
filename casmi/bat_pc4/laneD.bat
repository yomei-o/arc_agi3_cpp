@echo off
rem lane D: v32 (pcfp = rev3_fp3c16k_big) on hasqry
call C:\prog\casmi\fwdmodel\pc4\run4.bat hasqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin has32_0 phys_v32pc 2 ",pcrev=tan,pcfp=rev3_fp3c16k_big.bin"
echo ALLDONE
