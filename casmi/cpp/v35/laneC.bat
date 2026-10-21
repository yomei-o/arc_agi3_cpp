@echo off
rem v35 lane C (2 threads each; see v35_STATUS.md)
call C:\prog\casmi\v35\run35.bat hasqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hasap15_0 2 ",pcrev=tan,pcfp=rev3_fp3c16k_big.bin,attn=1,pcrank=rev,attpool=1.5"
call C:\prog\casmi\v35\run35.bat hasqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hasap25_0 2 ",pcrev=tan,pcfp=rev3_fp3c16k_big.bin,attn=1,pcrank=rev,attpool=2.5"
call C:\prog\casmi\v35\run35.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hcap25_0 2 ",pcrev=tan,pcfp=rev3_fp3c16k_big.bin,attn=1,pcrank=rev,attpool=2.5"
call C:\prog\casmi\v35\run35.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hcap25_1 2 ",rm=1,pcrev=tan,pcfp=rev3_fp3c16k_big.bin,attn=1,pcrank=rev,attpool=2.5"
echo ALLDONE
